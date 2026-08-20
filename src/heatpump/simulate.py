"""Closed-loop transients and a small CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from heatpump.control import LinearMPC, NonlinearMPC, make_cascade, make_mpc
from heatpump.design import (
    DesignReport,
    SystemDesign,
    design_air_conditioner,
    design_heat_pump,
    design_system,
)
from heatpump.plant import (
    PlantSpec,
    apply_operating_mode,
    diagnostics,
    initial_state,
    make_rhs,
    project_state,
    remap_state,
)
from heatpump.requirements import DesignRequest, TimeSeries
from heatpump.solver import TRBDF2, integrate, integrate_qss
from heatpump.thermo import build_tables, list_refrigerants, resolve_fluid


@dataclass
class SimResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    meas: dict[str, np.ndarray]
    spec: PlantSpec
    design: DesignReport | SystemDesign | None = None
    mode: str = "heating"


def _sh_fn(spec, tables):
    def sh(y):
        d = diagnostics(spec, tables, y, jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, 0.0]))
        return d["SH"]

    return sh


def make_controller(name: str, spec: PlantSpec, tables, rhs, Tsp: float, constraints=None):
    name = name.lower()
    mode = spec.operating_mode
    if name in ("pid", "hysteresis", "thermostat", "bangbang", "bang-bang"):
        kind = "hysteresis" if name in ("hysteresis", "thermostat") else ("bangbang" if "bang" in name else "pid")
        return make_cascade(mode, kind, Tsp, spec, constraints)
    project = lambda y: project_state(y, tables, spec.layout)
    if name in ("mpc", "lmpc", "nmpc", "nonlinear-mpc"):
        return make_mpc(
            rhs,
            project,
            spec.layout.i_tz,
            _sh_fn(spec, tables),
            Tsp,
            nonlinear=name in ("nmpc", "nonlinear-mpc"),
            mode=mode,
        )
    raise ValueError(f"unknown controller {name!r}")


def _q_design(design, operate: str) -> float:
    if design is None:
        return 0.0
    if isinstance(design, DesignReport):
        return float(design.Q_load)
    heat = getattr(design, "heating", None)
    cool = getattr(design, "cooling", None)
    if operate == "cooling" and cool is not None:
        return float(cool.Q_load)
    if operate == "heating" and heat is not None:
        return float(heat.Q_load)
    if heat is not None:
        return float(heat.Q_load)
    if cool is not None:
        return float(cool.Q_load)
    return 0.0


def _mode_segments(
    t_final: float,
    timeseries: TimeSeries | None,
    spec: PlantSpec,
    default_mode: str,
) -> list[tuple[float, float, str]]:
    """Split the horizon at reversing-valve changes.

    Uses the time-series ``mode`` column when present (1=heating, 0=cooling).
    On a reversible unit without that column, infer from ``Q_gain`` with a
    200 W deadband and a 5 min minimum dwell (no chatter).
    """
    reversible = spec.indoor is not None and spec.outdoor is not None
    if not reversible or timeseries is None:
        return [(0.0, float(t_final), default_mode)]

    def as_mode(v: float) -> str:
        return "heating" if v >= 0.5 else "cooling"

    knots: list[tuple[float, str]] = []
    if timeseries.mode is not None:
        t = np.asarray(timeseries.t, dtype=float)
        m = np.asarray(timeseries.mode, dtype=float)
        knots.append((0.0, as_mode(float(np.interp(0.0, t, m)))))
        for i in range(1, t.size):
            if as_mode(float(m[i])) != as_mode(float(m[i - 1])) and 0.0 < t[i] < t_final:
                knots.append((float(t[i]), as_mode(float(m[i]))))
    else:
        dead, dwell = 200.0, 300.0
        t_grid = np.unique(
            np.concatenate(
                [np.asarray(timeseries.t, dtype=float), np.arange(0.0, t_final + 1.0, 60.0)]
            )
        )
        t_grid = t_grid[(t_grid >= 0.0) & (t_grid <= t_final)]
        cur = default_mode
        last = -1.0e9
        knots.append((0.0, cur))
        for tt in t_grid:
            Q = float(np.interp(tt, timeseries.t, timeseries.Q_gain))
            want = "cooling" if Q > dead else ("heating" if Q < -dead else cur)
            if want != cur and (tt - last) >= dwell:
                knots.append((float(tt), want))
                cur, last = want, float(tt)

    knots = [(t, md) for t, md in knots if t < t_final - 1e-9]
    if not knots:
        return [(0.0, float(t_final), default_mode)]
    if knots[0][0] > 0.0:
        knots.insert(0, (0.0, default_mode))
    segs: list[tuple[float, float, str]] = []
    for i, (t0, md) in enumerate(knots):
        t1 = knots[i + 1][0] if i + 1 < len(knots) else float(t_final)
        if t1 > t0 + 1e-9:
            segs.append((t0, t1, md))
    return segs or [(0.0, float(t_final), default_mode)]


def _run_span(
    rhs,
    y0,
    u_abs,
    duration: float,
    rec: float,
    kind: str,
    spec: PlantSpec,
    max_steps,
    last: dict,
    t_offset: float,
):
    """Integrate ``duration`` seconds. ``u_abs`` is called with absolute time."""

    def u_loc(t_loc: float):
        return u_abs(t_loc + t_offset)

    def _on(_t, y):
        last["y"] = y

    project = lambda y: project_state(y, last["tables"], spec.layout)
    if kind == "qss":
        warm = min(180.0, duration)
        t, Y = integrate(
            rhs,
            y0,
            u_loc,
            t_final=warm,
            dt0=0.25,
            project=project,
            solver=TRBDF2(),
            record_dt=min(rec, 5.0),
            max_steps=max_steps,
            on_accept=_on,
        )
        if duration > warm + 1e-9:
            refresh = rec if duration < 7200.0 else (
                min(600.0, max(rec, 30.0)) if duration >= 86400.0 else min(120.0, max(rec, 30.0))
            )
            t_q, Y_q = integrate_qss(
                rhs,
                jnp.asarray(Y[-1]),
                lambda t_loc: u_abs(t_loc + t_offset + warm),
                duration - warm,
                i_tz=spec.layout.i_tz,
                slow_idx=spec.layout.slow_idx(),
                project=project,
                record_dt=rec,
                refresh_s=refresh,
                on_accept=_on,
            )
            t = np.concatenate([t, t_q[1:] + warm])
            Y = np.vstack([Y, Y_q[1:]])
    else:
        t, Y = integrate(
            rhs,
            y0,
            u_loc,
            t_final=duration,
            dt0=0.25,
            project=project,
            solver=TRBDF2(),
            record_dt=rec,
            max_steps=max_steps,
            on_accept=_on,
        )
    return np.asarray(t) + t_offset, np.asarray(Y)


def simulate(
    controller: str | object = "pid",
    t_final: float = 600.0,
    record_dt: float = 2.0,
    T_out: float = 273.15,
    T_zone0: float | None = None,
    Tsp: float = 293.15,
    Q_load: float = 0.0,
    refrigerant: str | None = None,
    Q_design: float | None = None,
    mode: str | None = None,
    cooling_tons: float | None = None,
    spec: PlantSpec | None = None,
    tables=None,
    design: DesignReport | SystemDesign | None = None,
    request: DesignRequest | None = None,
    timeseries: TimeSeries | None = None,
    max_steps: int | None = None,
    reduction: str = "auto",
) -> SimResult:
    """Integrate a closed-loop heating or cooling transient.

    ``Q_load`` / timeseries ``Q_gain`` is heat into the zone [W] on top of
    ``UA_env (T_out - T_z)``. If ``spec`` is omitted the plant is sized from
    ``request`` or from ``refrigerant`` + ``Q_design`` / ``cooling_tons``.

    ``reduction``:
        * ``full`` — finite-volume DAE for the whole horizon
        * ``qss`` — short DAE warmup, then zone ODE + periodic cycle relax
          (hours to days)
        * ``auto`` — ``qss`` when ``t_final >= 3600`` s, else ``full``
    """
    if request is not None:
        timeseries = timeseries or request.timeseries
        design = design or design_system(request)
        spec = spec or design.spec
        if isinstance(controller, str):
            if controller == "auto":
                controller = design.controller
            elif controller == "pid" and request.controller not in ("auto", "pid"):
                controller = request.controller
        if mode is None:
            mode = request.mode if request.mode != "heat_pump" else spec.operating_mode
        Tsp = request.T_zone
        constraints = request.constraints
    else:
        constraints = None

    if spec is None:
        fluid = resolve_fluid(refrigerant or "R32")
        operate = (mode or "heating")
        if operate == "cooling" or cooling_tons is not None:
            Qd = Q_design if Q_design is not None else None
            design = design_air_conditioner(fluid, Qd, T_out=T_out, T_zone=Tsp, cooling_tons=cooling_tons)
            spec = design.spec
        else:
            Qd = 5500.0 if Q_design is None else float(Q_design)
            design = design_heat_pump(fluid, Qd, T_out=T_out, T_zone=Tsp)
            spec = design.spec

    operate = spec.operating_mode if mode in (None, "heat_pump", "auto") else mode
    if operate != spec.operating_mode and spec.indoor is not None:
        spec = apply_operating_mode(spec, operate)
    T0_was_none = T_zone0 is None
    if T_zone0 is None:
        T_zone0 = Tsp + (2.0 if spec.operating_mode == "cooling" else -5.0)

    if isinstance(controller, str) and controller == "auto":
        controller = "pid"

    tables = tables or build_tables(spec.fluid)
    ctl_name = controller.lower() if isinstance(controller, str) else ""
    is_mpc = isinstance(controller, (LinearMPC, NonlinearMPC)) or ctl_name in (
        "mpc",
        "lmpc",
        "nmpc",
        "nonlinear-mpc",
    )
    kind = (reduction or "auto").lower()
    if is_mpc:
        # MPC unrolls implicit Euler on the full residual; QSS would be a different plant.
        kind = "full"
    elif kind == "auto":
        kind = "qss" if t_final >= 3600.0 else "full"
    rec = float(record_dt)
    if kind == "qss" and t_final >= 7200.0 and rec < 30.0:
        rec = 60.0

    segments = _mode_segments(t_final, timeseries, spec, spec.operating_mode)
    if spec.indoor is not None:
        spec = apply_operating_mode(spec, segments[0][2])
    if T0_was_none:
        T_zone0 = Tsp + (2.0 if spec.operating_mode == "cooling" else -5.0)
    y0 = initial_state(spec, tables, T_out=T_out, T_zone=T_zone0)
    last = {"t": -1.0, "u": None, "y": y0, "tables": tables, "spec": spec}
    u_log: list[np.ndarray] = []
    mode_log: list[tuple[float, str]] = []

    def exo(t: float) -> tuple[float, float, float, float, float, float | None]:
        if timeseries is not None:
            s = timeseries.at(t)
            return (
                s["T_out"],
                s["Q_gain"],
                s.get("Tsp", Tsp),
                s.get("W_gain", 0.0),
                s.get("defrost", 0.0),
                s.get("RH_out", last["spec"].RH_out),
            )
        return T_out, Q_load, Tsp, 0.0, 0.0, last["spec"].RH_out

    ctl = None
    rhs = None

    def bind(new_spec: PlantSpec, y):
        nonlocal ctl, rhs, spec
        spec = new_spec
        last["spec"] = spec
        rhs = make_rhs(spec, tables)
        if ctl is None:
            ctl = (
                controller
                if not isinstance(controller, str)
                else make_controller(controller, spec, tables, rhs, Tsp, constraints)
            )
            if hasattr(ctl, "reset"):
                ctl.reset()
        elif hasattr(ctl, "set_mode"):
            ctl.set_mode(spec.operating_mode)
        if hasattr(ctl, "Q_design"):
            ctl.Q_design = _q_design(design, spec.operating_mode)
        if hasattr(ctl, "UA"):
            ctl.UA = float(spec.UA_env)
        last["y"] = y
        return y

    y = bind(spec, y0)

    def u_abs(t: float):
        if last["u"] is None or t + 1e-9 >= last["t"] + rec:
            y = last["y"]
            Tamb, Qg, tsp, Wg, dfr, RHo = exo(t)
            if hasattr(ctl, "Tsp"):
                ctl.Tsp = tsp
            u_guess = (
                last["u"]
                if last["u"] is not None
                else np.array([40.0, 0.4, 1.0, 1.0, Tamb, Qg])
            )
            meas = {k: float(v) for k, v in diagnostics(spec, tables, y, u_guess).items()}
            meas["T_out"] = Tamb
            meas["Q_gain"] = Qg
            dt_c = rec
            if isinstance(ctl, (LinearMPC, NonlinearMPC)):
                ctl.Tsp = tsp
                u_exog = jnp.asarray(u_guess)
                if u_exog.size < 6:
                    u_exog = jnp.array([40.0, 0.4, 1.0, 1.0, Tamb, Qg])
                else:
                    u_exog = u_exog.at[4].set(Tamb).at[5].set(Qg)
                out = ctl.update(t, meas, dt_c, y, u_exog)
            else:
                out = ctl.update(t, meas, dt_c)
            if spec.moist or spec.frost:
                if RHo is None:
                    raise ValueError(
                        "moist/frost simulate needs RH_out on the spec or timeseries"
                    )
                u = out.as_input(Tamb, Qg, W_gain=Wg, defrost=dfr, RH_out=RHo)
            else:
                u = out.as_input(Tamb, Qg)
            last["t"] = t
            last["u"] = u
            u_log.append(np.concatenate([[t], u]))
        return last["u"]

    t_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for i, (t0, t1, md) in enumerate(segments):
        if spec.operating_mode != md and spec.indoor is not None:
            new_spec = apply_operating_mode(spec, md)
            y = remap_state(jnp.asarray(last["y"]), spec.layout, new_spec.layout)
            y = project_state(y, tables, new_spec.layout)
            last["u"] = None
            y = bind(new_spec, y)
        mode_log.append((t0, spec.operating_mode))
        t_s, Y_s = _run_span(
            rhs, jnp.asarray(last["y"]), u_abs, t1 - t0, rec, kind, spec, max_steps, last, t0
        )
        if i > 0:
            t_s, Y_s = t_s[1:], Y_s[1:]
        t_parts.append(t_s)
        y_parts.append(Y_s)
        last["y"] = Y_s[-1]

    t = np.concatenate(t_parts)
    Y = np.vstack(y_parts)
    nu = 9 if (spec.moist or spec.frost) else 6
    U = np.zeros((len(t), nu))
    if u_log:
        ut = np.array(u_log)
        ncol = min(nu, ut.shape[1] - 1)
        for j in range(ncol):
            U[:, j] = np.interp(t, ut[:, 0], ut[:, j + 1])
    mode_hist = np.zeros(len(t))
    for t0, md in mode_log:
        mode_hist[t >= t0] = 1.0 if md == "heating" else 0.0
    keys = [
        "p_e",
        "p_c",
        "T_z",
        "SH",
        "SC",
        "mdot_comp",
        "mdot_eev",
        "power",
        "charge",
        "x_e_mean",
        "x_c_mean",
        "x_e_in",
        "x_e_out",
        "x_c_in",
        "x_c_out",
        "pr",
        "Tsat_e",
        "Tsat_c",
        "T_disch",
        "T_e_out",
        "T_c_out",
        "Q_zone",
        "Q_evap",
        "COP",
        "h_disch",
        "h_suct",
        "h_ll",
        "Tw_e_mean",
        "Tw_c_mean",
        "T_e_mean",
        "T_c_mean",
        "W_z",
        "m_fr",
        "Q_lat",
        "delta_fr",
        "mode",
    ]
    meas = {k: np.zeros(len(t)) for k in keys}
    # Diagnostics use the coil map of the mode that was active at that time.
    spec_at = []
    for t0, md in mode_log:
        spec_at.append((t0, apply_operating_mode(spec, md) if spec.indoor is not None else spec))
    for i in range(len(t)):
        sp = spec
        for t0, sp_i in spec_at:
            if t[i] + 1e-12 >= t0:
                sp = sp_i
        d = diagnostics(sp, tables, jnp.asarray(Y[i]), jnp.asarray(U[i]))
        for k in keys:
            if k == "mode":
                meas[k][i] = mode_hist[i]
            else:
                meas[k][i] = float(d[k])
    return SimResult(
        t=np.asarray(t),
        y=np.asarray(Y),
        u=U,
        meas=meas,
        spec=spec,
        design=design,
        mode=spec.operating_mode,
    )


def plot_result(res: SimResult, path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    tmin = res.t / 60.0
    fig, ax = plt.subplots(3, 2, figsize=(10.5, 8.2), sharex=True)
    ax[0, 0].plot(tmin, res.meas["T_z"] - 273.15, color="#1d4ed8")
    ax[0, 0].set_ylabel("Zone temperature (°C)")
    ax[0, 1].plot(tmin, res.meas["SH"], color="#b45309", label="superheat")
    ax[0, 1].plot(tmin, res.meas["SC"], color="#0f766e", label="subcool")
    ax[0, 1].legend(frameon=False)
    ax[0, 1].set_ylabel("K")
    ax[1, 0].plot(tmin, res.meas["p_e"] / 1e5, color="#1d4ed8", label="evap")
    ax[1, 0].plot(tmin, res.meas["p_c"] / 1e5, color="#b91c1c", label="cond")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].set_ylabel("Pressure (bar)")
    ax[1, 1].plot(tmin, res.u[:, 0], color="#334155")
    ax[1, 1].set_ylabel("Compressor (Hz)")
    ax[2, 0].plot(tmin, res.u[:, 1], color="#334155")
    ax[2, 0].set_ylabel("EEV opening")
    ax[2, 0].set_xlabel("Time (min)")
    ax[2, 1].plot(tmin, res.meas["x_e_mean"], color="#1d4ed8", label="evap")
    ax[2, 1].plot(tmin, res.meas["x_c_mean"], color="#b91c1c", label="cond")
    ax[2, 1].legend(frameon=False)
    ax[2, 1].set_ylabel("Mean quality")
    ax[2, 1].set_xlabel("Time (min)")
    fig.suptitle(title or f"{res.spec.fluid} {res.mode} transient")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(
        description=(
            "Size and simulate an air-source heat pump or air conditioner. "
            "Omit --load/--weather to run the 5.5 kW heating demo."
        )
    )
    p.add_argument(
        "--controller",
        default="auto",
        choices=["auto", "pid", "hysteresis", "bangbang", "mpc", "lmpc", "nmpc"],
    )
    p.add_argument("--mode", default="auto", choices=["auto", "heating", "cooling", "heat_pump"])
    p.add_argument("--refrigerant", "-r", default="R32", help="CoolProp fluid name")
    p.add_argument("--load", type=float, default=None, help="optional nameplate override [W]; omit to size from --weather")
    p.add_argument("--load-tons", type=float, default=None, help="optional cooling override [tons]; omit to size from --weather")
    p.add_argument("--T-out", type=float, default=None, help="optional design outdoor [°C]; default is the peak-load hour in --weather")
    p.add_argument("--T-zone", type=float, default=21.0, help="zone setpoint [°C] (the design target)")
    p.add_argument("--T-zone0", type=float, default=None, help="initial zone temperature [°C]")
    p.add_argument("--weather", type=str, default=None, help="CSV of t, T_out, Q (see TimeSeries.from_csv)")
    p.add_argument("--load-kind", default="gain", choices=["gain", "cooling_load", "heating_load"])
    p.add_argument("--t-final", type=float, default=600.0)
    p.add_argument(
        "--reduction",
        default="auto",
        choices=["auto", "full", "qss"],
        help="full DAE, quasi-steady zone (hours/days), or auto (qss if t>1 h)",
    )
    p.add_argument("--list-fluids", action="store_true")
    p.add_argument("--out", default="output")
    p.add_argument("--report", default=None, help="write design package (.md + .json)")
    p.add_argument("--design-only", action="store_true", help="size and write the package, skip the transient")
    p.add_argument("--moist", action="store_true", help="add zone humidity to the DAE (requires --RH-out and --RH-zone)")
    p.add_argument("--frost", action="store_true", help="frost mass on the outdoor coil (implies --moist; requires --RH-out)")
    p.add_argument("--RH-out", type=float, default=None, help="outdoor RH 0–1 (required for --moist/--frost; not defaulted)")
    p.add_argument("--RH-zone", type=float, default=None, help="initial indoor RH 0–1 (required for --moist; not defaulted)")
    p.add_argument("--W-defrost", type=float, default=0.0, help="electric defrost power [W]; 0 = no heater")
    p.add_argument("--ahri540", default=None, help="AHRI 540 coefficient JSON (published or user; no silent default)")
    p.add_argument("--fan-table", default=None, help="CSV of speed,mdot_kg_s (user/manufacturer; no invented curve)")
    p.add_argument("--catalog", default=None, help="user catalog JSON (example lists only the cited Lee 2021 map)")
    p.add_argument("--seasonal", action="store_true", help="bin the --weather time series (hours from the record, not AHRI tables)")
    p.add_argument("--bin-width", type=float, default=5.0, help="seasonal outdoor-T bin width [K]")
    args = p.parse_args(argv)

    if args.list_fluids:
        print("\n".join(list_refrigerants()))
        return

    ts = None
    if args.weather:
        ts = TimeSeries.from_csv(args.weather, load_kind=args.load_kind)
        t_final = max(args.t_final, ts.duration)
    else:
        t_final = args.t_final

    Tsp = args.T_zone + 273.15
    T0 = None if args.T_zone0 is None else args.T_zone0 + 273.15
    T_out_cli = None if args.T_out is None else args.T_out + 273.15

    Q_heat = Q_cool = None
    tons = args.load_tons
    if ts is None and args.load is None and tons is None:
        if args.mode in ("auto", "heating", "heat_pump"):
            Q_heat = 5500.0
        if args.mode in ("cooling", "heat_pump"):
            Q_cool = 5500.0
        mode = "heating" if args.mode == "auto" else args.mode
    else:
        mode = args.mode
        if args.load is not None:
            if mode == "cooling":
                Q_cool = args.load
            elif mode == "heating":
                Q_heat = args.load
            else:
                Q_heat = Q_cool = args.load

    req = DesignRequest(
        refrigerant=args.refrigerant,
        mode=mode,
        T_zone=Tsp,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        cooling_tons=tons,
        T_out_heat=T_out_cli if T_out_cli is not None and mode != "cooling" else 273.15,
        T_out_cool=T_out_cli if T_out_cli is not None and mode != "heating" else 308.15,
        controller=args.controller,
        timeseries=ts,
    )
    design = design_system(req)
    operate = design.request.mode
    if operate == "heat_pump":
        qh = design.heating.Q_load if design.heating else 0.0
        qc = design.cooling.Q_load if design.cooling else 0.0
        operate = "cooling" if qc >= qh else "heating"
    spec = apply_operating_mode(design.spec, operate)

    if args.frost:
        args.moist = True
    if args.moist or args.frost:
        rh_out = args.RH_out
        rh_z = args.RH_zone
        if rh_out is None and ts is not None and ts.RH_out is not None:
            rh_out = float(ts.RH_out[0])
        if rh_out is None:
            raise SystemExit("--moist/--frost requires --RH-out or an RH_out column in --weather")
        if args.moist and rh_z is None:
            raise SystemExit("--moist requires --RH-zone (indoor humidity is not defaulted)")
        if rh_out > 1.5:
            rh_out = rh_out / 100.0
        if rh_z is not None and rh_z > 1.5:
            rh_z = rh_z / 100.0
        spec = replace(
            spec,
            moist=bool(args.moist),
            frost=bool(args.frost),
            RH_out=rh_out,
            RH_zone0=rh_z,
            W_defrost=float(args.W_defrost),
        )
    if args.ahri540:
        from heatpump.devices import AHRI540Compressor

        spec = replace(
            spec,
            compressor=AHRI540Compressor.from_file(args.ahri540),
            ahri540_path=args.ahri540,
        )
    if args.fan_table:
        from heatpump.devices import TableFan

        fan = TableFan.from_file(args.fan_table)
        spec = replace(spec, fan_indoor=fan, fan_outdoor=fan, fan_path=args.fan_table)
    if args.catalog:
        from heatpump.catalog import load_catalog
        from heatpump.devices import AHRI540Compressor

        cat = load_catalog(args.catalog)
        print(f"catalog: {cat.citation}")
        for it in cat.items:
            print(f"  {it.id}  {it.kind}  {it.path.name}")
        if args.ahri540 is None:
            maps = cat.of_kind("compressor_map")
            if maps:
                spec = replace(
                    spec,
                    compressor=AHRI540Compressor.from_file(maps[0].path),
                    ahri540_path=str(maps[0].path),
                )
    if ts is not None:
        T_out = float(ts.T_out[0])
    elif T_out_cli is not None:
        T_out = T_out_cli
    else:
        T_out = design.request.T_out_cool if operate == "cooling" else design.request.T_out_heat
    print(design.summary())
    pkg = design.as_report()
    report_path = Path(args.report) if args.report else (Path(args.out) / "design.md" if args.design_only else None)
    if report_path is not None:
        pkg.write(report_path)
        pkg.plot(report_path.with_name(report_path.stem + "_map.png"))
        print(f"wrote {report_path} and {report_path.with_suffix('.json')}")
    if args.seasonal:
        if ts is None:
            raise SystemExit("--seasonal needs --weather; AHRI bin-hour tables are not used")
        from heatpump.seasonal import bin_timeseries

        bins = bin_timeseries(
            ts,
            width_K=args.bin_width,
            heating_map=pkg.heating_map,
            cooling_map=pkg.cooling_map,
        )
        dest = Path(args.out) / "seasonal.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(bins.to_markdown(), encoding="utf-8")
        dest.with_suffix(".json").write_text(json.dumps(bins.to_json(), indent=2), encoding="utf-8")
        print(f"wrote {dest} ({len(bins.bins)} bins, {bins.hours_total:.1f} h from the record)")
    if args.design_only:
        return

    tables = build_tables(spec.fluid)
    ctl = design.controller if args.controller == "auto" else args.controller
    out = Path(args.out)
    res = simulate(
        ctl,
        t_final=t_final,
        spec=spec,
        tables=tables,
        design=design,
        T_out=T_out,
        Tsp=Tsp,
        T_zone0=T0,
        timeseries=ts,
        reduction=args.reduction,
    )
    plot_result(res, out / f"{ctl}.png", title=f"{ctl} · {spec.fluid} {spec.mode}")
    print(
        f"t={res.t[-1]:.0f}s  T_z {res.meas['T_z'][-1]-273.15:.2f} C  "
        f"SH {res.meas['SH'][-1]:.2f} K  "
        f"p_e {res.meas['p_e'][-1]/1e5:.2f} bar  p_c {res.meas['p_c'][-1]/1e5:.2f} bar  "
        f"COP {res.meas['COP'][-1]:.2f}"
    )


if __name__ == "__main__":
    main()
