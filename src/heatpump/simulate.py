"""Closed-loop transients and a small CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
)
from heatpump.requirements import DesignRequest, TimeSeries
from heatpump.solver import TRBDF2, integrate
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
    max_steps: int = 40000,
) -> SimResult:
    """Integrate a closed-loop heating or cooling transient.

    ``Q_load`` / timeseries ``Q_gain`` is heat into the zone [W] on top of
    ``UA_env (T_out - T_z)``. If ``spec`` is omitted the plant is sized from
    ``request`` or from ``refrigerant`` + ``Q_design`` / ``cooling_tons``.
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
    if T_zone0 is None:
        T_zone0 = Tsp + (2.0 if spec.operating_mode == "cooling" else -5.0)

    if isinstance(controller, str) and controller == "auto":
        controller = "pid"

    tables = tables or build_tables(spec.fluid)
    rhs = make_rhs(spec, tables)
    y0 = initial_state(spec, tables, T_out=T_out, T_zone=T_zone0)
    ctl = (
        controller
        if not isinstance(controller, str)
        else make_controller(controller, spec, tables, rhs, Tsp, constraints)
    )
    if hasattr(ctl, "reset"):
        ctl.reset()

    u_log: list[np.ndarray] = []
    last = {"t": -1.0, "u": None, "y": y0}

    def exo(t: float) -> tuple[float, float, float]:
        if timeseries is not None:
            s = timeseries.at(t)
            return s["T_out"], s["Q_gain"], s.get("Tsp", Tsp)
        return T_out, Q_load, Tsp

    def u_of_t(t: float):
        if last["u"] is None or t + 1e-9 >= last["t"] + record_dt:
            y = last["y"]
            Tamb, Qg, tsp = exo(t)
            if hasattr(ctl, "Tsp"):
                ctl.Tsp = tsp
            meas = {
                k: float(v)
                for k, v in diagnostics(
                    spec,
                    tables,
                    y,
                    last["u"]
                    if last["u"] is not None
                    else np.array([40.0, 0.4, 1.0, 1.0, Tamb, Qg]),
                ).items()
            }
            dt_c = record_dt if last["u"] is not None else record_dt
            if isinstance(ctl, (LinearMPC, NonlinearMPC)):
                ctl.Tsp = tsp
                u_exog = jnp.array([40.0, 0.4, 1.0, 1.0, Tamb, Qg])
                out = ctl.update(t, meas, dt_c, y, u_exog)
            else:
                out = ctl.update(t, meas, dt_c)
            u = out.as_input(Tamb, Qg)
            last["t"] = t
            last["u"] = u
            u_log.append(np.concatenate([[t], u]))
        return last["u"]

    project = lambda y: project_state(y, tables, spec.layout)

    t, Y = integrate(
        rhs,
        y0,
        u_of_t,
        t_final=t_final,
        dt0=0.15,
        project=project,
        solver=TRBDF2(),
        record_dt=record_dt,
        max_steps=max_steps,
        on_accept=lambda _t, y: last.__setitem__("y", y),
    )
    U = np.zeros((len(t), 6))
    if u_log:
        ut = np.array(u_log)
        for j in range(6):
            U[:, j] = np.interp(t, ut[:, 0], ut[:, j + 1])
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
    ]
    meas = {k: np.zeros(len(t)) for k in keys}
    for i in range(len(t)):
        d = diagnostics(spec, tables, jnp.asarray(Y[i]), jnp.asarray(U[i]))
        for k in keys:
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
    p = argparse.ArgumentParser(
        description="Size and simulate an air-source heat pump or air conditioner"
    )
    p.add_argument(
        "--controller",
        default="auto",
        choices=["auto", "pid", "hysteresis", "bangbang", "mpc", "nmpc"],
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
    p.add_argument("--compare", action="store_true", help="run PID / hysteresis / MPC")
    p.add_argument("--list-fluids", action="store_true")
    p.add_argument("--out", default="examples/out")
    p.add_argument("--report", default=None, help="write design package (.md + .json)")
    p.add_argument("--design-only", action="store_true", help="size and write the package, skip the transient")
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
    if args.design_only:
        return

    tables = build_tables(spec.fluid)
    ctl = design.controller if args.controller == "auto" else args.controller
    out = Path(args.out)
    if args.compare:
        for name in ("pid", "hysteresis", "mpc"):
            print(f"running {name} ...")
            res = simulate(
                name,
                t_final=t_final,
                spec=spec,
                tables=tables,
                design=design,
                T_out=T_out,
                Tsp=Tsp,
                T_zone0=T0,
                timeseries=ts,
            )
            plot_result(res, out / f"{name}.png", title=f"{name} · {spec.fluid} {spec.mode}")
            print(
                f"  T_z {res.meas['T_z'][-1]-273.15:.2f} C  SH {res.meas['SH'][-1]:.2f} K  "
                f"charge {res.meas['charge'][-1]:.3f} kg"
            )
    else:
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
        )
        plot_result(res, out / f"{ctl}.png", title=f"{ctl} · {spec.fluid} {spec.mode}")
        print(
            f"T_z {res.meas['T_z'][-1]-273.15:.2f} C  SH {res.meas['SH'][-1]:.2f} K  "
            f"p_e {res.meas['p_e'][-1]/1e5:.2f} bar  p_c {res.meas['p_c'][-1]/1e5:.2f} bar  "
            f"COP {res.meas['COP'][-1]:.2f}"
        )


if __name__ == "__main__":
    main()
