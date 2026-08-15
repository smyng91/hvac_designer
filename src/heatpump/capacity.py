"""Off-design useful capacity and COP versus outdoor temperature.

A fixed machine (displacement + coil geometry) is closed at each outdoor
temperature so refrigerant-side ṁ Δh equals the ε-NTU coil heat rate.
Saturation temperatures are solved; they are not held at the design
approach. The load line is Newton cooling UA |T_z − T_out|, or the
timeseries Q interpolated against outdoor temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

import numpy as np

from heatpump.hx import air_props, coil_Q, htc_air_bank, htc_ref_at, overall_UA
from heatpump.requirements import Constraints, TimeSeries

if TYPE_CHECKING:
    from heatpump.plant import PlantSpec


@dataclass(frozen=True)
class CapacityPoint:
    T_out: float
    Q_cap: float
    Q_load: float
    COP: float
    W: float
    mdot: float
    pr: float
    T_disch: float
    T_e: float
    T_c: float
    feasible: bool
    Q_min: float
    Q_max: float
    note: str = ""


@dataclass(frozen=True)
class CapacityMap:
    kind: str
    T_zone: float
    T_design: float
    Q_design: float
    T_out: np.ndarray
    points: tuple[CapacityPoint, ...]
    T_balance: float | None
    margin_design: float
    notes: tuple[str, ...]

    def at(self, T_out: float) -> CapacityPoint:
        T = np.asarray([p.T_out for p in self.points])
        i = int(np.argmin(np.abs(T - T_out)))
        return self.points[i]


def load_line(
    T_out: np.ndarray,
    *,
    kind: str,
    T_zone: float,
    Q_design: float,
    UA: float,
    timeseries: TimeSeries | None = None,
) -> np.ndarray:
    """Zone load the machine must meet at each outdoor temperature [W]."""
    T_out = np.asarray(T_out, dtype=float)
    if UA > 0.0:
        if kind == "heating":
            return UA * np.maximum(T_zone - T_out, 0.0)
        return UA * np.maximum(T_out - T_zone, 0.0)
    if timeseries is not None and timeseries.t.size >= 2:
        if kind == "heating":
            Q = np.maximum(-timeseries.Q_gain, 0.0)
        else:
            Q = np.maximum(timeseries.Q_gain, 0.0)
        T = timeseries.T_out
        order = np.argsort(T)
        T_s, Q_s = T[order], Q[order]
        Tu, inv = np.unique(np.round(T_s, 6), return_inverse=True)
        Q_u = np.zeros(Tu.shape)
        cnt = np.zeros(Tu.shape)
        np.add.at(Q_u, inv, Q_s)
        np.add.at(cnt, inv, 1.0)
        Q_u /= np.maximum(cnt, 1.0)
        return np.interp(T_out, Tu, Q_u)
    return np.full(T_out.shape, float(Q_design))


def _crossing(T: np.ndarray, d: np.ndarray) -> float | None:
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            return float(T[i])
        if d[i] * d[i + 1] < 0.0:
            w = abs(d[i]) / max(abs(d[i] - d[i + 1]), 1e-12)
            return float(T[i] + w * (T[i + 1] - T[i]))
    return None


def _air_temps(kind: str, T_out: float, T_zone: float) -> tuple[float, float]:
    if kind == "cooling":
        return T_zone, T_out
    return T_out, T_zone


def _cycle(fluid: str, kind: str, T_out: float, T_zone: float, T_e: float, T_c: float, SH: float, SC: float, eta_is: float):
    from heatpump.design import design_cycle

    T_air_e, T_air_c = _air_temps(kind, T_out, T_zone)
    return design_cycle(
        fluid,
        mode=kind,
        T_air_evap=T_air_e,
        T_air_cond=T_air_c,
        SH=SH,
        SC=SC,
        DT_evap=T_air_e - T_e,
        DT_cond=T_c - T_air_c,
        eta_is=eta_is,
    )


def _mdot(states: dict, meta: dict, V_disp: float, N_hz: float, C_loss: float) -> float:
    pr = meta["pr"]
    gamma = meta["gamma"]
    eta_v = 1.0 - C_loss * (pr ** (1.0 / gamma) - 1.0)
    if eta_v <= 0.0 or N_hz <= 0.0:
        return 0.0
    return eta_v * max(states["1"].rho, 0.0) * V_disp * N_hz


def _coil_Q(fluid: str, spec: PlantSpec, p: float, h: float, mdot: float, evaporating: bool, T_air: float, T_ref: float) -> float:
    if evaporating:
        D, L, n, fin, m_air = spec.D_e, spec.L_e, spec.n_tubes_e, spec.fin_e, spec.mdot_air_e0
    else:
        D, L, n, fin, m_air = spec.D_c, spec.L_c, spec.n_tubes_c, spec.fin_c, spec.mdot_air_c0
    A_ref = n * pi * D * L
    A_air = A_ref * fin
    G = mdot / max(n * 0.25 * pi * D**2, 1e-12)
    h_r = htc_ref_at(fluid, p, h, G, D, evaporating, spec.p_crit)
    h_a = htc_air_bank(m_air, T_air, D, n, L)
    UA = overall_UA(h_r, A_ref, h_a, A_air)
    return coil_Q(T_air, T_ref, m_air, air_props(T_air)["cp"], UA)


def _residual(
    Te: float,
    Tc: float,
    *,
    fluid: str,
    kind: str,
    T_out: float,
    T_zone: float,
    spec: PlantSpec,
    N_hz: float,
    SH: float,
    SC: float,
    eta_is: float,
) -> np.ndarray:
    _info, states, meta, _n = _cycle(fluid, kind, T_out, T_zone, Te, Tc, SH, SC, eta_is)
    mdot = _mdot(states, meta, spec.V_disp, N_hz, spec.C_loss)
    Qe = mdot * meta["q_evap"]
    Qc = mdot * meta["q_cond"]
    T_air_e, T_air_c = _air_temps(kind, T_out, T_zone)
    Qhx_e = _coil_Q(fluid, spec, meta["p_e"], 0.5 * (states["4"].h + states["1"].h), mdot, True, T_air_e, Te)
    Qhx_c = _coil_Q(fluid, spec, meta["p_c"], 0.5 * (states["2"].h + states["3"].h), mdot, False, T_air_c, Tc)
    return np.array([Qhx_e - Qe, -Qhx_c - Qc], dtype=float)


def _close(
    fluid: str,
    kind: str,
    T_out: float,
    T_zone: float,
    spec: PlantSpec,
    N_hz: float,
    SH: float,
    SC: float,
    eta_is: float,
    DT_evap: float,
    DT_cond: float,
) -> tuple[dict, dict, float] | None:
    T_air_e, T_air_c = _air_temps(kind, T_out, T_zone)
    x = np.array([T_air_e - DT_evap, T_air_c + DT_cond], dtype=float)
    r = None
    for _ in range(12):
        try:
            r = _residual(
                float(x[0]),
                float(x[1]),
                fluid=fluid,
                kind=kind,
                T_out=T_out,
                T_zone=T_zone,
                spec=spec,
                N_hz=N_hz,
                SH=SH,
                SC=SC,
                eta_is=eta_is,
            )
        except (ValueError, TypeError):
            return None
        if float(np.linalg.norm(r)) < 40.0:
            break
        J = np.eye(2)
        for i in range(2):
            dx = np.zeros(2)
            dx[i] = 0.25
            try:
                J[:, i] = (
                    _residual(
                        float(x[0] + dx[0]),
                        float(x[1] + dx[1]),
                        fluid=fluid,
                        kind=kind,
                        T_out=T_out,
                        T_zone=T_zone,
                        spec=spec,
                        N_hz=N_hz,
                        SH=SH,
                        SC=SC,
                        eta_is=eta_is,
                    )
                    - r
                ) / 0.25
            except (ValueError, TypeError):
                pass
        try:
            step = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
        x = x - np.clip(step, -4.0, 4.0)
        x[0] = min(float(x[0]), T_air_e - 1.0)
        x[1] = max(float(x[1]), T_air_c + 1.0)
    if r is None or float(np.linalg.norm(r)) > 250.0:
        return None
    try:
        _info, states, meta, _n = _cycle(fluid, kind, T_out, T_zone, float(x[0]), float(x[1]), SH, SC, eta_is)
    except (ValueError, TypeError):
        return None
    return states, meta, float(np.linalg.norm(r))


def _empty_point(T: float, q_l: float, note: str) -> CapacityPoint:
    return CapacityPoint(
        T_out=float(T),
        Q_cap=0.0,
        Q_load=float(q_l),
        COP=0.0,
        W=0.0,
        mdot=0.0,
        pr=0.0,
        T_disch=float("nan"),
        T_e=float("nan"),
        T_c=float("nan"),
        feasible=False,
        Q_min=0.0,
        Q_max=0.0,
        note=note,
    )


def capacity_map(
    *,
    fluid: str,
    kind: str,
    T_zone: float,
    T_design: float,
    Q_design: float,
    spec: PlantSpec,
    SH: float,
    SC: float,
    DT_evap: float,
    DT_cond: float,
    N_design: float,
    constraints: Constraints | None = None,
    UA: float = 0.0,
    timeseries: TimeSeries | None = None,
    n_points: int = 9,
    V_disp: float | None = None,
    C_loss: float | None = None,
    eta_is: float | None = None,
) -> CapacityMap:
    """Useful capacity of a fixed machine versus outdoor temperature."""
    cons = constraints or Constraints()
    kind = "cooling" if kind == "cooling" else "heating"
    V_disp = spec.V_disp if V_disp is None else V_disp
    C_loss = spec.C_loss if C_loss is None else C_loss
    eta_is = spec.eta_is0 if eta_is is None else eta_is
    if kind == "heating":
        T_lo = min(T_design - 15.0, T_zone - 35.0)
        T_hi = T_zone - 2.0
    else:
        T_lo = T_zone + 2.0
        T_hi = max(T_design + 8.0, T_zone + 20.0)
    T_grid = np.linspace(T_lo, T_hi, n_points)
    Q_load = load_line(
        T_grid,
        kind=kind,
        T_zone=T_zone,
        Q_design=Q_design,
        UA=UA,
        timeseries=timeseries,
    )
    points: list[CapacityPoint] = []
    notes: list[str] = [
        "Saturation temperatures are closed so ṁ Δh equals the ε-NTU coil "
        "heat rate (Dittus–Boelter / Shah refrigerant, Zhukauskas air).",
        "Load line is UA |T_zone − T_out| when UA > 0, otherwise the "
        "timeseries Q interpolated on outdoor temperature.",
    ]
    for T, q_l in zip(T_grid, Q_load):
        got = _close(
            fluid,
            kind,
            float(T),
            T_zone,
            spec,
            N_design,
            SH,
            SC,
            eta_is,
            DT_evap,
            DT_cond,
        )
        if got is None:
            points.append(_empty_point(float(T), float(q_l), "coil/cycle close failed"))
            continue
        states, meta, _res = got
        mdot = _mdot(states, meta, V_disp, N_design, C_loss)
        Q_cond = mdot * meta["q_cond"]
        Q_evap = mdot * meta["q_evap"]
        W = mdot * meta["w"]
        Q = Q_evap if kind == "cooling" else Q_cond
        scale_min = cons.N_min / N_design if N_design > 0.0 else 0.0
        scale_max = cons.N_max / N_design if N_design > 0.0 else 0.0
        cop = Q / W if W > 1.0 else 0.0
        points.append(
            CapacityPoint(
                T_out=float(T),
                Q_cap=float(Q),
                Q_load=float(q_l),
                COP=float(cop),
                W=float(W),
                mdot=float(mdot),
                pr=float(meta["pr"]),
                T_disch=float(states["2"].T),
                T_e=float(meta["T_e"]),
                T_c=float(meta["T_c"]),
                feasible=True,
                Q_min=float(Q * max(scale_min, 0.0)),
                Q_max=float(Q * max(scale_max, 0.0)),
            )
        )
    T = np.array([p.T_out for p in points])
    cap = np.array([p.Q_cap for p in points])
    load = np.array([p.Q_load for p in points])
    feas = np.array([p.feasible for p in points])
    T_bal = _crossing(T[feas], (cap - load)[feas]) if feas.any() else None
    i_des = int(np.argmin(np.abs(T - T_design)))
    margin = float(cap[i_des] / max(load[i_des], 1.0)) if points[i_des].feasible else 0.0
    if T_bal is None:
        if feas.any() and np.all((cap - load)[feas] > 0):
            notes.append("Capacity exceeds the load line over the plotted outdoor range.")
        elif feas.any() and np.all((cap - load)[feas] < 0):
            notes.append("Capacity is below the load line over the plotted outdoor range.")
    return CapacityMap(
        kind=kind,
        T_zone=float(T_zone),
        T_design=float(T_design),
        Q_design=float(Q_design),
        T_out=T,
        points=tuple(points),
        T_balance=T_bal,
        margin_design=margin,
        notes=tuple(notes),
    )
