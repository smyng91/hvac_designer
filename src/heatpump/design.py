"""Load-based air-source heat-pump sizing for an arbitrary CoolProp refrigerant."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from math import ceil, pi
from typing import TYPE_CHECKING

import CoolProp.CoolProp as CP
import numpy as np

from heatpump.gates import GateSet, evaluate_gates, raise_if_failed
from heatpump.hx import air_props, charge_from_profile, size_coil, zone_capacitance
from heatpump.requirements import Constraints, DesignRequest, cooling_tons_to_w
from heatpump.thermo import FluidInfo, fluid_info, make_state, resolve_fluid

if TYPE_CHECKING:
    from heatpump.plant import PlantSpec


@dataclass(frozen=True)
class CyclePoint:
    p: float
    T: float
    h: float
    s: float
    rho: float
    x: float


@dataclass(frozen=True)
class DesignReport:
    fluid: str
    Q_load: float
    Q_heat: float
    Q_evap: float
    W: float
    COP: float
    mdot: float
    p_e: float
    p_c: float
    T_e: float
    T_c: float
    T_suct: float
    T_disch: float
    SH: float
    SC: float
    glide_e: float
    glide_c: float
    pr: float
    gamma: float
    V_disp: float
    A_eev: float
    n_tubes_e: int
    n_tubes_c: int
    L_e: float
    L_c: float
    D: float
    spec: PlantSpec
    notes: tuple[str, ...]
    states: dict[str, CyclePoint]
    kind: str = "heating"
    Q_cool: float = 0.0
    controller: str = "pid"
    G_e: float = 0.0
    G_c: float = 0.0
    charge_kg: float = 0.0
    gates: GateSet | None = None

    def summary(self) -> str:
        s = self.states
        useful = self.Q_cool if self.kind == "cooling" else self.Q_heat
        lines = [
            f"Refrigerant {self.fluid}  ·  {self.kind} load {self.Q_load/1e3:.2f} kW  "
            f"·  unit {useful/1e3:.2f} kW",
            f"  Cycle  Te={self.T_e-273.15:.1f}°C  Tc={self.T_c-273.15:.1f}°C  "
            f"SH={self.SH:.1f} K  SC={self.SC:.1f} K  PR={self.pr:.2f}",
            f"  p_e={self.p_e/1e5:.2f} bar  p_c={self.p_c/1e5:.2f} bar  "
            f"T_disch={self.T_disch-273.15:.1f}°C  COP={self.COP:.2f}",
            f"  mdot={self.mdot*1e3:.1f} g/s  W={self.W/1e3:.2f} kW  "
            f"Q_evap={self.Q_evap/1e3:.2f} kW",
            f"  Compressor V_disp={self.V_disp*1e6:.1f} cm³/rev  "
            f"EEV A={self.A_eev*1e6:.2f} mm²  γ={self.gamma:.3f}",
            f"  Evap  {self.n_tubes_e} × Ø{self.D*1e3:.1f} mm × {self.L_e:.2f} m",
            f"  Cond  {self.n_tubes_c} × Ø{self.D*1e3:.1f} mm × {self.L_c:.2f} m",
            f"  1 (suct)  T={s['1'].T-273.15:.1f}°C  h={s['1'].h/1e3:.1f} kJ/kg  "
            f"ρ={s['1'].rho:.2f} kg/m³",
            f"  2 (disch) T={s['2'].T-273.15:.1f}°C  h={s['2'].h/1e3:.1f} kJ/kg",
            f"  3 (liquid) T={s['3'].T-273.15:.1f}°C  h={s['3'].h/1e3:.1f} kJ/kg  "
            f"ρ={s['3'].rho:.1f} kg/m³",
            f"  4 (eev)   h={s['4'].h/1e3:.1f} kJ/kg  x={s['4'].x:.3f}",
        ]
        if self.gates is not None:
            lines.append("  Gates:")
            lines.append(self.gates.summary())
        if self.notes:
            lines.append("  Notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def _point(AS: CP.AbstractState, x: float = -1.0) -> CyclePoint:
    q = float(AS.Q()) if 0.0 <= float(AS.Q()) <= 1.0 else x
    return CyclePoint(
        p=float(AS.p()),
        T=float(AS.T()),
        h=float(AS.hmass()),
        s=float(AS.smass()),
        rho=float(AS.rhomass()),
        x=q,
    )


def design_cycle(
    fluid: str,
    T_out: float | None = None,
    T_zone: float | None = None,
    *,
    mode: str = "heating",
    T_air_evap: float | None = None,
    T_air_cond: float | None = None,
    SH: float = 6.0,
    SC: float = 4.0,
    DT_evap: float = 10.0,
    DT_cond: float = 12.0,
    eta_is: float = 0.70,
    constraints=None,
) -> tuple[FluidInfo, dict[str, CyclePoint], dict[str, float], list[str]]:
    """Subcritical vapor-compression cycle from CoolProp.

    Heating: evaporator sees outdoor air, condenser sees the zone.
    Cooling: evaporator sees the zone, condenser sees outdoor air.
    """
    info = fluid_info(fluid)
    AS = make_state(info.name)
    notes: list[str] = []
    mode = (mode or "heating").lower()
    if T_air_evap is None or T_air_cond is None:
        if T_out is None or T_zone is None:
            raise TypeError("design_cycle needs T_out/T_zone or T_air_evap/T_air_cond")
        if mode == "cooling":
            T_air_evap = T_zone
            T_air_cond = T_out
        else:
            T_air_evap = T_out
            T_air_cond = T_zone

    T_e = T_air_evap - DT_evap
    T_c = T_air_cond + DT_cond
    if T_e < info.Tmin + 1.0:
        raise ValueError(
            f"{info.name}: evaporating temperature {T_e-273.15:.1f}°C is below "
            f"the EOS limit {info.Tmin-273.15:.1f}°C."
        )
    if T_c >= info.Tc:
        raise ValueError(
            f"{info.name} cannot condense subcritically at "
            f"{T_c-273.15:.1f}°C (Tc={info.Tc-273.15:.1f}°C). "
            "Pick a higher-Tc refrigerant or a lower zone temperature. "
            "This plant is two-phase / subcritical only (no transcritical CO2)."
        )

    AS.update(CP.QT_INPUTS, 1.0, T_e)
    p_e = float(AS.p())
    T_dew_e = float(AS.T())
    AS.update(CP.QT_INPUTS, 0.0, T_e)
    T_bubble_e = float(AS.T())

    AS.update(CP.QT_INPUTS, 0.0, T_c)
    p_c = float(AS.p())
    T_bubble_c = float(AS.T())
    AS.update(CP.QT_INPUTS, 1.0, T_c)
    T_dew_c = float(AS.T())

    if p_c <= p_e:
        raise ValueError(
            f"{info.name}: condenser pressure {p_c/1e5:.2f} bar is not above "
            f"evaporator {p_e/1e5:.2f} bar at this duty."
        )

    T1 = T_dew_e + SH
    AS.update(CP.PT_INPUTS, p_e, T1)
    st1 = _point(AS, x=1.0)
    gamma = float(np.clip(AS.cpmass() / max(AS.cvmass(), 1.0), 1.05, 1.8))

    AS.update(CP.PSmass_INPUTS, p_c, st1.s)
    h2s = float(AS.hmass())
    h2 = st1.h + (h2s - st1.h) / max(eta_is, 0.35)
    AS.update(CP.HmassP_INPUTS, h2, p_c)
    st2 = _point(AS, x=1.0)

    T3 = T_bubble_c - SC
    if T3 <= info.Tmin:
        raise ValueError(
            f"{info.name}: liquid temperature {T3-273.15:.1f}°C is below "
            f"the EOS limit {info.Tmin-273.15:.1f}°C."
        )
    AS.update(CP.PT_INPUTS, p_c, T3)
    st3 = _point(AS, x=0.0)

    AS.update(CP.HmassP_INPUTS, st3.h, p_e)
    st4 = _point(AS)

    q_cond = st2.h - st3.h
    q_evap = st1.h - st4.h
    w = st2.h - st1.h
    if q_cond <= 1.0e4 or w <= 1.0e3:
        raise ValueError(f"{info.name}: cycle enthalpies are not a valid heat-pump loop.")

    glide_e = T_dew_e - T_bubble_e
    glide_c = T_dew_c - T_bubble_c
    meta = {
        "T_e": T_e,
        "T_c": T_c,
        "p_e": p_e,
        "p_c": p_c,
        "q_cond": q_cond,
        "q_evap": q_evap,
        "w": w,
        "COP_h": q_cond / w,
        "COP_c": q_evap / w,
        "COP": (q_evap / w) if mode == "cooling" else (q_cond / w),
        "gamma": gamma,
        "glide_e": glide_e,
        "glide_c": glide_c,
        "pr": p_c / p_e,
        "SH": st1.T - T_dew_e,
        "SC": T_bubble_c - st3.T,
    }

    if p_c >= info.pc:
        raise ValueError(
            f"{info.name}: condensing pressure {p_c/1e5:.2f} bar is not subcritical "
            f"(p_crit={info.pc/1e5:.2f} bar)."
        )
    if glide_c > 2.0 or glide_e > 2.0:
        notes.append(
            f"Zeotropic glide {max(glide_e, glide_c):.1f} K — superheat uses dew, "
            "subcool uses bubble."
        )
    if info.name in {"CO2", "R744"}:
        notes.append("CO2 is a poor fit for this subcritical two-phase plant.")

    states = {"1": st1, "2": st2, "2s": CyclePoint(p_c, float("nan"), h2s, st1.s, float("nan"), 1.0), "3": st3, "4": st4}
    return info, states, meta, notes


# Default tube geometry (not a function of load).
_D_TUBE = 0.007
_L_EVAP = 1.20
_L_COND = 1.50
_FIN_EVAP = 16.0
_FIN_COND = 18.0
_U_EEV_DESIGN = 0.40
_V_ZONE_DEFAULT = 50.0


def _coil(n_cells: int, sized: dict, fin: float, cp_air: float):
    from heatpump.plant import CoilSpec

    return CoilSpec(
        n=n_cells,
        D=float(sized["D"]),
        L=float(sized["L"]),
        n_tubes=int(sized["n_tubes"]),
        fin=fin,
        V_header=float(sized["V_header"]),
        C_w=float(sized["C_w"]),
        mdot_air0=float(sized["mdot_air"]),
        htc_air=float(sized["htc_air"]),
        cp_air=float(cp_air),
    )


def _envelope_UA(Q_load: float, T_zone: float, T_out: float, UA_env: float | None) -> float:
    if UA_env is not None:
        return float(UA_env)
    dT = abs(float(T_zone) - float(T_out))
    if dT < 0.25:
        raise ValueError(
            "UA_env = Q_load / |T_zone - T_out| is undefined when those "
            "temperatures are equal. Pass UA_env explicitly."
        )
    return float(Q_load) / dT


def _zone_C(T_zone: float, C_zone: float | None, V_zone: float | None, notes: list[str]) -> float:
    if C_zone is not None:
        return float(C_zone)
    V = float(V_zone) if V_zone is not None else _V_ZONE_DEFAULT
    if V_zone is None:
        notes.append(
            f"Zone thermal mass is ρ_air c_p V with V = {_V_ZONE_DEFAULT:.0f} m³ "
            "(pass V_zone or C_zone to set the room)."
        )
    return zone_capacitance(T_zone, V)


def _merge_coil(a, b):
    from heatpump.plant import CoilSpec

    D = max(a.D, b.D)
    L = max(a.L, b.L)
    A = max(a.A_ref, b.A_ref)
    n_tubes = max(a.n_tubes, b.n_tubes, int(ceil(A / (pi * D * L))))
    return CoilSpec(
        n=max(a.n, b.n),
        D=D,
        L=L,
        n_tubes=n_tubes,
        fin=max(a.fin, b.fin),
        V_header=max(a.V_header, b.V_header),
        C_w=max(a.C_w, b.C_w),
        mdot_air0=max(a.mdot_air0, b.mdot_air0),
        htc_air=max(a.htc_air, b.htc_air),
    )


def _design_mode(
    refrigerant: str,
    kind: str,
    Q_load: float,
    T_out: float,
    T_zone: float,
    *,
    oversize: float = 1.0,
    SH: float = 6.0,
    SC: float = 4.0,
    DT_evap: float = 10.0,
    DT_cond: float = 12.0,
    N_hz: float = 50.0,
    n_e: int = 6,
    n_c: int = 6,
    eta_is: float = 0.70,
    C_loss: float = 0.075,
    Cd: float = 0.70,
    constraints=None,
    UA_env: float | None = None,
    C_zone: float | None = None,
    V_zone: float | None = None,
    D: float = _D_TUBE,
    L_e: float = _L_EVAP,
    L_c: float = _L_COND,
    fin_e: float = _FIN_EVAP,
    fin_c: float = _FIN_COND,
    **spec_overrides,
) -> DesignReport:
    from heatpump.plant import PlantSpec, apply_operating_mode

    if Q_load <= 200.0:
        raise ValueError(f"Q_load must be a positive thermal duty in watts, got {Q_load}")

    cons = constraints or Constraints()
    fluid = resolve_fluid(refrigerant)
    info, states, meta, notes = design_cycle(
        fluid,
        T_out,
        T_zone,
        mode=kind,
        SH=SH,
        SC=SC,
        DT_evap=DT_evap,
        DT_cond=DT_cond,
        eta_is=eta_is,
        constraints=cons,
    )
    Q_unit = float(oversize) * float(Q_load)
    if kind == "cooling":
        mdot = Q_unit / meta["q_evap"]
        Q_evap = Q_unit
        Q_cond = mdot * meta["q_cond"]
    else:
        mdot = Q_unit / meta["q_cond"]
        Q_cond = Q_unit
        Q_evap = mdot * meta["q_evap"]
    W = mdot * meta["w"]

    st1, st2, st3, st4 = states["1"], states["2"], states["3"], states["4"]
    pr = meta["pr"]
    gamma = meta["gamma"]
    eta_v = 1.0 - C_loss * (pr ** (1.0 / gamma) - 1.0)
    if eta_v <= 0.0:
        raise ValueError(
            f"{info.name}: clearance volumetric efficiency is non-positive at PR={pr:.2f}."
        )
    V_disp = mdot / (eta_v * max(st1.rho, 0.5) * max(N_hz, 10.0))

    dp = max(meta["p_c"] - meta["p_e"], 1.0e5)
    A_eev = mdot / (Cd * _U_EEV_DESIGN * np.sqrt(2.0 * max(st3.rho, 50.0) * dp))

    T_air_e = T_zone if kind == "cooling" else T_out
    T_air_c = T_out if kind == "cooling" else T_zone
    evap_s = size_coil(
        Q_evap,
        T_air_e,
        meta["T_e"],
        mdot,
        fluid=info.name,
        p=meta["p_e"],
        h=0.5 * (st4.h + st1.h),
        pc=info.pc,
        evaporating=True,
        D=D,
        L=L_e,
        fin=fin_e,
    )
    cond_s = size_coil(
        Q_cond,
        T_air_c,
        meta["T_c"],
        mdot,
        fluid=info.name,
        p=meta["p_c"],
        h=0.5 * (st2.h + st3.h),
        pc=info.pc,
        evaporating=False,
        D=D,
        L=L_c,
        fin=fin_c,
    )
    evap = _coil(n_e, evap_s, fin_e, air_props(T_air_e)["cp"])
    cond = _coil(n_c, cond_s, fin_c, air_props(T_air_c)["cp"])
    if kind == "cooling":
        indoor, outdoor = evap, cond
    else:
        indoor, outdoor = cond, evap
    UA = _envelope_UA(Q_load, T_zone, T_out, UA_env)
    Cz = _zone_C(T_zone, C_zone, V_zone, notes)

    spec = PlantSpec(
        fluid=info.name,
        mode=kind,
        p_crit=info.pc,
        T_crit=info.Tc,
        gamma=gamma,
        indoor=indoor,
        outdoor=outdoor,
        V_disp=float(V_disp),
        C_loss=C_loss,
        eta_is0=eta_is,
        A_eev=float(A_eev),
        Cd=Cd,
        C_zone=Cz,
        UA_env=UA,
        N_design=N_hz,
    )
    spec = apply_operating_mode(spec, kind)
    if spec_overrides:
        spec = replace(spec, **spec_overrides)

    G_e = mdot / max(spec.n_tubes_e * 0.25 * pi * spec.D_e**2, 1e-10)
    G_c = mdot / max(spec.n_tubes_c * 0.25 * pi * spec.D_c**2, 1e-10)
    charge = charge_from_profile(
        info.name,
        states,
        spec.V_e,
        spec.V_c,
        spec.V_header_e,
        spec.V_header_c,
        spec.n_e,
        spec.n_c,
    )
    gates = evaluate_gates(
        info=info,
        p_c=meta["p_c"],
        pr=meta["pr"],
        T_disch=states["2"].T,
        SH=meta["SH"],
        SC=meta["SC"],
        G_e=G_e,
        G_c=G_c,
        constraints=cons,
    )
    raise_if_failed(gates)

    return DesignReport(
        fluid=info.name,
        Q_load=float(Q_load),
        Q_heat=float(Q_cond),
        Q_evap=float(Q_evap),
        W=float(W),
        COP=float(meta["COP"]),
        mdot=float(mdot),
        p_e=float(meta["p_e"]),
        p_c=float(meta["p_c"]),
        T_e=float(meta["T_e"]),
        T_c=float(meta["T_c"]),
        T_suct=states["1"].T,
        T_disch=states["2"].T,
        SH=float(meta["SH"]),
        SC=float(meta["SC"]),
        glide_e=float(meta["glide_e"]),
        glide_c=float(meta["glide_c"]),
        pr=float(meta["pr"]),
        gamma=float(gamma),
        V_disp=float(spec.V_disp),
        A_eev=float(spec.A_eev),
        n_tubes_e=spec.n_tubes_e,
        n_tubes_c=spec.n_tubes_c,
        L_e=spec.L_e,
        L_c=spec.L_c,
        D=spec.D_e,
        spec=spec,
        notes=tuple(notes),
        states=states,
        kind=kind,
        Q_cool=float(Q_evap if kind == "cooling" else 0.0),
        G_e=float(G_e),
        G_c=float(G_c),
        charge_kg=float(charge),
        gates=gates,
    )


def design_heat_pump(
    refrigerant: str,
    Q_load: float,
    T_out: float = 273.15,
    T_zone: float = 293.15,
    **kwargs,
) -> DesignReport:
    """Size compressor, EEV, coils, and zone for a heating load [W]."""
    return _design_mode(refrigerant, "heating", Q_load, T_out, T_zone, **kwargs)


def design_air_conditioner(
    refrigerant: str,
    Q_load: float | None = None,
    T_out: float = 308.15,
    T_zone: float = 297.15,
    *,
    cooling_tons: float | None = None,
    **kwargs,
) -> DesignReport:
    """Size an air conditioner for a cooling load [W] or refrigeration tons."""
    if Q_load is None:
        if cooling_tons is None:
            raise ValueError("provide Q_load [W] or cooling_tons")
        Q_load = cooling_tons_to_w(cooling_tons)
    return _design_mode(refrigerant, "cooling", Q_load, T_out, T_zone, **kwargs)


def choose_controller(request: DesignRequest) -> str:
    name = (request.controller or "pid").lower()
    if name in ("auto", ""):
        return "pid"
    return name


def _merge_gates(a: GateSet | None, b: GateSet | None) -> GateSet:
    if a is None:
        return b or GateSet(())
    if b is None:
        return a
    heat = tuple(replace(g, name=f"heating.{g.name}") for g in a.gates)
    cool = tuple(replace(g, name=f"cooling.{g.name}") for g in b.gates)
    return GateSet(heat + cool)


@dataclass
class SystemDesign:
    """Hardware + controller for heating, cooling, or a reversible heat pump."""

    request: DesignRequest
    spec: PlantSpec
    controller: str
    heating: DesignReport | None
    cooling: DesignReport | None
    notes: tuple[str, ...]
    gates: GateSet = field(default_factory=lambda: GateSet(()))

    @property
    def fluid(self) -> str:
        return self.spec.fluid

    def summary(self) -> str:
        lines = [
            f"System  {self.fluid}  ·  mode={self.request.mode}  ·  controller={self.controller}",
            f"  Compressor V_disp={self.spec.V_disp*1e6:.1f} cm³/rev  EEV A={self.spec.A_eev*1e6:.2f} mm²",
            f"  Indoor  {self.spec.indoor.n_tubes if self.spec.indoor else self.spec.n_tubes_c} tubes  "
            f"outdoor {self.spec.outdoor.n_tubes if self.spec.outdoor else self.spec.n_tubes_e} tubes",
            f"  Zone UA={self.spec.UA_env:.1f} W/K  C={self.spec.C_zone/1e3:.1f} kJ/K",
        ]
        if self.heating is not None:
            lines.append("  Heating cycle:")
            lines.append("    " + self.heating.summary().replace("\n", "\n    "))
        if self.cooling is not None:
            lines.append("  Cooling cycle:")
            lines.append("    " + self.cooling.summary().replace("\n", "\n    "))
        if self.gates.gates:
            lines.append("  Gates:")
            lines.append(self.gates.summary())
        if self.notes:
            lines.append("  Notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)

    def as_report(self):
        from heatpump.report import build_report

        return build_report(self)


def design_system(request: DesignRequest | None = None, **kwargs) -> SystemDesign:
    """Size a heating, cooling, or reversible unit from a ``DesignRequest``."""
    from heatpump.plant import PlantSpec, apply_operating_mode

    req = request if request is not None else DesignRequest(**kwargs)
    if req.mode == "auto":
        raise ValueError(
            "mode='auto' needs a load/ambient timeseries so capacity can be inferred "
            "from the setpoint. Pass timeseries= or --weather, or set mode explicitly."
        )
    notes: list[str] = []
    if req.inferred_from_profile:
        if req.Q_cool:
            notes.append(
                f"Cooling capacity {req.Q_cool/1e3:.2f} kW inferred to hold "
                f"{req.T_zone_cool-273.15:.1f}°C at {req.T_out_cool-273.15:.1f}°C outdoor."
            )
        if req.Q_heat:
            notes.append(
                f"Heating capacity {req.Q_heat/1e3:.2f} kW inferred to hold "
                f"{req.T_zone_heat-273.15:.1f}°C at {req.T_out_heat-273.15:.1f}°C outdoor."
            )
    heating = cooling = None
    common = dict(
        oversize=req.oversize,
        SH=req.SH,
        SC=req.SC,
        DT_evap=req.DT_evap,
        DT_cond=req.DT_cond,
        N_hz=req.N_hz,
        n_e=req.n_cells,
        n_c=req.n_cells,
        constraints=req.constraints,
        UA_env=req.UA_env,
        C_zone=req.C_zone,
        V_zone=req.V_zone,
    )

    if req.mode in ("heating", "heat_pump"):
        if req.Q_heat is None or req.Q_heat < 200.0:
            raise ValueError(
                "No heating duty to size from. Provide a load/ambient timeseries "
                "(positive heating_load or negative Q_gain) at the setpoint, or Q_heat [W]."
            )
        heating = _design_mode(req.refrigerant, "heating", req.Q_heat, req.T_out_heat, req.T_zone_heat, **common)
        notes.extend(heating.notes)
    if req.mode in ("cooling", "heat_pump"):
        if req.Q_cool is None or req.Q_cool < 200.0:
            raise ValueError(
                "No cooling duty to size from. Provide a load/ambient timeseries "
                "(positive Q_gain / cooling_load) at the setpoint, or Q_cool [W] / cooling_tons."
            )
        cooling = _design_mode(req.refrigerant, "cooling", req.Q_cool, req.T_out_cool, req.T_zone_cool, **common)
        notes.extend(cooling.notes)

    if heating is not None and cooling is not None:
        indoor = _merge_coil(heating.spec.indoor, cooling.spec.indoor)
        outdoor = _merge_coil(heating.spec.outdoor, cooling.spec.outdoor)
        V_disp = max(heating.V_disp, cooling.V_disp)
        A_eev = max(heating.A_eev, cooling.A_eev)
        gamma = 0.5 * (heating.gamma + cooling.gamma)
        UA = req.UA_env if req.UA_env is not None else 0.5 * (heating.spec.UA_env + cooling.spec.UA_env)
        Cz = req.C_zone if req.C_zone is not None else max(heating.spec.C_zone, cooling.spec.C_zone)
        operate = "heating"
        spec = PlantSpec(
            fluid=heating.fluid,
            mode=operate,
            p_crit=heating.spec.p_crit,
            T_crit=heating.spec.T_crit,
            gamma=gamma,
            indoor=indoor,
            outdoor=outdoor,
            V_disp=V_disp,
            A_eev=A_eev,
            C_loss=heating.spec.C_loss,
            eta_is0=heating.spec.eta_is0,
            Cd=heating.spec.Cd,
            C_zone=Cz,
            UA_env=UA if req.use_envelope else 0.0,
            N_design=req.N_hz,
        )
        spec = apply_operating_mode(spec, operate)
        notes.append("Reversible unit: indoor/outdoor coils and compressor sized for the harder of the two duties.")
    elif heating is not None:
        spec = heating.spec
        if not req.use_envelope:
            spec = replace(spec, UA_env=0.0)
    else:
        spec = cooling.spec
        if not req.use_envelope:
            spec = replace(spec, UA_env=0.0)

    controller = choose_controller(req)
    if heating is not None:
        heating = replace(heating, controller=controller)
    if cooling is not None:
        cooling = replace(cooling, controller=controller)
    return SystemDesign(
        request=req,
        spec=spec,
        controller=controller,
        heating=heating,
        cooling=cooling,
        notes=tuple(dict.fromkeys(notes)),
        gates=_merge_gates(
            heating.gates if heating is not None else None,
            cooling.gates if cooling is not None else None,
        ),
    )


_DESIGN_KEYS = {
    "T_out",
    "T_zone",
    "oversize",
    "SH",
    "SC",
    "DT_evap",
    "DT_cond",
    "N_hz",
    "n_e",
    "n_c",
    "eta_is",
    "C_loss",
    "Cd",
    "UA_env",
    "C_zone",
    "V_zone",
    "constraints",
    "D",
    "L_e",
    "L_c",
    "fin_e",
    "fin_c",
}


def heating_spec(
    fluid: str = "R32",
    Q_load: float = 5500.0,
    **kwargs,
) -> PlantSpec:
    """Size a heating plant, then apply any explicit ``PlantSpec`` overrides."""
    design_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _DESIGN_KEYS}
    report = design_heat_pump(fluid, Q_load, **design_kw)
    if kwargs:
        allowed = {f.name for f in fields(type(report.spec))}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"unexpected PlantSpec fields: {sorted(unknown)}")
        return replace(report.spec, **kwargs)
    return report.spec
