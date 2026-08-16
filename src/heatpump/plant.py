"""Finite-volume two-phase vapor-compression plant (method of lines).

Each heat exchanger is a 1-D homogeneous-equilibrium channel with a single
pressure (acoustic equilibrium) and a distributed enthalpy / wall-temperature
field. Cell mass and energy are the integral form of

    ∂ρ/∂t + ∂(ρ v)/∂z = 0
    ∂(ρ h)/∂t + ∂(ρ v h)/∂z = ∂p/∂t + (P/A) q''

Internal mass flow is closed by a linear profile between the known port
flows (compressor, valve); overall mass then determines dp/dt. This is the
standard robust HVAC finite-volume model (Bendapudi, Rasmussen, Qiao).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import Callable, NamedTuple

import jax.numpy as jnp
from jax import Array, vmap

from heatpump.components import compressor_mdot_h, eev_mdot, htc_two_phase
from heatpump.devices.air import series_ua_air_q
from heatpump.thermo import PropertyTables, eval_ph, p_sat_from_tables, sat_from_tables

# Wall temperature is slaved to the series-UA equilibrium. The relaxation
# time is the physical C_w / (hA) floor so the DAE stays integrable on
# hour-scale steps; heat to the refrigerant uses the equilibrium Q.
_TAU_W_MIN = 2.0


class Layout(NamedTuple):
    n_e: int
    n_c: int
    moist: int = 0
    frost: int = 0

    @property
    def n_state(self) -> int:
        return 3 + 2 * self.n_e + 2 * self.n_c + self.moist + self.frost

    @property
    def i_pe(self) -> int:
        return 0

    @property
    def sl_he(self) -> slice:
        return slice(1, 1 + self.n_e)

    @property
    def sl_twe(self) -> slice:
        return slice(1 + self.n_e, 1 + 2 * self.n_e)

    @property
    def i_pc(self) -> int:
        return 1 + 2 * self.n_e

    @property
    def sl_hc(self) -> slice:
        a = 2 + 2 * self.n_e
        return slice(a, a + self.n_c)

    @property
    def sl_twc(self) -> slice:
        a = 2 + 2 * self.n_e + self.n_c
        return slice(a, a + self.n_c)

    @property
    def i_tz(self) -> int:
        return 2 + 2 * self.n_e + 2 * self.n_c

    @property
    def i_wz(self) -> int:
        return self.i_tz + 1

    @property
    def i_fr(self) -> int:
        return self.i_tz + 1 + self.moist

    def slow_idx(self) -> tuple[int, ...]:
        idx = [self.i_tz]
        if self.moist:
            idx.append(self.i_wz)
        if self.frost:
            idx.append(self.i_fr)
        return tuple(idx)


@dataclass
class CoilSpec:
    """One air-to-refrigerant coil (indoor or outdoor)."""

    n: int = 6
    D: float = 0.007
    L: float = 1.20
    n_tubes: int = 56
    fin: float = 16.0
    V_header: float = 3.0e-4
    C_w: float = 2200.0
    mdot_air0: float = 0.70
    htc_air: float = 90.0
    cp_air: float = 1006.0

    @property
    def V(self) -> float:
        return float(self.n_tubes * 0.25 * pi * self.D**2 * self.L)

    @property
    def A_ref(self) -> float:
        return float(self.n_tubes * pi * self.D * self.L)

    @property
    def A_air(self) -> float:
        return self.A_ref * self.fin


def _norm_mode(mode: str) -> str:
    m = (mode or "heating").strip().lower()
    if m in ("cool", "cooling", "ac"):
        return "cooling"
    return "heating"


@dataclass
class PlantSpec:
    """Geometry and maps for an air-source heat pump / air conditioner.

    Thermodynamic coils ``*_e`` / ``*_c`` are the evaporator and condenser
    *in the current operating mode*. Indoor / outdoor hardware is stored
    separately so a reversible unit can be remapped by ``apply_operating_mode``.

    Device slots (``compressor``, ``expansion``, ``htc_ref``, ``air_side``,
    ``zone_model``, ``fan_indoor``, ``fan_outdoor``) are optional objects.
    ``None`` uses the built-in kernel (no registry, no dispatch). Assign a
    replacement to retrofit: ``replace(spec, compressor=MyMap(...))``.
    """

    fluid: str = "R32"
    mode: str = "heating"
    p_crit: float = 5.783e6
    T_crit: float = 351.26
    gamma: float = 1.25
    indoor: CoilSpec | None = None
    outdoor: CoilSpec | None = None
    n_e: int = 6
    n_c: int = 6
    D_e: float = 0.007
    D_c: float = 0.007
    L_e: float = 1.20
    L_c: float = 1.50
    n_tubes_e: int = 56
    n_tubes_c: int = 80
    fin_e: float = 16.0
    fin_c: float = 18.0
    V_header_e: float = 3.0e-4
    V_header_c: float = 7.0e-4
    V_disp: float = 3.8e-5
    C_loss: float = 0.075
    eta_is0: float = 0.70
    A_eev: float = 1.3e-6
    Cd: float = 0.70
    C_w_e: float = 2200.0
    C_w_c: float = 2800.0
    C_zone: float = 1.9e5
    UA_env: float = 110.0
    mdot_air_e0: float = 0.75
    mdot_air_c0: float = 0.70
    htc_air_e: float = 80.0
    htc_air_c: float = 115.0
    cp_air_e: float = 1006.0
    cp_air_c: float = 1006.0
    N_design: float = 50.0
    compressor: object | None = None
    expansion: object | None = None
    htc_ref: object | None = None
    zone_model: object | None = None
    air_side: object | None = None
    fan_indoor: object | None = None
    fan_outdoor: object | None = None
    moist: bool = False
    frost: bool = False
    RH_zone0: float | None = None
    RH_out: float | None = None
    W_defrost: float = 0.0
    frost_closure: str = "hayashi"
    V_zone: float | None = None
    ahri540_path: str | None = None
    fan_path: str | None = None
    fan_citation: str | None = None

    @property
    def layout(self) -> Layout:
        return Layout(self.n_e, self.n_c, int(bool(self.moist)), int(bool(self.frost)))

    @property
    def V_e(self) -> float:
        return float(self.n_tubes_e * 0.25 * pi * self.D_e**2 * self.L_e)

    @property
    def V_c(self) -> float:
        return float(self.n_tubes_c * 0.25 * pi * self.D_c**2 * self.L_c)

    @property
    def A_ref_e(self) -> float:
        return float(self.n_tubes_e * pi * self.D_e * self.L_e)

    @property
    def A_ref_c(self) -> float:
        return float(self.n_tubes_c * pi * self.D_c * self.L_c)

    @property
    def A_air_e(self) -> float:
        return self.A_ref_e * self.fin_e

    @property
    def A_air_c(self) -> float:
        return self.A_ref_c * self.fin_c

    @property
    def operating_mode(self) -> str:
        return _norm_mode(self.mode)


def _coil_to_evap(coil: CoilSpec) -> dict:
    return {
        "n_e": coil.n,
        "D_e": coil.D,
        "L_e": coil.L,
        "n_tubes_e": coil.n_tubes,
        "fin_e": coil.fin,
        "V_header_e": coil.V_header,
        "C_w_e": coil.C_w,
        "mdot_air_e0": coil.mdot_air0,
        "htc_air_e": coil.htc_air,
        "cp_air_e": coil.cp_air,
    }


def _coil_to_cond(coil: CoilSpec) -> dict:
    return {
        "n_c": coil.n,
        "D_c": coil.D,
        "L_c": coil.L,
        "n_tubes_c": coil.n_tubes,
        "fin_c": coil.fin,
        "V_header_c": coil.V_header,
        "C_w_c": coil.C_w,
        "mdot_air_c0": coil.mdot_air0,
        "htc_air_c": coil.htc_air,
        "cp_air_c": coil.cp_air,
    }


def apply_operating_mode(spec: PlantSpec, mode: str) -> PlantSpec:
    """Map indoor/outdoor hardware onto evaporator/condenser for ``mode``."""
    mode = _norm_mode(mode)
    if spec.indoor is None or spec.outdoor is None:
        return replace(spec, mode=mode)
    if mode == "cooling":
        evap, cond = spec.indoor, spec.outdoor
    else:
        evap, cond = spec.outdoor, spec.indoor
    n = max(evap.n, cond.n)
    if evap.n != n:
        evap = replace(evap, n=n)
    if cond.n != n:
        cond = replace(cond, n=n)
    return replace(spec, mode=mode, **_coil_to_evap(evap), **_coil_to_cond(cond))


def pack_state(p_e, h_e, Tw_e, p_c, h_c, Tw_c, T_z, W_z=None, m_fr=None) -> Array:
    parts = [
        jnp.asarray([p_e], dtype=jnp.float64),
        jnp.asarray(h_e, dtype=jnp.float64),
        jnp.asarray(Tw_e, dtype=jnp.float64),
        jnp.asarray([p_c], dtype=jnp.float64),
        jnp.asarray(h_c, dtype=jnp.float64),
        jnp.asarray(Tw_c, dtype=jnp.float64),
        jnp.asarray([T_z], dtype=jnp.float64),
    ]
    if W_z is not None:
        parts.append(jnp.asarray([W_z], dtype=jnp.float64))
    if m_fr is not None:
        parts.append(jnp.asarray([m_fr], dtype=jnp.float64))
    return jnp.concatenate(parts)


def _resample(arr: Array, n: int) -> Array:
    arr = jnp.asarray(arr, dtype=jnp.float64).reshape(-1)
    if arr.size == n:
        return arr
    x0 = jnp.linspace(0.0, 1.0, arr.size)
    x1 = jnp.linspace(0.0, 1.0, n)
    return jnp.interp(x1, x0, arr)


def remap_state(y: Array, lay_from: Layout, lay_to: Layout) -> Array:
    """Swap evaporator/condenser fields after a reversing-valve change.

    Indoor and outdoor coils keep their refrigerant and wall states; they
    trade evap/cond roles. Humidity and frost mass stay on the zone /
    outdoor coil (scalar). Cell counts may differ; profiles are interpolated.
    """
    s = unpack_state(y, lay_from)
    return pack_state(
        s["p_c"],
        _resample(s["h_c"], lay_to.n_e),
        _resample(s["Tw_c"], lay_to.n_e),
        s["p_e"],
        _resample(s["h_e"], lay_to.n_c),
        _resample(s["Tw_e"], lay_to.n_c),
        s["T_z"],
        s.get("W_z"),
        s.get("m_fr"),
    )


def unpack_state(y: Array, lay: Layout) -> dict[str, Array]:
    out = {
        "p_e": y[lay.i_pe],
        "h_e": y[lay.sl_he],
        "Tw_e": y[lay.sl_twe],
        "p_c": y[lay.i_pc],
        "h_c": y[lay.sl_hc],
        "Tw_c": y[lay.sl_twc],
        "T_z": y[lay.i_tz],
    }
    if lay.moist:
        out["W_z"] = y[lay.i_wz]
    if lay.frost:
        out["m_fr"] = y[lay.i_fr]
    return out


def project_state(y: Array, tables: PropertyTables, lay: Layout) -> Array:
    y = y.at[lay.i_pe].set(jnp.clip(y[lay.i_pe], tables.p[0], tables.p[-1]))
    y = y.at[lay.i_pc].set(jnp.clip(y[lay.i_pc], tables.p[0], tables.p[-1]))
    y = y.at[lay.sl_he].set(jnp.clip(y[lay.sl_he], tables.h[0], tables.h[-1]))
    y = y.at[lay.sl_hc].set(jnp.clip(y[lay.sl_hc], tables.h[0], tables.h[-1]))
    y = y.at[lay.sl_twe].set(jnp.clip(y[lay.sl_twe], 200.0, 360.0))
    y = y.at[lay.sl_twc].set(jnp.clip(y[lay.sl_twc], 200.0, 360.0))
    y = y.at[lay.i_tz].set(jnp.clip(y[lay.i_tz], 250.0, 320.0))
    if lay.moist:
        y = y.at[lay.i_wz].set(jnp.clip(y[lay.i_wz], 1.0e-6, 0.04))
    if lay.frost:
        y = y.at[lay.i_fr].set(jnp.clip(y[lay.i_fr], 0.0, 80.0))
    return y


def _component_maps(spec: PlantSpec):
    """Bind residual closures. Defaults are the kernels; slots replace them.

    Captured ``None`` checks are Python-static, so the JIT default path is
    a direct call into ``components`` / ``series_ua_air_q``.
    """
    comp = spec.compressor
    if comp is None:
        Vd, Cl, eta, gam = spec.V_disp, spec.C_loss, spec.eta_is0, spec.gamma

        def compressor(p_s, p_d, h_s, rho_s, T_s, N, Tsat_s, Tsat_d):
            del Tsat_s, Tsat_d
            return compressor_mdot_h(p_s, p_d, h_s, rho_s, T_s, N, Vd, Cl, eta, gam)
    else:

        def compressor(p_s, p_d, h_s, rho_s, T_s, N, Tsat_s, Tsat_d):
            return comp.map(p_s, p_d, h_s, rho_s, T_s, N, Tsat_s, Tsat_d)

    exp = spec.expansion
    if exp is None:
        A, Cd = spec.A_eev, spec.Cd

        def eev(p_in, p_out, rho_in, opening):
            return eev_mdot(p_in, p_out, rho_in, opening, A, Cd)
    else:

        def eev(p_in, p_out, rho_in, opening):
            return exp.map(p_in, p_out, rho_in, opening)

    htc_obj = spec.htc_ref
    if htc_obj is None:

        def htc(G, D, mu, k, cp, x, p_r, evaporating):
            return htc_two_phase(G, D, mu, k, cp, x, p_r, evaporating)
    else:

        def htc(G, D, mu, k, cp, x, p_r, evaporating):
            return htc_obj.htc(G, D, mu, k, cp, x, p_r, evaporating)

    air = spec.air_side
    if air is None:

        def air_q(T_air, T_ref, h_ref, A_ref, h_air, A_air, mdot_air, cp_air):
            return series_ua_air_q(T_air, T_ref, h_ref, A_ref, h_air, A_air, mdot_air, cp_air)
    else:

        def air_q(T_air, T_ref, h_ref, A_ref, h_air, A_air, mdot_air, cp_air):
            return air.heat_rate(T_air, T_ref, h_ref, A_ref, h_air, A_air, mdot_air, cp_air)

    zone = spec.zone_model
    if zone is None:
        C, UA = spec.C_zone, spec.UA_env

        def dTdt(T_z, T_out, Q_hvac, Q_gain):
            return (Q_hvac + Q_gain + UA * (T_out - T_z)) / C
    else:

        def dTdt(T_z, T_out, Q_hvac, Q_gain):
            return zone.dTdt(T_z, T_out, Q_hvac, Q_gain)

    fan_i, fan_o = spec.fan_indoor, spec.fan_outdoor

    def mdot_i(speed, m0):
        return fan_i.mdot(speed, m0) if fan_i is not None else m0 * speed

    def mdot_o(speed, m0):
        return fan_o.mdot(speed, m0) if fan_o is not None else m0 * speed

    return compressor, eev, htc, air_q, dTdt, mdot_i, mdot_o


def _require_humid(spec: PlantSpec) -> None:
    if spec.frost and not spec.moist:
        raise ValueError(
            "frost requires moist=True; frost mass grows from outdoor humidity"
        )
    if spec.moist or spec.frost:
        if spec.RH_out is None:
            raise ValueError(
                "moist/frost plant requires user RH_out; outdoor humidity is not defaulted"
            )
        if spec.moist and spec.RH_zone0 is None:
            raise ValueError(
                "moist plant requires user RH_zone0; indoor humidity is not defaulted"
            )


def hx_derivatives(
    h: Array,
    props,
    V_cells: Array,
    V_header: float,
    m_in: Array,
    m_out: Array,
    h_in: Array,
    Q_to_ref: Array,
) -> tuple[Array, Array]:
    """Linear port-flow closure plus overall mass conservation.

    Energy per cell (well-mixed, upwind inlet)::

        ρ V dh/dt = ṁ_in (h_up − h) + Q + V dp/dt

    Header mass is V_h ⟨ρ⟩ and is included in both ∂ρ/∂p and ∂ρ/∂h terms so
    the coil inventory stays conservative.
    """
    n = h.size
    frac = jnp.arange(n + 1, dtype=h.dtype) / jnp.float64(n)
    m_iface = m_in * (1.0 - frac) + m_out * frac
    m_cell_in = m_iface[:-1]
    h_prev = jnp.concatenate([jnp.reshape(h_in, (1,)), h[:-1]])
    adv = m_cell_in * (h_prev - h)
    rhoV = jnp.maximum(props.rho * V_cells, 1.0e-8)
    inv_rho = 1.0 / jnp.maximum(props.rho, 1.0)
    rhs_h = (adv + Q_to_ref) / rhoV
    dM = m_in - m_out
    cap_p = jnp.sum(V_cells * props.drho_dp) + V_header * jnp.mean(props.drho_dp)
    num = (
        dM
        - jnp.sum(V_cells * props.drho_dh * rhs_h)
        - V_header * jnp.mean(props.drho_dh * rhs_h)
    )
    den = (
        cap_p
        + jnp.sum(V_cells * props.drho_dh * inv_rho)
        + V_header * jnp.mean(props.drho_dh * inv_rho)
    )
    pdot = num / jnp.clip(den, 1.0e-10, 1.0e3)
    hdot = rhs_h + pdot * inv_rho
    return pdot, hdot


def make_rhs(
    spec: PlantSpec, tables: PropertyTables
) -> Callable[[Array, Array, Array], Array]:
    """Return ``rhs(t, y, u)`` with ``u = [N, eev, fan_i, fan_o, T_out, Q_load]``.

    Optional extras (moist / frost): ``u[6:]= [W_gain, defrost, RH_out]``.
    Dry plants keep the six-vector; humidity is never defaulted.
    """
    _require_humid(spec)
    compressor, eev, htc, air_q, dTdt, mdot_i, mdot_o = _component_maps(spec)
    lay = spec.layout
    n_e, n_c = spec.n_e, spec.n_c
    V_e = jnp.full((n_e,), spec.V_e / n_e)
    V_c = jnp.full((n_c,), spec.V_c / n_c)
    A_ref_e = jnp.full((n_e,), spec.A_ref_e / n_e)
    A_ref_c = jnp.full((n_c,), spec.A_ref_c / n_c)
    A_air_e = jnp.full((n_e,), spec.A_air_e / n_e)
    A_air_c = jnp.full((n_c,), spec.A_air_c / n_c)
    A_cross_e = float(spec.n_tubes_e * 0.25 * pi * spec.D_e**2)
    A_cross_c = float(spec.n_tubes_c * 0.25 * pi * spec.D_c**2)
    pc_crit = float(tables.pc)
    cooling = spec.operating_mode == "cooling"
    tau_w_e = max(float(spec.C_w_e) / max(n_e * 400.0, 1.0), _TAU_W_MIN)
    tau_w_c = max(float(spec.C_w_c) / max(n_c * 400.0, 1.0), _TAU_W_MIN)
    moist = bool(spec.moist)
    frost = bool(spec.frost)
    RH_out0 = None if spec.RH_out is None else float(spec.RH_out)
    W_defrost = float(spec.W_defrost)
    frost_closure = spec.frost_closure
    A_air_e_tot = float(spec.A_air_e)
    A_air_c_tot = float(spec.A_air_c)

    humid_fns = None
    rhoV_z = 1.0
    if moist or frost:
        from heatpump.hx import air_props
        from heatpump.psychro import build_humid_tables, jax_humid_fns

        humid_fns = jax_humid_fns(build_humid_tables())
        T_ref = 293.15
        ap = air_props(T_ref)
        if spec.V_zone is not None:
            Vz = float(spec.V_zone)
        else:
            Vz = float(spec.C_zone) / max(ap["rho"] * ap["cp"], 1.0)
        rhoV_z = ap["rho"] * Vz

    def rhs(t: Array, y: Array, u: Array) -> Array:
        del t
        y = project_state(y, tables, lay)
        s = unpack_state(y, lay)
        p_e, h_e, Tw_e = s["p_e"], s["h_e"], s["Tw_e"]
        p_c, h_c, Tw_c = s["p_c"], s["h_c"], s["Tw_c"]
        T_z = s["T_z"]

        N, eev_u, fan_i, fan_o = u[0], u[1], u[2], u[3]
        T_out, Q_load = u[4], u[5]
        W_gain = u[6] if u.size > 6 else 0.0
        defrost = u[7] if u.size > 7 else 0.0
        RH_out = u[8] if u.size > 8 else (0.0 if RH_out0 is None else RH_out0)
        fan_i = jnp.clip(fan_i, 0.15, 1.2)
        fan_o = jnp.clip(fan_o, 0.15, 1.2)

        pe = vmap(lambda hi: eval_ph(tables, p_e, hi))(h_e)
        pc = vmap(lambda hi: eval_ph(tables, p_c, hi))(h_c)
        suct = eval_ph(tables, p_e, h_e[-1])
        c_out = eval_ph(tables, p_c, h_c[-1])

        m_comp, h_disch, _pwr = compressor(
            p_e, p_c, h_e[-1], suct.rho, suct.T, N, suct.Tsat, pc.Tsat[0]
        )
        m_eev = eev(p_c, p_e, c_out.rho, eev_u)
        h_eev = h_c[-1]

        G_e = 0.5 * (m_eev + m_comp) / max(A_cross_e, 1.0e-8)
        G_c = 0.5 * (m_comp + m_eev) / max(A_cross_c, 1.0e-8)
        htc_e = htc(G_e, spec.D_e, pe.mu, pe.k, pe.cp, pe.x, p_e / pc_crit, True)
        htc_c = htc(G_c, spec.D_c, pc.mu, pc.k, pc.cp, pc.x, p_c / pc_crit, False)

        if cooling:
            T_air_e, T_air_c = T_z, T_out
            mdot_e = mdot_i(fan_i, spec.mdot_air_e0)
            mdot_c = mdot_o(fan_o, spec.mdot_air_c0)
            fan_e, fan_c = fan_i, fan_o
        else:
            T_air_e, T_air_c = T_out, T_z
            mdot_e = mdot_o(fan_o, spec.mdot_air_e0)
            mdot_c = mdot_i(fan_i, spec.mdot_air_c0)
            fan_e, fan_c = fan_o, fan_i
        htc_ae = jnp.full((n_e,), spec.htc_air_e * fan_e)
        htc_ac = jnp.full((n_c,), spec.htc_air_c * fan_c)

        Q_lat_e = jnp.zeros((n_e,))
        Q_lat_c = jnp.zeros((n_c,))
        W_indoor_out = s["W_z"] if moist else 0.0
        dm_fr = 0.0
        if moist or frost:
            from heatpump.devices.frost import frost_layer
            from heatpump.psychro import wet_coil_march

            W_sat_T, h_fg_T, W_of, T_dp, h_if = humid_fns
            W_z = s["W_z"] if moist else W_of(T_z, jnp.asarray(0.5))
            W_amb = W_of(T_out, jnp.clip(RH_out, 0.05, 0.99))
            if cooling:
                W_in_e, W_in_c = W_z, W_amb
            else:
                W_in_e, W_in_c = W_amb, W_z
            if frost:
                Tw_out = Tw_e if not cooling else Tw_c
                A_out = A_air_e_tot if not cooling else A_air_c_tot
                delta, k_fr, _rho = frost_layer(
                    jnp.mean(Tw_out), s["m_fr"], A_out, frost_closure
                )
                r_fr = delta / jnp.maximum(k_fr, 1.0e-6)
                if cooling:
                    htc_ac = 1.0 / (1.0 / jnp.maximum(htc_ac, 1.0e-6) + r_fr)
                else:
                    htc_ae = 1.0 / (1.0 / jnp.maximum(htc_ae, 1.0e-6) + r_fr)
            W_e_out, Q_lat_e = wet_coil_march(pe.T, W_in_e, mdot_e, W_sat_T, h_fg_T, T_dp)
            W_c_out, Q_lat_c = wet_coil_march(pc.T, W_in_c, mdot_c, W_sat_T, h_fg_T, T_dp)
            W_indoor_out = W_e_out if cooling else W_c_out
            if frost:
                W_out_air = W_e_out if not cooling else W_c_out
                W_in_o = W_in_e if not cooling else W_in_c
                Tw_mean = jnp.mean(Tw_e if not cooling else Tw_c)
                grow = jnp.where(
                    Tw_mean < 273.15,
                    jnp.maximum(mdot_e if not cooling else mdot_c, 0.0)
                    * jnp.maximum(W_in_o - W_out_air, 0.0),
                    0.0,
                )
                melt = jnp.where(defrost > 0.5, W_defrost / max(h_if, 1.0), 0.0)
                dm_fr = grow - melt

        Q_air_e, _ = air_q(
            T_air_e, pe.T, htc_e, A_ref_e, htc_ae, A_air_e, mdot_e, spec.cp_air_e
        )
        Q_air_c, _ = air_q(
            T_air_c, pc.T, htc_c, A_ref_c, htc_ac, A_air_c, mdot_c, spec.cp_air_c
        )
        Q_ref_e = Q_air_e + Q_lat_e
        Q_ref_c = Q_air_c + Q_lat_c
        Tw_ss_e = pe.T + Q_ref_e / jnp.maximum(htc_e * A_ref_e, 1.0e-6)
        Tw_ss_c = pc.T + Q_ref_c / jnp.maximum(htc_c * A_ref_c, 1.0e-6)

        pdot_e, hdot_e = hx_derivatives(
            h_e, pe, V_e, spec.V_header_e, m_eev, m_comp, h_eev, Q_ref_e
        )
        pdot_c, hdot_c = hx_derivatives(
            h_c, pc, V_c, spec.V_header_c, m_comp, m_eev, h_disch, Q_ref_c
        )

        Twdot_e = (Tw_ss_e - Tw_e) / tau_w_e
        Twdot_c = (Tw_ss_c - Tw_c) / tau_w_c
        Q_to_zone = -jnp.sum(Q_air_e + Q_lat_e) if cooling else -jnp.sum(Q_air_c)
        Tzdot = dTdt(T_z, T_out, Q_to_zone, Q_load)

        extras = [Tzdot.reshape((1,))]
        if moist:
            mdot_zone = mdot_e if cooling else mdot_c
            W_z = s["W_z"]
            Wzdot = (mdot_zone * (W_indoor_out - W_z) + W_gain) / max(rhoV_z, 1.0e-6)
            extras.append(Wzdot.reshape((1,)))
        if frost:
            extras.append(jnp.reshape(dm_fr, (1,)))

        return jnp.concatenate(
            [
                pdot_e.reshape((1,)),
                hdot_e,
                Twdot_e,
                pdot_c.reshape((1,)),
                hdot_c,
                Twdot_c,
                *extras,
            ]
        )

    return rhs


def diagnostics(spec: PlantSpec, tables: PropertyTables, y: Array, u: Array) -> dict[str, Array]:
    _require_humid(spec)
    compressor, eev, htc, air_q, _dTdt, mdot_i, mdot_o = _component_maps(spec)
    lay = spec.layout
    y = project_state(y, tables, lay)
    s = unpack_state(y, lay)
    p_e, h_e, Tw_e = s["p_e"], s["h_e"], s["Tw_e"]
    p_c, h_c, Tw_c = s["p_c"], s["h_c"], s["Tw_c"]
    T_z = s["T_z"]
    pe = vmap(lambda hi: eval_ph(tables, p_e, hi))(h_e)
    pc = vmap(lambda hi: eval_ph(tables, p_c, hi))(h_c)
    suct = eval_ph(tables, p_e, h_e[-1])
    c_out = eval_ph(tables, p_c, h_c[-1])
    m_comp, h_disch, power = compressor(
        p_e, p_c, h_e[-1], suct.rho, suct.T, u[0], suct.Tsat, pc.Tsat[0]
    )
    m_eev = eev(p_c, p_e, c_out.rho, u[1])
    disch = eval_ph(tables, p_c, h_disch)
    V_e = spec.V_e / spec.n_e
    V_c = spec.V_c / spec.n_c
    charge = (
        jnp.sum(pe.rho) * V_e
        + jnp.sum(pc.rho) * V_c
        + spec.V_header_e * jnp.mean(pe.rho)
        + spec.V_header_c * jnp.mean(pc.rho)
    )
    n_e, n_c = spec.n_e, spec.n_c
    A_ref_e = jnp.full((n_e,), spec.A_ref_e / n_e)
    A_ref_c = jnp.full((n_c,), spec.A_ref_c / n_c)
    A_air_e = jnp.full((n_e,), spec.A_air_e / n_e)
    A_air_c = jnp.full((n_c,), spec.A_air_c / n_c)
    fan_i = jnp.clip(u[2], 0.15, 1.2)
    fan_o = jnp.clip(u[3], 0.15, 1.2)
    T_out = u[4]
    cooling = spec.operating_mode == "cooling"
    if cooling:
        T_air_e, T_air_c = T_z, T_out
        fan_e, fan_c = fan_i, fan_o
        mdot_e = mdot_i(fan_i, spec.mdot_air_e0)
        mdot_c = mdot_o(fan_o, spec.mdot_air_c0)
    else:
        T_air_e, T_air_c = T_out, T_z
        fan_e, fan_c = fan_o, fan_i
        mdot_e = mdot_o(fan_o, spec.mdot_air_e0)
        mdot_c = mdot_i(fan_i, spec.mdot_air_c0)
    G_e = 0.5 * (m_eev + m_comp) / max(spec.n_tubes_e * 0.25 * pi * spec.D_e**2, 1e-8)
    G_c = 0.5 * (m_comp + m_eev) / max(spec.n_tubes_c * 0.25 * pi * spec.D_c**2, 1e-8)
    htc_e = htc(G_e, spec.D_e, pe.mu, pe.k, pe.cp, pe.x, p_e / tables.pc, True)
    htc_c = htc(G_c, spec.D_c, pc.mu, pc.k, pc.cp, pc.x, p_c / tables.pc, False)
    htc_ae = jnp.full((n_e,), spec.htc_air_e * fan_e)
    htc_ac = jnp.full((n_c,), spec.htc_air_c * fan_c)
    Q_lat = jnp.asarray(0.0)
    delta_fr = jnp.asarray(0.0)
    Q_lat_e = jnp.zeros((n_e,))
    if spec.moist or spec.frost:
        from heatpump.devices.frost import frost_layer
        from heatpump.psychro import build_humid_tables, jax_humid_fns, wet_coil_march

        W_sat_T, h_fg_T, W_of, T_dp, _h_if = jax_humid_fns(build_humid_tables())
        RH_out = u[8] if u.size > 8 else (0.0 if spec.RH_out is None else float(spec.RH_out))
        W_z = s["W_z"] if lay.moist else W_of(T_z, jnp.asarray(0.5))
        W_amb = W_of(T_out, jnp.clip(RH_out, 0.05, 0.99))
        W_in_e, W_in_c = (W_z, W_amb) if cooling else (W_amb, W_z)
        if spec.frost:
            Tw_out = Tw_e if not cooling else Tw_c
            A_out = float(spec.A_air_e if not cooling else spec.A_air_c)
            delta_fr, k_fr, _rho = frost_layer(
                jnp.mean(Tw_out), s["m_fr"], A_out, spec.frost_closure
            )
            r_fr = delta_fr / jnp.maximum(k_fr, 1.0e-6)
            if cooling:
                htc_ac = 1.0 / (1.0 / jnp.maximum(htc_ac, 1.0e-6) + r_fr)
            else:
                htc_ae = 1.0 / (1.0 / jnp.maximum(htc_ae, 1.0e-6) + r_fr)
        _W_e, Q_lat_e = wet_coil_march(pe.T, W_in_e, mdot_e, W_sat_T, h_fg_T, T_dp)
        _W_c, Q_lat_c = wet_coil_march(pc.T, W_in_c, mdot_c, W_sat_T, h_fg_T, T_dp)
        Q_lat = jnp.sum(Q_lat_e) if cooling else jnp.sum(Q_lat_c)
    Q_air_e, _ = air_q(
        T_air_e, pe.T, htc_e, A_ref_e, htc_ae, A_air_e, mdot_e, spec.cp_air_e,
    )
    Q_air_c, _ = air_q(
        T_air_c, pc.T, htc_c, A_ref_c, htc_ac, A_air_c, mdot_c, spec.cp_air_c,
    )
    Q_zone = -jnp.sum(Q_air_e + Q_lat_e) if cooling else -jnp.sum(Q_air_c)
    Q_evap = jnp.sum(Q_air_e + Q_lat_e)
    Q_ref_e = Q_evap
    q_useful = -Q_zone if cooling else Q_zone
    cop = q_useful / jnp.maximum(power, 1.0)
    return {
        "p_e": p_e,
        "p_c": p_c,
        "T_z": T_z,
        "SH": suct.T - suct.Tsat,
        "SC": c_out.T_bubble - c_out.T,
        "T_e_out": suct.T,
        "T_c_out": c_out.T,
        "Tsat_e": suct.Tsat,
        "Tsat_c": c_out.Tsat,
        "T_disch": disch.T,
        "x_e_mean": jnp.mean(pe.x),
        "x_c_mean": jnp.mean(pc.x),
        "x_e_in": pe.x[0],
        "x_e_out": pe.x[-1],
        "x_c_in": pc.x[0],
        "x_c_out": pc.x[-1],
        "mdot_comp": m_comp,
        "mdot_eev": m_eev,
        "power": power,
        "h_disch": h_disch,
        "h_suct": h_e[-1],
        "h_ll": h_c[-1],
        "charge": charge,
        "pr": p_c / jnp.maximum(p_e, 1.0e4),
        "Q_zone": Q_zone,
        "Q_evap": Q_evap,
        "Q_ref_e": Q_ref_e,
        "COP": cop,
        "Tw_e_mean": jnp.mean(Tw_e),
        "Tw_c_mean": jnp.mean(Tw_c),
        "T_e_mean": jnp.mean(pe.T),
        "T_c_mean": jnp.mean(pc.T),
        "W_z": s["W_z"] if lay.moist else jnp.asarray(0.0),
        "m_fr": s["m_fr"] if lay.frost else jnp.asarray(0.0),
        "Q_lat": Q_lat,
        "delta_fr": delta_fr,
    }


def initial_state(
    spec: PlantSpec,
    tables: PropertyTables,
    T_out: float = 273.15,
    T_zone: float = 291.15,
) -> Array:
    """Two-phase evaporator, condensing high side, mild superheat / subcool."""
    _require_humid(spec)
    T_lo = float(tables.Tsat[0]) + 2.0
    T_hi = float(tables.Tsat[-1]) - 2.0
    if spec.operating_mode == "cooling":
        T_evap = min(max(T_zone - 8.0, T_lo), T_hi)
        T_cond = min(max(T_out + 12.0, T_lo), T_hi)
    else:
        T_evap = min(max(T_out - 8.0, T_lo), T_hi)
        T_cond = min(max(T_zone + 12.0, T_lo), T_hi)
    if T_cond <= T_evap + 4.0:
        T_cond = min(T_evap + 8.0, T_hi)
    p_e = p_sat_from_tables(tables, T_evap, "dew")
    p_c = p_sat_from_tables(tables, T_cond, "bubble")
    env_e = sat_from_tables(tables, p_e)
    env_c = sat_from_tables(tables, p_c)
    hf_e, hg_e = env_e["hf"], env_e["hg"]
    hf_c, hg_c = env_c["hf"], env_c["hg"]
    n_e, n_c = spec.n_e, spec.n_c
    x_e = jnp.linspace(0.22, 0.92, n_e)
    h_e = hf_e + x_e * (hg_e - hf_e)
    h_e = h_e.at[-1].set(hg_e + 8.0e3)
    h_c = jnp.linspace(hg_c + 3.5e4, hf_c - 8.0e3, n_c)
    if spec.operating_mode == "cooling":
        Tw_e = jnp.linspace(T_evap + 1.0, T_zone - 2.0, n_e)
        Tw_c = jnp.linspace(T_cond + 2.0, T_out + 4.0, n_c)
    else:
        Tw_e = jnp.linspace(T_evap + 2.0, T_out - 1.0, n_e)
        Tw_c = jnp.linspace(T_cond + 4.0, T_zone + 6.0, n_c)
    W0 = m0 = None
    if spec.moist:
        from heatpump.psychro import indoor_air

        W0 = indoor_air(T_zone, spec.RH_zone0).W
    if spec.frost:
        m0 = 0.0
    return project_state(
        pack_state(p_e, h_e, Tw_e, p_c, h_c, Tw_c, T_zone, W0, m0),
        tables,
        spec.layout,
    )
