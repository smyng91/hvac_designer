"""Heat-exchanger closures used by the sizer and the off-design map.

Refrigerant HTC is Dittus–Boelter (single-phase) or Shah (two-phase).
Air-side HTC is Zhukauskas cross-flow over a tube bank. Overall UA is
the series of the two resistances. Heat rate is ε-NTU with C_min on
the air (two-phase refrigerant has effectively infinite capacity rate).
"""

from __future__ import annotations

from math import exp, pi

import CoolProp.CoolProp as CP
import numpy as np

from heatpump.thermo import flash_ph, sat_at_p

# Copper tube wall (geometry + material properties, not a load rule).
RHO_CU = 8960.0
CP_CU = 385.0
T_WALL = 4.0e-4
# Staggered-bank transverse pitch / tube OD (geometry).
PITCH_OVER_D = 2.5
# Header stub length per tube (geometry).
L_HEADER = 0.08


def air_props(T: float, P: float = 101325.0) -> dict[str, float]:
    T = float(np.clip(T, 200.0, 350.0))
    return {
        "rho": float(CP.PropsSI("D", "T", T, "P", P, "Air")),
        "mu": float(CP.PropsSI("V", "T", T, "P", P, "Air")),
        "k": float(CP.PropsSI("L", "T", T, "P", P, "Air")),
        "cp": float(CP.PropsSI("C", "T", T, "P", P, "Air")),
    }


def htc_air_bank(mdot_air: float, T_air: float, D: float, n_tubes: int, L: float) -> float:
    """Zhukauskas Nu = 0.27 Re^{0.63} Pr^{0.36} for a tube bank in cross-flow."""
    ap = air_props(T_air)
    A_face = max(n_tubes * L * PITCH_OVER_D * D, 1e-9)
    v = float(mdot_air) / (ap["rho"] * A_face)
    Re = ap["rho"] * abs(v) * D / max(ap["mu"], 1e-8)
    Pr = ap["mu"] * ap["cp"] / max(ap["k"], 1e-6)
    Nu = 0.27 * max(Re, 1.0) ** 0.63 * max(Pr, 0.1) ** 0.36
    return float(Nu * ap["k"] / D)


def htc_refrigerant(
    G: float,
    D: float,
    mu: float,
    k: float,
    cp: float,
    x: float,
    p_r: float,
    evaporating: bool,
) -> float:
    """Dittus–Boelter, or Shah two-phase multiplier when 0 < x < 1."""
    Re = min(max(abs(G) * D / max(mu, 1e-6), 300.0), 1.0e6)
    Pr = min(max(mu * cp / max(k, 1e-3), 0.4), 8.0)
    n_pr = 0.4 if evaporating else 0.3
    h_sp = (0.023 * Re**0.8 * Pr**n_pr) * k / D
    if x <= 0.0 or x >= 1.0:
        return float(h_sp)
    xs = min(max(x, 1e-3), 1.0 - 1e-3)
    xtt = ((1.0 - xs) / xs) ** 0.9 * min(max(p_r, 0.02), 0.9) ** 0.15
    if evaporating:
        F = 1.0 + 1.8 / max(xtt, 1e-3) ** 0.7 + 8.0 * xs * (1.0 - xs)
    else:
        F = (1.0 - xs) ** 0.8 + 3.8 * xs**0.76 * (1.0 - xs) ** 0.04 / min(max(p_r, 0.05), 0.95) ** 0.38
    return float(h_sp * max(F, 1.0))


def htc_ref_at(fluid: str, p: float, h: float, G: float, D: float, evaporating: bool, pc: float) -> float:
    st = flash_ph(fluid, p, h)
    return htc_refrigerant(G, D, st["mu"], st["k"], st["cp"], st["x"], p / max(pc, 1.0), evaporating)


def overall_UA(h_ref: float, A_ref: float, h_air: float, A_air: float) -> float:
    r = 1.0 / max(h_ref * A_ref, 1e-6) + 1.0 / max(h_air * A_air, 1e-6)
    return 1.0 / r


def eps_ntu(UA: float, m_air: float, cp: float) -> float:
    NTU = UA / max(m_air * cp, 1e-6)
    return float(1.0 - exp(-min(NTU, 20.0)))


def coil_Q(T_air: float, T_ref: float, m_air: float, cp: float, UA: float) -> float:
    """Heat from air to refrigerant (positive when the air is warmer)."""
    return eps_ntu(UA, m_air, cp) * m_air * cp * (T_air - T_ref)


def size_coil(
    Q: float,
    T_air: float,
    T_ref: float,
    mdot_ref: float,
    *,
    fluid: str,
    p: float,
    h: float,
    pc: float,
    evaporating: bool,
    D: float,
    L: float,
    fin: float,
) -> dict[str, float]:
    """Iterate tube count until ε-NTU heat rate equals the cycle duty ``Q``."""
    dT = abs(T_air - T_ref)
    if dT < 0.5 or Q <= 0.0:
        raise ValueError("coil sizing needs a positive duty and a nonzero air–refrigerant ΔT")
    ap = air_props(T_air)
    cp = ap["cp"]
    n = 16
    m_air = Q / (cp * dT)
    h_r = h_a = 200.0
    for _ in range(16):
        n = max(int(n), 1)
        A_ref = n * pi * D * L
        A_air = A_ref * fin
        G = mdot_ref / max(n * 0.25 * pi * D**2, 1e-12)
        h_r = htc_ref_at(fluid, p, h, G, D, evaporating, pc)
        h_a = htc_air_bank(m_air, T_air, D, n, L)
        UA = overall_UA(h_r, A_ref, h_a, A_air)
        eps = eps_ntu(UA, m_air, cp)
        Q_hx = eps * m_air * cp * dT
        m_air = Q / max(eps * cp * dT, 1.0)
        if Q_hx > 1.0:
            n = max(1, int(np.ceil(n * Q / Q_hx)))
        if n > 400:
            raise ValueError(
                "coil sizing exceeded 400 tubes; the air–refrigerant ΔT cannot "
                "carry this duty with the stated tube geometry"
            )
        if abs(Q_hx - Q) / Q < 0.02:
            break
    A_ref = n * pi * D * L
    V = n * 0.25 * pi * D**2 * L
    return {
        "n_tubes": float(n),
        "L": float(L),
        "D": float(D),
        "A_ref": float(A_ref),
        "A_air": float(A_ref * fin),
        "V": float(V),
        "V_header": float(n * 0.25 * pi * D**2 * L_HEADER),
        "C_w": float(RHO_CU * CP_CU * A_ref * T_WALL),
        "mdot_air": float(m_air),
        "htc_air": float(h_a),
        "htc_ref": float(h_r),
        "cp_air": float(cp),
        "G": float(mdot_ref / max(n * 0.25 * pi * D**2, 1e-12)),
    }


def zone_capacitance(T: float, V: float) -> float:
    """Dry-air thermal mass ρ c_p V at the zone temperature."""
    ap = air_props(T)
    return ap["rho"] * ap["cp"] * float(V)


def charge_from_profile(
    fluid: str,
    states: dict,
    V_e: float,
    V_c: float,
    V_he: float,
    V_hc: float,
    n_e: int,
    n_c: int,
) -> float:
    """Inventory from the design enthalpy profile (same cells as the plant)."""
    from heatpump.thermo import zivi_density

    p_e, p_c = states["1"].p, states["3"].p
    env_e, env_c = sat_at_p(fluid, p_e), sat_at_p(fluid, p_c)
    h_e = np.linspace(states["4"].h, states["1"].h, n_e)
    h_c = np.linspace(states["2"].h, states["3"].h, n_c)
    rho_e = []
    for h in h_e:
        st = flash_ph(fluid, p_e, float(h))
        if 0.0 < st["x"] < 1.0:
            rho_e.append(float(zivi_density(st["x"], env_e["rhof"], env_e["rhog"])))
        else:
            rho_e.append(st["rho"])
    rho_c = []
    for h in h_c:
        st = flash_ph(fluid, p_c, float(h))
        if 0.0 < st["x"] < 1.0:
            rho_c.append(float(zivi_density(st["x"], env_c["rhof"], env_c["rhog"])))
        else:
            rho_c.append(st["rho"])
    return (
        float(np.mean(rho_e)) * V_e
        + float(np.mean(rho_c)) * V_c
        + float(np.mean(rho_e)) * V_he
        + float(np.mean(rho_c)) * V_hc
    )
