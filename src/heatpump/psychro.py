"""Humid-air states and JAX tables (CoolProp HA, never inside the residual).

Design-package SHR is an output from the coil duty, indoor state, and
apparatus dew point. The optional moist plant interpolates the tables
built here; CoolProp is not called from the JIT residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import CoolProp.CoolProp as CP
import jax.numpy as jnp
import numpy as np
from CoolProp.HumidAirProp import HAPropsSI
from jax import Array

P_ATM = 101325.0

# IAPWS enthalpy of fusion of ice Ih at 273.15 K (used if CoolProp Ice fails).
H_IF_IAPWS = 333550.0


@dataclass(frozen=True)
class IndoorAir:
    T_db: float
    RH: float
    T_wb: float
    T_dp: float
    W: float
    h: float
    cp: float
    P: float = P_ATM


@dataclass(frozen=True)
class CoolingPsychro:
    indoor: IndoorAir
    SHR: float
    Q_total: float
    Q_sensible: float
    Q_latent: float
    mdot_air: float
    condensate_kg_s: float
    T_adp: float
    wet: bool

    def summary(self) -> str:
        coil = "wet" if self.wet else "dry"
        return (
            f"Indoor {self.indoor.T_db-273.15:.1f}°C DB / "
            f"{self.indoor.T_wb-273.15:.1f}°C WB / {100*self.indoor.RH:.0f}% RH  "
            f"{coil} coil  SHR={self.SHR:.2f}  latent {self.Q_latent/1e3:.2f} kW  "
            f"condensate {self.condensate_kg_s*3600:.2f} kg/h"
        )


def indoor_air(T_db: float, RH: float = 0.50, P: float = P_ATM) -> IndoorAir:
    rh = float(min(max(RH, 0.05), 0.99))
    T = float(T_db)
    return IndoorAir(
        T_db=T,
        RH=rh,
        T_wb=float(HAPropsSI("Twb", "T", T, "P", P, "R", rh)),
        T_dp=float(HAPropsSI("Tdp", "T", T, "P", P, "R", rh)),
        W=float(HAPropsSI("W", "T", T, "P", P, "R", rh)),
        h=float(HAPropsSI("H", "T", T, "P", P, "R", rh)),
        cp=float(HAPropsSI("C", "T", T, "P", P, "R", rh)),
        P=P,
    )


def _h_fg(T: float) -> float:
    T = float(min(max(T, 274.0), 370.0))
    return float(
        CP.PropsSI("H", "T", T, "Q", 1, "Water") - CP.PropsSI("H", "T", T, "Q", 0, "Water")
    )


def cooling_psychro(
    T_zone: float,
    Q_coil: float,
    *,
    RH: float = 0.50,
    T_adp: float,
    mdot_air: float | None = None,
) -> CoolingPsychro:
    """Sensible / latent split from humid-air balances.

    Apparatus dew point is the evaporating temperature. The coil is wet
    when that temperature is below the indoor dew point; leaving humidity
    is then saturation at the ADP. Latent heat uses water h_fg(T).
    """
    air = indoor_air(T_zone, RH)
    Q_tot = float(Q_coil)
    adp = float(T_adp)
    wet = adp < air.T_dp
    if wet:
        W_out = float(HAPropsSI("W", "T", adp, "P", air.P, "R", 1.0))
    else:
        W_out = air.W
    if mdot_air is None:
        dT = max(air.T_db - adp, 0.5)
        m_air = Q_tot / max(air.cp * dT, 1.0)
    else:
        m_air = float(mdot_air)
    h_fg = _h_fg(adp if wet else air.T_db)
    Q_lat = m_air * max(air.W - W_out, 0.0) * h_fg
    if Q_lat > Q_tot:
        Q_lat = Q_tot
        W_out = air.W - Q_lat / max(m_air * h_fg, 1.0)
    Q_sens = Q_tot - Q_lat
    shr = Q_sens / Q_tot if Q_tot > 0.0 else 1.0
    m_w = Q_lat / h_fg if h_fg > 0.0 else 0.0
    return CoolingPsychro(
        indoor=air,
        SHR=float(shr),
        Q_total=Q_tot,
        Q_sensible=float(Q_sens),
        Q_latent=float(Q_lat),
        mdot_air=m_air,
        condensate_kg_s=float(m_w),
        T_adp=adp,
        wet=wet,
    )


@dataclass(frozen=True)
class HumidTables:
    """Saturation humidity and h_fg on a T grid; W(T, RH) on a 2-D grid."""

    T: np.ndarray
    W_sat: np.ndarray
    h_fg: np.ndarray
    RH: np.ndarray
    W_TRH: np.ndarray
    h_if: float
    P: float


def fusion_enthalpy() -> float:
    """h_if of ice Ih at the ice–liquid equilibrium [J/kg]."""
    for ice in ("Ice", "IF97::IceIh"):
        try:
            h_l = float(CP.PropsSI("H", "T", 273.16, "Q", 0, "Water"))
            h_i = float(CP.PropsSI("H", "T", 273.16, "P", 611.657, ice))
            val = h_l - h_i
            if 2.5e5 < val < 4.0e5:
                return val
        except (ValueError, TypeError, RuntimeError):
            continue
    return H_IF_IAPWS


@lru_cache(maxsize=4)
def build_humid_tables(P: float = P_ATM, n_T: int = 91, n_RH: int = 19) -> HumidTables:
    """Flash humid air with CoolProp once; the residual only interpolates."""
    T = np.linspace(240.0, 330.0, n_T)
    RH = np.linspace(0.05, 0.99, n_RH)
    W_sat = np.full(n_T, np.nan)
    h_fg = np.full(n_T, np.nan)
    W_TRH = np.full((n_T, n_RH), np.nan)
    for i, t in enumerate(T):
        try:
            W_sat[i] = float(HAPropsSI("W", "T", float(t), "P", P, "R", 1.0))
        except (ValueError, TypeError, RuntimeError):
            pass
        try:
            h_fg[i] = _h_fg(float(t))
        except (ValueError, TypeError, RuntimeError):
            pass
        for j, rh in enumerate(RH):
            try:
                W_TRH[i, j] = float(HAPropsSI("W", "T", float(t), "P", P, "R", float(rh)))
            except (ValueError, TypeError, RuntimeError):
                pass
    if np.isnan(W_sat).all() or np.isnan(W_TRH).all():
        raise RuntimeError("CoolProp HAPropsSI returned no humid-air states on the table grid")
    W_sat = _fill_1d(T, W_sat)
    h_fg = _fill_1d(T, h_fg)
    for j in range(n_RH):
        W_TRH[:, j] = _fill_1d(T, W_TRH[:, j])
    return HumidTables(
        T=T,
        W_sat=W_sat,
        h_fg=h_fg,
        RH=RH,
        W_TRH=W_TRH,
        h_if=fusion_enthalpy(),
        P=float(P),
    )


def _fill_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    if not np.any(ok):
        raise RuntimeError("no finite CoolProp samples to interpolate")
    return np.interp(x, x[ok], y[ok])


def _interp2(x: Array, y: Array, xg: Array, yg: Array, z: Array) -> Array:
    x = jnp.clip(x, xg[0], xg[-1])
    y = jnp.clip(y, yg[0], yg[-1])
    i = jnp.clip(jnp.searchsorted(xg, x, side="right") - 1, 0, xg.size - 2)
    j = jnp.clip(jnp.searchsorted(yg, y, side="right") - 1, 0, yg.size - 2)
    x0, x1 = xg[i], xg[i + 1]
    y0, y1 = yg[j], yg[j + 1]
    tx = (x - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    ty = (y - y0) / jnp.maximum(y1 - y0, 1.0e-12)
    return (
        (1.0 - tx) * (1.0 - ty) * z[i, j]
        + tx * (1.0 - ty) * z[i + 1, j]
        + (1.0 - tx) * ty * z[i, j + 1]
        + tx * ty * z[i + 1, j + 1]
    )


def jax_humid_fns(tables: HumidTables):
    """Return JAX interpolants ``W_sat(T), h_fg(T), W(T,RH), T_dp(W), h_if``."""
    Tg = jnp.asarray(tables.T)
    Ws = jnp.asarray(tables.W_sat)
    hfg = jnp.asarray(tables.h_fg)
    RHg = jnp.asarray(tables.RH)
    W2 = jnp.asarray(tables.W_TRH)
    h_if = float(tables.h_if)

    def W_sat_T(T: Array) -> Array:
        return jnp.interp(T, Tg, Ws)

    def h_fg_T(T: Array) -> Array:
        return jnp.interp(T, Tg, hfg)

    def W_of(T: Array, RH: Array) -> Array:
        return _interp2(T, RH, Tg, RHg, W2)

    def T_dp(W: Array) -> Array:
        return jnp.interp(W, Ws, Tg)

    return W_sat_T, h_fg_T, W_of, T_dp, h_if


def wet_coil_march(T_ref: Array, W_in: Array, mdot: Array, W_sat_T, h_fg_T, T_dp):
    """Sequential wet-coil humidity march. Returns ``(W_out, Q_lat_per_cell)``.

    A cell condenses when ``T_ref`` is below the local dew point; leaving
    humidity is then saturation at ``T_ref``. Latent heat uses water h_fg(T).
    """
    from jax import lax

    def body(W, T):
        Ws = W_sat_T(T)
        W_next = jnp.where(T < T_dp(W), jnp.minimum(Ws, W), W)
        Ql = mdot * (W - W_next) * h_fg_T(T)
        return W_next, Ql

    W_out, Q_lat = lax.scan(body, W_in, T_ref)
    return W_out, Q_lat
