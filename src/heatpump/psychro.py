"""Indoor moist-air state for cooling design (CoolProp humid air).

The transient plant is still dry. This module reports the sensible /
latent split that follows from the coil duty, indoor state, and
apparatus dew point (evaporating temperature). SHR is an output.
"""

from __future__ import annotations

from dataclasses import dataclass

import CoolProp.CoolProp as CP
from CoolProp.HumidAirProp import HAPropsSI

P_ATM = 101325.0


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
