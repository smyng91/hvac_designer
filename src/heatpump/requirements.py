"""Design requirements, operating constraints, and exogenous timeseries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 1 refrigeration ton = 12 000 Btu/h
TON_W = 3516.8528420667


def cooling_tons_to_w(tons: float) -> float:
    return float(tons) * TON_W


@dataclass
class Constraints:
    """Hard/soft limits the sizer and controllers must respect."""

    SH_min: float = 4.0
    SH_max: float = 10.0
    SH_sp: float = 6.0
    SC_min: float = 0.0
    T_disch_max: float = 273.15 + 115.0
    pr_max: float = 7.5
    p_c_frac_crit: float = 0.90
    N_min: float = 0.0
    N_max: float = 70.0
    eev_min: float = 0.10
    eev_max: float = 0.72
    min_on_s: float = 60.0
    min_off_s: float = 90.0
    T_zone_band: float = 0.5
    G_max: float | None = None


@dataclass
class TimeSeries:
    """Exogenous weather / load. ``Q_gain`` is heat into the zone [W]."""

    t: np.ndarray
    T_out: np.ndarray
    Q_gain: np.ndarray
    Tsp: np.ndarray | None = None
    mode: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.T_out = np.asarray(self.T_out, dtype=float).reshape(-1)
        self.Q_gain = np.asarray(self.Q_gain, dtype=float).reshape(-1)
        n = self.t.size
        if self.T_out.size != n or self.Q_gain.size != n:
            raise ValueError("TimeSeries t, T_out, and Q_gain must have the same length")
        if self.Tsp is not None:
            self.Tsp = np.asarray(self.Tsp, dtype=float).reshape(-1)
            if self.Tsp.size != n:
                raise ValueError("TimeSeries Tsp length must match t")
        if self.mode is not None:
            self.mode = np.asarray(self.mode, dtype=float).reshape(-1)
            if self.mode.size != n:
                raise ValueError("TimeSeries mode length must match t")
        order = np.argsort(self.t)
        self.t = self.t[order]
        self.T_out = self.T_out[order]
        self.Q_gain = self.Q_gain[order]
        if self.Tsp is not None:
            self.Tsp = self.Tsp[order]
        if self.mode is not None:
            self.mode = self.mode[order]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    def at(self, t: float) -> dict[str, float]:
        x = float(t)
        out = {
            "T_out": float(np.interp(x, self.t, self.T_out)),
            "Q_gain": float(np.interp(x, self.t, self.Q_gain)),
        }
        if self.Tsp is not None:
            out["Tsp"] = float(np.interp(x, self.t, self.Tsp))
        if self.mode is not None:
            out["mode"] = float(np.interp(x, self.t, self.mode) >= 0.5)
        return out

    def design_peaks(self, T_zone: float) -> dict[str, float]:
        """Infer design duty from the load profile at the given setpoint.

        ``Q_gain`` is heat into the zone. Holding ``T_z = Tsp`` with the
        profile as the complete load (no extra envelope) requires HVAC
        cooling ``max(Q_gain, 0)`` and heating ``max(-Q_gain, 0)``. Design
        outdoor temperature is the ambient at that peak hour, not the
        extreme of the whole record.
        """
        if self.Tsp is not None:
            tsp = np.asarray(self.Tsp, dtype=float)
        else:
            tsp = np.full(self.t.shape, float(T_zone))
        cool = np.maximum(self.Q_gain, 0.0)
        heat = np.maximum(-self.Q_gain, 0.0)
        i_cool = int(np.argmax(cool)) if cool.size else 0
        i_heat = int(np.argmax(heat)) if heat.size else 0
        if float(np.max(cool, initial=0.0)) < 1.0:
            i_cool = int(np.argmax(self.T_out))
        if float(np.max(heat, initial=0.0)) < 1.0:
            i_heat = int(np.argmin(self.T_out))
        return {
            "Q_cool": float(np.max(cool, initial=0.0)),
            "Q_heat": float(np.max(heat, initial=0.0)),
            "T_out_cool": float(self.T_out[i_cool]),
            "T_out_heat": float(self.T_out[i_heat]),
            "T_zone_cool": float(tsp[i_cool]),
            "T_zone_heat": float(tsp[i_heat]),
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        load_kind: str = "gain",
        time_unit: str = "s",
        T_unit: str = "C",
        Q_unit: str = "W",
    ) -> TimeSeries:
        """Load a CSV with a header.

        Required columns (case-insensitive, aliases accepted)::

            t | time | t_s | t_min
            T_out | T_out_C | Tamb | ambient
            Q | Q_W | Q_kW | Q_gain | load

        Optional: ``Tsp``, ``Tsp_C``, ``mode`` (1=heating, 0=cooling).

        ``load_kind``:
            * ``gain`` — positive Q heats the zone
            * ``cooling_load`` — positive Q is cooling duty (heat to remove) → +gain
            * ``heating_load`` — positive Q is heating duty (heat loss) → −gain
        """
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if raw.dtype.names is None:
            raise ValueError(f"{path}: CSV needs a header row")
        names = {n.lower(): n for n in raw.dtype.names}

        def col(*aliases: str) -> np.ndarray | None:
            for a in aliases:
                if a.lower() in names:
                    return np.asarray(raw[names[a.lower()]], dtype=float)
            return None

        t = col("t", "time", "t_s", "t_min", "minutes", "hours")
        T = col("t_out", "t_out_c", "tamb", "ambient", "t_amb", "t_outdoor")
        Q = col("q", "q_w", "q_kw", "q_gain", "load", "q_load", "cooling_load", "heating_load")
        if t is None or T is None or Q is None:
            raise ValueError(
                f"{path}: need t, T_out, and Q columns (got {list(raw.dtype.names)})"
            )
        if time_unit in ("min", "minutes") or "t_min" in names:
            t = t * 60.0
        elif time_unit in ("h", "hour", "hours") or "hours" in names:
            t = t * 3600.0
        if T_unit.lower() in ("c", "degc", "celsius") or "t_out_c" in names or "tsp_c" in names:
            T = T + 273.15
        q_is_kw = Q_unit.lower() in ("kw", "kilowatt") or "q_kw" in names
        q_is_ton = Q_unit.lower() in ("ton", "tons", "rt") or "q_ton" in names or "q_tons" in names
        if q_is_kw:
            Q = Q * 1000.0
        elif q_is_ton:
            Q = Q * TON_W
        kind = load_kind.lower()
        if kind in ("heating_load", "heating"):
            Q = -np.abs(Q)
        elif kind in ("cooling_load", "cooling"):
            Q = np.abs(Q)
        Tsp = col("tsp", "tsp_c", "t_sp", "setpoint")
        if Tsp is not None and T_unit.lower() in ("c", "degc", "celsius"):
            Tsp = Tsp + 273.15
        mode = col("mode", "hp_mode")
        return cls(t=t, T_out=T, Q_gain=Q, Tsp=Tsp, mode=mode)


@dataclass
class DesignRequest:
    """Everything the sizer needs to pick a fluid, hardware, and controller."""

    refrigerant: str
    mode: str = "heating"
    T_zone: float = 294.15
    Q_heat: float | None = None
    Q_cool: float | None = None
    cooling_tons: float | None = None
    T_out_heat: float = 273.15
    T_out_cool: float = 308.15
    T_zone_heat: float | None = None
    T_zone_cool: float | None = None
    oversize: float = 1.0
    SH: float = 6.0
    SC: float = 4.0
    DT_evap: float = 10.0
    DT_cond: float = 12.0
    N_hz: float = 50.0
    n_cells: int = 6
    controller: str = "auto"
    use_envelope: bool = True
    constraints: Constraints = field(default_factory=Constraints)
    timeseries: TimeSeries | None = None
    UA_env: float | None = None
    C_zone: float | None = None
    inferred_from_profile: bool = False
    indoor_RH: float = 0.50
    voltage: float = 230.0
    phases: int = 1
    eta_motor: float | None = None
    V_zone: float | None = None

    def __post_init__(self) -> None:
        self.mode = _norm_mode(self.mode)
        if self.cooling_tons is not None and self.Q_cool is None:
            self.Q_cool = cooling_tons_to_w(self.cooling_tons)
        if self.T_zone_heat is None:
            self.T_zone_heat = self.T_zone
        if self.T_zone_cool is None:
            self.T_zone_cool = self.T_zone
        if self.timeseries is not None:
            self._apply_profile(self.timeseries.design_peaks(self.T_zone))

    def _apply_profile(self, peaks: dict[str, float]) -> None:
        """Fill missing duties and design weather from the load / ambient traces."""
        had_heat = self.Q_heat is not None
        had_cool = self.Q_cool is not None or self.cooling_tons is not None
        if not had_heat:
            qh = peaks["Q_heat"]
            self.Q_heat = qh if qh >= 200.0 else None
            self.T_out_heat = peaks["T_out_heat"]
            self.T_zone_heat = peaks["T_zone_heat"]
        if not had_cool:
            qc = peaks["Q_cool"]
            self.Q_cool = qc if qc >= 200.0 else None
            self.T_out_cool = peaks["T_out_cool"]
            self.T_zone_cool = peaks["T_zone_cool"]
        if not had_heat and not had_cool:
            self.inferred_from_profile = True
            if self.UA_env is None:
                # Profile is the duty to hold the setpoint; do not add UA on top.
                self.use_envelope = False
        if self.mode == "auto":
            has_h = self.Q_heat is not None and self.Q_heat >= 200.0
            has_c = self.Q_cool is not None and self.Q_cool >= 200.0
            if has_h and has_c:
                self.mode = "heat_pump"
            elif has_c:
                self.mode = "cooling"
            elif has_h:
                self.mode = "heating"
            else:
                raise ValueError(
                    "Load profile has no heating or cooling duty above 200 W at the "
                    f"setpoint {self.T_zone - 273.15:.1f} °C. Check Q sign / load_kind."
                )


def _norm_mode(mode: str) -> str:
    m = mode.strip().lower().replace("-", "_")
    if m in ("heat", "heating", "hp_heat"):
        return "heating"
    if m in ("cool", "cooling", "ac", "air_conditioning"):
        return "cooling"
    if m in ("heat_pump", "heatpump", "both", "reversible"):
        return "heat_pump"
    if m in ("auto", "from_profile", "infer"):
        return "auto"
    raise ValueError(f"mode must be heating, cooling, heat_pump, or auto, got {mode!r}")
