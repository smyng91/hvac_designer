"""Seasonal bins from a *user* outdoor timeseries.

Hours in each bin are the dwell time of the supplied record. AHRI 210/240
bin-hour tables are not copied (copyright, and they are not this climate).
Capacity and power are taken from a closed capacity map when the caller
passes one; they are left unset when no map is given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from heatpump.capacity import CapacityMap
from heatpump.requirements import TimeSeries


@dataclass(frozen=True)
class OutdoorBin:
    T_lo: float
    T_hi: float
    T_mean: float
    hours: float
    Q_gain_mean: float
    n_samples: int
    Q_cap: float | None
    W: float | None
    COP: float | None


@dataclass(frozen=True)
class SeasonalBins:
    width_K: float
    hours_total: float
    bins: tuple[OutdoorBin, ...]
    notes: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "width_K": self.width_K,
            "hours_total": self.hours_total,
            "bins": [asdict(b) for b in self.bins],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Seasonal bins from the supplied timeseries",
            "",
            f"Bin width {self.width_K:.1f} K. Total {self.hours_total:.1f} h "
            "from the record dwell times (not an AHRI climate table).",
            "",
            "| T_lo °C | T_hi °C | T_mean °C | hours | Q_gain kW | Q_cap kW | W kW | COP | n |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for b in self.bins:
            qc = f"{b.Q_cap/1e3:.2f}" if b.Q_cap is not None else "—"
            w = f"{b.W/1e3:.2f}" if b.W is not None else "—"
            cop = f"{b.COP:.2f}" if b.COP is not None else "—"
            lines.append(
                f"| {b.T_lo-273.15:.1f} | {b.T_hi-273.15:.1f} | {b.T_mean-273.15:.1f} | "
                f"{b.hours:.2f} | {b.Q_gain_mean/1e3:.2f} | {qc} | {w} | {cop} | {b.n_samples} |"
            )
        lines.append("")
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines).rstrip() + "\n"


def _sample_hours(t: np.ndarray) -> np.ndarray:
    """Hours represented by each sample; weights sum to the record duration."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return t
    if t.size == 1:
        raise ValueError("TimeSeries needs at least two samples to measure dwell time")
    d = np.diff(t)
    if np.any(d < 0):
        raise ValueError("TimeSeries t must be non-decreasing")
    w = np.zeros_like(t)
    w[0] = 0.5 * d[0]
    w[-1] = 0.5 * d[-1]
    if t.size > 2:
        w[1:-1] = 0.5 * d[:-1] + 0.5 * d[1:]
    return w / 3600.0


def _map_at(cmap: CapacityMap, T: float) -> tuple[float, float, float] | None:
    T_grid = np.asarray([p.T_out for p in cmap.points], dtype=float)
    if T < T_grid.min() - 1.0e-6 or T > T_grid.max() + 1.0e-6:
        return None
    Q = np.interp(T, T_grid, np.asarray([p.Q_cap for p in cmap.points], dtype=float))
    W = np.interp(T, T_grid, np.asarray([p.W for p in cmap.points], dtype=float))
    cop = float(Q / W) if W > 1.0 else 0.0
    return float(Q), float(W), cop


def bin_timeseries(
    ts: TimeSeries,
    *,
    width_K: float = 5.0,
    heating_map: CapacityMap | None = None,
    cooling_map: CapacityMap | None = None,
) -> SeasonalBins:
    """Histogram outdoor temperature with hours from the record itself."""
    if width_K <= 0.0:
        raise ValueError("bin width must be positive")
    hours = _sample_hours(ts.t)
    T = np.asarray(ts.T_out, dtype=float)
    Q = np.asarray(ts.Q_gain, dtype=float)
    T_lo0 = float(np.floor(T.min() / width_K) * width_K)
    T_hi0 = float(np.ceil(T.max() / width_K) * width_K)
    if T_hi0 <= T_lo0:
        T_hi0 = T_lo0 + width_K
    edges = np.arange(T_lo0, T_hi0 + 0.5 * width_K, width_K)
    bins: list[OutdoorBin] = []
    notes = [
        "Hours are the dwell time of the supplied timeseries, not AHRI 210/240 bin hours.",
        "Q_cap / W are interpolated from the closed capacity map when one is passed; "
        "they are omitted when no map is available (not invented).",
    ]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (T >= lo) & (T < hi) if hi < edges[-1] else (T >= lo) & (T <= hi)
        if not np.any(mask):
            continue
        Tm = float(np.average(T[mask], weights=np.maximum(hours[mask], 1e-12)))
        Qm = float(np.average(Q[mask], weights=np.maximum(hours[mask], 1e-12)))
        cmap = cooling_map if Qm >= 0.0 else heating_map
        got = _map_at(cmap, Tm) if cmap is not None else None
        bins.append(
            OutdoorBin(
                T_lo=float(lo),
                T_hi=float(hi),
                T_mean=Tm,
                hours=float(np.sum(hours[mask])),
                Q_gain_mean=Qm,
                n_samples=int(np.sum(mask)),
                Q_cap=None if got is None else got[0],
                W=None if got is None else got[1],
                COP=None if got is None else got[2],
            )
        )
    return SeasonalBins(
        width_K=float(width_K),
        hours_total=float(np.sum(hours)),
        bins=tuple(bins),
        notes=tuple(notes),
    )
