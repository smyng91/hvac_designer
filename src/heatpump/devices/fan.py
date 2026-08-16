"""Indoor / outdoor air-flow maps.

The default is a linear speed fraction times the coil's design ṁ.
A tabulated fan is loaded from a user or manufacturer CSV — no curve
is invented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class LinearFan:
    """ṁ = ṁ0 · φ, φ ∈ [0, 1.2]."""

    def mdot(self, speed: Array, mdot0: float) -> Array:
        return mdot0 * jnp.clip(speed, 0.0, 1.2)


@dataclass(frozen=True)
class TableFan:
    """Piecewise-linear ṁ(φ) from a two-column table (speed_frac, kg/s)."""

    speed: tuple[float, ...]
    mdot_kg_s: tuple[float, ...]
    citation: str

    def __post_init__(self):
        if len(self.speed) < 2 or len(self.speed) != len(self.mdot_kg_s):
            raise ValueError("TableFan needs ≥2 (speed, mdot) points")
        if not (self.citation or "").strip():
            raise ValueError("TableFan must name the source of the airflow points")

    @classmethod
    def from_file(cls, path: str | Path, citation: str | None = None) -> TableFan:
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = {n.lower(): n for n in (raw.dtype.names or ())}
        try:
            s_key = next(names[k] for k in names if k in ("speed", "speed_frac", "phi", "rpm_frac"))
            m_key = next(names[k] for k in names if k in ("mdot", "mdot_kg_s", "kg_s", "flow"))
        except StopIteration as exc:
            raise ValueError(
                f"{path}: TableFan CSV needs speed and mdot columns (got {list(raw.dtype.names or ())})"
            ) from exc
        src = citation or f"user table {Path(path).name}"
        return cls(
            speed=tuple(float(x) for x in np.asarray(raw[s_key], dtype=float)),
            mdot_kg_s=tuple(float(x) for x in np.asarray(raw[m_key], dtype=float)),
            citation=src,
        )

    def mdot(self, speed: Array, mdot0: float) -> Array:
        del mdot0
        return jnp.interp(speed, jnp.asarray(self.speed), jnp.asarray(self.mdot_kg_s))
