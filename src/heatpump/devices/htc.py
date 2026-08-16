"""Built-in refrigerant HTC closures."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from heatpump.components import htc_two_phase


@dataclass(frozen=True)
class ShahDittusHTC:
    """Dittus–Boelter in single-phase; Shah multiplier in the two-phase dome."""

    def htc(
        self,
        G: Array,
        D: float,
        mu: Array,
        k: Array,
        cp: Array,
        x: Array,
        p_r: Array,
        evaporating: bool,
    ) -> Array:
        return htc_two_phase(G, D, mu, k, cp, x, p_r, evaporating)
