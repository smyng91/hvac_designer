"""Built-in air-side coil closures."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from heatpump.components import air_march


def series_ua_air_q(
    T_air: Array,
    T_ref: Array,
    h_ref: Array,
    A_ref: Array,
    h_air: Array,
    A_air: Array,
    mdot_air: Array,
    cp_air: float,
) -> tuple[Array, Array]:
    """Quasi-steady wall: Q from air to refrigerant is the series UA.

    1/UA = 1/(h_r A_r) + 1/(h_a A_a). Air is marched with that UA against
    the refrigerant temperature (no wall ODE in the energy close).
    """
    r = 1.0 / jnp.maximum(h_ref * A_ref, 1.0e-6) + 1.0 / jnp.maximum(h_air * A_air, 1.0e-6)
    UA = 1.0 / r
    htc_eff = UA / jnp.maximum(A_air, 1.0e-9)
    return air_march(T_air, T_ref, htc_eff, A_air, mdot_air, cp_air)


@dataclass(frozen=True)
class SeriesUAAir:
    """Default air-side map: series UA + downstream air march."""

    def heat_rate(
        self,
        T_air: Array,
        T_ref: Array,
        h_ref: Array,
        A_ref: Array,
        h_air: Array,
        A_air: Array,
        mdot_air: Array,
        cp_air: float,
    ) -> tuple[Array, Array]:
        return series_ua_air_q(T_air, T_ref, h_ref, A_ref, h_air, A_air, mdot_air, cp_air)
