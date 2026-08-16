"""Built-in expansion devices."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from heatpump.components import eev_mdot


@dataclass(frozen=True)
class OrificeEEV:
    """Isenthalpic orifice: ṁ = Cd A u √(2ρ Δp)."""

    A_max: float
    Cd: float = 0.70

    @classmethod
    def from_plant(cls, spec) -> OrificeEEV:
        return cls(spec.A_eev, spec.Cd)

    def map(self, p_in: Array, p_out: Array, rho_in: Array, opening: Array) -> Array:
        return eev_mdot(p_in, p_out, rho_in, opening, self.A_max, self.Cd)
