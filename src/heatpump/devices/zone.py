"""Built-in zone energy balances."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class LumpedZone:
    """Single dry capacitance: C Ṫ = Q_hvac + Q_gain + UA (T_out − T_z)."""

    C: float
    UA: float

    @classmethod
    def from_plant(cls, spec) -> LumpedZone:
        return cls(spec.C_zone, spec.UA_env)

    def dTdt(self, T_z: Array, T_out: Array, Q_hvac: Array, Q_gain: Array) -> Array:
        return (Q_hvac + Q_gain + self.UA * (T_out - T_z)) / self.C
