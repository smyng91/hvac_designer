"""Duck types for optional plant components.

The residual does not import these at run time. They document the methods
a replacement object should implement. Assign the object on ``PlantSpec``.
"""

from __future__ import annotations

from typing import Protocol

from jax import Array


class Compressor(Protocol):
    """Suction → discharge mass flow, enthalpy, and shaft power."""

    def map(
        self,
        p_s: Array,
        p_d: Array,
        h_s: Array,
        rho_s: Array,
        T_s: Array,
        N_hz: Array,
        Tsat_s: Array,
        Tsat_d: Array,
    ) -> tuple[Array, Array, Array]:
        """Return ``(mdot, h_discharge, power)``.

        ``Tsat_s`` / ``Tsat_d`` are dew-point temperatures [K] (AHRI 540).
        Clearance maps ignore them.
        """


class ExpansionValve(Protocol):
    def map(self, p_in: Array, p_out: Array, rho_in: Array, opening: Array) -> Array:
        """Return mass flow [kg/s]."""


class RefrigerantHTC(Protocol):
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
    ) -> Array: ...


class AirSide(Protocol):
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
        """Return ``(Q_air_to_ref, T_air_out)`` per cell."""


class Fan(Protocol):
    def mdot(self, speed: Array, mdot0: float) -> Array:
        """Return air ṁ [kg/s] at fan speed fraction ``speed``."""


class ZoneModel(Protocol):
    def dTdt(self, T_z: Array, T_out: Array, Q_hvac: Array, Q_gain: Array) -> Array:
        """Return dT_zone/dt [K/s]."""
