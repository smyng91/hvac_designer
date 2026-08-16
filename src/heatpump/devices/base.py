"""Protocols for swappable plant devices.

Each device is a JAX-pure map from port states to flows or heat rates.
The plant residual calls these objects when the matching ``PlantSpec``
slot is set; otherwise it uses the built-in kernels.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
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
        Tsat_s: Array | None = None,
        Tsat_d: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Return ``(mdot, h_discharge, power)``.

        ``Tsat_s`` / ``Tsat_d`` are dew-point temperatures [K] at the
        suction and discharge pressures (AHRI 540). Clearance maps ignore them.
        """


@runtime_checkable
class ExpansionValve(Protocol):
    """Isenthalpic (or user-defined) expansion device."""

    def map(
        self,
        p_in: Array,
        p_out: Array,
        rho_in: Array,
        opening: Array,
    ) -> Array:
        """Return mass flow [kg/s]."""


@runtime_checkable
class RefrigerantHTC(Protocol):
    """Refrigerant-side heat-transfer coefficient [W/m²K]."""

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


@runtime_checkable
class AirSide(Protocol):
    """Air-to-refrigerant heat rate for one coil pass."""

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


@runtime_checkable
class Fan(Protocol):
    """Indoor or outdoor air mass-flow map."""

    def mdot(self, speed: Array, mdot0: float) -> Array:
        """Return air ṁ [kg/s] at fan speed fraction ``speed``."""


@runtime_checkable
class ZoneModel(Protocol):
    """Lumped or user-defined zone energy balance."""

    def dTdt(
        self,
        T_z: Array,
        T_out: Array,
        Q_hvac: Array,
        Q_gain: Array,
    ) -> Array:
        """Return dT_zone/dt [K/s]."""
