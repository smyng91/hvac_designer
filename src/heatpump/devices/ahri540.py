"""AHRI 540 10-coefficient compressor map.

Coefficients are never defaulted. Load a published or manufacturer file
(``from_file``) or pass the ten power and mass-flow numbers explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jax import Array


def ahri540_poly(Ts: Array, Td: Array, C: Array) -> Array:
    """AHRI 540 (SI/I-P) cubic in suction / discharge dew-point temperature.

    X = C1 + C2 S + C3 D + C4 S² + C5 S D + C6 D² + C7 S³ + C8 S² D + C9 S D² + C10 D³
    """
    S, D = Ts, Td
    return (
        C[0]
        + C[1] * S
        + C[2] * D
        + C[3] * S * S
        + C[4] * S * D
        + C[5] * D * D
        + C[6] * S * S * S
        + C[7] * S * S * D
        + C[8] * S * D * D
        + C[9] * D * D * D
    )


@dataclass(frozen=True)
class AHRI540Compressor:
    """Hermetic map: ṁ and electrical power from dew-point temperatures.

    ``power_C`` must yield watts; ``mdot_C`` must yield kg/s (convert in
    ``from_file`` if the source file is in g/s). ``N_rated_hz`` is optional;
    if set, ṁ and W scale with N / N_rated (fixed-speed maps omit it).
    Discharge enthalpy assumes a hermetic machine: h_d = h_s + W / ṁ.
    """

    power_C: tuple[float, ...]
    mdot_C: tuple[float, ...]
    citation: str
    T_unit: str = "C"
    N_rated_hz: float | None = None

    def __post_init__(self):
        if len(self.power_C) != 10 or len(self.mdot_C) != 10:
            raise ValueError("AHRI 540 maps need exactly 10 power and 10 mass-flow coefficients")
        if not (self.citation or "").strip():
            raise ValueError("AHRI 540 map must name its source (paper, datasheet, or lab file)")

    @classmethod
    def from_file(cls, path: str | Path) -> AHRI540Compressor:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "power" not in raw or "mdot" not in raw:
            raise ValueError(f"{path}: AHRI 540 file must contain 'power' and 'mdot' (10 coefficients each)")
        if not str(raw.get("citation", "")).strip():
            raise ValueError(f"{path}: AHRI 540 file must name its source in 'citation'")
        mdot = list(raw["mdot"])
        if raw.get("mdot_unit", "kg/s") in ("g/s", "g_s"):
            mdot = [c * 1.0e-3 for c in mdot]
        return cls(
            power_C=tuple(float(c) for c in raw["power"]),
            mdot_C=tuple(float(c) for c in mdot),
            citation=str(raw["citation"]),
            T_unit=str(raw.get("T_unit", raw.get("units", {}).get("T", "C"))),
            N_rated_hz=raw.get("N_rated_hz"),
        )

    @classmethod
    def from_plant(cls, spec) -> AHRI540Compressor:
        path = getattr(spec, "ahri540_path", None)
        if not path:
            raise TypeError(
                "AHRI 540 requires a published or user coefficient file "
                "(PlantSpec.ahri540_path). No default map is invented."
            )
        return cls.from_file(path)

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
        del p_s, p_d, rho_s
        if Tsat_s is None or Tsat_d is None:
            raise ValueError("AHRI 540 needs dew-point temperatures Tsat_s and Tsat_d")
        Ts = Tsat_s - 273.15 if self.T_unit.upper() in ("C", "DEG_C") else Tsat_s
        Td = Tsat_d - 273.15 if self.T_unit.upper() in ("C", "DEG_C") else Tsat_d
        Cp = jnp.asarray(self.power_C, dtype=Ts.dtype)
        Cm = jnp.asarray(self.mdot_C, dtype=Ts.dtype)
        power = jnp.maximum(ahri540_poly(Ts, Td, Cp), 1.0)
        mdot = jnp.maximum(ahri540_poly(Ts, Td, Cm), 1.0e-6)
        if self.N_rated_hz is not None:
            fac = jnp.clip(N_hz / float(self.N_rated_hz), 0.0, 1.6)
            power = power * fac
            mdot = mdot * fac
        # Off below ~4 Hz (same cutoff as the clearance map). Fixed-speed
        # maps omit N_rated; speed still gates the machine off.
        from heatpump.components import jax_sigmoid

        on = jax_sigmoid((N_hz - 4.0) * 1.5)
        power = power * on
        mdot = mdot * on
        h_d = h_s + jnp.where(on > 0.05, power / jnp.maximum(mdot, 1.0e-9), 0.0)
        return mdot, h_d, power
