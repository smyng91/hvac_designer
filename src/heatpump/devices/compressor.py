"""Built-in compressor maps."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from heatpump.components import compressor_mdot_h


@dataclass(frozen=True)
class ClearanceCompressor:
    """Clearance volumetric efficiency plus a polytropic isentropic rise.

    Swap this for an AHRI 540 polynomial (or any ``Compressor``) without
    changing the plant residual.
    """

    V_disp: float
    C_loss: float = 0.075
    eta_is: float = 0.70
    gamma: float = 1.25

    def map(
        self,
        p_s: Array,
        p_d: Array,
        h_s: Array,
        rho_s: Array,
        T_s: Array,
        N_hz: Array,
        Tsat_s=None,
        Tsat_d=None,
    ) -> tuple[Array, Array, Array]:
        del Tsat_s, Tsat_d
        return compressor_mdot_h(
            p_s, p_d, h_s, rho_s, T_s, N_hz, self.V_disp, self.C_loss, self.eta_is, self.gamma
        )
