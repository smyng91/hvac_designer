"""Published frost-layer closures (density, conductivity).

These are material-property correlations, not capacity derate tables.
Growth itself is a humidity mass balance in the plant residual.
"""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

# Hayashi, Aoki, Adachi, Hori, "Study of frost properties correlating with
# frost formation types," J. Heat Transfer 99 (1977) 239–245.
# ρ = 650 exp(0.277 T_s) with T_s the frost-surface temperature in °C.
HAYASHI_RHO0 = 650.0
HAYASHI_B = 0.277

# Yonko & Sepsy, "An investigation of the thermal conductivity of frost
# while forming on a flat horizontal plate," ASHRAE Trans. 73 (1967).
# k = 0.001202 ρ^0.963  [W/m·K] with ρ in kg/m³.
YONKO_A = 0.001202
YONKO_N = 0.963

# IAPWS ice Ih at 0 °C (used only when the user asks for solid ice, not frost).
ICE_RHO = 916.7
ICE_K = 2.22


def hayashi_density(T_w: Array) -> Array:
    """Hayashi et al. (1977) frost density [kg/m³]. ``T_w`` in K."""
    Ts = jnp.clip(T_w - 273.15, -30.0, 0.0)
    return HAYASHI_RHO0 * jnp.exp(HAYASHI_B * Ts)


def yonko_sepsy_k(rho: Array) -> Array:
    """Yonko & Sepsy (1967) frost conductivity [W/m·K]."""
    return YONKO_A * jnp.maximum(rho, 1.0) ** YONKO_N


def frost_layer(T_w: Array, m_fr: Array, A: Array | float, closure: str) -> tuple[Array, Array, Array]:
    """Return ``(δ, k, ρ)`` for a uniform layer of mass ``m_fr`` on area ``A``."""
    if closure == "ice":
        rho = jnp.full_like(T_w, ICE_RHO)
        k = jnp.full_like(T_w, ICE_K)
    elif closure == "hayashi":
        rho = hayashi_density(T_w)
        k = yonko_sepsy_k(rho)
    else:
        raise ValueError(
            f"unknown frost closure {closure!r}; use 'hayashi' (Hayashi 1977 + "
            "Yonko–Sepsy 1967) or 'ice' (IAPWS ice Ih at 0 °C)"
        )
    delta = jnp.maximum(m_fr, 0.0) / jnp.maximum(rho * A, 1.0e-9)
    return delta, k, rho
