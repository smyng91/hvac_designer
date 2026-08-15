"""Compressor, isenthalpic EEV, and heat-transfer closures."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def soft_pos(x: Array, eps: float = 1.0e4) -> Array:
    """Smooth positive part, C1, used for valve pressure drop."""
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


def compressor_mdot_h(
    p_s: Array,
    p_d: Array,
    h_s: Array,
    rho_s: Array,
    T_s: Array,
    N_hz: Array,
    V_disp: float,
    C_loss: float,
    eta_is0: float,
    gamma: float = 1.25,
) -> tuple[Array, Array, Array]:
    """Clearance volumetric map plus a polytropic isentropic rise.

    Returns (mdot, discharge enthalpy, shaft power). Volumetric efficiency
    is the clearance form; discharge enthalpy uses a polytropic estimate
    from the suction state so the residual stays JAX-smooth (no entropy
    inversion). Isentropic efficiency is the user-supplied η_is.
    """
    pr = jnp.clip(p_d / jnp.maximum(p_s, 1.0e4), 1.01, 12.0)
    eta_v = jnp.clip(1.0 - C_loss * (pr ** (1.0 / gamma) - 1.0), 0.25, 0.97)
    N_eff = N_hz * jax_sigmoid((N_hz - 4.0) * 1.5)
    mdot = eta_v * rho_s * V_disp * N_eff

    # Polytropic isentropic enthalpy change: Δh_is = [γ/(γ-1)] (p/ρ) (pr^{(γ-1)/γ}-1)
    dh_is = (gamma / (gamma - 1.0)) * (p_s / jnp.maximum(rho_s, 1.0)) * (
        pr ** ((gamma - 1.0) / gamma) - 1.0
    )
    eta_is = jnp.clip(eta_is0, 0.20, 0.95)
    h_d = h_s + dh_is / eta_is
    power = mdot * (h_d - h_s)
    return mdot, h_d, power


def jax_sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(z, -40.0, 40.0)))


def eev_mdot(
    p_in: Array,
    p_out: Array,
    rho_in: Array,
    opening: Array,
    A_max: float,
    Cd: float,
) -> Array:
    """Isenthalpic electronic expansion valve (orifice + two-phase inlet).

    Mass flow is Cd A sqrt(2 ρ Δp) with A = A_max u (geometric opening)
    and a regularized Δp so the Jacobian exists at zero pressure difference.
    """
    u = jnp.clip(opening, 0.0, 1.0)
    area = A_max * u
    dp = soft_pos(p_in - p_out, 2.0e4)
    return Cd * area * jnp.sqrt(2.0 * jnp.maximum(rho_in, 5.0) * dp)


def dittus_boelter(G: Array, D: float, mu: Array, k: Array, cp: Array, n_pr: float) -> Array:
    Re = jnp.clip(jnp.abs(G) * D / jnp.maximum(mu, 1.0e-6), 300.0, 1.0e6)
    Pr = jnp.clip(mu * cp / jnp.maximum(k, 1.0e-3), 0.4, 8.0)
    Nu = 0.023 * Re**0.8 * Pr**n_pr
    return Nu * k / D


def htc_two_phase(
    G: Array,
    D: float,
    mu: Array,
    k: Array,
    cp: Array,
    x: Array,
    p_r: Array,
    evaporating: bool,
) -> Array:
    """Dittus–Boelter when x is 0 or 1; Shah two-phase multiplier inside the dome."""
    h_sp = dittus_boelter(G, D, mu, k, cp, 0.4 if evaporating else 0.3)
    x = jnp.clip(x, 0.0, 1.0)
    xs = jnp.clip(x, 1.0e-3, 1.0 - 1.0e-3)
    xtt = jnp.power((1.0 - xs) / xs, 0.9) * jnp.power(jnp.clip(p_r, 0.02, 0.9), 0.15)
    if evaporating:
        F = 1.0 + 1.8 / jnp.power(jnp.maximum(xtt, 1e-3), 0.7) + 8.0 * xs * (1.0 - xs)
    else:
        F = (1.0 - xs) ** 0.8 + 3.8 * jnp.power(xs, 0.76) * jnp.power(1.0 - xs, 0.04) / jnp.power(
            jnp.clip(p_r, 0.05, 0.95), 0.38
        )
    tp = (x > 1.0e-3) & (x < 1.0 - 1.0e-3)
    return jnp.where(tp, h_sp * jnp.maximum(F, 1.0), h_sp)


def air_march(
    T_in: Array,
    Tw: Array,
    htc_a: Array,
    area: Array,
    mdot_air: Array,
    cp_air: float = 1006.0,
) -> tuple[Array, Array]:
    """Quasi-steady air stream past a row of wall cells (unrolled, static length)."""
    n = Tw.shape[0]
    T = T_in
    q_to_wall = []
    for i in range(n):
        q = htc_a[i] * area[i] * (T - Tw[i])
        q_to_wall.append(q)
        T = T - q / jnp.maximum(mdot_air * cp_air, 1.0)
    return jnp.stack(q_to_wall), T
