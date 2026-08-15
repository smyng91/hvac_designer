"""L-stable TR-BDF2 integrator with damped Newton and an exact JAX Jacobian.

TR-BDF2 (Hosea & Shampine, 1996) is the method behind MATLAB's ode23tb:
a trapezoidal stage to t+γh followed by a BDF2 completion. It is stiffly
accurate and L-stable, which vapor-compression pressure dynamics need.
Each implicit stage is solved by Newton with ``jacfwd`` and a backtracking
line search. The step is rejected and cut when the residual stalls or the
embedded trapezoidal estimate exceeds the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

GAMMA = 2.0 - 2.0**0.5
_A = 1.0 / (GAMMA * (2.0 - GAMMA))
_B = -((1.0 - GAMMA) ** 2) / (GAMMA * (2.0 - GAMMA))
_C = (1.0 - GAMMA) / (2.0 - GAMMA)

RHS = Callable[[Array, Array, Array], Array]
PROJ = Callable[[Array], Array]


def _newton(g: Callable[[Array], Array], y: Array, n_iter: int) -> tuple[Array, Array]:
    damps = jnp.array([1.0, 0.5, 0.25, 0.125])

    def body(yk, _):
        r = g(yk)
        J = jax.jacfwd(g)(yk) + 1.0e-10 * jnp.eye(yk.size)
        dy = jnp.linalg.solve(J, -r)
        r0 = jnp.linalg.norm(r)

        def consider(damp, best):
            yb, rb = best
            yt = yk + damp * dy
            rt = jnp.linalg.norm(g(yt))
            take = rt < rb
            return jnp.where(take, yt, yb), jnp.where(take, rt, rb)

        y_best, r_best = yk, r0
        for d in damps:
            y_best, r_best = consider(d, (y_best, r_best))
        return y_best, r_best

    y, rn = jax.lax.scan(body, y, jnp.arange(n_iter))
    return y, rn[-1]


def implicit_euler_step(
    rhs: RHS,
    t: Array,
    y: Array,
    u: Array,
    dt: Array,
    project: PROJ | None = None,
    n_newton: int = 5,
) -> Array:
    """Differentiable implicit Euler (unrolled Newton) for NMPC."""
    y0 = y

    def g(yk):
        return yk - y0 - dt * rhs(t + dt, yk, u)

    y1, _ = _newton(g, y0, n_newton)
    return project(y1) if project is not None else y1


def make_stepper(
    rhs: RHS,
    project: PROJ | None,
    n_newton: int,
    rtol: float,
    atol: float,
    dt_min: float,
    dt_max: float,
    safety: float,
):
    """Compile a single TR-BDF2 attempted step ``(t, y, u, dt) -> ...``."""

    def step(t, y, u, dt):
        f0 = rhs(t, y, u)
        h = dt
        g1s = 0.5 * GAMMA * h

        def g1(y1):
            return y1 - y - g1s * (f0 + rhs(t + GAMMA * h, y1, u))

        y1, rn1 = _newton(g1, y + GAMMA * h * f0, n_newton)

        def g2(y2):
            return y2 - _A * y1 - _B * y - _C * h * rhs(t + h, y2, u)

        y2, rn2 = _newton(g2, y1, n_newton)
        if project is not None:
            y2 = project(y2)
        f2 = rhs(t + h, y2, u)
        y_tr = y + 0.5 * h * (f0 + f2)
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y2))
        err = jnp.linalg.norm((y2 - y_tr) / scale) / jnp.sqrt(y.size)
        newton_ok = jnp.maximum(rn1, rn2) < 1.0e-3 * (1.0 + jnp.linalg.norm(y))
        accepted = jnp.logical_and(err < 1.25, newton_ok)
        fac = jnp.clip(safety * jnp.power(1.0 / jnp.maximum(err, 1e-12), 0.5), 0.3, 2.2)
        dt_next = jnp.clip(h * jnp.where(accepted, fac, 0.45), dt_min, dt_max)
        y_out = jnp.where(accepted, y2, y)
        t_out = jnp.where(accepted, t + h, t)
        return y_out, t_out, dt_next, accepted

    return jax.jit(step)


@dataclass
class TRBDF2:
    rtol: float = 3.0e-4
    atol: float = 3.0e-6
    dt_min: float = 8.0e-4
    dt_max: float = 1.5
    n_newton: int = 6
    safety: float = 0.85
    _step = None
    _rhs_id = None

    def bind(self, rhs: RHS, project: PROJ | None = None):
        self._step = make_stepper(
            rhs, project, self.n_newton, self.rtol, self.atol, self.dt_min, self.dt_max, self.safety
        )
        return self

    def step(self, t, y, u, dt):
        if self._step is None:
            raise RuntimeError("Call TRBDF2.bind(rhs, project) before stepping")
        y_n, t_n, dt_n, ok = self._step(t, y, u, dt)
        return y_n, float(t_n), float(dt_n), bool(np.asarray(ok))


def integrate(
    rhs: RHS,
    y0: Array,
    u_of_t: Callable[[float], Array],
    t_final: float,
    dt0: float = 0.2,
    project: PROJ | None = None,
    solver: TRBDF2 | None = None,
    record_dt: float = 1.0,
    max_steps: int = 40000,
    on_accept: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive TR-BDF2. Returns ``(t_hist, y_hist)`` on a uniform record grid."""
    solver = (solver or TRBDF2()).bind(rhs, project)
    t = 0.0
    y = y0
    dt = dt0
    rec_t = [0.0]
    rec_y = [np.asarray(y0)]
    next_rec = record_dt
    steps = 0
    rejects = 0
    while t < t_final - 1e-12 and steps < max_steps:
        steps += 1
        dt = min(float(dt), t_final - t)
        if next_rec - t > solver.dt_min:
            dt = min(dt, next_rec - t)
        u = jnp.asarray(u_of_t(t))
        y_n, t_n, dt_n, ok = solver.step(t, y, u, dt)
        if ok:
            y, t = y_n, t_n
            if on_accept is not None:
                on_accept(t, y)
            while next_rec <= t + 1e-12 and next_rec <= t_final + 1e-12:
                rec_t.append(float(next_rec))
                rec_y.append(np.asarray(y))
                next_rec += record_dt
        else:
            rejects += 1
        dt = dt_n
    return np.asarray(rec_t), np.stack(rec_y)
