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
    rtol: float = 1.0e-3
    atol: float = 1.0e-5
    dt_min: float = 5.0e-3
    dt_max: float = 8.0
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


def _record(rec_t, rec_y, next_rec, record_dt, t_final, t, y):
    while next_rec <= t + 1e-12 and next_rec <= t_final + 1e-12:
        rec_t.append(float(next_rec))
        rec_y.append(np.asarray(y))
        next_rec += record_dt
    return next_rec


def make_euler(rhs: RHS, project: PROJ | None, n_newton: int = 5):
    """JIT implicit Euler used as a guaranteed-progress fallback and for QSS."""

    def step(t, y, u, dt):
        return implicit_euler_step(rhs, t, y, u, dt, project, n_newton)

    return jax.jit(step)


def make_qss_relax(
    rhs: RHS,
    project: PROJ | None,
    i_tz: int,
    n_relax: int,
    dt_relax: float,
    n_newton: int = 5,
    slow_idx: tuple[int, ...] | None = None,
):
    """JIT a few implicit-Euler cycle refreshes with slow states held."""
    hold = jnp.asarray(slow_idx if slow_idx is not None else (i_tz,))

    def relax(t, y, u):
        held = y[hold]

        def body(yk, _):
            yk = implicit_euler_step(rhs, t, yk, u, dt_relax, project, n_newton)
            return yk.at[hold].set(held), None

        y1, _ = jax.lax.scan(body, y, jnp.arange(n_relax))
        return y1

    return jax.jit(relax)


def integrate(
    rhs: RHS,
    y0: Array,
    u_of_t: Callable[[float], Array],
    t_final: float,
    dt0: float = 0.25,
    project: PROJ | None = None,
    solver: TRBDF2 | None = None,
    record_dt: float = 1.0,
    max_steps: int | None = None,
    on_accept: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive TR-BDF2. Returns ``(t_hist, y_hist)`` on a uniform record grid.

    Steps are *not* cut to the record grid (that was stalling long runs).
    A rejected step at ``dt_min`` is accepted as implicit Euler so time
    always advances.
    """
    solver = (solver or TRBDF2()).bind(rhs, project)
    euler = make_euler(rhs, project)
    if max_steps is None:
        max_steps = int(max(t_final / max(solver.dt_min, 1e-3) * 1.3 + 2000, 20000))
        max_steps = min(max_steps, 2_000_000)
    t = 0.0
    y = y0
    dt = min(float(dt0), solver.dt_max)
    rec_t = [0.0]
    rec_y = [np.asarray(y0)]
    next_rec = record_dt
    steps = 0
    while t < t_final - 1e-12 and steps < max_steps:
        steps += 1
        dt = min(float(dt), t_final - t, solver.dt_max)
        dt = max(dt, solver.dt_min if t_final - t > solver.dt_min else t_final - t)
        u = jnp.asarray(u_of_t(t))
        y_n, t_n, dt_n, ok = solver.step(t, y, u, dt)
        if ok:
            y, t = y_n, t_n
        elif dt <= solver.dt_min * 1.05:
            y = euler(jnp.float64(t), y, u, jnp.float64(dt))
            if project is not None:
                y = project(y)
            t = t + dt
            dt_n = min(dt * 1.6, solver.dt_max)
        # else: rejected, dt_n already cut; do not advance t
        if on_accept is not None:
            on_accept(t, y)
        next_rec = _record(rec_t, rec_y, next_rec, record_dt, t_final, t, y)
        dt = dt_n
    if rec_t[-1] < t - 1e-9 and t >= t_final - record_dt:
        rec_t.append(float(t))
        rec_y.append(np.asarray(y))
    return np.asarray(rec_t), np.stack(rec_y)


def integrate_qss(
    rhs: RHS,
    y0: Array,
    u_of_t: Callable[[float], Array],
    t_final: float,
    *,
    i_tz: int,
    project: PROJ | None = None,
    record_dt: float = 60.0,
    refresh_s: float = 120.0,
    n_relax: int = 12,
    dt_relax: float = 2.0,
    on_accept: Callable | None = None,
    slow_idx: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hour-to-day integrator: slow ODEs plus periodic refrigerant relax.

    Between refreshes the refrigerant state is held and only the slow
    states (zone temperature, optional humidity and frost mass) advance
    with the residual. Every ``refresh_s`` the full DAE is relaxed with
    implicit Euler (slow states held) so the cycle tracks outdoor T and N.
    """
    hold = tuple(slow_idx) if slow_idx is not None else (i_tz,)
    relax = make_qss_relax(rhs, project, i_tz, n_relax, dt_relax, slow_idx=hold)
    hold_arr = jnp.asarray(hold)
    t = 0.0
    y = y0
    rec_t = [0.0]
    rec_y = [np.asarray(y0)]
    next_rec = record_dt
    last_refresh = -1e9
    if on_accept is not None:
        on_accept(t, y)
    while t < t_final - 1e-12:
        dt = min(record_dt, t_final - t)
        u = jnp.asarray(u_of_t(t))
        if t - last_refresh >= refresh_s - 1e-12:
            y = relax(jnp.float64(t), y, u)
            if project is not None:
                y = project(y)
            last_refresh = t
        dy = rhs(jnp.float64(t), y, u)
        y = y.at[hold_arr].set(y[hold_arr] + dt * dy[hold_arr])
        if project is not None:
            y = project(y)
        t = t + dt
        if on_accept is not None:
            on_accept(t, y)
        next_rec = _record(rec_t, rec_y, next_rec, record_dt, t_final, t, y)
    if rec_t[-1] < t - 1e-9:
        rec_t.append(float(t))
        rec_y.append(np.asarray(y))
    return np.asarray(rec_t), np.stack(rec_y)
