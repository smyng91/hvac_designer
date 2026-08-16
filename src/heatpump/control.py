"""Zone and superheat controllers: PID, hysteresis, bang-bang, linear and nonlinear MPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from heatpump.solver import implicit_euler_step


@dataclass
class ControlOutput:
    N: float
    eev: float
    fan_i: float = 1.0
    fan_o: float = 1.0

    def as_input(
        self,
        T_out: float,
        Q_load: float,
        W_gain: float = 0.0,
        defrost: float = 0.0,
        RH_out: float | None = None,
    ) -> np.ndarray:
        core = [self.N, self.eev, self.fan_i, self.fan_o, T_out, Q_load]
        if RH_out is None and W_gain == 0.0 and defrost == 0.0:
            return np.array(core, dtype=np.float64)
        return np.array(core + [W_gain, defrost, 0.0 if RH_out is None else RH_out], dtype=np.float64)


@dataclass
class PID:
    """ISA PID with back-calculation anti-windup and derivative on measurement."""

    kp: float
    ki: float
    kd: float = 0.0
    umin: float = 0.0
    umax: float = 1.0
    kaw: float = 1.0
    d_tau: float = 0.4
    _i: float = 0.0
    _y_f: float | None = None

    def reset(self) -> None:
        self._i = 0.0
        self._y_f = None

    def update(self, sp: float, y: float, dt: float) -> float:
        e = sp - y
        # QSS / coarse ZOH can pass 30–60 s; do not integrate that as a single dt.
        dt_i = float(np.clip(dt, 1e-6, 5.0))
        if self._y_f is None:
            self._y_f = y
        a = dt_i / (self.d_tau + dt_i)
        y_prev = self._y_f
        self._y_f = (1.0 - a) * self._y_f + a * y
        d = -self.kd * (self._y_f - y_prev) / dt_i
        u_unsat = self.kp * e + self._i + d
        u = float(np.clip(u_unsat, self.umin, self.umax))
        self._i += self.ki * e * dt_i + self.kaw * (u - u_unsat) * dt_i
        return u


@dataclass
class SuperheatEEV:
    """Inner-loop EEV: low superheat (flooding) closes the valve.

    A compressor-speed feedforward keeps the port from slamming shut on a
    high-side pack-out; the PID is a slow trim around that map. Opening is
    rate-limited so the two-phase inventory can move.
    """

    pid: PID = field(
        default_factory=lambda: PID(
            kp=-0.025, ki=-0.004, kd=-0.008, umin=-0.25, umax=0.25, kaw=0.8
        )
    )
    sh_sp: float = 6.0
    close_when_off: bool = True
    max_rate: float = 0.06
    _u: float = 0.34

    def reset(self) -> None:
        self.pid.reset()
        self._u = 0.34

    def update(self, sh: float, dt: float, compressor_on: bool, N: float = 40.0) -> float:
        if self.close_when_off and not compressor_on:
            self.pid.reset()
            target = 0.06
        else:
            u_ff = 0.18 + 0.28 * float(np.clip(N, 0.0, 80.0)) / 80.0
            trim = self.pid.update(self.sh_sp, sh, dt)
            target = float(np.clip(u_ff + trim, 0.10, 0.72))
        du = float(np.clip(target - self._u, -self.max_rate * dt, self.max_rate * dt))
        self._u = float(np.clip(self._u + du, 0.06, 0.75))
        return self._u


@dataclass
class HysteresisThermostat:
    """On/off compressor with deadband and minimum cycle times."""

    deadband: float = 1.2
    N_on: float = 55.0
    min_on: float = 60.0
    min_off: float = 90.0
    mode: str = "heating"
    on: bool = False
    t_switch: float = -1.0e9

    def reset(self) -> None:
        self.on = False
        self.t_switch = -1.0e9

    def update(self, t: float, T: float, Tsp: float) -> float:
        cooling = self.mode == "cooling"
        if self.on:
            done = (T < Tsp - 0.5 * self.deadband) if cooling else (T > Tsp + 0.5 * self.deadband)
            if done and (t - self.t_switch) >= self.min_on:
                self.on = False
                self.t_switch = t
        else:
            need = (T > Tsp + 0.5 * self.deadband) if cooling else (T < Tsp - 0.5 * self.deadband)
            if need and (t - self.t_switch) >= self.min_off:
                self.on = True
                self.t_switch = t
        return self.N_on if self.on else 0.0


@dataclass
class BangBang:
    """Deadband switch without minimum cycle time (chatters on a fast plant)."""

    deadband: float = 0.6
    N_on: float = 55.0
    mode: str = "heating"
    on: bool = False

    def reset(self) -> None:
        self.on = False

    def update(self, T: float, Tsp: float) -> float:
        lo, hi = Tsp - 0.5 * self.deadband, Tsp + 0.5 * self.deadband
        if self.mode == "cooling":
            if T > hi:
                self.on = True
            elif T < lo:
                self.on = False
        else:
            if T < lo:
                self.on = True
            elif T > hi:
                self.on = False
        return self.N_on if self.on else 0.0


@dataclass
class Cascade:
    """Outer zone loop (any speed law) + inner superheat EEV."""

    speed: object
    eev: SuperheatEEV = field(default_factory=SuperheatEEV)
    Tsp: float = 293.15
    fan_i: float = 1.0
    fan_o: float = 1.0
    kind: str = "pid"  # pid | hysteresis | bangbang
    mode: str = "heating"
    N_rate: float = 8.0
    N_max: float = 80.0
    N_design: float = 50.0
    UA: float = 0.0
    Q_design: float = 0.0
    _N: float = 0.0

    def reset(self) -> None:
        if hasattr(self.speed, "reset"):
            self.speed.reset()
        self.eev.reset()
        self._N = 0.0

    def set_mode(self, mode: str) -> None:
        """Flip heating/cooling action and clear windup. Used on reverse."""
        mode = "cooling" if mode == "cooling" else "heating"
        if mode == self.mode:
            return
        self.mode = mode
        if hasattr(self.speed, "mode"):
            self.speed.mode = mode
        if self.kind == "pid" and hasattr(self.speed, "kp"):
            s = -1.0 if mode == "cooling" else 1.0
            self.speed.kp = s * abs(self.speed.kp)
            self.speed.ki = s * abs(self.speed.ki)
            self.speed.kd = s * abs(self.speed.kd)
        self.reset()

    def _feedforward(self, meas: dict) -> float:
        """Speed that holds the setpoint against UA and Q_gain (trim is PID)."""
        if self.kind != "pid":
            return 0.0
        T_out = meas.get("T_out")
        if T_out is None or self.UA <= 0.0:
            return 0.0
        Qg = float(meas.get("Q_gain", 0.0))
        if self.mode == "cooling":
            Q_need = self.UA * (float(T_out) - self.Tsp) + Qg
        else:
            Q_need = self.UA * (self.Tsp - float(T_out)) - Qg
        Q_ref = self.Q_design if self.Q_design > 200.0 else max(self.UA * 20.0, 500.0)
        return float(np.clip(Q_need / Q_ref, 0.0, 1.4) * self.N_design)

    def update(self, t: float, meas: dict, dt: float) -> ControlOutput:
        Tz = float(meas["T_z"])
        if self.kind == "pid":
            N_cmd = self._feedforward(meas) + self.speed.update(self.Tsp, Tz, dt)
        elif self.kind == "hysteresis":
            N_cmd = self.speed.update(t, Tz, self.Tsp)
        elif self.kind == "bangbang":
            N_cmd = self.speed.update(Tz, self.Tsp)
        else:
            raise ValueError(self.kind)
        N_cmd = float(np.clip(N_cmd, 0.0, self.N_max))
        dN = float(np.clip(N_cmd - self._N, -self.N_rate * min(dt, 5.0), self.N_rate * min(dt, 5.0)))
        self._N = float(np.clip(self._N + dN, 0.0, self.N_max))
        eev = self.eev.update(float(meas["SH"]), dt, compressor_on=self._N > 6.0, N=self._N)
        return ControlOutput(N=self._N, eev=float(eev), fan_i=self.fan_i, fan_o=self.fan_o)


def scaled_pid_gains(spec, constraints, mode: str) -> tuple[float, float, float]:
    """Trim around load feedforward. Signs: heating +e → +N, cooling +e → −N."""
    scale = constraints.N_max / 70.0
    kp, ki, kd = 4.0 * scale, 0.10 * scale, 1.2 * scale
    if mode == "cooling":
        return -abs(kp), -abs(ki), -abs(kd)
    return kp, ki, kd


def make_cascade(
    mode: str = "heating",
    kind: str = "pid",
    Tsp: float = 293.15,
    spec=None,
    constraints=None,
    Q_design: float = 0.0,
) -> Cascade:
    from heatpump.requirements import Constraints

    cons = constraints or Constraints()
    mode = "cooling" if mode == "cooling" else "heating"
    if kind == "pid":
        if spec is not None:
            kp, ki, kd = scaled_pid_gains(spec, cons, mode)
        else:
            s = -1.0 if mode == "cooling" else 1.0
            kp, ki, kd = s * 4.0, s * 0.10, s * 1.2
        # Trim around feedforward; full range so a large zone error can still saturate.
        speed = PID(kp=kp, ki=ki, kd=kd, umin=-cons.N_max, umax=cons.N_max, kaw=0.4)
    elif kind == "hysteresis":
        speed = HysteresisThermostat(
            deadband=max(1.2, 2.0 * cons.T_zone_band),
            N_on=min(55.0, cons.N_max),
            min_on=cons.min_on_s,
            min_off=cons.min_off_s,
            mode=mode,
        )
    else:
        speed = BangBang(N_on=min(55.0, cons.N_max), mode=mode)
    N_des = float(getattr(spec, "N_design", 50.0) or 50.0) if spec is not None else 50.0
    UA = float(getattr(spec, "UA_env", 0.0) or 0.0) if spec is not None else 0.0
    return Cascade(
        speed=speed,
        eev=SuperheatEEV(sh_sp=cons.SH_sp),
        Tsp=Tsp,
        kind=kind if kind != "bang-bang" else "bangbang",
        mode=mode,
        N_max=cons.N_max,
        N_design=N_des,
        UA=UA,
        Q_design=float(Q_design),
    )


# ---------------------------------------------------------------------------
# MPC
# ---------------------------------------------------------------------------


@dataclass
class LinearMPC:
    """Affine finite-horizon MPC about the current (y, u) of the full plant.

    Decision variables are compressor speed and EEV opening. The discrete
    model is the implicit-Euler linearization

        y_{k+1} = A y_k + B u_k + c

    with A, B from ``jacfwd`` of the residual. The QP is dense least-squares
    on the stacked inputs, then projected onto box constraints.
    """

    rhs: Callable
    project: Callable
    i_tz: int
    sh_fn: Callable[[Array], Array]
    dt: float = 3.0
    horizon: int = 8
    Tsp: float = 293.15
    sh_sp: float = 6.0
    q_T: float = 8.0
    q_sh: float = 0.25
    r_N: float = 8e-4
    r_eev: float = 12.0
    s_N: float = 2e-5
    N_bounds: tuple[float, float] = (0.0, 70.0)
    eev_bounds: tuple[float, float] = (0.10, 0.72)
    fan_i: float = 1.0
    fan_o: float = 1.0
    mode: str = "heating"
    _U: np.ndarray | None = None
    _lin = None

    def reset(self) -> None:
        self._U = None

    def set_mode(self, mode: str) -> None:
        self.mode = "cooling" if mode == "cooling" else "heating"
        self.reset()

    def _ensure_lin(self):
        if self._lin is not None:
            return

        def f(y, u):
            return self.rhs(jnp.float64(0.0), y, u)

        dfdy = jax.jit(jax.jacfwd(f, 0))
        dfdu = jax.jit(jax.jacfwd(f, 1))
        f_j = jax.jit(f)
        self._lin = (f_j, dfdy, dfdu)

    def _implicit_maps(self, Af: np.ndarray, h: float) -> np.ndarray:
        """Discrete A = (I - h Af)^{-1}. Regularize; never fall back to explicit Euler."""
        n = Af.shape[0]
        eye = np.eye(n)
        ridge = 0.0
        for _ in range(6):
            try:
                return np.linalg.solve(eye - h * Af + ridge * eye, eye)
            except np.linalg.LinAlgError:
                ridge = 1e-8 if ridge == 0.0 else ridge * 10.0
        raise np.linalg.LinAlgError("implicit-Euler linearization is singular")

    def update(self, t: float, meas: dict, dt: float, y: Array, u_exog: Array) -> ControlOutput:
        del t, meas
        self._ensure_lin()
        f_j, dfdy, dfdu = self._lin
        y = jnp.asarray(y)
        u0 = jnp.asarray(u_exog)
        Af = np.nan_to_num(np.asarray(dfdy(y, u0)), nan=0.0, posinf=0.0, neginf=0.0)
        Bf = np.nan_to_num(np.asarray(dfdu(y, u0))[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        f0 = np.nan_to_num(np.asarray(f_j(y, u0)), nan=0.0, posinf=0.0, neginf=0.0)
        y0 = np.asarray(self.project(y))
        u0c = np.asarray(u0)[:2]
        h = float(dt) if dt is not None and dt > 0.0 else float(self.dt)
        self.dt = h
        # Implicit Euler linearization of the residual (not TR-BDF2).
        try:
            A = self._implicit_maps(Af, h)
        except np.linalg.LinAlgError:
            eT = float(y0[self.i_tz] - self.Tsp)
            sign = 8.0 if self.mode == "cooling" else -8.0
            N = float(np.clip(u0c[0] + sign * eT, *self.N_bounds))
            eev = float(np.clip(u0c[1], *self.eev_bounds))
            return ControlOutput(N=N, eev=eev, fan_i=self.fan_i, fan_o=self.fan_o)
        B = A @ (h * Bf)
        c = A @ (h * (f0 - Af @ y0 - Bf @ u0c))

        H = self.horizon
        nx, nu = y0.size, 2
        G = np.zeros((H * nx, H * nu))
        d = np.zeros(H * nx)
        xpred = y0.copy()
        for k in range(H):
            xpred = np.asarray(self.project(jnp.asarray(A @ xpred + c)))
            d[k * nx : (k + 1) * nx] = xpred
            Apow = np.eye(nx)
            for j in range(k, -1, -1):
                G[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = Apow @ B
                Apow = Apow @ A

        C = np.zeros((2, nx))
        C[0, self.i_tz] = 1.0
        sh0 = float(self.sh_fn(y))
        if not hasattr(self, "_sh_grad"):
            self._sh_grad = jax.jit(jax.grad(lambda z: self.sh_fn(z)))
        C[1, :] = np.nan_to_num(np.asarray(self._sh_grad(y)), nan=0.0)

        Q = np.diag([self.q_T, self.q_sh])
        # stacked output map
        Cbar = np.kron(np.eye(H), C)
        Qbar = np.kron(np.eye(H), Q)
        zbar = np.tile(np.array([self.Tsp, self.sh_sp]), H)
        # SH affine: C (x - y0) + sh0
        z_off = np.zeros(2 * H)
        for k in range(H):
            z_off[2 * k + 1] = sh0 - C[1] @ y0
        # y_stack = G U + d  →  z = Cbar (G U + d) + z_off
        Hm = Cbar @ G
        z0 = Cbar @ d + z_off
        # input rate penalty
        D = np.eye(H * nu)
        for k in range(1, H):
            D[k * nu : (k + 1) * nu, (k - 1) * nu : k * nu] -= np.eye(nu)
        R = np.diag(np.tile([self.r_N, self.r_eev], H))
        S = np.diag(np.tile([self.s_N, 0.4], H))
        # J = (Hm U + z0 - zbar)^T Q (·) + U^T S U + (D U)^T R (D U)
        e0 = z0 - zbar
        Hqp = Hm.T @ Qbar @ Hm + S + D.T @ R @ D
        gqp = Hm.T @ Qbar @ e0
        try:
            U = -np.linalg.solve(Hqp + 1e-8 * np.eye(H * nu), gqp)
        except np.linalg.LinAlgError:
            U = np.zeros(H * nu)
        U = U.reshape(H, nu)
        if not np.isfinite(U).all():
            eT = float(y0[self.i_tz] - self.Tsp)
            sign = 8.0 if self.mode == "cooling" else -8.0
            U[:, 0] = float(np.clip(40.0 + sign * eT, *self.N_bounds))
            U[:, 1] = 0.35
        U[:, 0] = np.clip(U[:, 0], *self.N_bounds)
        U[:, 1] = np.clip(U[:, 1], *self.eev_bounds)
        self._U = U
        return ControlOutput(N=float(U[0, 0]), eev=float(U[0, 1]), fan_i=self.fan_i, fan_o=self.fan_o)


@dataclass
class NonlinearMPC:
    """Shooting NMPC: implicit-Euler rollout, projected gradient on U."""

    rhs: Callable
    project: Callable
    i_tz: int
    sh_fn: Callable[[Array], Array]
    dt: float = 8.0
    horizon: int = 6
    Tsp: float = 293.15
    sh_sp: float = 6.0
    q_T: float = 6.0
    q_sh: float = 0.2
    r_N: float = 6e-4
    r_eev: float = 10.0
    n_iter: int = 8
    n_newton: int = 6
    N_bounds: tuple[float, float] = (0.0, 70.0)
    eev_bounds: tuple[float, float] = (0.10, 0.72)
    fan_i: float = 1.0
    fan_o: float = 1.0
    mode: str = "heating"
    _U: np.ndarray | None = None
    _grad = None

    def reset(self) -> None:
        self._U = None

    def set_mode(self, mode: str) -> None:
        mode = "cooling" if mode == "cooling" else "heating"
        if mode == self.mode:
            return
        self.mode = mode
        self.reset()

    def _ensure(self):
        if self._grad is not None:
            return
        i_tz, sh_fn, rhs, project = self.i_tz, self.sh_fn, self.rhs, self.project
        Tsp, sh_sp = self.Tsp, self.sh_sp
        q_T, q_sh, r_N, r_eev = self.q_T, self.q_sh, self.r_N, self.r_eev
        n_newton = int(self.n_newton)

        def cost(U, y0, u_base, h):
            def body(y, ue):
                u = u_base.at[0].set(ue[0]).at[1].set(ue[1])
                yn = implicit_euler_step(rhs, 0.0, y, u, h, project, n_newton=n_newton)
                eT = yn[i_tz] - Tsp
                eS = sh_fn(yn) - sh_sp
                return yn, q_T * eT**2 + q_sh * eS**2

            _, stage = jax.lax.scan(body, y0, U)
            dU = jnp.diff(U, axis=0, prepend=U[:1])
            return jnp.sum(stage) + r_N * jnp.sum(dU[:, 0] ** 2) + r_eev * jnp.sum(dU[:, 1] ** 2)

        self._grad = jax.jit(jax.grad(cost, argnums=0))
        self._cost = jax.jit(cost)

    def update(self, t: float, meas: dict, dt: float, y: Array, u_exog: Array) -> ControlOutput:
        del t, meas
        self._ensure()
        H = self.horizon
        h = float(dt) if dt is not None and dt > 0.0 else float(self.dt)
        self.dt = h
        if self._U is None:
            u0 = np.asarray(u_exog)
            self._U = np.tile(np.array([float(u0[0]), float(u0[1])]), (H, 1))
        U = jnp.asarray(self._U)
        y0 = jnp.asarray(self.project(y))
        u_base = jnp.asarray(u_exog)
        lo = jnp.array([self.N_bounds[0], self.eev_bounds[0]])
        hi = jnp.array([self.N_bounds[1], self.eev_bounds[1]])
        lr = 0.25
        for _ in range(self.n_iter):
            g = self._grad(U, y0, u_base, h)
            U = jnp.clip(U - lr * g, lo, hi)
            lr *= 0.9
        self._U = np.array(U)
        self._U = np.vstack([self._U[1:], self._U[-1:]])
        return ControlOutput(N=float(U[0, 0]), eev=float(U[0, 1]), fan_i=self.fan_i, fan_o=self.fan_o)


def make_mpc(
    rhs,
    project,
    i_tz: int,
    sh_fn: Callable[[Array], Array],
    Tsp: float = 293.15,
    nonlinear: bool = False,
    mode: str = "heating",
    N_bounds: tuple[float, float] = (0.0, 70.0),
    eev_bounds: tuple[float, float] = (0.10, 0.72),
):
    kw = dict(
        rhs=rhs,
        project=project,
        i_tz=i_tz,
        sh_fn=sh_fn,
        Tsp=Tsp,
        N_bounds=N_bounds,
        eev_bounds=eev_bounds,
        mode=mode,
    )
    if nonlinear:
        return NonlinearMPC(**kw)
    return LinearMPC(**kw)
