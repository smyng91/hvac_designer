import jax
import jax.numpy as jnp
import numpy as np

from heatpump.control import (
    BangBang,
    HysteresisThermostat,
    LinearMPC,
    NonlinearMPC,
    PID,
    make_cascade,
    make_mpc,
)
from heatpump.design import design_air_conditioner, heating_spec
from heatpump.plant import diagnostics, initial_state, make_rhs, project_state
from heatpump.simulate import simulate
from heatpump.solver import implicit_euler_step
from heatpump.thermo import build_tables

jax.config.update("jax_enable_x64", True)


def test_hysteresis_turns_on_when_cold():
    h = HysteresisThermostat(deadband=1.0, N_on=50.0, min_on=0.0, min_off=0.0)
    assert h.update(0.0, 288.0, 293.0) == 50.0
    assert h.update(10.0, 294.0, 293.0) == 0.0


def test_bangbang_deadband():
    b = BangBang(deadband=1.0, N_on=40.0)
    assert b.update(287.0, 293.0) == 40.0
    assert b.update(293.0, 293.0) == 40.0
    assert b.update(294.0, 293.0) == 0.0


def test_pid_moves_toward_setpoint_direction():
    pid = PID(kp=2.0, ki=0.0, kd=0.0, umin=-10.0, umax=10.0)
    u = pid.update(1.0, 0.0, 0.1)
    assert u > 0.0


def test_cooling_hysteresis_turns_on_when_hot():
    h = HysteresisThermostat(deadband=1.0, N_on=50.0, min_on=0.0, min_off=0.0, mode="cooling")
    assert h.update(0.0, 298.0, 297.0) == 50.0
    assert h.update(10.0, 296.0, 297.0) == 0.0


def test_cascade_set_mode_flips_pid_sign():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    c = make_cascade("heating", "pid", 293.15, spec)
    assert c.speed.kp > 0.0
    c.set_mode("cooling")
    assert c.mode == "cooling"
    assert c.speed.kp < 0.0


def test_pid_heating_moves_zone_toward_setpoint():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    res = simulate(
        "pid",
        t_final=90.0,
        spec=spec,
        tables=tables,
        T_out=273.15,
        Tsp=293.15,
        record_dt=5.0,
        reduction="full",
    )
    assert res.meas["T_z"][-1] > res.meas["T_z"][0] + 0.3
    assert res.u[-1, 0] > 8.0
    assert res.meas["p_c"][-1] > res.meas["p_e"][-1]


def test_pid_cooling_moves_zone_toward_setpoint():
    design = design_air_conditioner("R410A", 6200.0, n_e=4, n_c=4)
    tables = build_tables(design.spec.fluid, n_p=32, n_h=48)
    res = simulate(
        "pid",
        t_final=90.0,
        spec=design.spec,
        tables=tables,
        design=design,
        T_out=308.15,
        Tsp=297.15,
        record_dt=5.0,
        reduction="full",
    )
    assert res.meas["T_z"][-1] < res.meas["T_z"][0] - 0.2
    assert res.meas["p_c"][-1] > res.meas["p_e"][-1]


def _small_plant():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    rhs = make_rhs(spec, tables)
    project = lambda z: project_state(z, tables, spec.layout)
    y = initial_state(spec, tables)
    u = jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, -1000.0])
    return spec, tables, rhs, project, y, u


def test_residual_jacfwd_is_finite():
    spec, tables, rhs, project, y, u = _small_plant()

    def f(yy, uu):
        return rhs(jnp.float64(0.0), yy, uu)

    Af = np.asarray(jax.jacfwd(f, 0)(y, u))
    Bf = np.asarray(jax.jacfwd(f, 1)(y, u))
    assert np.all(np.isfinite(Af))
    assert np.all(np.isfinite(Bf))
    assert Af.shape == (y.size, y.size)
    assert Bf.shape[0] == y.size


def test_residual_du_matches_finite_difference():
    spec, tables, rhs, project, y, u = _small_plant()

    def f(uu):
        return rhs(jnp.float64(0.0), y, uu)

    g_ad = np.asarray(jax.jacfwd(f)(u))[:, 0]
    eps = 1e-5
    g_fd = (np.asarray(f(u.at[0].set(u[0] + eps))) - np.asarray(f(u.at[0].set(u[0] - eps)))) / (
        2.0 * eps
    )
    scale = np.maximum(np.abs(g_ad), 1e-8)
    rel = np.linalg.norm((g_ad - g_fd) / scale) / np.sqrt(g_ad.size)
    assert rel < 0.05


def test_lmpc_onestep_matches_implicit_euler():
    spec, tables, rhs, project, y, u = _small_plant()
    ctl = make_mpc(rhs, project, spec.layout.i_tz, lambda z: diagnostics(spec, tables, z, u)["SH"])
    h = 2.0
    out = ctl.update(0.0, {}, h, y, u)
    assert np.isfinite(out.N) and np.isfinite(out.eev)
    assert ctl.dt == h

    def f(yy, uu):
        return rhs(jnp.float64(0.0), yy, uu)

    Af = np.asarray(jax.jacfwd(f, 0)(y, u))
    Bf = np.asarray(jax.jacfwd(f, 1)(y, u))[:, :2]
    f0 = np.asarray(f(y, u))
    y0 = np.asarray(y)
    A = np.linalg.solve(np.eye(y0.size) - h * Af, np.eye(y0.size))
    B = A @ (h * Bf)
    c = A @ (h * (f0 - Af @ y0 - Bf @ np.asarray(u)[:2]))
    y_lin = A @ y0 + B @ np.asarray(u)[:2] + c
    y_nl = np.asarray(implicit_euler_step(rhs, 0.0, y, u, h, project, n_newton=6))
    rel = np.linalg.norm(y_lin - y_nl) / max(np.linalg.norm(y_nl), 1.0)
    assert rel < 0.05


def test_nmpc_set_mode_records_cooling():
    spec, tables, rhs, project, y, u = _small_plant()
    ctl = make_mpc(
        rhs,
        project,
        spec.layout.i_tz,
        lambda z: diagnostics(spec, tables, z, u)["SH"],
        nonlinear=True,
        mode="heating",
    )
    assert isinstance(ctl, NonlinearMPC)
    ctl.set_mode("cooling")
    assert ctl.mode == "cooling"


def test_lmpc_heating_closed_loop_is_finite():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    res = simulate(
        "mpc",
        t_final=90.0,
        spec=spec,
        tables=tables,
        T_out=273.15,
        Tsp=293.15,
        record_dt=5.0,
        reduction="qss",
    )
    assert np.all(np.isfinite(res.meas["T_z"]))
    assert np.all(np.isfinite(res.u[:, 0]))
    assert res.meas["p_c"][-1] > res.meas["p_e"][-1]
    assert res.t[-1] >= 89.0
