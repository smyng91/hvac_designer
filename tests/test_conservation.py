"""Closed-loop mass inventory and a short implicit step."""

import jax
import jax.numpy as jnp
import numpy as np

from heatpump.design import heating_spec
from heatpump.plant import diagnostics, initial_state, make_rhs, project_state
from heatpump.thermo import build_tables

jax.config.update("jax_enable_x64", True)


def test_charge_finite_and_rhs_finite():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables)
    u = jnp.array([45.0, 0.35, 1.0, 1.0, 273.15, -1200.0])
    rhs = make_rhs(spec, tables)
    dy = rhs(0.0, y, u)
    assert np.all(np.isfinite(np.asarray(dy)))
    d = diagnostics(spec, tables, y, u)
    assert 0.2 < float(d["charge"]) < 4.0
    assert float(d["p_c"]) > float(d["p_e"])


def test_mass_almost_closed_over_implicit_euler():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables)
    u = jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, -1000.0])
    rhs = make_rhs(spec, tables)
    project = lambda z: project_state(z, tables, spec.layout)
    from heatpump.solver import implicit_euler_step

    m0 = float(diagnostics(spec, tables, y, u)["charge"])
    y1 = implicit_euler_step(rhs, 0.0, y, u, 0.25, project, n_newton=5)
    m1 = float(diagnostics(spec, tables, y1, u)["charge"])
    assert abs(m1 - m0) / m0 < 5e-3


def test_mass_almost_closed_over_trbdf2():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables)
    u = jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, -1000.0])
    rhs = make_rhs(spec, tables)
    project = lambda z: project_state(z, tables, spec.layout)
    from heatpump.solver import integrate

    m0 = float(diagnostics(spec, tables, y, u)["charge"])
    t, Y = integrate(rhs, y, lambda _t: u, t_final=20.0, dt0=0.25, project=project, record_dt=2.0)
    assert t[-1] >= 19.0
    m1 = float(diagnostics(spec, tables, jnp.asarray(Y[-1]), u)["charge"])
    assert abs(m1 - m0) / m0 < 0.02


def test_pid_antiwindup_saturates():
    from heatpump.control import PID

    pid = PID(kp=10.0, ki=20.0, umin=0.0, umax=1.0)
    u = 0.0
    for _ in range(40):
        u = pid.update(1.0, 0.0, 0.1)
    assert 0.0 <= u <= 1.0
    assert pid._i < 50.0
