"""Integrator must reach the requested horizon (hours via QSS)."""

import numpy as np

from heatpump.design import heating_spec
from heatpump.plant import initial_state, make_rhs, project_state
from heatpump.simulate import simulate
from heatpump.solver import integrate, integrate_qss
from heatpump.thermo import build_tables


def _plant():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    return spec, tables


def test_full_dae_reaches_past_old_stall():
    """Previously the DAE stalled near 30 s; 90 s must complete."""
    spec, tables = _plant()
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
    assert res.t[-1] >= 85.0
    assert np.isfinite(res.meas["T_z"][-1])
    assert res.meas["p_c"][-1] > res.meas["p_e"][-1]


def test_qss_hour_reaches_horizon():
    spec, tables = _plant()
    res = simulate(
        "pid",
        t_final=3600.0,
        spec=spec,
        tables=tables,
        T_out=273.15,
        Tsp=293.15,
        record_dt=60.0,
        reduction="qss",
    )
    assert res.t[-1] >= 3500.0
    assert np.isfinite(res.meas["T_z"][-1])
    assert np.isfinite(res.meas["COP"][-1])
    # Controller + QSS must heat the zone, not drift with a frozen u(t).
    assert res.meas["T_z"][-1] > res.meas["T_z"][0]
    # With polytropic design the QSS plant must heat; 4 K is a loose bound
    # (n_e=4 skips DAE displacement matching).
    assert res.meas["T_z"][-1] > 293.15 - 4.0


def test_integrate_qss_direct():
    spec, tables = _plant()
    rhs = make_rhs(spec, tables)
    y0 = initial_state(spec, tables, T_out=273.15, T_zone=288.15)
    u = np.array([50.0, 0.40, 1.0, 1.0, 273.15, -2000.0])
    t, Y = integrate_qss(
        rhs,
        y0,
        lambda _t: u,
        1800.0,
        i_tz=spec.layout.i_tz,
        project=lambda z: project_state(z, tables, spec.layout),
        record_dt=60.0,
        refresh_s=120.0,
    )
    assert t[-1] >= 1740.0
    assert np.all(np.isfinite(Y[-1]))


def test_integrate_advances_when_steps_rejected():
    spec, tables = _plant()
    rhs = make_rhs(spec, tables)
    y0 = initial_state(spec, tables)
    u = np.array([40.0, 0.40, 1.0, 1.0, 273.15, -1000.0])
    t, Y = integrate(
        rhs,
        y0,
        lambda _t: u,
        t_final=20.0,
        dt0=0.25,
        project=lambda z: project_state(z, tables, spec.layout),
        record_dt=2.0,
    )
    assert t[-1] >= 18.0
    assert np.all(np.isfinite(Y[-1]))
