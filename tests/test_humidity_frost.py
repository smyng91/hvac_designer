"""Optional moist / frost states. Dry default is unchanged; no invented RH."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from heatpump.design import heating_spec
from heatpump.plant import diagnostics, initial_state, make_rhs, unpack_state
from heatpump.thermo import build_tables


def test_dry_layout_unchanged():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    assert spec.moist is False and spec.frost is False
    assert spec.layout.n_state == 3 + 8 + 8
    assert spec.layout.slow_idx() == (spec.layout.i_tz,)
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables, T_out=273.15, T_zone=291.15)
    u = jnp.array([45.0, 0.35, 1.0, 1.0, 273.15, -1200.0])
    d = diagnostics(spec, tables, y, u)
    assert float(d["Q_lat"]) == 0.0
    assert float(d["delta_fr"]) == 0.0


def test_moist_requires_user_humidity():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    with pytest.raises(ValueError, match="RH_out"):
        make_rhs(replace(spec, moist=True, RH_zone0=0.45), tables)
    with pytest.raises(ValueError, match="RH_zone0"):
        make_rhs(replace(spec, moist=True, RH_out=0.70), tables)
    with pytest.raises(ValueError, match="frost requires moist"):
        make_rhs(replace(spec, frost=True, RH_out=0.80), tables)


def test_moist_rhs_finite_and_adds_humidity_state():
    spec = replace(
        heating_spec("R32", 5500.0, n_e=4, n_c=4),
        moist=True,
        RH_out=0.70,
        RH_zone0=0.45,
    )
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables, T_out=273.15, T_zone=291.15)
    assert y.size == spec.layout.n_state
    s = unpack_state(y, spec.layout)
    assert 0.001 < float(s["W_z"]) < 0.02
    u = jnp.array([45.0, 0.35, 1.0, 1.0, 273.15, -1200.0, 0.0, 0.0, 0.70])
    dy = make_rhs(spec, tables)(0.0, y, u)
    assert dy.size == y.size
    assert np.all(np.isfinite(np.asarray(dy)))
    d = diagnostics(spec, tables, y, u)
    assert np.isfinite(float(d["Q_lat"]))
    assert np.isfinite(float(d["Q_zone"]))
    assert float(d["delta_fr"]) == 0.0


def test_frost_grows_only_from_humidity_when_wall_is_below_freezing():
    spec = replace(
        heating_spec("R32", 5500.0, n_e=4, n_c=4),
        moist=True,
        frost=True,
        RH_out=0.85,
        RH_zone0=0.40,
        W_defrost=0.0,
    )
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables, T_out=263.15, T_zone=291.15)
    assert float(unpack_state(y, spec.layout)["m_fr"]) == 0.0
    u = jnp.array([50.0, 0.35, 1.0, 1.0, 263.15, -2000.0, 0.0, 0.0, 0.85])
    dy = make_rhs(spec, tables)(0.0, y, u)
    assert np.all(np.isfinite(np.asarray(dy)))
    # W_defrost = 0: no invented heater. Growth is ≥ 0 from the humidity balance.
    assert float(dy[spec.layout.i_fr]) >= -1.0e-12


def test_defrost_flag_without_heater_does_not_invent_melt():
    spec = replace(
        heating_spec("R32", 5500.0, n_e=4, n_c=4),
        moist=True,
        frost=True,
        RH_out=0.80,
        RH_zone0=0.40,
        W_defrost=0.0,
    )
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables, T_out=263.15, T_zone=291.15)
    y = y.at[spec.layout.i_fr].set(0.05)
    u_off = jnp.array([40.0, 0.35, 1.0, 1.0, 263.15, -1500.0, 0.0, 0.0, 0.80])
    u_flag = jnp.array([40.0, 0.35, 1.0, 1.0, 263.15, -1500.0, 0.0, 1.0, 0.80])
    rhs = make_rhs(spec, tables)
    dm0 = float(rhs(0.0, y, u_off)[spec.layout.i_fr])
    dm1 = float(rhs(0.0, y, u_flag)[spec.layout.i_fr])
    assert dm1 == pytest.approx(dm0, abs=1e-9)
    d = diagnostics(spec, tables, y, u_off)
    assert float(d["delta_fr"]) > 0.0
    assert np.isfinite(float(d["Q_lat"]))
    assert np.isfinite(float(d["Q_zone"]))
