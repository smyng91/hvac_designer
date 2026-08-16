"""AHRI 540 and fan tables: published or user data only, no silent defaults."""

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from heatpump.design import heating_spec
from heatpump.devices import AHRI540Compressor, LinearFan, TableFan
from heatpump.plant import diagnostics, initial_state, make_rhs
from heatpump.thermo import build_tables
from heatpump.validation import (
    LEE2021_TABLE5_NEW_MDOT_G_S,
    LEE2021_TABLE5_NEW_POWER,
    ahri540_poly_np,
    compare_lee2021_map,
    default_maps_dir,
)

LEE = default_maps_dir() / "lee2021_iop1180_012041.json"


def test_ahri540_rejects_missing_citation_and_short_maps():
    with pytest.raises(ValueError, match="source"):
        AHRI540Compressor(power_C=(0.0,) * 10, mdot_C=(0.0,) * 10, citation="")
    with pytest.raises(ValueError, match="10"):
        AHRI540Compressor(power_C=(1.0, 2.0), mdot_C=(1.0,) * 10, citation="x")


def test_lee2021_json_matches_table5_and_poly_is_independent():
    report = compare_lee2021_map(LEE)
    assert report["coefficients_match_table5"]
    assert report["all_positive"]
    t5 = report["table4_test5"]
    assert t5["Te_C"] == 12.3 and t5["Tc_C"] == 48.4
    assert t5["power_W"] > 0.0 and t5["mdot_g_s"] > 0.0
    # Independent numpy poly vs stored coefficients (not the JAX device).
    W = ahri540_poly_np(12.3, 48.4, LEE2021_TABLE5_NEW_POWER)
    m = ahri540_poly_np(12.3, 48.4, LEE2021_TABLE5_NEW_MDOT_G_S)
    assert W == pytest.approx(t5["power_W"])
    assert m == pytest.approx(t5["mdot_g_s"])


def test_ahri540_device_matches_numpy_poly_at_table4_test5():
    comp = AHRI540Compressor.from_file(LEE)
    Ts = jnp.float64(12.3 + 273.15)
    Td = jnp.float64(48.4 + 273.15)
    mdot, h_d, power = comp.map(
        jnp.float64(1e6),
        jnp.float64(2e6),
        jnp.float64(4.0e5),
        jnp.float64(20.0),
        Ts,
        jnp.float64(50.0),
        Ts,
        Td,
    )
    W = ahri540_poly_np(12.3, 48.4, LEE2021_TABLE5_NEW_POWER)
    m = ahri540_poly_np(12.3, 48.4, LEE2021_TABLE5_NEW_MDOT_G_S) * 1.0e-3
    assert float(power) == pytest.approx(W, rel=1e-6)
    assert float(mdot) == pytest.approx(m, rel=1e-6)
    assert float(h_d) == pytest.approx(4.0e5 + W / m, rel=1e-6)


def test_ahri540_on_plant_rhs_finite():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    spec = replace(spec, compressor=AHRI540Compressor.from_file(LEE), ahri540_path=str(LEE))
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables)
    u = jnp.array([50.0, 0.35, 1.0, 1.0, 273.15, -1200.0])
    dy = make_rhs(spec, tables)(0.0, y, u)
    assert np.all(np.isfinite(np.asarray(dy)))
    d = diagnostics(spec, tables, y, u)
    assert float(d["mdot_comp"]) > 0.0


def test_ahri540_is_assigned_not_registered():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    spec = replace(spec, compressor=AHRI540Compressor.from_file(LEE))
    assert isinstance(spec.compressor, AHRI540Compressor)


def test_table_fan_requires_citation_and_points():
    with pytest.raises(ValueError, match="source"):
        TableFan(speed=(0.0, 1.0), mdot_kg_s=(0.1, 0.5), citation="")
    fan = LinearFan()
    assert float(fan.mdot(jnp.float64(0.5), 0.8)) == pytest.approx(0.4)
