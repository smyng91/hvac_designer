"""CoolProp refrigerant properties and JAX (p, h) tables."""

import numpy as np
import pytest

from heatpump.thermo import (
    build_tables,
    eval_ph,
    fluid_info,
    p_sat,
    resolve_fluid,
    sat_at_T,
    sat_from_tables,
)


def test_resolve_aliases():
    assert resolve_fluid("R-32") == "R32"
    assert resolve_fluid("410a") == "R410A"
    assert resolve_fluid("propane") == "R290"
    assert resolve_fluid("R134a") == "R134a"


def test_unknown_fluid():
    with pytest.raises(ValueError, match="Unknown refrigerant"):
        resolve_fluid("not-a-refrigerant")


def test_r32_iir_reference_at_0c():
    env = sat_at_T("R32", 273.15, q=0.0)
    assert abs(env["hf"] / 1e3 - 200.00) < 0.05
    assert abs(env["p"] / 1e3 - 813.10) < 1.0
    assert abs(env["rhof"] - 1055.3) < 1.0
    dew = sat_at_T("R32", 273.15, q=1.0)
    assert abs(dew["hg"] / 1e3 - 515.30) < 0.05
    assert 15.0 < dew["rhog"] < 30.0


def test_r32_critical():
    info = fluid_info("R32")
    assert abs(info.pc - 5.782645e6) / 5.782645e6 < 5e-4
    assert abs(info.Tc - 351.255) < 0.02


def test_psat_r32():
    assert abs(p_sat("R32", 273.15) - 813100.0) / 813100.0 < 5e-4
    assert abs(p_sat("R32", 246.35) - 311510.0) / 311510.0 < 2e-3


def test_tables_r32_dome():
    tables = build_tables("R32", n_p=32, n_h=48)
    assert tables.fluid == "R32"
    env = sat_from_tables(tables, 813097.0)
    assert abs(env["Tsat"] - 273.15) < 0.4
    assert 180e3 < env["hf"] < 220e3
    assert 500e3 < env["hg"] < 530e3
    st = eval_ph(tables, 813097.0, env["hf"])
    assert abs(float(st.T) - env["Tsat"]) < 2.5
    assert float(st.x) < 0.08


def test_tables_other_fluids():
    for name in ("R134a", "R410A", "R290"):
        tables = build_tables(name, n_p=24, n_h=36)
        assert tables.fluid == name
        assert tables.pc > 2.0e6
        assert float(tables.p[-1]) < 0.95 * tables.pc
        st = eval_ph(tables, float(tables.p[len(tables.p) // 2]), float(tables.h[len(tables.h) // 2]))
        assert np.isfinite(float(st.T))
        assert float(st.rho) > 0.5


def test_eval_ph_reads_each_field_from_its_own_grid():
    """eval_ph interpolates T, x, and rho independently (not a coupled flash)."""
    tables = build_tables("R32", n_p=32, n_h=48)
    i, j = 10, 20
    p = float(tables.p[i])
    h = float(tables.h[j])
    st = eval_ph(tables, p, h)
    assert abs(float(st.T) - float(tables.T[i, j])) < 1e-8
    assert abs(float(st.rho) - float(tables.rho[i, j])) < 1e-8
    assert abs(float(st.x) - float(np.clip(tables.x[i, j], 0.0, 1.0))) < 1e-8


def test_eval_ph_dome_quality_tracks_enthalpy():
    tables = build_tables("R32", n_p=32, n_h=48)
    p = 813097.0
    env = sat_from_tables(tables, p)
    hf, hg = env["hf"], env["hg"]
    xs = []
    rhos = []
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        st = eval_ph(tables, p, hf + frac * (hg - hf))
        xs.append(float(st.x))
        rhos.append(float(st.rho))
        assert abs(float(st.x) - frac) < 0.12
    assert xs[-1] > xs[0]
    assert rhos[0] > rhos[-1]
