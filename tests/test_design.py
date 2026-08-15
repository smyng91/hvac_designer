"""Auto-sizing for an arbitrary refrigerant and heating load."""

import numpy as np
import pytest

from heatpump.design import design_air_conditioner, design_heat_pump, design_system, heating_spec
from heatpump.requirements import DesignRequest, TimeSeries, cooling_tons_to_w


def test_design_r32_matches_small_split():
    rep = design_heat_pump("R32", 5500.0, T_out=273.15, T_zone=293.15)
    assert rep.fluid == "R32"
    assert rep.p_c > rep.p_e * 2.0
    assert 2.5 < rep.COP < 6.5
    assert 1.0e-5 < rep.V_disp < 1.2e-4
    assert rep.n_tubes_e >= 8 and rep.n_tubes_c >= 8
    assert rep.spec.fluid == "R32"
    assert abs(rep.spec.UA_env - 5500.0 / 20.0) < 1.0


def test_low_density_fluid_gets_larger_compressor():
    r32 = design_heat_pump("R32", 6000.0)
    r134a = design_heat_pump("R134a", 6000.0)
    assert r134a.V_disp > r32.V_disp
    assert r134a.mdot != r32.mdot
    assert r134a.spec.fluid == "R134a"


def test_propane_and_blend_size():
    for fluid in ("R290", "R410A", "R1234yf"):
        rep = design_heat_pump(fluid, 4000.0, T_out=268.15, T_zone=293.15)
        assert rep.Q_heat == pytest.approx(4000.0)
        assert rep.spec.V_disp > 0.0
        assert rep.spec.A_eev > 0.0
        assert rep.p_c < 0.92 * rep.spec.p_crit


def test_co2_rejected_for_room_heating():
    with pytest.raises(ValueError, match="subcritically"):
        design_heat_pump("CO2", 5000.0, T_out=273.15, T_zone=293.15)


def test_heating_spec_overrides_cells():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    assert spec.n_e == 4 and spec.n_c == 4
    assert spec.fluid == "R32"


def test_cooling_tons_and_cycle():
    assert cooling_tons_to_w(2.0) == pytest.approx(7033.7, rel=1e-3)
    rep = design_air_conditioner("R410A", cooling_tons=2.0, T_out=308.15, T_zone=297.15)
    assert rep.kind == "cooling"
    assert rep.spec.mode == "cooling"
    assert rep.p_c > rep.p_e
    assert rep.COP > 2.0
    assert rep.Q_cool == pytest.approx(cooling_tons_to_w(2.0), rel=1e-6)


def test_heat_pump_merges_hardware():
    sys = design_system(
        DesignRequest(
            refrigerant="R32",
            mode="heat_pump",
            Q_heat=5000.0,
            Q_cool=7000.0,
            T_out_heat=268.15,
            T_out_cool=308.15,
            T_zone=294.15,
        )
    )
    assert sys.heating is not None and sys.cooling is not None
    assert sys.spec.V_disp >= max(sys.heating.V_disp, sys.cooling.V_disp) - 1e-12
    assert sys.controller == "pid"
    cool_spec = sys.spec
    from heatpump.plant import apply_operating_mode

    heat = apply_operating_mode(cool_spec, "heating")
    cool = apply_operating_mode(cool_spec, "cooling")
    assert heat.operating_mode == "heating"
    assert cool.operating_mode == "cooling"


def test_cooling_zone_heat_is_negative():
    import jax.numpy as jnp
    from heatpump.plant import diagnostics, initial_state, make_rhs
    from heatpump.thermo import build_tables

    rep = design_air_conditioner("R32", 5000.0, T_out=308.15, T_zone=297.15, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=24, n_h=36)
    y = initial_state(rep.spec, tables, T_out=308.15, T_zone=299.15)
    u = jnp.array([50.0, 0.4, 1.0, 1.0, 308.15, 0.0])
    d = diagnostics(rep.spec, tables, y, u)
    assert float(d["Q_zone"]) < 0.0
    assert np.all(np.isfinite(np.asarray(make_rhs(rep.spec, tables)(0.0, y, u))))


def test_design_from_setpoint_and_profile(tmp_path):
    csv = tmp_path / "cool.csv"
    csv.write_text("t,T_out_C,Q_kW,Tsp_C\n0,28,3.5,24\n600,35,6.2,24\n1200,32,4.0,24\n")
    ts = TimeSeries.from_csv(csv)
    req = DesignRequest(refrigerant="R410A", mode="auto", T_zone=297.15, timeseries=ts)
    assert req.mode == "cooling"
    assert req.inferred_from_profile
    assert req.Q_cool == pytest.approx(6200.0)
    assert req.T_out_cool == pytest.approx(308.15)
    assert req.use_envelope is False
    sys = design_system(req)
    assert sys.cooling is not None and sys.heating is None
    assert sys.spec.UA_env == 0.0
    assert any("inferred" in n.lower() for n in sys.notes)


def test_timeseries_interpolation(tmp_path):
    csv = tmp_path / "w.csv"
    csv.write_text("t,T_out_C,Q_kW\n0,30,4.0\n600,35,6.0\n")
    ts = TimeSeries.from_csv(csv, Q_unit="kW")
    mid = ts.at(300.0)
    assert 303.15 < mid["T_out"] < 308.15
    assert 4000.0 < mid["Q_gain"] < 6000.0
