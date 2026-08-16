"""Reversing valve: state remap and a short cool→heat closed loop."""

import numpy as np

from heatpump.design import design_system, heating_spec
from heatpump.plant import apply_operating_mode, initial_state, remap_state, unpack_state
from heatpump.requirements import DesignRequest, TimeSeries
from heatpump.simulate import simulate
from heatpump.thermo import build_tables


def test_remap_swaps_coil_states_and_keeps_the_zone():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    y = initial_state(spec, tables, T_out=273.15, T_zone=288.15)
    cool = apply_operating_mode(spec, "cooling")
    y2 = remap_state(y, spec.layout, cool.layout)
    s0 = unpack_state(y, spec.layout)
    s1 = unpack_state(y2, cool.layout)
    assert float(s1["p_e"]) == float(s0["p_c"])
    assert float(s1["p_c"]) == float(s0["p_e"])
    assert float(s1["T_z"]) == float(s0["T_z"])
    assert y2.size == cool.layout.n_state


def test_closed_loop_cool_then_heat_rebuilds_pressure_ratio():
    ts = TimeSeries(
        t=np.array([0.0, 40.0, 40.01, 90.0]),
        T_out=np.array([308.15, 308.15, 273.15, 273.15]),
        Q_gain=np.array([3500.0, 3500.0, -2000.0, -2000.0]),
        Tsp=np.array([297.15, 297.15, 293.15, 293.15]),
        mode=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    req = DesignRequest(
        refrigerant="R32",
        mode="heat_pump",
        Q_heat=5500.0,
        Q_cool=5500.0,
        T_zone=293.15,
        T_zone_cool=297.15,
        n_cells=4,
        timeseries=ts,
    )
    sys = design_system(req)
    tables = build_tables(sys.spec.fluid, n_p=32, n_h=48)
    res = simulate(
        "pid",
        t_final=90.0,
        spec=sys.spec,
        tables=tables,
        design=sys,
        request=req,
        timeseries=ts,
        record_dt=5.0,
        reduction="full",
    )
    assert res.t[-1] >= 85.0
    heat = res.t >= 50.0
    assert np.mean(res.meas["mode"][heat]) > 0.5
    assert res.meas["p_c"][-1] > res.meas["p_e"][-1]
    assert np.isfinite(res.meas["T_z"][-1])
