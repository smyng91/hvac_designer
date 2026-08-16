from heatpump.control import BangBang, HysteresisThermostat, PID, make_cascade
from heatpump.design import design_air_conditioner, heating_spec
from heatpump.simulate import simulate
from heatpump.thermo import build_tables


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
