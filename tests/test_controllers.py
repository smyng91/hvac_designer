from heatpump.control import BangBang, HysteresisThermostat, PID


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
