"""Users retrofit components by assigning objects on PlantSpec."""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from heatpump.design import heating_spec
from heatpump.devices import ClearanceCompressor, LumpedZone
from heatpump.plant import diagnostics, initial_state, make_rhs
from heatpump.thermo import build_tables


def test_default_path_does_not_need_a_device_object():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    assert spec.compressor is None
    assert spec.zone_model is None
    tables = build_tables("R32", n_p=32, n_h=48)
    dy = make_rhs(spec, tables)(
        0.0, initial_state(spec, tables), jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, 0.0])
    )
    assert np.all(np.isfinite(np.asarray(dy)))


def test_swap_compressor_object():
    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    tables = build_tables("R32", n_p=32, n_h=48)
    scaled = ClearanceCompressor(
        V_disp=spec.V_disp * 1.2,
        C_loss=spec.C_loss,
        eta_is=spec.eta_is0,
        gamma=spec.gamma,
    )
    spec = replace(spec, compressor=scaled)
    y = initial_state(spec, tables)
    u = jnp.array([45.0, 0.35, 1.0, 1.0, 273.15, -1200.0])
    dy = make_rhs(spec, tables)(0.0, y, u)
    assert np.all(np.isfinite(np.asarray(dy)))
    d = diagnostics(spec, tables, y, u)
    assert float(d["mdot_comp"]) > 0.0
    assert float(d["p_c"]) > float(d["p_e"])


def test_swap_zone_object():
    @dataclass(frozen=True)
    class DummyZone:
        C: float
        UA: float

        def dTdt(self, T_z, T_out, Q_hvac, Q_gain):
            return (Q_hvac + Q_gain + self.UA * (T_out - T_z)) / self.C

    spec = heating_spec("R32", 5500.0, n_e=4, n_c=4)
    spec = replace(spec, zone_model=DummyZone(spec.C_zone, spec.UA_env))
    assert isinstance(spec.zone_model, DummyZone)
    tables = build_tables("R32", n_p=32, n_h=48)
    dy = make_rhs(spec, tables)(
        0.0, initial_state(spec, tables), jnp.array([40.0, 0.4, 1.0, 1.0, 273.15, 0.0])
    )
    assert np.all(np.isfinite(np.asarray(dy)))
    assert isinstance(LumpedZone(spec.C_zone, spec.UA_env), LumpedZone)
