"""Transient two-phase vapor-compression heat pump / air conditioner (JAX)."""

from heatpump.control import (
    BangBang,
    Cascade,
    ControlOutput,
    HysteresisThermostat,
    LinearMPC,
    NonlinearMPC,
    PID,
    SuperheatEEV,
)
from heatpump.capacity import CapacityMap, capacity_map
from heatpump.design import (
    DesignReport,
    SystemDesign,
    design_air_conditioner,
    design_heat_pump,
    design_system,
    heating_spec,
)
from heatpump.gates import DesignGateError, GateSet
from heatpump.report import DesignPackage
from heatpump.plant import CoilSpec, PlantSpec, apply_operating_mode, make_rhs
from heatpump.requirements import Constraints, DesignRequest, TimeSeries, cooling_tons_to_w
from heatpump.simulate import SimResult, simulate
from heatpump.solver import TRBDF2
from heatpump.thermo import (
    COMMON_REFRIGERANTS,
    PropertyTables,
    build_tables,
    eval_ph,
    fluid_info,
    list_refrigerants,
    resolve_fluid,
)

__version__ = "0.1.0"
__all__ = [
    "COMMON_REFRIGERANTS",
    "BangBang",
    "CapacityMap",
    "Cascade",
    "CoilSpec",
    "Constraints",
    "ControlOutput",
    "DesignGateError",
    "DesignPackage",
    "DesignReport",
    "DesignRequest",
    "GateSet",
    "HysteresisThermostat",
    "LinearMPC",
    "NonlinearMPC",
    "PID",
    "PlantSpec",
    "PropertyTables",
    "SimResult",
    "SuperheatEEV",
    "SystemDesign",
    "TRBDF2",
    "TimeSeries",
    "apply_operating_mode",
    "build_tables",
    "capacity_map",
    "cooling_tons_to_w",
    "design_air_conditioner",
    "design_heat_pump",
    "design_system",
    "eval_ph",
    "fluid_info",
    "heating_spec",
    "list_refrigerants",
    "make_rhs",
    "resolve_fluid",
    "simulate",
]
