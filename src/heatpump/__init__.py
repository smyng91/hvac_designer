"""Transient two-phase vapor-compression heat pump / air conditioner (JAX)."""

from heatpump.capacity import CapacityMap, CapacityPoint, capacity_at, capacity_map
from heatpump.catalog import Catalog, load_catalog
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
from heatpump.design import (
    DesignReport,
    SystemDesign,
    design_air_conditioner,
    design_heat_pump,
    design_system,
    heating_spec,
)
from heatpump.devices import (
    AHRI540Compressor,
    ClearanceCompressor,
    LinearFan,
    LumpedZone,
    OrificeEEV,
    SeriesUAAir,
    ShahDittusHTC,
    TableFan,
)
from heatpump.gates import DesignGateError, GateSet
from heatpump.plant import (
    CoilSpec,
    PlantSpec,
    apply_operating_mode,
    diagnostics,
    make_rhs,
    remap_state,
)
from heatpump.report import DesignPackage
from heatpump.requirements import Constraints, DesignRequest, TimeSeries, cooling_tons_to_w
from heatpump.seasonal import SeasonalBins, bin_timeseries
from heatpump.simulate import SimResult, simulate
from heatpump.solver import TRBDF2, integrate, integrate_qss
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
    "AHRI540Compressor",
    "BangBang",
    "COMMON_REFRIGERANTS",
    "CapacityMap",
    "CapacityPoint",
    "Cascade",
    "Catalog",
    "ClearanceCompressor",
    "CoilSpec",
    "Constraints",
    "ControlOutput",
    "DesignGateError",
    "DesignPackage",
    "DesignReport",
    "DesignRequest",
    "GateSet",
    "HysteresisThermostat",
    "LinearFan",
    "LinearMPC",
    "LumpedZone",
    "NonlinearMPC",
    "OrificeEEV",
    "PID",
    "PlantSpec",
    "PropertyTables",
    "SeasonalBins",
    "SeriesUAAir",
    "ShahDittusHTC",
    "SimResult",
    "SuperheatEEV",
    "SystemDesign",
    "TRBDF2",
    "TableFan",
    "TimeSeries",
    "apply_operating_mode",
    "bin_timeseries",
    "build_tables",
    "capacity_at",
    "capacity_map",
    "cooling_tons_to_w",
    "design_air_conditioner",
    "design_heat_pump",
    "design_system",
    "diagnostics",
    "eval_ph",
    "fluid_info",
    "heating_spec",
    "integrate",
    "integrate_qss",
    "list_refrigerants",
    "load_catalog",
    "make_rhs",
    "remap_state",
    "resolve_fluid",
    "simulate",
]
