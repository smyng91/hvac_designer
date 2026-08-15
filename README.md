# Transient vapor-compression heat pump

JAX model of a **subcritical air-source heat pump or air conditioner**:
CoolProp refrigerant properties, finite-volume two-phase coils, scroll
compressor, isenthalpic EEV, and a stiff TR-BDF2 integrator. Hardware is
sized from a zone setpoint plus a load/ambient timeseries (nameplate kW
or tons are optional overrides). The same residual used by the integrator
is the plant inside MPC.

The physics, DAE, sizing, and controllers are described in
[docs/model.md](docs/model.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# size from setpoint + weather/load CSV (no kW / tons required)
PYTHONPATH=src python -m heatpump.simulate -r R410A --T-zone 24 \
  --weather examples/weather_cooling.csv

# optional nameplate override
PYTHONPATH=src python -m heatpump.simulate --mode heating -r R32 \
  --load 5500 --T-out 0 --T-zone 20

# design package only (gates, Q(T_out), balance point, psychrometrics)
PYTHONPATH=src python -m heatpump.simulate --design-only --mode heating \
  -r R32 --load 5500 --T-out 0 --T-zone 20 --report examples/out/design.md

PYTHONPATH=src pytest
```

CSV header required: time, outdoor temperature, load. Optional setpoint.

```
t,T_out_C,Q_kW,Tsp_C
0,28,3.5,24
600,35,6.2,24
```

`Q` is heat **into** the zone unless `--load-kind cooling_load` or
`heating_load`. With a profile and no nameplate, envelope `UA` is not
added on top of the CSV.

```python
from heatpump import DesignRequest, TimeSeries, design_system, simulate

req = DesignRequest(
    refrigerant="R410A",
    mode="auto",
    T_zone=297.15,
    timeseries=TimeSeries.from_csv("examples/weather_cooling.csv"),
)
sys = design_system(req)
pkg = sys.as_report()
pkg.write("examples/out/design.md")
res = simulate(sys.controller, spec=sys.spec, request=req, t_final=1800.0)
```

The design package closes the vapor-compression balances with CoolProp,
ε-NTU coils, a clearance compressor, and an orifice EEV. It is not an
AHRI/EN rating.

Not modeled in the transient plant: humidity DAE, frost growth, ducts,
multi-zone, transcritical CO2, or reversing mid-run. Cooling latent
load in the design package is a humid-air balance at the evaporating
temperature, not a humidity state in the DAE.
