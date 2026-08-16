# Transient vapor-compression heat pump

JAX plant for a **subcritical air-source heat pump or air conditioner**:
CoolProp properties, finite-volume two-phase coils, clearance or AHRI 540
compressor, isenthalpic EEV, and a stiff TR-BDF2 integrator (QSS for
hour-to-day runs). Hardware is sized from a zone setpoint plus a
weather/load record. The same residual is the plant inside MPC.

| | |
|---|---|
| Physics | [docs/model.md](docs/model.md) |
| Cases | [examples/](examples/README.md) |
| Literature | [validation/](validation/README.md) |
| Cited maps | [data/](data/README.md) |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
python examples/heating.py          # R32, 5.5 kW, 0 °C / 20 °C, 1 h
python examples/run_all.py          # heating, cooling, reverse, weather
pytest
```

Outputs go to `examples/out/`. `--t-final 90` keeps the full DAE.

## Library

```python
from dataclasses import replace
from heatpump import (
    AHRI540Compressor, DesignRequest, TimeSeries, design_system, simulate,
)

req = DesignRequest(
    refrigerant="R410A",
    mode="auto",
    T_zone=297.15,
    timeseries=TimeSeries.from_csv("examples/weather_cooling.csv"),
)
sys = design_system(req)
sys.as_report().write("examples/out/design.md")
res = simulate(sys.controller, spec=sys.spec, request=req, t_final=3600.0)

spec = replace(
    sys.spec,
    compressor=AHRI540Compressor.from_file("data/maps/lee2021_iop1180_012041.json"),
)
```

Empty `PlantSpec` slots call the built-in kernels. Assign an object to
retrofit; there is no name registry.

## CLI

```bash
python -m heatpump.simulate -r R410A --T-zone 24 \
  --weather examples/weather_cooling.csv

python -m heatpump.simulate --design-only --mode heating -r R32 \
  --load 5500 --T-out 0 --T-zone 20 --report examples/out/design.md

python -m heatpump.simulate --mode heating -r R32 --load 5500 \
  --T-out 0 --T-zone 20 --t-final 86400 --reduction qss

python -m heatpump.simulate --mode heating -r R32 --load 5500 \
  --T-out 0 --T-zone 20 --moist --frost --RH-out 0.8 --RH-zone 0.4 --t-final 90

python -m heatpump.simulate --mode heating -r R32 --load 5500 \
  --T-out 0 --T-zone 20 --ahri540 data/maps/lee2021_iop1180_012041.json

python -m heatpump.simulate --design-only --seasonal \
  --weather examples/weather_cooling.csv --out examples/out

python validation/run.py
```

`--load` / `--load-tons` are optional nameplate overrides. With only
`--weather`, duty is the peak of that record. With neither, the CLI
sizes a 5.5 kW demo plant.

## Weather CSV

Required header: time, outdoor temperature, load. Optional: `Tsp_C`,
`mode` (1=heating, 0=cooling), `RH_out`, `W_gain`, `defrost`.

```
t,T_out_C,Q_kW,Tsp_C
0,28,3.5,24
600,35,6.2,24
```

`Q` is heat **into** the zone unless `--load-kind cooling_load` or
`heating_load`. When the CSV is the complete load, envelope `UA` is not
added on top of it. Seasonal bins use this record’s dwell time, not
copied AHRI 210/240 hours.

## Scope

The design package closes CoolProp balances, ε-NTU coils, a clearance
compressor, and an orifice EEV. It is not an AHRI/EN rating.

Off by default: zone humidity and outdoor-coil frost. Both need user RH
(no default humidity, no frost derate table, no automatic defrost).
AHRI 540 and fan tables are used only when a cited file is supplied.
The example catalog lists only the published Lee 2021 map.

Reverse (heating ↔ cooling) remaps the coils mid-run on a reversible
unit. Not modeled: ducts, multi-zone, transcritical CO2.

## License

MIT. See [LICENSE](LICENSE).
