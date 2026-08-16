# Quick start

Python 3.14. After [install](../README.md#install):

```bash
source .venv/bin/activate
python examples/design.py           # size only → output/design.md
python examples/heating.py
pytest
```

`--t-final 90` is the full finite-volume DAE. Default 1 h uses QSS after
a short warmup. Plots and design files go to `output/`.

## Cases

| Script | What it does |
|---|---|
| `examples/design.py` | Size from conditions (no transient) → `output/design.md` |
| `examples/heating.py` | R32 heat pump, 5.5 kW, \(0^\circ\mathrm{C}\) / \(20^\circ\mathrm{C}\) |
| `examples/cooling.py` | R410A AC, 6.2 kW, \(35^\circ\mathrm{C}\) / \(24^\circ\mathrm{C}\) |
| `examples/reverse.py` | One reversible unit: cool, then heat |
| `examples/weather.py` | Size and run from a CSV; bins = this record’s dwell |
| `examples/run_all.py` | Design, heating, cooling, reverse, weather cooling |

Weather CSVs are scenarios, not lab traces. All example and CLI results
go to `output/` (gitignored). Literature:
[`validation/`](../validation/README.md).

## Size a plant

```bash
python examples/design.py
python examples/design.py --mode cooling --load 6200 --T-out 35 --T-zone 24
python examples/design.py --mode heat_pump --load-heat 5500 --load-cool 6200
python examples/design.py --weather examples/weather_heating.csv
```

Writes `output/design.md`, `output/design.json`, and `output/design_map.png`.
No transient. Optional: `--SH`, `--SC`, `--DT-evap`, `--DT-cond`, `--V-zone`,
`--load-tons`. Default conditions match `heating.py`.

## Library

```python
from heatpump import DesignRequest, TimeSeries, design_system, simulate

req = DesignRequest(
    refrigerant="R410A",
    mode="auto",
    T_zone=297.15,
    timeseries=TimeSeries.from_csv("examples/weather_cooling.csv"),
)
sys = design_system(req)
sys.as_report().write("output/design.md")
res = simulate(sys.controller, spec=sys.spec, request=req, t_final=3600.0)
```

Empty `PlantSpec` slots use the built-in kernels. Retrofit by assignment:

```python
from dataclasses import replace
from heatpump import AHRI540Compressor

spec = replace(
    sys.spec,
    compressor=AHRI540Compressor.from_file("data/maps/lee2021_iop1180_012041.json"),
)
```

## CLI

```bash
python -m heatpump.simulate -r R410A --T-zone 24 \
  --weather examples/weather_cooling.csv

python -m heatpump.simulate --design-only --mode heating -r R32 \
  --load 5500 --T-out 0 --T-zone 20 --report output/design.md

python -m heatpump.simulate --mode heating -r R32 --load 5500 \
  --T-out 0 --T-zone 20 --t-final 86400 --reduction qss
```

| Flag | Role |
|---|---|
| `--weather` | CSV of \(t\), \(T_\mathrm{out}\), \(Q\); duty = peak of the record |
| `--load` / `--load-tons` | optional nameplate override |
| `--reduction auto\|full\|qss` | `auto` → QSS if \(t\ge 3600\,\mathrm{s}\) |
| `--moist --frost --RH-out --RH-zone` | humidity / frost (RH is not defaulted) |
| `--ahri540` | cited AHRI 540 JSON (no silent map) |
| `--seasonal` | bins from this record’s dwell, not AHRI hours |

With neither `--weather` nor `--load`, the CLI sizes a 5.5 kW heating demo.

## Weather CSV

Required: time, outdoor temperature, load. Optional: `Tsp_C`, `mode`
(\(1\)=heating, \(0\)=cooling), `RH_out`, `W_gain`, `defrost`.

```
t,T_out_C,Q_kW,Tsp_C
0,28,3.5,24
600,35,6.2,24
```

\(Q\) is heat **into** the zone unless `--load-kind cooling_load` or
`heating_load`. When the CSV is the complete load, envelope \(UA=0\).

## Validation

```bash
python validation/run.py
python validation/lee2021_map.py
```

Unfitted designer vs downloaded lab files. Citations:
[`validation/data/SOURCES.md`](../validation/data/SOURCES.md).

Physics and every symbol: [model.md](model.md).
