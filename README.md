# Transient vapor-compression heat pump

JAX plant for a **subcritical air-source heat pump or air conditioner**.
CoolProp properties, finite-volume two-phase coils, and a stiff TR-BDF2
integrator (QSS for hour-to-day runs). The same residual is the plant
inside MPC.

Equations render on the
**[wiki](https://github.com/smyng91/hvac_designer/wiki)**.

| | Source | Wiki |
|---|---|---|
| Quick start | [docs/quickstart.md](docs/quickstart.md) | [Quick start](https://github.com/smyng91/hvac_designer/wiki/Quick-start) |
| Model | [docs/model.md](docs/model.md) | [Model](https://github.com/smyng91/hvac_designer/wiki/Model) |
| Examples | [examples/](examples/README.md) | [Examples](https://github.com/smyng91/hvac_designer/wiki/Examples) |
| Validation | [validation/](validation/README.md) | [Validation](https://github.com/smyng91/hvac_designer/wiki/Validation) |
| Cited maps | [data/](data/README.md) | [Data](https://github.com/smyng91/hvac_designer/wiki/Data) |

## Install

Requires **Python 3.14**.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
python examples/design.py           # size R32, 5.5 kW, 0 °C / 20 °C
python examples/heating.py          # same plant, 1 h closed-loop
python examples/run_all.py
pytest
```

Outputs: `output/`. `--t-final 90` keeps the full DAE.
Tutorial: [Quick start](docs/quickstart.md).

## Scope

Sized from a zone setpoint plus a weather/load record (nameplate kW or
tons optional). Design closes CoolProp balances, ε-NTU coils, a
clearance compressor, and an orifice EEV — not an AHRI/EN rating.

Off by default: humidity and frost (user RH required). AHRI 540 and fan
tables only when a cited file is supplied. Reverse remaps coils mid-run.
Not modeled: ducts, multi-zone, transcritical CO2.

## License

MIT. See [LICENSE](LICENSE).
