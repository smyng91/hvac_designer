# Examples

Four operating cases. Each sizes a plant and runs it. Weather CSVs are
scenarios, not laboratory traces.

```bash
source .venv/bin/activate
python examples/heating.py
python examples/run_all.py
```

Default horizon is 1 h (QSS after a short DAE warmup). `--t-final 90`
keeps the full finite-volume residual. Outputs go to `examples/out/`.

| Case | Script | Plant |
|---|---|---|
| Winter heating | `heating.py` | R32, 5.5 kW, 0 °C / 20 °C |
| Summer cooling | `cooling.py` | R410A, 6.2 kW, 35 °C / 24 °C |
| Reverse | `reverse.py` | One HP: cool, then heat |
| Weather | `weather.py` | Duty from a CSV; bins = this record’s dwell |

Literature comparisons are in [`validation/`](../validation/README.md).
