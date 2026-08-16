# Examples

Operating cases plus a design-only sizing script. Weather CSVs are
scenarios, not laboratory traces.

```bash
source .venv/bin/activate
python examples/design.py
python examples/heating.py
python examples/run_all.py
```

Default horizon $`t=3600\,\mathrm{s}`$ (QSS after a short DAE warmup).
`--t-final 90` keeps the full residual. Outputs: `output/`.

Zone thermal mass is dry air, $`C_z=\rho c_p V`$ with $`V=50\,\mathrm{m}^3`$
($`\approx 60.6\,\mathrm{kJ/K}`$ at $`20^\circ\mathrm{C}`$,
$`\approx 59.8\,\mathrm{kJ/K}`$ at $`24^\circ\mathrm{C}`$).

| Case | Script | Plant |
|---|---|---|
| Size | `design.py` | Conditions → package in `output/` (no transient) |
| Winter heating | `heating.py` | R32, $`5.5\,\mathrm{kW}`$, $`0^\circ\mathrm{C}`$ / $`20^\circ\mathrm{C}`$ |
| Summer cooling | `cooling.py` | R410A, $`6.2\,\mathrm{kW}`$, $`35^\circ\mathrm{C}`$ / $`24^\circ\mathrm{C}`$ |
| Reverse | `reverse.py` | One HP: cool, then heat (`mode` column) |
| Weather | `weather.py` | Duty from a CSV; bins = this record’s dwell |

`design.py` defaults to the heating conditions. Override with
`--mode`, `--load`, `--T-out`, `--T-zone`, `--weather`, `--V-zone`.

Tutorial: [docs/quickstart.md](../docs/quickstart.md).
Literature: [validation/](../validation/README.md).
