# Validation

Downloaded laboratory files, the scripts that compare the **unfitted**
designer to them, and the numbers those scripts write. Not example
scenarios.

```bash
source .venv/bin/activate
python validation/run.py
python validation/lee2021_map.py
python validation/audit.py
```

| Path | Contents |
|---|---|
| `data/` | Ramírez workbook, optional NREL HIL traces, Lee 2021 map |
| `data/SOURCES.md` | Citations, licenses, SHA-256, download commands |
| `results/` | MAPE tables, on-period means, map evaluation |
| `run.py` | Ramírez + NREL + Lee → `results/` |
| `lee2021_map.py` | Table 5 at published Table 4 (Te, Tc) |
| `audit.py` | Closed-loop heating physics check (not a lab twin) |

Ramírez is a nameplate-class R410A check (refrigerant known). NREL does
not name the refrigerant; the model assumes R410A and says so. Lee
Table 6 system capacities are not scored (Te/Tc are not tabulated).
Physics is in [docs/model.md](../docs/model.md).
