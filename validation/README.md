# Validation

Downloaded laboratory files, scripts that compare the **unfitted**
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
| `lee2021_map.py` | Table 5 at published Table 4 $`(T_e,T_c)`$ |
| `audit.py` | Closed-loop heating physics check (not a lab twin) |

| Set | What is scored | What is not |
|---|---|---|
| Ramírez 2019 | Nameplate-class R410A $`Q,W,\mathrm{COP},p`$ at 16 runs | Fitted geometry / charge |
| NREL HIL 2024 | On-period $`Q,W`$ vs 3-ton R410A (refrigerant assumed) | Cycling, aux heat, named fluid |
| Lee 2021 | AHRI 540 Table 5 at Table 4 setpoints | Table 6 (no $`T_e,T_c`$) |

Physics: [docs/model.md](../docs/model.md).
