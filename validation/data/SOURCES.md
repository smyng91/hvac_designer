# Validation sources

Nothing in this folder is synthetic. Files were downloaded from the
publishers cited below. The designer is **not** fitted to either set.
How the unfitted plant is scored: [validation/README.md](../README.md).
Physics: [docs/model.md](../../docs/model.md).

## Ramírez-León et al. 2019 (primary — refrigerant known)

- H. Ramírez-León, J. Jiménez-Cabas, A. Bula, “Experimental data for an
  air-conditioning system identification,” *Data in Brief* 25 (2019) 104316.
- DOI: [10.1016/j.dib.2019.104316](https://doi.org/10.1016/j.dib.2019.104316)
- License: CC BY 4.0
- Unit: 3.5 kW inverter mini-split, R410A, SEER 17, Universidad del Norte
- File: `ramirez2019_mmc1.xlsx` (publisher supplementary)
- URL: https://ars.els-cdn.com/content/image/1-s2.0-S2352340919306705-mmc1.xlsx
- SHA-256: `9f352c6f51ceaf00c68ee419596e0d49c80052768df40d23203d69d47fbbbf66`
- Run metadata (fan, T_in, RH) is Table 3 of that paper.

## Ramaraj & Sparn 2024 (system Q/W — refrigerant not named)

- S. Ramaraj and B. Sparn, “BENEFIT with Northeastern University: HVAC
  Hardware-in-the-Loop Experimental Testing of Heat Pump and Air Conditioner,”
  NLR Data Catalog (2024).
- DOI: [10.7799/2440214](https://doi.org/10.7799/2440214)
- Unit: 3-ton single-speed ASHP, SEER 16, HSPF 9.5, 15 kW aux, NREL SPL
- Raw 1 Hz CSVs are large and gitignored. Re-download:

```bash
mkdir -p validation/data/nrel_hil
cd validation/data/nrel_hil
base=https://data.nlr.gov/system/files/246/1725922071
for f in Test_Matrix.xlsx \
  HP_Cool_OAT95F_SP76F72F68F.csv HP_Cool_OAT75F_SP72F68F.csv \
  HP_Heat_OAT45F_SP68F72F.csv HP_Heat_OAT5F_SP72F.csv
 do curl -fL -o "$f" "$base-$f"; done
```

The NREL readme does **not** name the refrigerant. Model comparisons use
R410A as an assumption (US residential 3-ton SEER-16 class in 2024) and
are labelled as such.

## Lee et al. 2021 (compressor map — not a cabinet twin)

- C.-Y. Lee, T. Cao, Y. Hwang, R. Radermacher, S. Shaffer, *IOP Conf.
  Ser.: Mater. Sci. Eng.* **1180** (2021) 012041, CC BY 3.0.
- Coefficients: `maps/lee2021_iop1180_012041.json` (Table 5 New
  Generated Map; same file as `data/maps/`). Validation evaluates the
  AHRI 540 polynomial at the published Table 4 (Te, Tc) setpoints.
- Table 6 VapCyc system capacities are **not** used: Te/Tc for those
  two points are not tabulated, so a system residual would require
  invented approaches.
- Ramírez 2019 has no published compressor map or coil circuiting for
  that mini-split; it remains a nameplate-class check.

## What was not used

- Yousaf / Bradshaw 2026 IJR 14 kW R410A ASHP tables — supplementary Excel
  is cited in the paper but was not retrieved as an open file here.
- Water-source / vapor-injection / high-temperature heat-pump sets (wrong
  cycle architecture for this plant).
