# Compressor maps

Nothing in this folder is a fitted or invented coefficient set.

## Lee et al. 2021 (AHRI 540, CC BY 3.0)

- C.-Y. Lee, T. Cao, Y. Hwang, R. Radermacher, S. Shaffer, “Development
  of accurate and widely applicable compressor performance maps,”
  *IOP Conf. Ser.: Mater. Sci. Eng.* **1180** (2021) 012041.
- DOI: [10.1088/1757-899X/1180/1/012041](https://doi.org/10.1088/1757-899X/1180/1/012041)
- License: CC BY 3.0
- Files:
  - `lee2021_iop1180_012041.json` — Table 5 *New Generated Map*
  - `lee2021_iop1180_012041_manufacturer.json` — Table 5 *Manufacturer's Map*
- Temperatures in the polynomial are suction / discharge dew point \(T_s,T_d\) in °C.
- Power is watts; mass flow in the paper is g/s (converted to kg/s on load).

\[
X=C_1+C_2 T_s+C_3 T_d+C_4 T_s^2+C_5 T_s T_d+C_6 T_d^2
+C_7 T_s^3+C_8 T_s^2 T_d+C_9 T_s T_d^2+C_{10} T_d^3.
\]
- Table 6 is a VapCyc *system* close. Te and Tc for those two points are
  not tabulated, so that table is not used as a residual check.
- The paper does not name the refrigerant. The map is a (Te, Tc) → (ṁ, W)
  polynomial and is fluid-independent at evaluation; hermetic
  `h_d = h_s + W/ṁ` uses whatever fluid the plant is running.
