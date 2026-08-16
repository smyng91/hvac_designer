"""Compare the plant to published laboratory measurements.

Sources (downloaded, not invented):

* Ramírez-León, Jiménez-Cabas, Bula, *Data in Brief* 25 (2019) 104316,
  doi:10.1016/j.dib.2019.104316, CC-BY. 3.5 kW R410A mini-split. Raw
  supplementary workbook ``mmc1.xlsx``.
* Ramaraj & Sparn, NLR Data Catalog (2024), doi:10.7799/2440214.
  3-ton single-speed ASHP, SEER 16 / HSPF 9.5, 1 s HIL traces.

The designer is **not** fitted to either set. Hardware is sized from the
published nameplate and the rating-condition air temperatures stated in
ISO 5151 / AHRI 210/240. Errors are reported as-is.

Compressor-map check (not a cabinet twin): Lee et al., IOP Conf. Ser.:
Mater. Sci. Eng. 1180 (2021) 012041, Table 5 coefficients, evaluated at
the published Table 4 (Te, Tc) setpoints. Table 6 is not used (Te/Tc
for those VapCyc points are not tabulated).
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np

from heatpump.capacity import capacity_at
from heatpump.design import design_air_conditioner, design_heat_pump

PSIA_PA = 6894.75729
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

# Table 3 of Ramírez-León et al., Data in Brief 25 (2019) 104316.
RAMIREZ_RUNS: dict[int, dict] = {
    1: {"fan": 1, "T_in_C": 28.0, "RH": 46.0},
    2: {"fan": 3, "T_in_C": 28.0, "RH": 55.0},
    3: {"fan": 1, "T_in_C": 23.0, "RH": 62.0},
    4: {"fan": 3, "T_in_C": 23.0, "RH": 75.0},
    5: {"fan": 3, "T_in_C": 28.0, "RH": 46.0},
    6: {"fan": 3, "T_in_C": 23.0, "RH": 62.0},
    7: {"fan": 1, "T_in_C": 23.0, "RH": 75.0},
    8: {"fan": 1, "T_in_C": 28.0, "RH": 55.0},
    9: {"fan": 1, "T_in_C": 28.0, "RH": 46.0},
    10: {"fan": 3, "T_in_C": 28.0, "RH": 55.0},
    11: {"fan": 1, "T_in_C": 23.0, "RH": 62.0},
    12: {"fan": 3, "T_in_C": 23.0, "RH": 75.0},
    13: {"fan": 3, "T_in_C": 28.0, "RH": 46.0},
    14: {"fan": 3, "T_in_C": 23.0, "RH": 62.0},
    15: {"fan": 1, "T_in_C": 23.0, "RH": 75.0},
    16: {"fan": 1, "T_in_C": 28.0, "RH": 55.0},
}

RAMIREZ_SHA256 = "9f352c6f51ceaf00c68ee419596e0d49c80052768df40d23203d69d47fbbbf66"
RAMIREZ_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S2352340919306705-mmc1.xlsx"
NREL_DOI = "10.7799/2440214"
NREL_BASE = "https://data.nlr.gov/system/files/246/1725922071"

# Lee et al. 2021 Table 4 cooling setpoints (evaporating / condensing °C).
# Superheat is listed as 2 K or 11 K; the AHRI 540 poly uses dew points.
LEE2021_TABLE4_COOLING = (
    (13.3, 58.1, 46.0),
    (13.1, 55.5, 43.0),
    (12.8, 52.9, 40.0),
    (12.6, 50.2, 37.0),
    (12.3, 48.4, 35.0),
    (12.2, 46.6, 33.0),
    (12.0, 44.8, 30.6),
    (11.8, 42.0, 27.8),
    (11.6, 40.2, 26.0),
)
LEE2021_TABLE5_NEW_POWER = (
    5.2020e2, -1.1830e1, -3.5960e0, 2.1350e-2, 2.0930e-1,
    2.7810e-2, 7.1760e-5, -7.4150e-4, -6.7700e-4, 1.2330e-4,
)
LEE2021_TABLE5_NEW_MDOT_G_S = (
    1.4850e2, 3.5180e0, -5.6360e0, 1.6610e-1, -1.2530e-1,
    8.6030e-2, 6.4650e-4, -2.1050e-3, 1.1140e-3, -3.8550e-4,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return path.name


def default_data_dir() -> Path:
    return _REPO_ROOT / "validation" / "data"


def default_results_dir() -> Path:
    return _REPO_ROOT / "validation" / "results"


def default_maps_dir() -> Path:
    pub = _REPO_ROOT / "validation" / "data" / "maps"
    if (pub / "lee2021_iop1180_012041.json").exists():
        return pub
    return _REPO_ROOT / "data" / "maps"


def ahri540_poly_np(Ts: float, Td: float, C) -> float:
    """Independent AHRI 540 cubic (numpy), used to check the JAX device."""
    C = np.asarray(C, dtype=float)
    S, D = float(Ts), float(Td)
    return float(
        C[0]
        + C[1] * S
        + C[2] * D
        + C[3] * S * S
        + C[4] * S * D
        + C[5] * D * D
        + C[6] * S * S * S
        + C[7] * S * S * D
        + C[8] * S * D * D
        + C[9] * D * D * D
    )


def compare_lee2021_map(path: Path | None = None) -> dict:
    """Evaluate the published Table 5 map at Table 4 (Te, Tc). No system close."""
    path = path or default_maps_dir() / "lee2021_iop1180_012041.json"
    raw = json.loads(path.read_text())
    power_C = tuple(float(c) for c in raw["power"])
    mdot_C = tuple(float(c) for c in raw["mdot"])
    if raw.get("mdot_unit") in ("g/s", "g_s"):
        mdot_si = tuple(c * 1.0e-3 for c in mdot_C)
    else:
        mdot_si = mdot_C
    coeff_ok = np.allclose(power_C, LEE2021_TABLE5_NEW_POWER) and np.allclose(
        mdot_C, LEE2021_TABLE5_NEW_MDOT_G_S
    )
    rows = []
    for Te, Tc, Tout in LEE2021_TABLE4_COOLING:
        W = ahri540_poly_np(Te, Tc, power_C)
        m_g = ahri540_poly_np(Te, Tc, mdot_C)
        rows.append(
            {
                "Te_C": Te,
                "Tc_C": Tc,
                "T_out_C": Tout,
                "source": "Lee et al. 2021 Table 4",
                "power_W": W,
                "mdot_g_s": m_g,
                "mdot_kg_s": m_g * 1.0e-3 if raw.get("mdot_unit") in ("g/s", "g_s") else m_g,
                "positive": bool(W > 0.0 and m_g > 0.0),
            }
        )
    t5 = next(r for r in rows if r["Te_C"] == 12.3 and r["Tc_C"] == 48.4)
    return {
        "source": {
            "citation": raw["citation"],
            "doi": raw.get("doi"),
            "file": _repo_rel(path),
            "license": raw.get("license"),
        },
        "coefficients_match_table5": bool(coeff_ok),
        "table4": rows,
        "table4_test5": t5,
        "all_positive": all(r["positive"] for r in rows),
        "notes": [
            "Polynomial evaluated at published Table 4 dew-point setpoints.",
            "Table 6 VapCyc system capacities are not scored: Te/Tc for those points are not tabulated.",
            "No refrigerant is named in the paper; the map is (Te, Tc) → (ṁ, W).",
            "This is a compressor-map check, not a geometry-matched cabinet twin.",
        ],
        "mdot_si_C": list(mdot_si),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _xlsx_shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall("m:si", NS):
        out.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    return out


def _xlsx_cell(c, shared: list[str]):
    t = c.attrib.get("t")
    v = c.find("m:v", NS)
    if v is None or v.text is None:
        return None
    if t == "s":
        return shared[int(v.text)]
    if t == "b":
        return bool(int(v.text))
    try:
        return float(v.text)
    except ValueError:
        return v.text


def _col_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_xlsx_sheets(path: Path) -> dict[str, list[dict]]:
    """Read an xlsx into ``{sheet: [row dict, ...]}`` (stdlib only)."""
    z = zipfile.ZipFile(path)
    shared = _xlsx_shared_strings(z)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
    sheets = {}
    for s in wb.findall("m:sheets/m:sheet", NS):
        name = s.attrib["name"]
        target = rid[s.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        rows_xml = root.findall("m:sheetData/m:row", NS)
        if not rows_xml:
            sheets[name] = []
            continue
        headers: dict[int, str] = {}
        for c in rows_xml[0].findall("m:c", NS):
            ref = c.attrib.get("r", "")
            headers[_col_index(ref)] = str(_xlsx_cell(c, shared))
        recs = []
        for row in rows_xml[1:]:
            rec = {}
            for c in row.findall("m:c", NS):
                ref = c.attrib.get("r", "")
                key = headers.get(_col_index(ref))
                if key is None:
                    continue
                rec[key] = _xlsx_cell(c, shared)
            if rec:
                recs.append(rec)
        sheets[name] = recs
    return sheets


def _col(row: dict, *names: str) -> float:
    for n in names:
        if n in row and row[n] is not None:
            return float(row[n])
    raise KeyError(names)


def _flash_pt(fluid: str, p_pa: float, T_k: float) -> dict[str, float]:
    AS = CP.AbstractState("HEOS", fluid)
    AS.update(CP.PT_INPUTS, p_pa, T_k)
    dew = CP.AbstractState("HEOS", fluid)
    dew.update(CP.PQ_INPUTS, p_pa, 1.0)
    bub = CP.AbstractState("HEOS", fluid)
    bub.update(CP.PQ_INPUTS, p_pa, 0.0)
    return {
        "T": AS.T(),
        "p": AS.p(),
        "h": AS.hmass(),
        "rho": AS.rhomass(),
        "Tsat_dew": dew.T(),
        "Tsat_bub": bub.T(),
    }


@dataclass
class RamirezRun:
    run: int
    fan: int
    T_in_set_C: float
    RH_set: float
    n: int
    P1_psia: float
    P2_psia: float
    P3_psia: float
    P4_psia: float
    W_comp_kW: float
    W_fan_c_kW: float
    W_fan_e_kW: float
    F1_mL_min: float
    T1_C: float
    T3_C: float
    T4_C: float
    T7_C: float
    T9_C: float
    T10_C: float
    p_e_Pa: float
    p_c_Pa: float
    SH_K: float
    SC_K: float
    Tsat_e_C: float
    Tsat_c_C: float
    mdot_kg_s: float
    Q_evap_W: float
    Q_cond_W: float
    W_enthalpy_W: float
    COP: float


def parse_ramirez(path: Path | None = None) -> tuple[list[RamirezRun], dict]:
    path = path or default_data_dir() / "ramirez2019_mmc1.xlsx"
    digest = file_sha256(path)
    meta = {
        "citation": "Ramírez-León, Jiménez-Cabas, Bula, Data in Brief 25 (2019) 104316",
        "doi": "10.1016/j.dib.2019.104316",
        "url": RAMIREZ_URL,
        "file": _repo_rel(path),
        "sha256": digest,
        "sha256_expected": RAMIREZ_SHA256,
        "fluid": "R410A",
        "nameplate_Q_W": 3500.0,
        "SEER": 17,
    }
    sheets = read_xlsx_sheets(path)
    runs: list[RamirezRun] = []
    for i in range(1, 17):
        rows = sheets[f"Run{i}"]
        info = RAMIREZ_RUNS[i]

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            if not vals:
                raise KeyError(key)
            return float(np.mean(vals))

        P1 = mean("P1 (PSIa)")
        P2 = mean("P2 (PSIa)")
        P3 = mean("P3 (PSIa)")
        P4 = mean("P4 (PSIa)")
        T1 = mean("T1 (°C)")
        T3 = mean("T3 (°C)")
        T4 = mean("T4 (°C)")
        T7 = mean("T7 (°C)")
        T9 = mean("T9 (°C)")
        T10 = mean("T10 (°C)")
        F1 = mean("F1 (mL/min)")
        W1 = mean("Power 1 (kW)")
        p_e = P1 * PSIA_PA
        p_c = P2 * PSIA_PA
        suct = _flash_pt("R410A", p_e, T1 + 273.15)
        disch = _flash_pt("R410A", p_c, T3 + 273.15)
        liq = _flash_pt("R410A", P3 * PSIA_PA, T7 + 273.15)
        mdot = (F1 * 1.0e-6 / 60.0) * liq["rho"]
        h1, h2, h3 = suct["h"], disch["h"], liq["h"]
        Q_e = mdot * (h1 - h3)
        Q_c = mdot * (h2 - h3)
        W_h = mdot * (h2 - h1)
        runs.append(
            RamirezRun(
                run=i,
                fan=info["fan"],
                T_in_set_C=info["T_in_C"],
                RH_set=info["RH"],
                n=len(rows),
                P1_psia=P1,
                P2_psia=P2,
                P3_psia=P3,
                P4_psia=P4,
                W_comp_kW=W1,
                W_fan_c_kW=mean("Power 2 (kW)"),
                W_fan_e_kW=mean("Power 3 (kW)"),
                F1_mL_min=F1,
                T1_C=T1,
                T3_C=T3,
                T4_C=T4,
                T7_C=T7,
                T9_C=T9,
                T10_C=T10,
                p_e_Pa=p_e,
                p_c_Pa=p_c,
                SH_K=suct["T"] - suct["Tsat_dew"],
                SC_K=liq["Tsat_bub"] - liq["T"],
                Tsat_e_C=suct["Tsat_dew"] - 273.15,
                Tsat_c_C=disch["Tsat_dew"] - 273.15,
                mdot_kg_s=mdot,
                Q_evap_W=Q_e,
                Q_cond_W=Q_c,
                W_enthalpy_W=W_h,
                COP=Q_e / max(W1 * 1000.0, 1.0),
            )
        )
    return runs, meta


def compare_ramirez(runs: list[RamirezRun] | None = None, src: dict | None = None) -> dict:
    """Size a 3.5 kW R410A AC at ISO 5151 T1 and compare each lab run."""
    if runs is None or src is None:
        runs, src = parse_ramirez()
    # ISO 5151 T1: indoor 27 °C, outdoor 35 °C; nameplate 3.5 kW.
    design = design_air_conditioner("R410A", 3500.0, T_out=308.15, T_zone=300.15)
    rows = []
    for r in runs:
        pred = capacity_at(
            fluid="R410A",
            kind="cooling",
            T_out=r.T4_C + 273.15,
            T_zone=r.T9_C + 273.15,
            spec=design.spec,
            SH=design.SH,
            SC=design.SC,
            DT_evap=10.0,
            DT_cond=12.0,
        )
        rows.append(
            {
                "run": r.run,
                "fan": r.fan,
                "T_out_C": r.T4_C,
                "T_in_C": r.T9_C,
                "meas_Q_W": r.Q_evap_W,
                "meas_W_W": r.W_comp_kW * 1000.0,
                "meas_COP": r.COP,
                "meas_p_e_bar": r.p_e_Pa / 1e5,
                "meas_p_c_bar": r.p_c_Pa / 1e5,
                "meas_SH_K": r.SH_K,
                "meas_SC_K": r.SC_K,
                "meas_mdot_g_s": r.mdot_kg_s * 1e3,
                "pred_Q_W": pred.Q_cap,
                "pred_W_W": pred.W,
                "pred_COP": pred.COP,
                "pred_p_e_bar": pred.p_e / 1e5 if pred.feasible else float("nan"),
                "pred_p_c_bar": pred.p_c / 1e5 if pred.feasible else float("nan"),
                "pred_T_e_C": pred.T_e - 273.15 if pred.feasible else float("nan"),
                "pred_T_c_C": pred.T_c - 273.15 if pred.feasible else float("nan"),
                "pred_feasible": pred.feasible,
                "err_Q_pct": 100.0 * (pred.Q_cap - r.Q_evap_W) / r.Q_evap_W if r.Q_evap_W else float("nan"),
                "err_W_pct": 100.0 * (pred.W - r.W_comp_kW * 1000.0) / max(r.W_comp_kW * 1000.0, 1.0),
                "err_COP_pct": 100.0 * (pred.COP - r.COP) / r.COP if r.COP else float("nan"),
            }
        )
    Qe = np.array([x["err_Q_pct"] for x in rows if np.isfinite(x["err_Q_pct"])])
    We = np.array([x["err_W_pct"] for x in rows if np.isfinite(x["err_W_pct"])])
    Ce = np.array([x["err_COP_pct"] for x in rows if np.isfinite(x["err_COP_pct"])])
    # Flag runs where I×120 V is far below the refrigerant enthalpy rise
    # (the paper computes power as current × 120 V, no power factor).
    screened = []
    for r, row in zip(runs, rows):
        if r.W_comp_kW * 1000.0 >= 0.45 * max(r.W_enthalpy_W, 1.0):
            screened.append(row)
    def _mape(items, key):
        a = np.array([x[key] for x in items if np.isfinite(x[key])])
        return float(np.mean(np.abs(a))) if a.size else float("nan")

    return {
        "source": src,
        "design": {
            "fluid": design.fluid,
            "Q_nameplate_W": 3500.0,
            "rating": "ISO 5151 T1 (27 °C indoor / 35 °C outdoor)",
            "V_disp_m3": design.V_disp,
            "COP_design": design.COP,
            "p_e_bar": design.p_e / 1e5,
            "p_c_bar": design.p_c / 1e5,
            "fitted": False,
        },
        "runs": rows,
        "measured": [asdict(r) for r in runs],
        "mape": {
            "Q_pct": float(np.mean(np.abs(Qe))) if Qe.size else float("nan"),
            "W_pct": float(np.mean(np.abs(We))) if We.size else float("nan"),
            "COP_pct": float(np.mean(np.abs(Ce))) if Ce.size else float("nan"),
            "n": len(rows),
        },
        "mape_electrical_ok": {
            "Q_pct": _mape(screened, "err_Q_pct"),
            "W_pct": _mape(screened, "err_W_pct"),
            "COP_pct": _mape(screened, "err_COP_pct"),
            "n": len(screened),
            "rule": "keep run if Power1 ≥ 0.45 × mdot (h_disch − h_suct)",
        },
        "notes": [
            "Q_evap is mdot·(h_suct − h_liquid) with CoolProp HEOS R410A on the measured (p,T).",
            "mdot is the liquid volume flow F1 times CoolProp density at (P3, T7).",
            "Power 1 is the paper's I×120 V compressor power, not a shaft map.",
            "The model is a first-principles 3.5 kW sizer, not this cabinet's geometry or inverter map.",
            "Indoor humidity is measured in the lab and is not a DAE state in the plant.",
        ],
    }


def _nrel_on_period(path: Path, kind: str) -> dict | None:
    cap_key = "Cooling Capacity (kW)" if kind == "cooling" else "Heating Capacity (kW)"
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows or cap_key not in rows[0]:
        return None
    P = np.array([float(r["Outdoor Unit Power (W)"]) for r in rows])
    Q = np.array([float(r[cap_key]) for r in rows])
    Tout = np.array([float(r["T_Outdoor (C)"]) for r in rows])
    Tin = np.array([float(r["T_Indoor (C)"]) for r in rows])
    Pin = np.array([float(r["Indoor Unit Power (W)"]) for r in rows])
    Ts = np.array([float(r["T_Supply (C)"]) for r in rows])
    Tr = np.array([float(r["T_Return (C)"]) for r in rows])
    cfm = np.array([float(r["Evaporator Airflow Rate (CFM)"]) for r in rows])
    on = (P > 800.0) & (Pin < 4000.0)
    if not np.any(on):
        return None
    # Published heating capacity in these CSVs is negative (zone-load sign).
    # Magnitude is the physical coil duty. Air-side Q uses T_supply, T_return, CFM.
    Q_pub = Q[on] * 1000.0
    mdot = cfm[on] * 0.00047194745 * 1.2
    Q_air = mdot * 1006.0 * (Ts[on] - Tr[on])
    if kind == "cooling":
        Q_use = np.abs(Q_pub)
        Q_air_use = np.abs(Q_air)
    else:
        Q_use = np.abs(Q_pub)
        Q_air_use = np.maximum(Q_air, 0.0)
    Po = P[on]
    return {
        "file": path.name,
        "kind": kind,
        "n_on": int(np.sum(on)),
        "n_total": int(len(rows)),
        "T_out_C": float(np.mean(Tout[on])),
        "T_in_C": float(np.mean(Tin[on])),
        "Q_W": float(np.mean(Q_use)),
        "Q_air_W": float(np.mean(Q_air_use)),
        "Q_published_signed_W": float(np.mean(Q_pub)),
        "W_out_W": float(np.mean(Po)),
        "W_in_W": float(np.mean(Pin[on])),
        "COP": float(np.mean(Q_use / np.maximum(Po, 1.0))),
        "on_rule": "Outdoor Unit Power > 800 W and indoor power < 4 kW; Q is |published capacity|",
    }


NREL_FILES = {
    "HP_Cool_OAT95F_SP76F72F68F.csv": "cooling",
    "HP_Cool_OAT75F_SP72F68F.csv": "cooling",
    "HP_Heat_OAT45F_SP68F72F.csv": "heating",
    "HP_Heat_OAT5F_SP72F.csv": "heating",
}


def parse_nrel(folder: Path | None = None) -> list[dict]:
    folder = folder or default_data_dir() / "nrel_hil"
    out = []
    for name, kind in NREL_FILES.items():
        path = folder / name
        if not path.exists():
            continue
        rec = _nrel_on_period(path, kind)
        if rec is not None:
            out.append(rec)
    return out


def compare_nrel(points: list[dict] | None = None) -> dict:
    """Nameplate-class comparison: 3-ton R410A vs NREL on-period means.

    The NREL readme does not name the refrigerant. The comparison uses
    R410A because that is the fluid of the Ramírez unit and of US
    residential 3-ton SEER-16 equipment of that generation — it is an
    assumption, labelled as such, not a measured fact from doi:10.7799/2440214.
    """
    if points is None:
        points = parse_nrel()
    Q_ton = 3.0 * 3516.8525
    cool = design_air_conditioner("R410A", Q_ton, T_out=308.15, T_zone=299.817)
    heat = design_heat_pump("R410A", Q_ton, T_out=281.483, T_zone=294.261)
    rows = []
    for p in points:
        spec = cool.spec if p["kind"] == "cooling" else heat.spec
        pred = capacity_at(
            fluid="R410A",
            kind=p["kind"],
            T_out=p["T_out_C"] + 273.15,
            T_zone=p["T_in_C"] + 273.15,
            spec=spec,
        )
        rows.append(
            {
                **p,
                "pred_Q_W": pred.Q_cap,
                "pred_W_W": pred.W,
                "pred_COP": pred.COP,
                "pred_feasible": pred.feasible,
                "err_Q_pct": 100.0 * (pred.Q_cap - p["Q_W"]) / p["Q_W"] if p["Q_W"] else float("nan"),
                "err_W_pct": 100.0 * (pred.W - p["W_out_W"]) / p["W_out_W"] if p["W_out_W"] else float("nan"),
                "err_COP_pct": 100.0 * (pred.COP - p["COP"]) / p["COP"] if p["COP"] else float("nan"),
            }
        )
    return {
        "source": {
            "citation": "Ramaraj & Sparn, NLR Data Catalog (2024)",
            "doi": NREL_DOI,
            "unit": "3-ton single-speed ASHP, SEER 16, HSPF 9.5, 15 kW aux",
            "refrigerant_in_source": None,
            "refrigerant_assumed_for_model": "R410A",
        },
        "points": rows,
        "notes": [
            "On-period means only; cycling / thermostat deadband is not scored.",
            "Published heating-capacity column is negative in the CSVs; |Q| is used for COP.",
            "Outdoor-unit power is compared to compressor shaft power from the clearance map (no fan/aux).",
            "Refrigerant is not stated in the NREL dataset; R410A is an assumption.",
        ],
    }


def write_derived(report: dict, nrel: dict, dest: Path | None = None) -> Path:
    dest = dest or default_results_dir()
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "ramirez2019_run_means.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report["measured"][0].keys()))
        w.writeheader()
        w.writerows(report["measured"])
    if nrel["points"]:
        keys = list(nrel["points"][0].keys())
        with (dest / "nrel_on_period.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(nrel["points"])
    payload = {"ramirez": report, "nrel": nrel}
    (dest / "validation_report.json").write_text(json.dumps(payload, indent=2, default=str))
    return dest


def run_validation(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or default_data_dir()
    runs, src = parse_ramirez(data_dir / "ramirez2019_mmc1.xlsx")
    ramirez = compare_ramirez(runs, src)
    nrel = compare_nrel(parse_nrel(data_dir / "nrel_hil"))
    lee = compare_lee2021_map()
    dest = default_results_dir()
    write_derived(ramirez, nrel, dest)
    (dest / "lee2021_table4.json").write_text(json.dumps(lee, indent=2, default=str))
    return {"ramirez": ramirez, "nrel": nrel, "lee2021": lee}


def _fmt_row(r: dict) -> str:
    return (
        f"  run {r['run']:2d}  T_out={r['T_out_C']:5.1f}°C  T_in={r['T_in_C']:5.1f}°C  "
        f"Q {r['meas_Q_W']/1e3:4.2f}/{r['pred_Q_W']/1e3:4.2f} kW  "
        f"W {r['meas_W_W']/1e3:4.2f}/{r['pred_W_W']/1e3:4.2f} kW  "
        f"COP {r['meas_COP']:4.2f}/{r['pred_COP']:4.2f}  "
        f"ΔQ {r['err_Q_pct']:+6.1f}%"
    )


def main(argv: list[str] | None = None) -> None:
    del argv
    out = run_validation()
    r = out["ramirez"]
    print(r["source"]["citation"])
    print(f"  doi:{r['source']['doi']}  sha256={r['source']['sha256'][:12]}…")
    print(f"  sized at {r['design']['rating']}, V_disp={r['design']['V_disp_m3']*1e6:.1f} cm³, not fitted")
    print(
        f"  MAPE all {r['mape']['n']} runs  Q={r['mape']['Q_pct']:.1f}%  "
        f"W={r['mape']['W_pct']:.1f}%  COP={r['mape']['COP_pct']:.1f}%"
    )
    s = r["mape_electrical_ok"]
    print(
        f"  MAPE {s['n']} runs with plausible I×120 V  Q={s['Q_pct']:.1f}%  "
        f"W={s['W_pct']:.1f}%  COP={s['COP_pct']:.1f}%"
    )
    print("  meas/pred per run:")
    for row in r["runs"]:
        print(_fmt_row(row))
    n = out["nrel"]
    print()
    print(n["source"]["citation"], "doi:" + n["source"]["doi"])
    if not n["points"]:
        print("  NREL CSVs not present; download from", NREL_BASE)
    else:
        print("  on-period means vs 3-ton R410A nameplate-class model (refrigerant assumed):")
        for p in n["points"]:
            print(
                f"  {p['file']:40s}  {p['kind']:8s}  T_out={p['T_out_C']:5.1f}°C  "
                f"Q {p['Q_W']/1e3:5.2f}/{p['pred_Q_W']/1e3:5.2f} kW  "
                f"COP {p['COP']:4.2f}/{p['pred_COP']:4.2f}  ΔQ {p['err_Q_pct']:+6.1f}%"
            )
    lee = out["lee2021"]
    print()
    print(lee["source"]["citation"])
    print(
        f"  Table 5 digits match file: {lee['coefficients_match_table5']}  "
        f"Table 4 all ṁ,W > 0: {lee['all_positive']}"
    )
    t5 = lee["table4_test5"]
    print(
        f"  Table 4 test 5  Te={t5['Te_C']}°C  Tc={t5['Tc_C']}°C  "
        f"W={t5['power_W']:.1f} W  ṁ={t5['mdot_g_s']:.2f} g/s"
    )
    print("  Table 6 system capacities are not scored (Te/Tc not tabulated).")


if __name__ == "__main__":
    main()
