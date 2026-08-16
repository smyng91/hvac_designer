"""Literature validation uses the downloaded Ramírez workbook, not invented points."""

from pathlib import Path

import numpy as np
import pytest

from heatpump.validation import (
    RAMIREZ_SHA256,
    compare_lee2021_map,
    compare_ramirez,
    default_data_dir,
    default_maps_dir,
    file_sha256,
    parse_nrel,
    parse_ramirez,
    read_xlsx_sheets,
)

XLSX = default_data_dir() / "ramirez2019_mmc1.xlsx"


@pytest.mark.skipif(not XLSX.exists(), reason="Ramírez supplementary xlsx not in tree")
def test_ramirez_file_is_the_published_workbook():
    assert file_sha256(XLSX) == RAMIREZ_SHA256
    sheets = read_xlsx_sheets(XLSX)
    assert set(sheets) == {f"Run{i}" for i in range(1, 17)}
    first = sheets["Run1"][0]
    assert first["P1 (PSIa)"] == pytest.approx(100.47)
    assert first["T9 (°C)"] == pytest.approx(27.94)
    assert first["Power 1 (kW)"] == pytest.approx(0.44145)


@pytest.mark.skipif(not XLSX.exists(), reason="Ramírez supplementary xlsx not in tree")
def test_ramirez_run_means_are_physical():
    runs, meta = parse_ramirez(XLSX)
    assert meta["fluid"] == "R410A"
    assert len(runs) == 16
    for r in runs:
        assert r.p_c_Pa > r.p_e_Pa
        assert np.isfinite(r.Q_evap_W) and r.Q_evap_W > 0
        assert np.isfinite(r.SH_K)
        assert r.mdot_kg_s > 0


@pytest.mark.skipif(not XLSX.exists(), reason="Ramírez supplementary xlsx not in tree")
def test_unsized_model_is_within_order_of_the_lab_capacity():
    """Not a fit: the 3.5 kW sizer should land near the measured 3.5–4 kW."""
    report = compare_ramirez()
    assert report["design"]["fitted"] is False
    assert report["mape"]["Q_pct"] < 25.0
    qs = [row["meas_Q_W"] for row in report["runs"]]
    assert 3000.0 < float(np.mean(qs)) < 4500.0
    closed = [row for row in report["runs"] if row["pred_feasible"]]
    assert closed
    for row in closed:
        assert np.isfinite(row["pred_p_e_bar"])
        assert np.isfinite(row["pred_p_c_bar"])
        assert row["pred_p_c_bar"] > row["pred_p_e_bar"] > 0.0
    assert "data/validation/" not in str(report["source"]["file"])
    assert str(report["source"]["file"]).startswith("validation/")


def test_lee2021_map_is_the_published_table5():
    path = default_maps_dir() / "lee2021_iop1180_012041.json"
    assert path.exists()
    report = compare_lee2021_map(path)
    assert report["coefficients_match_table5"]
    assert report["all_positive"]
    assert "Table 6" in report["notes"][1]


def test_nrel_parser_skips_or_reads_downloaded_csvs():
    pts = parse_nrel()
    if not pts:
        pytest.skip("NREL CSVs not downloaded (see validation/data/SOURCES.md)")
    kinds = {p["kind"] for p in pts}
    assert kinds <= {"cooling", "heating"}
    for p in pts:
        assert p["n_on"] > 0
        assert p["Q_W"] > 1000.0
        assert p["W_out_W"] > 500.0
