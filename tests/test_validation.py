"""Literature validation uses the downloaded Ramírez workbook, not invented points."""

import numpy as np
import pytest

from heatpump.validation import (
    RAMIREZ_SHA256,
    _tex_sci,
    compare_lee2021_map,
    compare_ramirez,
    default_data_dir,
    default_maps_dir,
    file_sha256,
    parse_nrel,
    parse_ramirez,
    read_xlsx_sheets,
    raw_sources_complete,
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
    assert report["source"]["primary_source"] is True
    assert report["source"]["cached_means"] is False
    assert report["mape"]["n"] == 16
    assert report["mape_enthalpy_band"]["n"] <= 16
    assert report["mape_enthalpy_band"]["rule"].startswith("SI only")


def test_lee2021_map_is_the_published_table5():
    path = default_maps_dir() / "lee2021_iop1180_012041.json"
    assert path.exists()
    report = compare_lee2021_map(path)
    assert report["coefficients_match_table5"]
    assert report["all_positive"]
    assert any("Table 6" in n for n in report["notes"])
    assert any("not a comparison to measured" in n for n in report["notes"])
    assert report["jax_vs_numpy"]["n"] == 9
    assert report["jax_vs_numpy"]["max_rel_power"] < 1.0e-10
    assert report["jax_vs_numpy"]["max_rel_mdot"] < 1.0e-10


def test_tex_sci_does_not_print_bare_zero():
    assert _tex_sci(0.0) == r"<10^{-16}"
    assert "10^{" in _tex_sci(1.23e-12)
    assert "0" != _tex_sci(1e-20)


def test_raw_sources_complete_when_xlsx_and_nrel_present():
    ok, missing = raw_sources_complete()
    if not XLSX.exists():
        assert ok is False
        return
    nrel = default_data_dir() / "nrel_hil"
    if not all((nrel / name).exists() for name in (
        "HP_Cool_OAT95F_SP76F72F68F.csv",
        "HP_Cool_OAT75F_SP72F68F.csv",
        "HP_Heat_OAT45F_SP68F72F.csv",
        "HP_Heat_OAT5F_SP72F.csv",
    )):
        assert ok is False
        return
    assert ok is True
    assert missing == []


def test_nrel_parser_skips_or_reads_downloaded_csvs():
    pts = parse_nrel()
    if not pts:
        pytest.skip("NREL CSVs not downloaded (see validation/data/SOURCES.md)")
    kinds = {p["kind"] for p in pts}
    assert kinds <= {"cooling", "heating"}
    from_cache = all(p.get("cached_on_period") for p in pts)
    for p in pts:
        assert p["n_on"] > 0
        assert p["Q_W"] > 1000.0
        assert p["W_out_W"] > 500.0
        assert p.get("Q_air_W") is not None
        assert p.get("sha256")
        if from_cache:
            # Raw 1 Hz HIL files are gitignored; CI uses validation/results/nrel_on_period.csv.
            assert p.get("primary_source") is False
            assert p.get("cached_on_period") is True
        else:
            assert p.get("primary_source") is True
            assert p.get("cached_on_period") is False
