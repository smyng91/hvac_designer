"""User catalog and timeseries bins — no fake SKUs or AHRI hour tables."""

from pathlib import Path

import numpy as np
import pytest

from heatpump.catalog import default_example_catalog, load_catalog
from heatpump.requirements import TimeSeries
from heatpump.seasonal import bin_timeseries

ROOT = Path(__file__).resolve().parents[1]


def test_example_catalog_is_only_the_cited_lee_map():
    cat = load_catalog(default_example_catalog())
    assert "Lee" in cat.citation
    assert len(cat.items) == 1
    it = cat.items[0]
    assert it.kind == "compressor_map"
    assert it.path.exists()
    assert "1180" in it.citation
    assert "SKU" not in it.id.upper()


def test_catalog_rejects_missing_citation(tmp_path):
    p = tmp_path / "anon.json"
    p.write_text('{"items": [{"id": "x", "kind": "compressor_map", "path": "n.json"}]}')
    with pytest.raises(ValueError, match="citation"):
        load_catalog(p)


def test_seasonal_bins_use_record_dwell_not_ahri_hours():
    t = np.array([0.0, 3600.0, 7200.0, 10800.0])
    T = np.array([268.15, 273.15, 278.15, 268.15])
    Q = np.array([-2000.0, -1500.0, 500.0, -1800.0])
    ts = TimeSeries(t=t, T_out=T, Q_gain=Q)
    bins = bin_timeseries(ts, width_K=5.0)
    assert bins.hours_total == pytest.approx(3.0)
    assert sum(b.hours for b in bins.bins) == pytest.approx(3.0)
    assert all(b.Q_cap is None and b.W is None for b in bins.bins)
    assert any("AHRI" in n for n in bins.notes)
    # Capacity is omitted when no map is passed — not invented.
    md = bins.to_markdown()
    assert "—" in md
