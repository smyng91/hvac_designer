"""Algebraic sizing case studies at published dry-bulb conditions."""

import math

import pytest

from heatpump.requirements import cooling_tons_to_w
from heatpump.sizing_cases import (
    APP_G_COOL_OVERSIZE,
    APP_G_HEAT_OVERSIZE,
    MANUAL_S_COOL_MAX,
    MANUAL_S_HEAT_MAX,
    case_definitions,
    run_cases,
)


def test_published_limit_constants():
    assert MANUAL_S_COOL_MAX == pytest.approx(1.15)
    assert MANUAL_S_HEAT_MAX == pytest.approx(1.40)
    assert APP_G_COOL_OVERSIZE == pytest.approx(1.15)
    assert APP_G_HEAT_OVERSIZE == pytest.approx(1.25)


def test_three_ton_is_iso_ton():
    assert cooling_tons_to_w(3.0) == pytest.approx(10550.558, rel=1e-6)


def test_run_cooling_heating_reverse_cases():
    cases = {c.key: c for c in run_cases()}
    assert "Ciso" in cases and "Hex" in cases and "Rdual" in cases
    cool = cases["Ciso"]
    assert cool.cooling is not None and cool.heating is None
    assert cool.fluid == "R410A"
    assert cool.cooling.feasible
    assert math.isfinite(cool.V_disp_m3) and cool.V_disp_m3 > 0.0
    assert 0.5 < cool.cooling.ratio < 1.5
    heat = cases["Hex"]
    assert heat.heating is not None and heat.cooling is None
    assert heat.fluid == "R32"
    assert heat.heating.feasible
    assert 0.5 < heat.heating.ratio < 1.5
    rev = cases["Rdual"]
    assert rev.heating is not None and rev.cooling is not None
    assert rev.V_disp_m3 > 0.0
    for c in cases.values():
        if c.rejected:
            assert c.gate_detail
            continue
        if c.heating is not None:
            assert c.heating.feasible
            assert math.isfinite(c.heating.ratio)
        if c.cooling is not None:
            assert c.cooling.feasible
            assert math.isfinite(c.cooling.ratio)


def test_case_definitions_include_rating_points():
    keys = [c["key"] for c in case_definitions()]
    assert keys[:8] == ["Ciso", "Cahri", "Cex", "Hex", "Hahri", "Hrej", "Rdual", "Rahri"]
