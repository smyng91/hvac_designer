"""Feasibility gates, capacity map, psychrometrics, design package."""

from pathlib import Path

import pytest

from heatpump.capacity import capacity_map
from heatpump.design import design_air_conditioner, design_heat_pump, design_system
from heatpump.gates import DesignGateError
from heatpump.psychro import cooling_psychro
from heatpump.requirements import Constraints, DesignRequest


def test_typical_r32_passes_gates():
    rep = design_heat_pump("R32", 5500.0, T_out=273.15, T_zone=293.15)
    assert rep.gates is not None and rep.gates.ok
    assert rep.charge_kg > 0.0
    assert rep.G_e > 0.0 and rep.G_c > 0.0
    assert rep.Q_heat == pytest.approx(5500.0)


def test_tight_discharge_is_a_hard_fail():
    with pytest.raises(DesignGateError, match="discharge"):
        design_heat_pump(
            "R32",
            5500.0,
            T_out=273.15,
            T_zone=293.15,
            constraints=Constraints(T_disch_max=330.0),
        )


def test_heating_capacity_rises_with_outdoor_t():
    rep = design_heat_pump("R32", 5500.0, T_out=273.15, T_zone=293.15)
    cmap = capacity_map(
        fluid=rep.fluid,
        kind="heating",
        T_zone=293.15,
        T_design=273.15,
        Q_design=5500.0,
        spec=rep.spec,
        SH=6.0,
        SC=4.0,
        DT_evap=10.0,
        DT_cond=12.0,
        N_design=50.0,
        UA=rep.spec.UA_env,
        n_points=7,
    )
    cold = cmap.at(263.15)
    mild = cmap.at(278.15)
    assert cold.feasible and mild.feasible
    assert mild.Q_cap > cold.Q_cap
    assert cmap.margin_design == pytest.approx(1.0, rel=0.25)


def test_cooling_balance_and_package(tmp_path: Path):
    sys = design_system(
        DesignRequest(
            refrigerant="R410A",
            mode="cooling",
            Q_cool=6200.0,
            T_out_cool=308.15,
            T_zone=297.15,
            indoor_RH=0.55,
        )
    )
    pkg = sys.as_report()
    assert pkg.ok
    assert pkg.cooling_map is not None
    assert pkg.psychro is not None
    assert 0.0 < pkg.psychro.SHR <= 1.0
    assert pkg.psychro.Q_sensible + pkg.psychro.Q_latent == pytest.approx(pkg.psychro.Q_total)
    md = pkg.to_markdown()
    assert "Feasibility gates" in md
    assert "Assumptions" in md
    assert "Balance point" in md
    assert "no time-based derate" in md.lower()
    assert "no default map" in md.lower() or "not invented" in md.lower()
    dest = tmp_path / "design.md"
    pkg.write(dest)
    assert dest.exists() and dest.with_suffix(".json").exists()


def test_psychro_shr_is_an_output():
    wet = cooling_psychro(297.15, 5000.0, RH=0.55, T_adp=283.15, mdot_air=0.40)
    assert wet.wet
    assert 0.0 < wet.SHR < 1.0
    assert wet.Q_sensible + wet.Q_latent == pytest.approx(wet.Q_total)
    assert wet.condensate_kg_s > 0.0
    dry = cooling_psychro(297.15, 5000.0, RH=0.30, T_adp=290.15, mdot_air=0.40)
    assert not dry.wet
    assert dry.Q_latent == pytest.approx(0.0)
    assert dry.SHR == pytest.approx(1.0)
