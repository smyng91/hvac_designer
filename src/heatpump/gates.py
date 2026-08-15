"""Pass/fail operating envelope for a sized cycle."""

from __future__ import annotations

from dataclasses import dataclass

from heatpump.requirements import Constraints
from heatpump.thermo import FluidInfo


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    limit: float
    passed: bool
    hard: bool
    detail: str
    unit: str = ""


@dataclass(frozen=True)
class GateSet:
    gates: tuple[Gate, ...]

    @property
    def ok(self) -> bool:
        return all(g.passed or not g.hard for g in self.gates)

    @property
    def hard_failures(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if g.hard and not g.passed)

    @property
    def soft_failures(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if not g.hard and not g.passed)

    def summary(self) -> str:
        lines = []
        for g in self.gates:
            mark = "PASS" if g.passed else ("FAIL" if g.hard else "WARN")
            lines.append(f"  [{mark}] {g.name}: {g.detail}")
        return "\n".join(lines)


class DesignGateError(ValueError):
    """Hard feasibility gate failed — the machine is not a valid design."""

    def __init__(self, gates: GateSet):
        self.gates = gates
        failed = gates.hard_failures
        body = "\n".join(f"  - {g.name}: {g.detail}" for g in failed)
        super().__init__("Design failed feasibility gates:\n" + body)


def evaluate_gates(
    *,
    info: FluidInfo,
    p_c: float,
    pr: float,
    T_disch: float,
    SH: float,
    SC: float,
    G_e: float,
    G_c: float,
    constraints: Constraints | None = None,
) -> GateSet:
    cons = constraints or Constraints()
    gates = (
        Gate(
            "condensing_pressure",
            p_c,
            cons.p_c_frac_crit * info.pc,
            p_c <= cons.p_c_frac_crit * info.pc,
            True,
            f"p_c={p_c/1e5:.2f} bar vs {cons.p_c_frac_crit:.0%} of "
            f"p_crit={info.pc/1e5:.2f} bar",
            "Pa",
        ),
        Gate(
            "discharge_temperature",
            T_disch,
            cons.T_disch_max,
            T_disch <= cons.T_disch_max,
            True,
            f"T_disch={T_disch-273.15:.1f}°C vs max {cons.T_disch_max-273.15:.0f}°C",
            "K",
        ),
        Gate(
            "pressure_ratio",
            pr,
            cons.pr_max,
            pr <= cons.pr_max,
            True,
            f"PR={pr:.2f} vs max {cons.pr_max:.2f}",
        ),
        Gate(
            "superheat",
            SH,
            cons.SH_min,
            cons.SH_min <= SH <= cons.SH_max,
            True,
            f"SH={SH:.2f} K in [{cons.SH_min:.1f}, {cons.SH_max:.1f}] K",
            "K",
        ),
        Gate(
            "subcooling",
            SC,
            cons.SC_min,
            SC >= cons.SC_min - 1e-6,
            True,
            f"SC={SC:.2f} K vs min {cons.SC_min:.1f} K",
            "K",
        ),
    )
    extra = []
    if cons.G_max is not None:
        extra = [
            Gate(
                "evap_mass_flux",
                G_e,
                cons.G_max,
                G_e <= cons.G_max,
                True,
                f"G_e={G_e:.0f} kg/m²s vs max {cons.G_max:.0f}",
                "kg/m²s",
            ),
            Gate(
                "cond_mass_flux",
                G_c,
                cons.G_max,
                G_c <= cons.G_max,
                True,
                f"G_c={G_c:.0f} kg/m²s vs max {cons.G_max:.0f}",
                "kg/m²s",
            ),
        ]
    return GateSet(gates + tuple(extra))


def raise_if_failed(gates: GateSet) -> None:
    if not gates.ok:
        raise DesignGateError(gates)
