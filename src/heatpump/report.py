"""Written design package: gates, capacity map, psychrometrics, assumptions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from heatpump.capacity import CapacityMap, capacity_map
from heatpump.gates import GateSet
from heatpump.psychro import CoolingPsychro, cooling_psychro

ASSUMPTIONS = """\
Results are the vapor-compression balances closed with CoolProp HEOS
(Bell et al., 2014) and the heat-transfer / compressor closures below.
This is not an AHRI 210/240 or EN 14511 rating.

- Saturation and (p, h) states: CoolProp Helmholtz EOS for the named fluid.
- Design compressor: displacement is inverted from the design mass flow
  using clearance volumetric efficiency. Discharge enthalpy at design uses
  the same polytropic isentropic rise as the JAX residual (optional HEOS
  isentropic close via compression='heos'). Default n_e=n_c=6 plants then
  scale V_disp and A_eev so a short residual settle meets Q_load; that
  scale is a plant consistency step, not a laboratory fit.
- EEV: isenthalpic orifice ṁ = Cd A u √(2ρ Δp) with A sized at the
  design opening (default Cd = 0.70, u_design = 0.40).
- Coils: tube count is iterated until ε-NTU heat rate equals the cycle
  duty. Refrigerant HTC is Dittus–Boelter (single-phase) or a Shah-type
  multiplier (two-phase), not the full Shah (1979) correlation. Air-side
  HTC is Zhukauskas cross-flow over a tube bank.
  Wall capacitance is ρ_cu c_p,cu A t_wall.
- Charge is the integral of Zivi / flashed density on the design
  enthalpy profile and the internal volume (same cells as the plant).
- Envelope UA is Q_load / |T_zone − T_out|. Zone capacitance is
  ρ_air c_p V.
- Off-design capacity re-closes T_e and T_c so ṁ Δh equals the ε-NTU
  coil rate at the sized geometry. The load line is Newton cooling or
  the time-series Q(T_out).
- Cooling SHR is computed from humid-air balances (CoolProp HA) with
  apparatus dew point = T_e. The transient plant is dry unless the user
  sets moist=True and supplies RH_out and RH_zone0 (no default humidity).
- Optional frost mass uses Hayashi (1977) density and Sanders (1974)
  conductivity, or IAPWS ice if requested. There is no time-based derate
  and no automatic defrost; melt uses the user W_defrost only.
- AHRI 540 and fan tables are used only when the user supplies a cited
  coefficient / airflow file. No default map or SKU is invented.
- Seasonal bins use the dwell time of the user time series. AHRI 210/240
  bin-hour tables are not copied.
- Electrical current is shaft power / voltage (or / V√3 if three-phase),
  divided by η_motor only when that efficiency is supplied.
"""


@dataclass(frozen=True)
class Electrical:
    W_shaft: float
    voltage: float
    phases: int
    I: float
    eta_motor: float | None
    W_electric: float


@dataclass(frozen=True)
class DesignPackage:
    fluid: str
    mode: str
    controller: str
    gates: GateSet
    heating_map: CapacityMap | None
    cooling_map: CapacityMap | None
    psychro: CoolingPsychro | None
    electrical: Electrical
    charge_kg: float
    notes: tuple[str, ...]
    summary_text: str
    assumptions: str = ASSUMPTIONS
    hardware: dict | None = None

    @property
    def ok(self) -> bool:
        return self.gates.ok

    def to_markdown(self) -> str:
        lines = [
            f"# Design package — {self.fluid} {self.mode}",
            "",
            f"Controller `{self.controller}`.",
            "",
            "## Feasibility gates",
            "",
            "```",
            self.gates.summary(),
            "```",
            "",
            "## Hardware",
            "",
            self.summary_text,
            "",
            f"Charge from the design enthalpy profile: {self.charge_kg:.3f} kg.",
            "",
        ]
        elec = self.electrical
        if elec.eta_motor is not None:
            lines.append(
                f"Electrical: shaft {elec.W_shaft/1e3:.2f} kW, "
                f"η_motor={elec.eta_motor:.2f} → {elec.W_electric/1e3:.2f} kW, "
                f"{elec.voltage:.0f} V / {elec.phases} ph, I = {elec.I:.1f} A."
            )
        else:
            lines.append(
                f"Electrical: shaft {elec.W_shaft/1e3:.2f} kW at "
                f"{elec.voltage:.0f} V / {elec.phases} ph, I = {elec.I:.1f} A "
                "(no motor efficiency supplied)."
            )
        lines.append("")
        if self.psychro is not None:
            lines += ["## Cooling psychrometrics", "", self.psychro.summary(), ""]
        for cmap, title in (
            (self.heating_map, "Heating capacity vs outdoor T"),
            (self.cooling_map, "Cooling capacity vs outdoor T"),
        ):
            if cmap is None:
                continue
            bal = (
                f"{cmap.T_balance-273.15:.1f}°C"
                if cmap.T_balance is not None
                else "none in range"
            )
            lines += [
                f"## {title}",
                "",
                f"Design margin Q_cap/Q_load = {cmap.margin_design:.2f} at "
                f"{cmap.T_design-273.15:.1f}°C outdoor. Balance point: {bal}.",
                "",
                "| T_out °C | Q_cap kW | Q_load kW | COP | T_e °C | T_c °C | p_e bar | p_c bar | PR | ok |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
            ]
            for p in cmap.points:
                te = f"{p.T_e-273.15:.1f}" if np.isfinite(p.T_e) else "—"
                tc = f"{p.T_c-273.15:.1f}" if np.isfinite(p.T_c) else "—"
                pe = f"{p.p_e/1e5:.2f}" if np.isfinite(p.p_e) else "—"
                pc = f"{p.p_c/1e5:.2f}" if np.isfinite(p.p_c) else "—"
                lines.append(
                    f"| {p.T_out-273.15:.1f} | {p.Q_cap/1e3:.2f} | "
                    f"{p.Q_load/1e3:.2f} | {p.COP:.2f} | {te} | {tc} | "
                    f"{pe} | {pc} | {p.pr:.2f} | {'yes' if p.feasible else 'no'} |"
                )
            lines.append("")
            for n in cmap.notes:
                lines.append(f"- {n}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            lines.extend(f"- {n}" for n in self.notes)
            lines.append("")
        lines += ["## Assumptions", "", self.assumptions]
        return "\n".join(lines).rstrip() + "\n"

    def to_json(self) -> dict:
        def cmap(m: CapacityMap | None):
            if m is None:
                return None
            return {
                "kind": m.kind,
                "T_zone": m.T_zone,
                "T_design": m.T_design,
                "Q_design": m.Q_design,
                "T_balance": m.T_balance,
                "margin_design": m.margin_design,
                "notes": list(m.notes),
                "points": [asdict(p) for p in m.points],
            }

        return {
            "fluid": self.fluid,
            "mode": self.mode,
            "controller": self.controller,
            "ok": self.ok,
            "gates": [asdict(g) for g in self.gates.gates],
            "hardware": self.hardware,
            "heating_map": cmap(self.heating_map),
            "cooling_map": cmap(self.cooling_map),
            "psychro": asdict(self.psychro) if self.psychro else None,
            "electrical": asdict(self.electrical),
            "charge_kg": self.charge_kg,
            "notes": list(self.notes),
            "assumptions": self.assumptions,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")
        return path

    def plot(self, path: str | Path) -> Path | None:
        maps = [m for m in (self.heating_map, self.cooling_map) if m is not None]
        if not maps:
            return None
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(len(maps), 1, figsize=(8.0, 3.6 * len(maps)), sharex=False)
        if len(maps) == 1:
            ax = [ax]
        for a, m in zip(ax, maps):
            T = np.array([p.T_out - 273.15 for p in m.points])
            a.plot(T, [p.Q_cap / 1e3 for p in m.points], color="#1d4ed8", label="capacity")
            a.plot(T, [p.Q_load / 1e3 for p in m.points], color="#b91c1c", label="load")
            if m.T_balance is not None:
                a.axvline(m.T_balance - 273.15, color="#334155", linestyle=":", label="balance")
            a.set_ylabel("kW")
            a.set_xlabel("Outdoor temperature (°C)")
            a.set_title(f"{m.kind}  ·  margin {m.margin_design:.2f}")
            a.legend(frameon=False)
        fig.suptitle(f"{self.fluid} capacity map")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def write_latex_macros(self, dest: str | Path) -> Path:
        """Hardware / map numbers for the manuscript (no invented values)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        hw = self.hardware or {}
        hm = self.heating_map
        t_bal = ""
        if hm is not None and hm.T_balance is not None:
            tb = float(hm.T_balance) - 273.15
            t_bal = f"{0.0 if abs(tb) < 0.05 else tb:.2f}"
        margin = "" if hm is None else f"{hm.margin_design:.2f}"
        lines = [
            f"\\newcommand{{\\DesignFluid}}{{{self.fluid}}}",
            f"\\newcommand{{\\DesignVdispCmThree}}{{{hw.get('V_disp_m3', float('nan')) * 1e6:.1f}}}",
            f"\\newcommand{{\\DesignAeevMmTwo}}{{{hw.get('A_eev_m2', float('nan')) * 1e6:.2f}}}",
            f"\\newcommand{{\\DesignCd}}{{{hw.get('Cd', float('nan')):.2f}}}",
            f"\\newcommand{{\\DesignCloss}}{{{hw.get('C_loss', float('nan')):.3f}}}",
            f"\\newcommand{{\\DesignEtaIs}}{{{hw.get('eta_is', float('nan')):.2f}}}",
            f"\\newcommand{{\\DesignNtubesIn}}{{{hw.get('n_tubes_indoor', 'NA')}}}",
            f"\\newcommand{{\\DesignNtubesOut}}{{{hw.get('n_tubes_outdoor', 'NA')}}}",
            f"\\newcommand{{\\DesignNe}}{{{hw.get('n_e', 'NA')}}}",
            f"\\newcommand{{\\DesignNc}}{{{hw.get('n_c', 'NA')}}}",
            f"\\newcommand{{\\DesignNstateDry}}{{{hw.get('n_state_dry', 'NA')}}}",
            f"\\newcommand{{\\DesignUAenv}}{{{hw.get('UA_env', float('nan')):.1f}}}",
            f"\\newcommand{{\\DesignCzonekJK}}{{{hw.get('C_zone', float('nan')) / 1e3:.1f}}}",
            f"\\newcommand{{\\DesignNHz}}{{{hw.get('N_design_Hz', float('nan')):.0f}}}",
            f"\\newcommand{{\\DesignChargeKg}}{{{self.charge_kg:.3f}}}",
            f"\\newcommand{{\\DesignTbalanceC}}{{{t_bal}}}",
            f"\\newcommand{{\\DesignMargin}}{{{margin}}}",
            f"\\newcommand{{\\DesignPlantScale}}{{{hw.get('plant_match_scale', 1.0):.3f}}}",
        ]
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dest


def _electrical(W: float, voltage: float, phases: int, eta_motor: float | None) -> Electrical:
    W_el = float(W) / eta_motor if eta_motor is not None and eta_motor > 0.0 else float(W)
    if phases >= 3:
        I = W_el / max(voltage * 3.0**0.5, 1.0)
    else:
        I = W_el / max(voltage, 1.0)
    return Electrical(
        W_shaft=float(W),
        voltage=voltage,
        phases=phases,
        I=I,
        eta_motor=eta_motor,
        W_electric=W_el,
    )


def _map_for(report, req, spec, kind: str) -> CapacityMap:
    T_zone = req.T_zone_cool if kind == "cooling" else req.T_zone_heat
    T_des = req.T_out_cool if kind == "cooling" else req.T_out_heat
    return capacity_map(
        fluid=report.fluid,
        kind=kind,
        T_zone=T_zone,
        T_design=T_des,
        Q_design=report.Q_load,
        spec=spec,
        SH=req.SH,
        SC=req.SC,
        DT_evap=req.DT_evap,
        DT_cond=req.DT_cond,
        N_design=req.N_hz,
        constraints=req.constraints,
        UA=spec.UA_env,
        timeseries=req.timeseries,
    )


def build_report(system) -> DesignPackage:
    """Assemble a package from a ``SystemDesign``."""
    req = system.request
    spec = system.spec
    gates = system.gates
    heat_m = cool_m = psych = None
    from heatpump.plant import apply_operating_mode

    if system.heating is not None:
        heat_m = _map_for(system.heating, req, apply_operating_mode(spec, "heating"), "heating")
    if system.cooling is not None:
        cool_m = _map_for(system.cooling, req, apply_operating_mode(spec, "cooling"), "cooling")
        psych = cooling_psychro(
            req.T_zone_cool,
            system.cooling.Q_evap,
            RH=req.indoor_RH,
            T_adp=system.cooling.T_e,
            mdot_air=system.cooling.spec.mdot_air_e0,
        )
    W = 0.0
    if system.heating is not None:
        W = max(W, system.heating.W)
    if system.cooling is not None:
        W = max(W, system.cooling.W)
    charge = (system.heating or system.cooling).charge_kg
    notes = list(system.notes)
    if system.heating is not None and system.cooling is not None:
        notes.append(
            f"Charge from the design enthalpy profile: heating "
            f"{system.heating.charge_kg:.3f} kg, cooling "
            f"{system.cooling.charge_kg:.3f} kg (same internal volume)."
        )
        charge = max(system.heating.charge_kg, system.cooling.charge_kg)
    indoor_n = spec.indoor.n_tubes if spec.indoor else spec.n_tubes_c
    outdoor_n = spec.outdoor.n_tubes if spec.outdoor else spec.n_tubes_e
    hardware_text = (
        f"Compressor V_disp = {spec.V_disp*1e6:.1f} cm³/rev, "
        f"EEV {spec.A_eev*1e6:.2f} mm². "
        f"Indoor {indoor_n} tubes, outdoor {outdoor_n} tubes. "
        f"Zone UA = {spec.UA_env:.1f} W/K, C = {spec.C_zone/1e3:.1f} kJ/K."
    )
    hardware = {
        "V_disp_m3": float(spec.V_disp),
        "A_eev_m2": float(spec.A_eev),
        "Cd": float(spec.Cd),
        "C_loss": float(spec.C_loss),
        "eta_is": float(spec.eta_is0),
        "n_tubes_indoor": int(indoor_n),
        "n_tubes_outdoor": int(outdoor_n),
        "n_e": int(spec.n_e),
        "n_c": int(spec.n_c),
        "n_state_dry": int(3 + 2 * spec.n_e + 2 * spec.n_c),
        "UA_env": float(spec.UA_env),
        "C_zone": float(spec.C_zone),
        "N_design_Hz": float(spec.N_design),
        "plant_match_scale": float(
            (system.heating or system.cooling).plant_match_scale
            if (system.heating or system.cooling) is not None
            else 1.0
        ),
    }
    return DesignPackage(
        fluid=spec.fluid,
        mode=req.mode,
        controller=system.controller,
        gates=gates,
        heating_map=heat_m,
        cooling_map=cool_m,
        psychro=psych,
        electrical=_electrical(W, req.voltage, req.phases, req.eta_motor),
        charge_kg=float(charge),
        notes=tuple(dict.fromkeys(notes)),
        summary_text=hardware_text,
        hardware=hardware,
    )
