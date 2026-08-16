#!/usr/bin/env python3
"""Closed-loop heating physics audit (model consistency, not a lab twin)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatpump.design import design_heat_pump
from heatpump.simulate import plot_result, simulate
from heatpump.thermo import build_tables, resolve_fluid
from heatpump.validation import default_results_dir


def audit(res, T_out: float, Tsp: float) -> dict:
    t = res.t
    m = res.meas
    late = t >= max(t[-1] - 60.0, 0.6 * t[-1])
    sl = slice(int(np.argmax(late)), None)

    def mean(key):
        return float(np.mean(m[key][sl]))

    Tz0 = float(m["T_z"][0] - 273.15)
    Tz1 = float(m["T_z"][-1] - 273.15)
    Tsat_e = mean("Tsat_e")
    Tsat_c = mean("Tsat_c")
    W = mean("power")
    Qz = mean("Q_zone")
    Qe = mean("Q_evap")
    cop = Qz / max(W, 1.0)
    cop_carnot = Tsat_c / max(Tsat_c - Tsat_e, 1.0)
    dM = float((m["charge"][-1] - m["charge"][0]) / m["charge"][0])
    mbal = float(
        np.mean(np.abs(m["mdot_comp"][sl] - m["mdot_eev"][sl])) / max(mean("mdot_comp"), 1e-9)
    )
    q_gap = abs(Qz - (Qe + W)) / max(abs(Qz), 1.0)
    pc = res.spec.p_crit
    checks = {
        "evap_colder_than_outdoor": Tsat_e < T_out - 1.0,
        "cond_hotter_than_zone": Tsat_c > mean("T_z") + 3.0,
        "superheat_in_band": 2.0 <= mean("SH") <= 12.0,
        "subcool_nonneg": mean("SC") > -1.5,
        "pr_typical": 2.5 <= mean("pr") <= 8.5,
        "cop_below_carnot": cop < cop_carnot,
        "zone_toward_setpoint": Tz1 > Tz0,
        "pressures_below_critical": mean("p_c") < 0.92 * pc,
        "mass_flow_balanced": mbal < 0.15,
        "energy_closes": q_gap < 0.20,
        "charge_drift_lt_4pct": abs(dM) < 0.04,
    }
    return {
        "fluid": res.spec.fluid,
        "t_final_s": float(t[-1]),
        "T_out_C": T_out - 273.15,
        "Tsp_C": Tsp - 273.15,
        "Tz_start_C": Tz0,
        "Tz_end_C": Tz1,
        "Tsat_e_C": Tsat_e - 273.15,
        "Tsat_c_C": Tsat_c - 273.15,
        "SH_K": mean("SH"),
        "SC_K": mean("SC"),
        "p_e_bar": mean("p_e") / 1e5,
        "p_c_bar": mean("p_c") / 1e5,
        "COP": cop,
        "COP_carnot": cop_carnot,
        "charge_drift": dM,
        "mdot_imbalance": mbal,
        "energy_gap": q_gap,
        "checks": checks,
        "n_fail": int(sum(1 for v in checks.values() if not v)),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refrigerant", "-r", default="R32")
    p.add_argument("--load", type=float, default=5500.0)
    p.add_argument("--T-out", type=float, default=0.0)
    p.add_argument("--T-zone", type=float, default=20.0)
    p.add_argument("--t-final", type=float, default=300.0)
    args = p.parse_args(argv)

    fluid = resolve_fluid(args.refrigerant)
    T_out, Tsp = args.T_out + 273.15, args.T_zone + 273.15
    design = design_heat_pump(fluid, args.load, T_out=T_out, T_zone=Tsp)
    print(design.summary())
    tables = build_tables(design.spec.fluid)
    res = simulate(
        "pid",
        t_final=args.t_final,
        record_dt=2.0 if args.t_final < 3600.0 else 30.0,
        spec=design.spec,
        tables=tables,
        design=design,
        T_out=T_out,
        Tsp=Tsp,
        reduction="auto",
    )
    dest = default_results_dir()
    dest.mkdir(parents=True, exist_ok=True)
    tag = "audit_1h" if args.t_final >= 3600.0 else "audit"
    plot_result(res, dest / f"{tag}.png", f"audit · {design.fluid}")
    report = audit(res, T_out, Tsp)
    (dest / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("failed checks:", [k for k, v in report["checks"].items() if not v])


if __name__ == "__main__":
    main()
