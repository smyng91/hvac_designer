#!/usr/bin/env python3
"""Case: winter heating.

Size an R32 air-source heat pump for 5.5 kW at 0 °C outdoor / 20 °C zone
and run one hour closed-loop (PID). Pass ``--t-final 90`` for the full DAE.
"""

from __future__ import annotations

import argparse

from _paths import HOUR_S, out_dir, sim_horizon
from heatpump import design_heat_pump, simulate
from heatpump.simulate import plot_result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-final", type=float, default=HOUR_S)
    p.add_argument("--controller", default="pid", choices=["pid", "hysteresis", "bangbang"])
    args = p.parse_args(argv)

    T_out, Tsp = 273.15, 293.15
    design = design_heat_pump("R32", 5500.0, T_out=T_out, T_zone=Tsp)
    print(design.summary())
    res = simulate(
        args.controller,
        spec=design.spec,
        design=design,
        T_out=T_out,
        Tsp=Tsp,
        **sim_horizon(args.t_final),
    )
    dest = out_dir() / "heating.png"
    plot_result(res, dest, f"Heating · {design.fluid} · {args.controller}")
    print(
        f"t={res.t[-1]/60:.0f} min  T_z {res.meas['T_z'][-1]-273.15:.2f}°C  "
        f"(setpoint {Tsp-273.15:.0f}°C)  "
        f"SH {res.meas['SH'][-1]:.1f} K  COP {res.meas['COP'][-1]:.2f}"
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
