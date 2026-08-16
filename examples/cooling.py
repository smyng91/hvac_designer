#!/usr/bin/env python3
"""Case: summer cooling.

Size an R410A air conditioner for 6.2 kW at 35 °C outdoor / 24 °C zone
and run one hour closed-loop (PID).
"""

from __future__ import annotations

import argparse

from _paths import HOUR_S, out_dir, sim_horizon
from heatpump import design_air_conditioner, simulate
from heatpump.simulate import plot_result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-final", type=float, default=HOUR_S)
    p.add_argument("--controller", default="pid", choices=["pid", "hysteresis", "bangbang"])
    args = p.parse_args(argv)

    T_out, Tsp = 308.15, 297.15
    design = design_air_conditioner("R410A", 6200.0, T_out=T_out, T_zone=Tsp)
    print(design.summary())
    res = simulate(
        args.controller,
        spec=design.spec,
        design=design,
        T_out=T_out,
        Tsp=Tsp,
        **sim_horizon(args.t_final),
    )
    dest = out_dir() / "cooling.png"
    plot_result(res, dest, f"Cooling · {design.fluid} · {args.controller}")
    print(
        f"t={res.t[-1]/60:.0f} min  T_z {res.meas['T_z'][-1]-273.15:.2f}°C  "
        f"(setpoint {Tsp-273.15:.0f}°C)  "
        f"SH {res.meas['SH'][-1]:.1f} K  COP {res.meas['COP'][-1]:.2f}"
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
