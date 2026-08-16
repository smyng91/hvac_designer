#!/usr/bin/env python3
"""Case: reversible heat pump, cooling then heating.

One machine, one run. The CSV ``mode`` column is 0=cooling, 1=heating.
At the changeover the coils swap roles and the controller flips sign.
"""

from __future__ import annotations

import argparse

from _paths import HOUR_S, WEATHER_REVERSE, out_dir, sim_horizon
from heatpump import DesignRequest, TimeSeries, design_system, simulate
from heatpump.simulate import plot_result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-final", type=float, default=HOUR_S)
    args = p.parse_args(argv)

    ts = TimeSeries.from_csv(WEATHER_REVERSE)
    req = DesignRequest(
        refrigerant="R32",
        mode="heat_pump",
        T_zone=293.15,
        T_zone_cool=297.15,
        timeseries=ts,
    )
    sys = design_system(req)
    print(sys.summary())
    res = simulate(
        sys.controller,
        spec=sys.spec,
        design=sys,
        request=req,
        timeseries=ts,
        **sim_horizon(min(args.t_final, ts.duration)),
    )
    dest = out_dir() / "reverse.png"
    plot_result(res, dest, "Reverse · cool then heat")
    cool = res.meas["mode"] < 0.5
    heat = res.meas["mode"] > 0.5
    print(
        f"t={res.t[-1]/60:.0f} min  "
        f"cooling T_z {res.meas['T_z'][cool][-1]-273.15:.2f}°C → "
        f"heating T_z {res.meas['T_z'][heat][-1]-273.15:.2f}°C  "
        f"final p_c/p_e {res.meas['p_c'][-1]/res.meas['p_e'][-1]:.2f}"
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
