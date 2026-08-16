#!/usr/bin/env python3
"""Case: size and operate from a weather / load record.

Hours in each outdoor-T bin are the dwell time of this file, not an
AHRI 210/240 climate table.
"""

from __future__ import annotations

import argparse

from _paths import HOUR_S, WEATHER_COOL, WEATHER_HEAT, out_dir, sim_horizon
from heatpump import DesignRequest, TimeSeries, bin_timeseries, design_system, simulate
from heatpump.simulate import plot_result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["cooling", "heating"], default="cooling")
    p.add_argument("--t-final", type=float, default=HOUR_S)
    args = p.parse_args(argv)

    weather = WEATHER_COOL if args.mode == "cooling" else WEATHER_HEAT
    ts = TimeSeries.from_csv(weather)
    req = DesignRequest(
        refrigerant="R410A" if args.mode == "cooling" else "R32",
        mode="auto",
        T_zone=297.15 if args.mode == "cooling" else 293.15,
        timeseries=ts,
    )
    sys = design_system(req)
    print(sys.summary())
    pkg = sys.as_report()
    dest = out_dir()
    pkg.write(dest / f"weather_{args.mode}_design.md")
    bins = bin_timeseries(
        ts,
        width_K=5.0,
        heating_map=pkg.heating_map,
        cooling_map=pkg.cooling_map,
    )
    (dest / f"weather_{args.mode}_seasonal.md").write_text(bins.to_markdown())
    print(f"seasonal: {len(bins.bins)} bins, {bins.hours_total:.2f} h from {weather.name}")

    res = simulate(
        sys.controller,
        spec=sys.spec,
        design=sys,
        request=req,
        timeseries=ts,
        **sim_horizon(min(args.t_final, ts.duration)),
    )
    plot_result(res, dest / f"weather_{args.mode}.png", f"{args.mode} · {weather.name}")
    print(
        f"t={res.t[-1]/60:.0f} min  T_z {res.meas['T_z'][-1]-273.15:.2f}°C  "
        f"COP {res.meas['COP'][-1]:.2f}"
    )


if __name__ == "__main__":
    main()
