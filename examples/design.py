#!/usr/bin/env python3
"""Case: size a heat pump from stated conditions.

No transient. Writes the design package (markdown, JSON, capacity map)
to ``output/``. Default: R32, 5.5 kW heating at 0 °C outdoor / 20 °C zone.
Pass ``--weather`` to infer duty from a load record instead of ``--load``.
"""

from __future__ import annotations

import argparse

from _paths import out_dir
from heatpump import DesignRequest, TimeSeries, design_system


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="heating", choices=["heating", "cooling", "heat_pump"])
    p.add_argument("-r", "--refrigerant", default=None, help="default R32 heat / R410A cool")
    p.add_argument("--load", type=float, default=None, help="nameplate duty [W]")
    p.add_argument("--load-heat", type=float, default=None, help="heating duty [W] (heat_pump)")
    p.add_argument("--load-cool", type=float, default=None, help="cooling duty [W] (heat_pump)")
    p.add_argument("--load-tons", type=float, default=None, help="cooling tons (overrides --load-cool)")
    p.add_argument("--T-out", type=float, default=None, help="design outdoor [°C]")
    p.add_argument("--T-zone", type=float, default=None, help="zone setpoint [°C]")
    p.add_argument("--SH", type=float, default=6.0)
    p.add_argument("--SC", type=float, default=4.0)
    p.add_argument("--DT-evap", type=float, default=10.0)
    p.add_argument("--DT-cond", type=float, default=12.0)
    p.add_argument("--V-zone", type=float, default=None, help="zone volume [m³]; default 50")
    p.add_argument("--weather", default=None, help="CSV; duty = peak of this record")
    args = p.parse_args(argv)

    ts = TimeSeries.from_csv(args.weather) if args.weather else None
    if args.mode == "cooling":
        fluid = args.refrigerant or "R410A"
        T_zone = 24.0 if args.T_zone is None else args.T_zone
        T_out = 35.0 if args.T_out is None else args.T_out
        Q_heat, Q_cool = None, args.load
        if ts is None and Q_cool is None and args.load_tons is None:
            Q_cool = 6200.0
    elif args.mode == "heat_pump":
        fluid = args.refrigerant or "R32"
        T_zone = 20.0 if args.T_zone is None else args.T_zone
        T_out = 0.0 if args.T_out is None else args.T_out
        Q_heat = args.load_heat if args.load_heat is not None else args.load
        Q_cool = args.load_cool
        if ts is None:
            if Q_heat is None:
                Q_heat = 5500.0
            if Q_cool is None and args.load_tons is None:
                Q_cool = 6200.0
    else:
        fluid = args.refrigerant or "R32"
        T_zone = 20.0 if args.T_zone is None else args.T_zone
        T_out = 0.0 if args.T_out is None else args.T_out
        Q_heat, Q_cool = args.load, None
        if ts is None and Q_heat is None:
            Q_heat = 5500.0

    req = DesignRequest(
        refrigerant=fluid,
        mode=args.mode,
        T_zone=T_zone + 273.15,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        cooling_tons=args.load_tons,
        T_out_heat=T_out + 273.15,
        T_out_cool=(35.0 if args.T_out is None and args.mode == "heat_pump" else T_out) + 273.15,
        SH=args.SH,
        SC=args.SC,
        DT_evap=args.DT_evap,
        DT_cond=args.DT_cond,
        V_zone=args.V_zone,
        timeseries=ts,
    )
    sys = design_system(req)
    print(sys.summary())
    dest = out_dir()
    pkg = sys.as_report()
    md = dest / "design.md"
    pkg.write(md)
    mapped = pkg.plot(dest / "design_map.png")
    print(f"wrote {md} and {md.with_suffix('.json')}")
    if mapped is not None:
        print(f"wrote {mapped}")


if __name__ == "__main__":
    main()
