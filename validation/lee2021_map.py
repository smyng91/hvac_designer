#!/usr/bin/env python3
"""Lee et al. 2021 AHRI 540 map at published Table 4 (Te, Tc).

Compressor-map check, not a cabinet twin. Table 6 is not scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatpump.validation import compare_lee2021_map, default_maps_dir, default_results_dir


def main() -> None:
    path = default_maps_dir() / "lee2021_iop1180_012041.json"
    rep = compare_lee2021_map(path)
    src = rep["source"]
    print(src["citation"])
    print(f"  {src.get('doi', '')}  ·  {src.get('license', '')}")
    print(f"  coefficients match Table 5: {rep['coefficients_match_table5']}")
    print()
    print(f"{'Te °C':>7} {'Tc °C':>7} {'T_out °C':>9} {'W':>8} {'ṁ g/s':>8}")
    for r in rep["table4"]:
        print(
            f"{r['Te_C']:7.1f} {r['Tc_C']:7.1f} {r['T_out_C']:9.1f} "
            f"{r['power_W']:8.1f} {r['mdot_g_s']:8.2f}"
        )
    t5 = rep["table4_test5"]
    print()
    print(
        f"Table 4 test 5 (12.3 / 48.4 °C): {t5['power_W']:.1f} W, "
        f"{t5['mdot_g_s']:.2f} g/s"
    )
    for n in rep["notes"]:
        print(f"- {n}")
    dest = default_results_dir()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "lee2021_table4.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
