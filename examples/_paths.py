"""Repo paths for the example cases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = ROOT / "output"
WEATHER_COOL = Path(__file__).resolve().parent / "weather_cooling.csv"
WEATHER_HEAT = Path(__file__).resolve().parent / "weather_heating.csv"
WEATHER_REVERSE = Path(__file__).resolve().parent / "weather_reverse.csv"
HOUR_S = 3600.0


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def sim_horizon(t_final: float, *, qss: bool = False) -> dict:
    """``simulate`` kwargs: 1 h and longer use QSS."""
    use_qss = bool(qss) or t_final >= HOUR_S
    if t_final >= 7200.0:
        rec = 60.0
    elif use_qss:
        rec = 30.0
    elif t_final <= 180.0:
        rec = 5.0
    else:
        rec = 10.0
    return {
        "t_final": float(t_final),
        "record_dt": rec,
        "reduction": "qss" if use_qss else "full",
    }
