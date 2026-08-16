#!/usr/bin/env python3
"""Run the four example cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = (
    ("heating", ["heating.py"]),
    ("cooling", ["cooling.py"]),
    ("reverse", ["reverse.py"]),
    ("weather cooling", ["weather.py", "--mode", "cooling"]),
)


def main() -> None:
    for title, argv in STEPS:
        print(f"\n=== {title} ===\n")
        subprocess.run([sys.executable, str(HERE / argv[0]), *argv[1:]], check=True)


if __name__ == "__main__":
    main()
