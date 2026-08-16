"""Repo Markdown must use GitHub math, not TeX \\( \\) / \\[ \\]."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"(```[\s\S]*?```)")
_DOCS = (
    ROOT / "docs" / "model.md",
    ROOT / "docs" / "quickstart.md",
    ROOT / "examples" / "README.md",
    ROOT / "data" / "maps" / "SOURCES.md",
    ROOT / "validation" / "README.md",
)


def _leftover_tex(text: str) -> int:
    n = 0
    for i, part in enumerate(_FENCE.split(text)):
        if i % 2 == 0:
            n += part.count(r"\(") + part.count(r"\[")
    return n


def test_docs_use_github_math_delimiters() -> None:
    for path in _DOCS:
        assert _leftover_tex(path.read_text()) == 0, path
