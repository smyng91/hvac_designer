"""GitHub-flavored math and wiki page generation.

GitHub renders ``$`...`$`` (inline) and fenced ``math`` blocks (display).
It does not treat LaTeX ``\\(...\\)`` / ``\\[...\\]`` as math in Markdown,
so those delimiters show as raw text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/smyng91/hvac_designer"

# (source relative to ROOT, wiki filename, optional banner source path)
WIKI_PAGES: list[tuple[str, str, str | None]] = [
    ("README.md", "Home.md", None),
    ("docs/quickstart.md", "Quick start.md", "docs/quickstart.md"),
    ("docs/model.md", "Model.md", "docs/model.md"),
    ("examples/README.md", "Examples.md", "examples/README.md"),
    ("validation/README.md", "Validation.md", "validation/README.md"),
    ("validation/data/SOURCES.md", "Validation sources.md", "validation/data/SOURCES.md"),
    ("data/README.md", "Data.md", "data/README.md"),
    ("data/maps/SOURCES.md", "Compressor maps.md", "data/maps/SOURCES.md"),
]

# Markdown link targets → wiki page (spaces become hyphens in wiki URLs).
_WIKI_HREF = {
    "docs/quickstart.md": "Quick-start",
    "quickstart.md": "Quick-start",
    "../docs/quickstart.md": "Quick-start",
    "docs/model.md": "Model",
    "model.md": "Model",
    "../docs/model.md": "Model",
    "../../docs/model.md": "Model",
    "README.md": "Home",
    "../README.md": "Home",
    "examples/README.md": "Examples",
    "../examples/README.md": "Examples",
    "validation/README.md": "Validation",
    "../validation/README.md": "Validation",
    "../validation/": "Validation",
    "validation/data/SOURCES.md": "Validation-sources",
    "../validation/data/SOURCES.md": "Validation-sources",
    "data/README.md": "Data",
    "../data/README.md": "Data",
    "data/maps/SOURCES.md": "Compressor-maps",
    "maps/SOURCES.md": "Compressor-maps",
    "../data/maps/SOURCES.md": "Compressor-maps",
    "../../../data/maps/SOURCES.md": "Compressor-maps",
    "LICENSE": f"{REPO}/blob/main/LICENSE",
}

_FENCE = re.compile(r"(```[\s\S]*?```)")
_DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)
_INLINE = re.compile(r"\\\((.*?)\\\)", re.S)
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

_MATH_MD = [
    ROOT / "docs" / "model.md",
    ROOT / "docs" / "quickstart.md",
    ROOT / "examples" / "README.md",
    ROOT / "data" / "maps" / "SOURCES.md",
    ROOT / "validation" / "README.md",
]


def _to_github_math(chunk: str) -> str:
    def display(m: re.Match[str]) -> str:
        body = m.group(1).strip("\n")
        return f"\n\n```math\n{body}\n```\n\n"

    chunk = _DISPLAY.sub(display, chunk)
    chunk = _INLINE.sub(lambda m: f"$`{m.group(1)}`$", chunk)
    return re.sub(r"\n{3,}", "\n\n", chunk)


def to_github_math(text: str) -> str:
    parts = _FENCE.split(text)
    return "".join(p if i % 2 else _to_github_math(p) for i, p in enumerate(parts))


def leftover_tex_delimiters(text: str) -> int:
    parts = _FENCE.split(text)
    n = 0
    for i, p in enumerate(parts):
        if i % 2:
            continue
        n += p.count(r"\(") + p.count(r"\[")
    return n


def convert_repo_docs() -> None:
    for path in _MATH_MD:
        text = path.read_text()
        converted = to_github_math(text)
        leftover = leftover_tex_delimiters(converted)
        if leftover:
            raise SystemExit(f"{path}: {leftover} TeX delimiters remain")
        if converted != text:
            path.write_text(converted)


def _rewrite_href(href: str) -> str:
    href = href.strip()
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    path, frag = href, ""
    if "#" in href:
        path, frag = href.split("#", 1)
        frag = "#" + frag
    mapped = _WIKI_HREF.get(path)
    if mapped:
        return mapped + frag
    if path.startswith(("docs/", "examples/", "validation/", "data/", "src/", "output/")):
        return f"{REPO}/blob/main/{path}{frag}"
    if path.startswith("../"):
        rel = path
        while rel.startswith("../"):
            rel = rel[3:]
        if rel:
            return f"{REPO}/blob/main/{rel}{frag}"
    return href


def rewrite_links(text: str) -> str:
    parts = _FENCE.split(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)
        else:
            out.append(_MD_LINK.sub(lambda m: f"[{m.group(1)}]({_rewrite_href(m.group(2))})", part))
    return "".join(out)


def _banner(src: str) -> str:
    return (
        f"> Published from [`{src}`]({REPO}/blob/main/{src}). "
        f"Edit that file in the repository.\n\n"
    )


_SIDEBAR = """**Documentation**

* [Home](Home)
* [Quick start](Quick-start)
* [Model](Model)
* [Examples](Examples)
* [Validation](Validation)
* [Validation sources](Validation-sources)
* [Data](Data)
* [Compressor maps](Compressor-maps)

[Repository](https://github.com/smyng91/hvac_designer)
"""


def build_wiki(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for src_rel, wiki_name, banner_src in WIKI_PAGES:
        text = (ROOT / src_rel).read_text()
        text = rewrite_links(text)
        if banner_src:
            text = _banner(banner_src) + text
        if wiki_name == "Home.md":
            for a, b in (
                (
                    "[docs/quickstart.md](Quick-start)",
                    f"[docs/quickstart.md]({REPO}/blob/main/docs/quickstart.md)",
                ),
                (
                    "[docs/model.md](Model)",
                    f"[docs/model.md]({REPO}/blob/main/docs/model.md)",
                ),
                (
                    "[examples/](Examples)",
                    f"[examples/]({REPO}/tree/main/examples)",
                ),
                (
                    "[validation/](Validation)",
                    f"[validation/]({REPO}/tree/main/validation)",
                ),
                (
                    "[data/](Data)",
                    f"[data/]({REPO}/tree/main/data)",
                ),
                (f"{REPO}/wiki/Quick-start", "Quick-start"),
                (f"{REPO}/wiki/Model", "Model"),
                (f"{REPO}/wiki/Examples", "Examples"),
                (f"{REPO}/wiki/Validation", "Validation"),
                (f"{REPO}/wiki/Data", "Data"),
            ):
                text = text.replace(a, b)
            text = (
                "> This wiki is the rendered documentation, including equations. "
                f"Edit the source files in the [repository]({REPO}).\n\n" + text
            )
        (out / wiki_name).write_text(text)
    (out / "_Sidebar.md").write_text(_SIDEBAR)
    (out / "_Footer.md").write_text(
        f"[{REPO.split('/')[-1]}]({REPO}) · source of truth is the repository\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--convert", action="store_true", help="rewrite repo Markdown math in place")
    p.add_argument("--wiki", type=Path, help="write wiki pages to this directory")
    args = p.parse_args(argv)
    if args.convert:
        convert_repo_docs()
    if args.wiki:
        build_wiki(args.wiki)
    if not args.convert and not args.wiki:
        p.error("pass --convert and/or --wiki DIR")


if __name__ == "__main__":
    main()
