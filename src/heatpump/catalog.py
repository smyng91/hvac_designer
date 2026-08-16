"""Load a user equipment catalog. Nothing is invented here.

A catalog is a JSON list of items the user (or a cited paper) supplied.
The example file in ``data/catalog/`` lists only the published Lee 2021
AHRI 540 compressor map — not fictional product SKUs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogItem:
    id: str
    kind: str
    path: Path
    citation: str
    extra: dict


@dataclass(frozen=True)
class Catalog:
    citation: str
    items: tuple[CatalogItem, ...]
    path: Path

    def by_id(self, item_id: str) -> CatalogItem:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(f"{item_id!r} is not in {self.path}; known: {[i.id for i in self.items]}")

    def of_kind(self, kind: str) -> tuple[CatalogItem, ...]:
        return tuple(it for it in self.items if it.kind == kind)


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    for cand in (base / p, base.parent / p, Path.cwd() / p):
        if cand.exists():
            return cand.resolve()
    return (base / p).resolve()


def load_catalog(path: str | Path) -> Catalog:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cite = str(raw.get("citation") or "").strip()
    if not cite:
        raise ValueError(f"{path}: catalog must name its source in 'citation' (no anonymous SKU list)")
    items = []
    for rec in raw.get("items") or []:
        iid = str(rec.get("id") or "").strip()
        kind = str(rec.get("kind") or "").strip()
        rel = rec.get("path")
        item_cite = str(rec.get("citation") or cite).strip()
        if not iid or not kind or not rel:
            raise ValueError(f"{path}: each item needs id, kind, and path")
        if not item_cite:
            raise ValueError(f"{path}: item {iid!r} needs a citation")
        extra = {k: v for k, v in rec.items() if k not in {"id", "kind", "path", "citation"}}
        items.append(
            CatalogItem(
                id=iid,
                kind=kind,
                path=_resolve(path.parent, str(rel)),
                citation=item_cite,
                extra=extra,
            )
        )
    if not items:
        raise ValueError(f"{path}: catalog 'items' is empty")
    return Catalog(citation=cite, items=tuple(items), path=path)


def default_example_catalog() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "catalog" / "lee2021_compressor.json"
