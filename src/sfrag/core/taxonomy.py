"""Taxonomy loading and alias normalisation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Product:
    key: str
    label: str
    aliases: tuple[str, ...]
    url_patterns: tuple[str, ...]
    identifier_patterns: tuple[re.Pattern[str], ...]
    alias_regex: re.Pattern[str] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Taxonomy:
    version: int
    products: dict[str, Product]
    categories: dict[str, tuple[str, ...]]
    category_personas: dict[str, tuple[str, ...]]
    category_intents: dict[str, tuple[str, ...]]

    def product_label(self, key: str) -> str:
        p = self.products.get(key)
        return p.label if p else key

    def normalize_product(self, value: str) -> str | None:
        """Map a free-text product mention (alias, label or key) to a product key."""
        v = value.strip().lower()
        for key, p in self.products.items():
            if v == key or v == p.label.lower() or any(v == a.lower() for a in p.aliases):
                return key
        return None


def _alias_regex(aliases: list[str]) -> re.Pattern[str] | None:
    if not aliases:
        return None
    # longest first so "Bulk API 2.0" wins over "Bulk API"
    parts = sorted({re.escape(a) for a in aliases}, key=len, reverse=True)
    return re.compile(r"(?<![\w])(" + "|".join(parts) + r")(?![\w])", re.IGNORECASE)


def load_taxonomy(path: Path) -> Taxonomy:
    data = yaml.safe_load(path.read_text())
    products = {}
    for key, p in data["products"].items():
        aliases = [p["label"], *p.get("aliases", [])]
        products[key] = Product(
            key=key,
            label=p["label"],
            aliases=tuple(aliases),
            url_patterns=tuple(s.lower() for s in p.get("url_patterns", [])),
            identifier_patterns=tuple(re.compile(x) for x in p.get("identifier_patterns", [])),
            alias_regex=_alias_regex(aliases),
        )
    return Taxonomy(
        version=int(data.get("version", 1)),
        products=products,
        categories={k: tuple(v) for k, v in data["categories"].items()},
        category_personas={k: tuple(v) for k, v in data["category_personas"].items()},
        category_intents={k: tuple(v) for k, v in data["category_intents"].items()},
    )


@lru_cache(maxsize=4)
def get_taxonomy(path: str) -> Taxonomy:
    return load_taxonomy(Path(path))
