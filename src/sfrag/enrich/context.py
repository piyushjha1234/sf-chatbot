"""Contextual chunk headers (embedding-time only; display text stays unchanged)."""

from __future__ import annotations

from sfrag.core.models import Chunk, Classification
from sfrag.core.taxonomy import Taxonomy


def contextual_header(chunk: Chunk, cls: Classification, doc_title: str, taxonomy: Taxonomy) -> str:
    product = taxonomy.product_label(cls.products[0]) if cls.products else ""
    path = [h for h in chunk.heading_path if h and h != doc_title]
    parts = [p for p in [product, doc_title, " › ".join(path)] if p]
    tail = [x for x in [cls.release, cls.content_type.value if cls.content_type else None] if x]
    head = " › ".join(parts)
    if tail:
        head += " | " + " | ".join(tail)
    return head


def embed_text(chunk: Chunk, cls: Classification, doc_title: str, taxonomy: Taxonomy) -> str:
    return f"{contextual_header(chunk, cls, doc_title, taxonomy)}\n\n{chunk.text}"
