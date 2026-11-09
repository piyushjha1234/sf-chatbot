"""Stable, content-derived identifiers."""

from __future__ import annotations

import hashlib
import uuid

_NAMESPACE = uuid.UUID("6f1c3a1e-5d0b-4c43-9b51-8a7d2f0e9c11")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def short_hash(*parts: str, length: int = 24) -> str:
    return sha256_text("\x1f".join(parts))[:length]


def doc_id_for(origin: str) -> str:
    return "d_" + short_hash(origin)


def parent_id_for(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}:p{ordinal:04d}"


def chunk_id_for(doc_id: str, heading_path: list[str], ordinal: int) -> str:
    return "c_" + short_hash(doc_id, " > ".join(heading_path), str(ordinal))


def point_id(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned ints; derive a UUIDv5 from the chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))
