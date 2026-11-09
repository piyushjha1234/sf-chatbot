"""Turns stored chunks into Qdrant points; incremental upserts and blue/green rebuilds."""

from __future__ import annotations

import json
import logging
from typing import Any

from sfrag.core.ids import point_id
from sfrag.core.models import Classification
from sfrag.core.settings import Settings
from sfrag.index.embedder import Embedder
from sfrag.index.store import VectorStore
from sfrag.storage import repo
from sfrag.storage.db import Database, now

log = logging.getLogger(__name__)


def payload_for(row: dict[str, Any], settings: Settings) -> dict[str, Any]:
    cls = Classification.model_validate_json(row["classification_json"])
    heading_path = row["heading_path"] if "heading_path" in row else json.loads(row["heading_path_json"])
    acl = row["acl"] if "acl" in row else json.loads(row["acl_json"])
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "parent_id": row["parent_id"],
        "ordinal": row["ordinal"],
        "url": row["origin"],
        "title": row["title"],
        "heading_path": heading_path,
        "heading": " › ".join(heading_path[1:]),
        "text": (row["context_prefix"] or "") + row["body"],
        "page": row["page"],
        "group_id": row["group_id"],
        "source_id": row["source_id"],
        "authority_tier": row["authority_tier"],
        "acl": acl,
        "quarantined": bool(row["quarantined"]),
        "products": cls.products,
        "categories": [c.value for c in cls.categories],
        "personas": [p.value for p in cls.personas],
        "intents": [i.value for i in cls.intents],
        "content_type": cls.content_type.value,
        "release": cls.release,
        "api_version": cls.api_version,
        "lifecycle": cls.lifecycle.value,
        "identifiers": cls.identifiers,
        "doc_version": row["version"],
        "pipeline_version": settings.pipeline_version,
        "embedding_model_version": settings.embedding_model_version,
    }


class Indexer:
    def __init__(self, settings: Settings, db: Database, store: VectorStore, embedder: Embedder) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.embedder = embedder

    def vector_params(self) -> tuple[int, bool]:
        """Collection shape from config (no model load needed): dense dim, IDF on sparse vectors."""
        return self.settings.embedding.dense_dim, self.settings.embedding.backend == "hashing"

    def live(self) -> str:
        name = self.store.ensure_live(*self.vector_params())
        self.db.execute(
            "INSERT OR IGNORE INTO index_generations(name, embedding_model_version, pipeline_version, state,"
            " created_at, activated_at) VALUES (?,?,?,?,?,?)",
            (name, self.settings.embedding_model_version, self.settings.pipeline_version, "live", now(), now()),
        )
        return name

    def index_doc(self, doc_id: str, removed_chunk_ids: list[str], collection: str | None = None) -> int:
        coll = collection or self.live()
        rows = self.db.query(
            "SELECT c.*, d.origin, d.title, d.source_id, d.authority_tier, d.acl_json, d.quarantined, d.version"
            " FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE c.doc_id=? ORDER BY c.ordinal",
            (doc_id,),
        )
        self.store.delete_points(coll, [point_id(c) for c in removed_chunk_ids])
        return self._upsert_rows(coll, [dict(r) for r in rows])

    def remove_doc(self, doc_id: str) -> None:
        coll = self.store.live_collection()
        if coll:
            self.store.delete_doc(coll, doc_id)

    def _upsert_rows(self, coll: str, rows: list[dict[str, Any]]) -> int:
        bs = max(1, self.settings.embedding.batch_size)
        n = 0
        for i in range(0, len(rows), bs * 4):
            batch = rows[i : i + bs * 4]
            embs = self.embedder.embed_documents([r["embed_text"] for r in batch])
            self.store.upsert(
                coll,
                [(point_id(r["chunk_id"]), e, payload_for(r, self.settings)) for r, e in zip(batch, embs, strict=True)],
            )
            n += len(batch)
        return n

    def rebuild(self, keep_previous: bool = True, min_ratio: float = 0.98) -> dict[str, Any]:
        """Blue/green: build a fresh generation from SQLite, validate, swap the alias, keep N-1 for rollback."""
        name = self.store.next_generation_name()
        self.db.execute(
            "INSERT INTO index_generations(name, embedding_model_version, pipeline_version, state, created_at)"
            " VALUES (?,?,?,?,?)",
            (name, self.settings.embedding_model_version, self.settings.pipeline_version, "building", now()),
        )
        self.store.create_collection(name, *self.vector_params())
        expected = 0
        buf: list[dict[str, Any]] = []
        for row in repo.iter_indexable_chunks(self.db):
            row["heading_path"] = json.loads(row["heading_path_json"])
            buf.append(row)
            expected += 1
            if len(buf) >= 512:
                self._upsert_rows(name, buf)
                buf = []
        if buf:
            self._upsert_rows(name, buf)
        got = self.store.count(name)
        if expected and got < expected * min_ratio:
            self.db.execute("UPDATE index_generations SET state='failed', point_count=? WHERE name=?", (got, name))
            self.store.drop_collection(name)
            raise RuntimeError(f"rebuild validation failed: {got}/{expected} points")
        old = self.store.swap_alias(name)
        self.db.execute(
            "UPDATE index_generations SET state='live', point_count=?, activated_at=? WHERE name=?", (got, now(), name)
        )
        if old:
            self.db.execute("UPDATE index_generations SET state='retired' WHERE name=?", (old,))
            # retain exactly one previous generation for rollback
            for r in self.db.query("SELECT name FROM index_generations WHERE state='retired' AND name<>?", (old,)):
                self.store.drop_collection(r["name"])
                self.db.execute("UPDATE index_generations SET state='dropped' WHERE name=?", (r["name"],))
            if not keep_previous:
                self.store.drop_collection(old)
        self.db.execute("DELETE FROM answer_cache")
        log.info("index rebuilt", extra={"generation": name, "points": got, "previous": old})
        return {"generation": name, "points": got, "previous": old}

    def rollback(self) -> str:
        row = self.db.one(
            "SELECT name FROM index_generations WHERE state='retired' ORDER BY activated_at DESC, "
            "created_at DESC LIMIT 1"
        )
        if not row or not self.store.client.collection_exists(row["name"]):
            raise RuntimeError("no previous generation available for rollback")
        current = self.store.swap_alias(row["name"])
        self.db.execute("UPDATE index_generations SET state='live', activated_at=? WHERE name=?", (now(), row["name"]))
        if current:
            self.db.execute("UPDATE index_generations SET state='retired' WHERE name=?", (current,))
        self.db.execute("DELETE FROM answer_cache")
        return str(row["name"])
