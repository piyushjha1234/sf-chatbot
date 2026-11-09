"""Repository functions over the SQLite schema (documents, sections, chunks, urls, sources)."""

from __future__ import annotations

import json
from typing import Any

from sfrag.core.models import Chunk, Classification, DocumentRecord, Section
from sfrag.storage.db import Database, dumps, now

# --------------------------------------------------------------------------- sources


def upsert_source(db: Database, source_id: str, cfg: dict[str, Any]) -> None:
    with db.tx() as c:
        c.execute(
            "INSERT INTO sources(source_id, name, kind, authority_tier, permitted, retain_raw, config_json, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, kind=excluded.kind,"
            " authority_tier=excluded.authority_tier, permitted=excluded.permitted, retain_raw=excluded.retain_raw,"
            " config_json=excluded.config_json, updated_at=excluded.updated_at",
            (
                source_id,
                cfg.get("name", source_id),
                cfg.get("kind", "web"),
                int(cfg.get("authority_tier", 2)),
                1 if cfg.get("permitted", True) else 0,
                1 if cfg.get("retain_raw", True) else 0,
                dumps(cfg),
                now(),
            ),
        )


def get_source(db: Database, source_id: str) -> dict[str, Any] | None:
    row = db.one("SELECT * FROM sources WHERE source_id=?", (source_id,))
    if not row:
        return None
    cfg = json.loads(row["config_json"])
    cfg.update(
        source_id=row["source_id"],
        authority_tier=row["authority_tier"],
        permitted=bool(row["permitted"]),
        retain_raw=bool(row["retain_raw"]),
    )
    return cfg


def list_sources(db: Database) -> list[dict[str, Any]]:
    return [s for r in db.query("SELECT source_id FROM sources ORDER BY source_id") if (s := get_source(db, r[0]))]


# --------------------------------------------------------------------------- urls


def register_url(
    db: Database,
    url: str,
    source_id: str,
    status: str = "discovered",
    reason: str | None = None,
    depth: int = 0,
    discovered_from: str | None = None,
    sitemap_lastmod: str | None = None,
) -> bool:
    """Insert a URL if new. Returns True if newly registered. Updates sitemap lastmod when known."""
    with db.tx() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO urls(url, source_id, status, reason, depth, discovered_from, sitemap_lastmod,"
            " first_seen) VALUES (?,?,?,?,?,?,?,?)",
            (url, source_id, status, reason, depth, discovered_from, sitemap_lastmod, now()),
        )
        if cur.rowcount == 0 and sitemap_lastmod:
            c.execute("UPDATE urls SET sitemap_lastmod=? WHERE url=?", (sitemap_lastmod, url))
        return cur.rowcount == 1


def set_url_status(db: Database, url: str, status: str, reason: str | None = None, **fields: Any) -> None:
    cols = ["status=?", "reason=?"]
    vals: list[Any] = [status, reason]
    for k, v in fields.items():
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(url)
    db.execute(f"UPDATE urls SET {', '.join(cols)} WHERE url=?", tuple(vals))


def get_url(db: Database, url: str) -> dict[str, Any] | None:
    row = db.one("SELECT * FROM urls WHERE url=?", (url,))
    return dict(row) if row else None


# --------------------------------------------------------------------------- documents


def get_document(db: Database, doc_id: str) -> dict[str, Any] | None:
    row = db.one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
    if not row:
        return None
    d = dict(row)
    d["acl"] = json.loads(d.pop("acl_json"))
    d["classification"] = Classification.model_validate_json(d.pop("classification_json"))
    return d


def save_document(
    db: Database,
    rec: DocumentRecord,
    raw_hash: str | None,
    minhash: bytes | None,
    status: str,
    normalized_path: str | None,
) -> None:
    with db.tx() as c:
        c.execute(
            "INSERT INTO documents(doc_id, origin, source_id, title, norm_hash, version, authority_tier, acl_json,"
            " classification_json, pipeline_version, canonical_of, quarantined, minhash, status, normalized_path,"
            " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET"
            " origin=excluded.origin, source_id=excluded.source_id, title=excluded.title,"
            " norm_hash=excluded.norm_hash, version=excluded.version, authority_tier=excluded.authority_tier,"
            " acl_json=excluded.acl_json, classification_json=excluded.classification_json,"
            " pipeline_version=excluded.pipeline_version, canonical_of=excluded.canonical_of,"
            " quarantined=excluded.quarantined, minhash=excluded.minhash, status=excluded.status,"
            " normalized_path=excluded.normalized_path, updated_at=excluded.updated_at",
            (
                rec.doc_id,
                rec.origin,
                rec.source_id,
                rec.title,
                rec.norm_hash,
                rec.version,
                rec.authority_tier,
                dumps(rec.acl),
                rec.classification.model_dump_json(),
                rec.pipeline_version,
                rec.canonical_of,
                1 if rec.quarantined else 0,
                minhash,
                status,
                normalized_path,
                now(),
            ),
        )
        c.execute(
            "INSERT OR IGNORE INTO document_versions(doc_id, version, norm_hash, raw_hash, pipeline_version,"
            " created_at)"
            " VALUES (?,?,?,?,?,?)",
            (rec.doc_id, rec.version, rec.norm_hash, raw_hash, rec.pipeline_version, now()),
        )


def set_document_status(db: Database, doc_id: str, status: str, canonical_of: str | None = None) -> None:
    db.execute(
        "UPDATE documents SET status=?, canonical_of=?, updated_at=? WHERE doc_id=?",
        (status, canonical_of, now(), doc_id),
    )


def add_relationship(db: Database, src: str, dst: str, kind: str) -> None:
    db.execute("INSERT OR IGNORE INTO relationships(src, dst, kind) VALUES (?,?,?)", (src, dst, kind))


# --------------------------------------------------------------------------- sections & chunks


def replace_sections_and_chunks(
    db: Database, doc_id: str, sections: list[Section], chunks: list[Chunk], embed_texts: dict[str, str]
) -> list[str]:
    """Atomically replace a document's parents/children. Returns chunk ids that no longer exist."""
    with db.tx() as c:
        old = {r[0] for r in c.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()}
        c.execute("DELETE FROM sections WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        c.executemany(
            "INSERT INTO sections(parent_id, doc_id, ordinal, heading_path_json, text, token_count, page_start,"
            " page_end) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    s.parent_id,
                    s.doc_id,
                    s.ordinal,
                    json.dumps(s.heading_path),
                    s.text,
                    s.token_count,
                    s.page_start,
                    s.page_end,
                )
                for s in sections
            ],
        )
        c.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, parent_id, ordinal, heading_path_json, body, context_prefix,"
            " token_count, content_type, group_id, page, classification_json, embed_text)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    ch.chunk_id,
                    ch.doc_id,
                    ch.parent_id,
                    ch.ordinal,
                    json.dumps(ch.heading_path),
                    ch.body,
                    ch.context_prefix,
                    ch.token_count,
                    ch.content_type.value,
                    ch.group_id,
                    ch.page,
                    (ch.classification or Classification()).model_dump_json(),
                    embed_texts[ch.chunk_id],
                )
                for ch in chunks
            ],
        )
    return sorted(old - {ch.chunk_id for ch in chunks})


def delete_document_content(db: Database, doc_id: str) -> list[str]:
    with db.tx() as c:
        old = [r[0] for r in c.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()]
        c.execute("DELETE FROM sections WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    return old


def get_section(db: Database, parent_id: str) -> dict[str, Any] | None:
    row = db.one("SELECT * FROM sections WHERE parent_id=?", (parent_id,))
    if not row:
        return None
    d = dict(row)
    d["heading_path"] = json.loads(d.pop("heading_path_json"))
    return d


def get_chunks_for_parent(db: Database, parent_id: str) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM chunks WHERE parent_id=? ORDER BY ordinal", (parent_id,))
    out = []
    for r in rows:
        d = dict(r)
        d["heading_path"] = json.loads(d.pop("heading_path_json"))
        out.append(d)
    return out


def iter_indexable_chunks(db: Database, batch: int = 512):
    """Yield (chunk_row, doc_row) for every chunk of an active document, in stable order."""
    last = ""
    while True:
        rows = db.query(
            "SELECT c.*, d.origin, d.title, d.source_id, d.authority_tier, d.acl_json, d.quarantined, d.version"
            " FROM chunks c JOIN documents d ON d.doc_id=c.doc_id"
            " WHERE d.status='active' AND c.chunk_id>? ORDER BY c.chunk_id LIMIT ?",
            (last, batch),
        )
        if not rows:
            return
        for r in rows:
            yield dict(r)
        last = rows[-1]["chunk_id"]
