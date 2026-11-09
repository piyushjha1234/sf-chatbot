"""SQLite system of record: registry, versions, lineage, sections/chunks, jobs, feedback.

WAL mode + one connection per thread. Schema migrations are ordered, idempotent SQL scripts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1 — core registry
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,              -- web | file
        authority_tier INTEGER NOT NULL DEFAULT 2,
        permitted INTEGER NOT NULL DEFAULT 1,
        retain_raw INTEGER NOT NULL DEFAULT 1,
        config_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS urls (
        url TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        status TEXT NOT NULL,            -- discovered | queued | fetched | unchanged | excluded | failed
                                         -- | needs_render | removed | redirected | external_review
        reason TEXT,
        depth INTEGER NOT NULL DEFAULT 0,
        discovered_from TEXT,
        sitemap_lastmod TEXT,
        etag TEXT,
        last_modified TEXT,
        raw_hash TEXT,
        doc_id TEXT,
        first_seen REAL NOT NULL,
        last_fetched REAL
    );
    CREATE INDEX IF NOT EXISTS ix_urls_source_status ON urls(source_id, status);
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        origin TEXT NOT NULL,
        source_id TEXT NOT NULL,
        title TEXT NOT NULL,
        norm_hash TEXT NOT NULL,
        version INTEGER NOT NULL,
        authority_tier INTEGER NOT NULL,
        acl_json TEXT NOT NULL,
        classification_json TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        canonical_of TEXT,
        quarantined INTEGER NOT NULL DEFAULT 0,
        minhash BLOB,
        status TEXT NOT NULL DEFAULT 'active',   -- active | duplicate | removed | failed
        normalized_path TEXT,
        updated_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_documents_source ON documents(source_id);
    CREATE TABLE IF NOT EXISTS document_versions (
        doc_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        norm_hash TEXT NOT NULL,
        raw_hash TEXT,
        pipeline_version TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (doc_id, version)
    );
    CREATE TABLE IF NOT EXISTS relationships (
        src TEXT NOT NULL,
        dst TEXT NOT NULL,
        kind TEXT NOT NULL,              -- links_to | duplicate_of | child_of | supersedes
        PRIMARY KEY (src, dst, kind)
    );
    CREATE TABLE IF NOT EXISTS sections (
        parent_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        heading_path_json TEXT NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        page_start INTEGER,
        page_end INTEGER
    );
    CREATE INDEX IF NOT EXISTS ix_sections_doc ON sections(doc_id);
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        heading_path_json TEXT NOT NULL,
        body TEXT NOT NULL,
        context_prefix TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        content_type TEXT NOT NULL,
        group_id TEXT,
        page INTEGER,
        classification_json TEXT NOT NULL,
        embed_text TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
    CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id);
    CREATE TABLE IF NOT EXISTS jobs (
        job_id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        idempotency_key TEXT UNIQUE,
        state TEXT NOT NULL,             -- pending | leased | done | failed | dlq
        priority INTEGER NOT NULL DEFAULT 100,
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL,
        lease_owner TEXT,
        lease_expires REAL,
        run_after REAL NOT NULL,
        last_error TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_jobs_ready ON jobs(state, run_after, priority);
    CREATE TABLE IF NOT EXISTS index_generations (
        name TEXT PRIMARY KEY,
        embedding_model_version TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        state TEXT NOT NULL,             -- building | live | retired | failed
        point_count INTEGER,
        created_at REAL NOT NULL,
        activated_at REAL
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        principal TEXT,
        question TEXT NOT NULL,
        answer TEXT,
        rating INTEGER NOT NULL,         -- +1 / -1
        reason TEXT,
        citations_json TEXT,
        kind TEXT NOT NULL DEFAULT 'rating',   -- rating | outdated
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS answer_cache (
        cache_key TEXT PRIMARY KEY,
        generation TEXT NOT NULL,
        answer_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS classification_cache (
        cache_key TEXT PRIMARY KEY,
        result_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """,
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.migrate()

    # ---- connection management ----------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """IMMEDIATE transaction: takes the write lock up front, avoiding upgrade deadlocks."""
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def migrate(self) -> None:
        c = self.conn()
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = c.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for i, script in enumerate(MIGRATIONS, start=1):
            if i <= current:
                continue
            c.executescript(f"BEGIN;\n{script}\nINSERT INTO schema_version(version) VALUES ({i});\nCOMMIT;")

    # ---- helpers --------------------------------------------------------------------
    def query(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> None:
        self.conn().execute(sql, params)


def now() -> float:
    return time.time()


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
