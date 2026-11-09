"""Coverage & reconciliation report: every discovered URL has a disposition; gaps are explicit."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sfrag.storage.db import Database


def coverage_report(db: Database) -> dict[str, Any]:
    url_status = db.query("SELECT source_id, status, COUNT(*) AS n FROM urls GROUP BY source_id, status")
    reasons = db.query(
        "SELECT source_id, status, COALESCE(reason,'') AS reason, COUNT(*) AS n FROM urls "
        "WHERE status IN ('excluded','failed','needs_render','removed','external_review') "
        "GROUP BY source_id, status, reason ORDER BY n DESC LIMIT 200"
    )
    docs = db.query("SELECT source_id, status, COUNT(*) AS n FROM documents GROUP BY source_id, status")
    chunks = db.one("SELECT COUNT(*) FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE d.status='active'")
    unclassified = db.one(
        "SELECT COUNT(*) FROM documents WHERE status='active' AND json_array_length(json_extract(classification_json,"
        " '$.products'))=0"
    )
    low_conf = db.one(
        "SELECT COUNT(*) FROM documents WHERE status='active' AND json_extract(classification_json,"
        " '$.product_confidence') < 0.3"
    )
    by_product = db.query(
        "SELECT j.value AS product, COUNT(*) AS n FROM documents d, json_each(json_extract(d.classification_json,"
        " '$.products')) j WHERE d.status='active' GROUP BY j.value ORDER BY n DESC"
    )
    jobs = db.query("SELECT state, kind, COUNT(*) AS n FROM jobs GROUP BY state, kind")
    gens = db.query(
        "SELECT name, state, point_count, embedding_model_version FROM index_generations ORDER BY created_at"
    )

    per_source: dict[str, dict[str, int]] = {}
    for r in url_status:
        per_source.setdefault(r["source_id"], {})[r["status"]] = r["n"]
    terminal = {
        "fetched",
        "unchanged",
        "excluded",
        "failed",
        "needs_render",
        "removed",
        "redirected",
        "external_review",
    }
    reconciliation = {}
    for sid, counts in per_source.items():
        total = sum(counts.values())
        done = sum(v for k, v in counts.items() if k in terminal)
        reconciliation[sid] = {
            "discovered": total,
            "terminal_disposition": done,
            "pending": total - done,
            "terminal_pct": round(100 * done / total, 1) if total else 0.0,
        }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reconciliation": reconciliation,
        "url_status": per_source,
        "exclusions_and_failures": [dict(r) for r in reasons],
        "documents": [dict(r) for r in docs],
        "indexed_chunks": chunks[0] if chunks else 0,
        "unclassified_documents": unclassified[0] if unclassified else 0,
        "low_confidence_documents": low_conf[0] if low_conf else 0,
        "documents_by_product": {r["product"]: r["n"] for r in by_product},
        "jobs": [dict(r) for r in jobs],
        "index_generations": [dict(r) for r in gens],
    }


def write_report(db: Database, reports_dir: Path) -> Path:
    rep = coverage_report(db)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"coverage-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(rep, indent=2))
    return path
