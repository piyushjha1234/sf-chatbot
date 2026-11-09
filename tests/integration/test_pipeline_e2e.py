from __future__ import annotations

from pathlib import Path

from sfrag.container import Container
from sfrag.core.ids import doc_id_for
from sfrag.ops.coverage import coverage_report
from sfrag.pipeline.orchestrator import Pipeline
from sfrag.sources.watcher import scan
from sfrag.storage import repo
from tests.conftest import D360, FakeWeb


def _run(c: Container) -> dict[str, int]:
    return Pipeline(c).run_worker(concurrency=2, drain=True)


def _crawl(c: Container) -> None:
    c.queue.enqueue("discover_source", {"source_id": "sf_dev_data360"}, idempotency_key="discover:1")
    _run(c)


def test_discovery_fetch_index_and_reconciliation(web_container: Container) -> None:
    c = web_container
    _crawl(c)
    db = c.db
    urls = {r["url"]: dict(r) for r in db.query("SELECT * FROM urls")}
    # sitemap urls + start url + followed link are fetched; out-of-scope excluded; 404 marked removed
    assert urls[f"{D360}/identity-resolution.html"]["status"] == "fetched"
    assert urls[f"{D360}/data-streams.html"]["status"] == "fetched"
    assert urls[f"{D360}/query-api.html"]["status"] == "fetched"
    assert urls[f"{D360}/missing.html"]["status"] == "removed"
    assert urls["https://developer.salesforce.com/docs/other/unrelated.html"]["status"] == "excluded"
    # near-duplicate mirror is linked to its canonical and not indexed
    mirror = repo.get_document(db, doc_id_for(f"{D360}/mirror/identity-resolution.html"))
    canon = repo.get_document(db, doc_id_for(f"{D360}/identity-resolution.html"))
    assert {mirror["status"], canon["status"]} == {"active", "duplicate"}
    rep = coverage_report(db)
    rec = rep["reconciliation"]["sf_dev_data360"]
    assert rec["pending"] == 0 and rec["terminal_pct"] == 100.0
    assert rep["documents_by_product"].get("data360", 0) >= 3
    assert c.store.count() == rep["indexed_chunks"] > 0


def test_refresh_uses_conditional_get_and_skips_unchanged(web_container: Container, fake_web: FakeWeb) -> None:
    c = web_container
    _crawl(c)
    before = c.store.count()
    url = f"{D360}/query-api.html"
    c.queue.enqueue("fetch_url", {"url": url, "source_id": "sf_dev_data360"}, idempotency_key="refetch-1")
    _run(c)
    assert fake_web.requests[-1].headers.get("if-none-match")
    assert repo.get_url(c.db, url)["status"] == "unchanged"
    # content change -> new version, re-chunked, index updated in place
    fake_web.pages[url] = fake_web.pages[url].replace(b"time limit", b"time limit of 120 seconds")
    c.queue.enqueue("fetch_url", {"url": url, "source_id": "sf_dev_data360"}, idempotency_key="refetch-2")
    _run(c)
    doc = repo.get_document(c.db, doc_id_for(url))
    assert doc["version"] == 2
    assert c.store.count() == before
    # page removed upstream -> tombstoned and dropped from index
    del fake_web.pages[url]
    c.queue.enqueue("fetch_url", {"url": url, "source_id": "sf_dev_data360"}, idempotency_key="refetch-3")
    _run(c)
    assert repo.get_document(c.db, doc_id_for(url))["status"] == "removed"
    assert c.store.count() < before


def test_user_files_are_private_and_unsupported_is_reported(container: Container) -> None:
    c = container
    inc = c.settings.incoming_dir
    (inc / "users" / "alice").mkdir(parents=True)
    (inc / "users" / "alice" / "notes.md").write_text(
        "# Alice private runbook\n\nZebra-quartz rotation procedure for the Data Cloud ingestion connector."
    )
    (inc / "public").mkdir()
    (inc / "public" / "bad.bin").write_bytes(b"\x00\x01\x02")
    assert scan(c.queue, inc) == 2
    stats = _run(c)
    assert stats["dlq"] == 1  # unsupported format is explicit, not silent
    dlq = c.queue.dlq()
    assert "UnsupportedFormat" in dlq[0]["last_error"] or "no parser" in dlq[0]["last_error"]

    engine = c.engine
    _, _, as_alice, _ = engine.gather("zebra-quartz rotation procedure", [], *_pd(), ["user:alice"])
    _, _, as_bob, _ = engine.gather("zebra-quartz rotation procedure", [], *_pd(), ["user:bob"])
    assert any("Alice private runbook" in x.payload["title"] for x in as_alice)
    assert not any("Alice private runbook" in x.payload["title"] for x in as_bob)
    # idempotent re-scan does not enqueue unchanged files again
    assert scan(c.queue, inc) == 0


def test_blue_green_rebuild_and_rollback(web_container: Container) -> None:
    c = web_container
    _crawl(c)
    first = c.store.live_collection()
    n = c.store.count()
    res = c.indexer.rebuild()
    assert res["previous"] == first and res["points"] == n
    assert c.store.live_collection() == res["generation"] != first
    assert c.indexer.rollback() == first
    assert c.store.live_collection() == first and c.store.count() == n


def test_worker_crash_recovery(web_container: Container) -> None:
    c = web_container
    c.queue.lease_seconds = 1
    c.queue.enqueue(
        "fetch_url", {"url": f"{D360}/query-api.html", "source_id": "sf_dev_data360"}, idempotency_key="crash"
    )
    crashed = c.queue.lease("dead-worker")
    assert crashed is not None
    import time

    time.sleep(1.1)
    _run(c)
    assert repo.get_url(c.db, f"{D360}/query-api.html")["status"] == "fetched"
    assert c.queue.stats().get("done") == 1


def _pd():  # type: ignore[no-untyped-def]
    from sfrag.core.models import Depth, Persona

    return Persona.AUTO, Depth.STANDARD


assert Path
