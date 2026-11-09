"""Single ingestion orchestrator: every source type (sitemaps, URLs, watched files, CLI/API uploads) becomes
jobs on the same durable queue and flows through the same processing path."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sfrag.container import Container
from sfrag.core.ids import doc_id_for, sha256_text
from sfrag.core.models import DocumentRecord
from sfrag.enrich.context import embed_text
from sfrag.pipeline.dedup import minhash_for, prefer_new, to_bytes
from sfrag.pipeline.extract import (
    ExtractionError,
    NeedsRender,
    UnsupportedFormat,
    detect_mime,
    parse_document,
)
from sfrag.pipeline.queue import Job
from sfrag.pipeline.security import injection_signals
from sfrag.sources.discovery import Scope, is_salesforce_reference, normalize_url, parse_sitemap
from sfrag.sources.fetch import FetchError
from sfrag.storage import repo

log = logging.getLogger(__name__)

EXTERNAL_SOURCE = "_external_review"


class NonRetryable(Exception):
    pass


def _day_bucket() -> int:
    return int(time.time() // 86400)


class Pipeline:
    def __init__(self, c: Container) -> None:
        self.c = c
        self.s = c.settings
        self.handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "discover_source": self.discover_source,
            "fetch_url": self.fetch_url,
            "ingest_file": self.ingest_file,
            "process_raw": self.process_raw,
            "delete_file": self.delete_file,
        }

    # ================================================================== discovery
    def discover_source(self, p: dict[str, Any]) -> None:
        src = repo.get_source(self.c.db, p["source_id"])
        if not src:
            raise NonRetryable(f"unknown source {p['source_id']}")
        if not src.get("permitted", True):
            log.info("source not permitted; skipping", extra={"source_id": src["source_id"]})
            return
        scope = Scope.from_config(src)
        sitemaps = list(src.get("sitemaps", []))
        if src.get("use_robots_sitemaps", True):
            for u in src.get("start_urls", []):
                sitemaps += self.c.fetcher.sitemaps_from_robots(u)
        seen: set[str] = set()
        queue = list(dict.fromkeys(sitemaps))
        n_urls = 0
        while queue:
            sm = queue.pop(0)
            if sm in seen or len(seen) > 5000:
                continue
            seen.add(sm)
            try:
                r = self.c.fetcher.fetch(sm)
            except FetchError as e:
                log.warning("sitemap fetch failed", extra={"sitemap": sm, "err": str(e)})
                continue
            if r.status != 200:
                continue
            for entry in parse_sitemap(r.content):
                if entry.is_index:
                    queue.append(entry.loc)
                else:
                    n_urls += self._register(src, scope, entry.loc, 0, sm, entry.lastmod)
        for u in src.get("start_urls", []):
            n_urls += self._register(src, scope, u, 0, "start_url", None)
        log.info("discovery done", extra={"source_id": src["source_id"], "sitemaps": len(seen), "urls": n_urls})

    def _register(
        self, src: dict[str, Any], scope: Scope, url: str, depth: int, discovered_from: str, lastmod: str | None
    ) -> int:
        url = normalize_url(url)
        ok, reason = scope.check(url)
        if not ok:
            if reason == "out_of_scope_host" and is_salesforce_reference(url, src.get("reference_domains", [])):
                repo.register_url(
                    self.c.db,
                    url,
                    EXTERNAL_SOURCE,
                    status="external_review",
                    reason=f"referenced by {src['source_id']}",
                    depth=depth,
                    discovered_from=discovered_from,
                )
            elif reason != "out_of_scope_host":
                repo.register_url(
                    self.c.db,
                    url,
                    src["source_id"],
                    status="excluded",
                    reason=reason,
                    depth=depth,
                    discovered_from=discovered_from,
                )
            return 0
        new = repo.register_url(
            self.c.db, url, src["source_id"], depth=depth, discovered_from=discovered_from, sitemap_lastmod=lastmod
        )
        row = repo.get_url(self.c.db, url) or {}
        stale = (
            new
            or row.get("status") in ("discovered", "failed")
            or (
                lastmod is not None
                and row.get("last_fetched") is not None
                and lastmod > time.strftime("%Y-%m-%d", time.gmtime(row["last_fetched"]))
            )
        )
        if stale:
            self.c.queue.enqueue(
                "fetch_url",
                {"url": url, "source_id": src["source_id"], "depth": depth},
                idempotency_key=f"fetch:{url}:{_day_bucket()}",
            )
            repo.set_url_status(self.c.db, url, "queued")
            return 1
        return 0

    # ================================================================== fetching
    def fetch_url(self, p: dict[str, Any]) -> None:
        url, source_id, depth = p["url"], p["source_id"], int(p.get("depth", 0))
        src = repo.get_source(self.c.db, source_id)
        if not src:
            raise NonRetryable(f"unknown source {source_id}")
        repo.register_url(self.c.db, url, source_id, status="queued", depth=depth, discovered_from="job")
        if not self.c.fetcher.allowed(url):
            repo.set_url_status(self.c.db, url, "excluded", "robots_disallowed")
            return
        prev = repo.get_url(self.c.db, url) or {}
        try:
            r = self.c.fetcher.fetch(url, etag=prev.get("etag"), last_modified=prev.get("last_modified"))
        except FetchError as e:
            repo.set_url_status(self.c.db, url, "failed", str(e)[:500])
            if e.retryable:
                raise
            raise NonRetryable(str(e)) from e
        t = time.time()
        if r.not_modified:
            repo.set_url_status(self.c.db, url, "unchanged", None, last_fetched=t)
            return
        if r.status in (404, 410):
            repo.set_url_status(self.c.db, url, "removed", f"HTTP {r.status}", last_fetched=t)
            if prev.get("doc_id"):
                self._tombstone(prev["doc_id"])
            return
        if r.status >= 400:
            repo.set_url_status(self.c.db, url, "failed", f"HTTP {r.status}", last_fetched=t)
            raise NonRetryable(f"HTTP {r.status}")
        final = normalize_url(r.url)
        if r.redirected and final != url:
            repo.set_url_status(self.c.db, url, "redirected", final, last_fetched=t)
            scope = Scope.from_config(src)
            ok, _ = scope.check(final)
            if not ok:
                return
            repo.register_url(self.c.db, final, source_id, status="queued", depth=depth, discovered_from=url)
            url = final
        meta = {"url": url, "content_type": r.content_type, "fetched_at": t}
        raw_hash = self.c.raw.put(r.content, meta)
        repo.set_url_status(
            self.c.db,
            url,
            "fetched",
            None,
            etag=r.etag,
            last_modified=r.last_modified,
            raw_hash=raw_hash,
            last_fetched=t,
        )
        self.process_raw(
            {
                "raw_hash": raw_hash,
                "origin": url,
                "source_id": source_id,
                "mime": r.content_type,
                "acl": ["public"],
                "depth": depth,
                "url": url,
            }
        )

    # ================================================================== files
    def ingest_file(self, p: dict[str, Any]) -> None:
        path = Path(p["path"])
        if not path.exists():
            raise NonRetryable(f"file vanished: {path}")
        data = path.read_bytes()
        raw_hash = self.c.raw.put(data, {"path": str(path), "rel": p["rel"], "owner": p.get("owner")})
        repo.upsert_source(
            self.c.db,
            "user_files",
            {"name": "User-provided files", "kind": "file", "authority_tier": 3, "retain_raw": True},
        )
        self.process_raw(
            {
                "raw_hash": raw_hash,
                "origin": f"file://{p['rel']}",
                "source_id": "user_files",
                "mime": None,
                "acl": p.get("acl", ["group:internal"]),
                "name": path.name,
            }
        )

    def delete_file(self, p: dict[str, Any]) -> None:
        self._tombstone(doc_id_for(f"file://{p['rel']}"))

    def _tombstone(self, doc_id: str) -> None:
        repo.delete_document_content(self.c.db, doc_id)
        self.c.indexer.remove_doc(doc_id)
        self.c.dedup.remove(doc_id)
        repo.set_document_status(self.c.db, doc_id, "removed")
        self.c.db.execute("DELETE FROM answer_cache")

    # ================================================================== processing
    def process_raw(self, p: dict[str, Any]) -> None:
        doc_id = doc_id_for(p["origin"])
        try:
            self._process_raw(p)
        except BaseException:
            # release a dedup reservation made by this attempt unless the doc is already active
            row = repo.get_document(self.c.db, doc_id)
            if not row or row["status"] != "active":
                self.c.dedup.remove(doc_id)
            raise

    def _process_raw(self, p: dict[str, Any]) -> None:
        db = self.c.db
        origin, source_id = p["origin"], p["source_id"]
        url = p.get("url")
        data = self.c.raw.get(p["raw_hash"])
        mime = detect_mime(p.get("name") or origin, data, p.get("mime"))
        try:
            doc = parse_document(data, origin, mime)
        except NeedsRender as e:
            if url:
                repo.set_url_status(db, url, "needs_render", str(e))
            return
        except (UnsupportedFormat, ExtractionError) as e:
            if url:
                repo.set_url_status(db, url, "failed", f"{type(e).__name__}: {e}")
            raise NonRetryable(f"{type(e).__name__}: {e}") from e

        src = repo.get_source(db, source_id) or {"authority_tier": 3, "retain_raw": True, "kind": "file"}
        doc_id = doc_id_for(origin)
        markdown = doc.markdown()
        norm_hash = sha256_text(markdown)
        existing = repo.get_document(db, doc_id)
        acl = list(p.get("acl") or ["public"])
        if (
            existing
            and existing["norm_hash"] == norm_hash
            and existing["pipeline_version"] == self.s.pipeline_version
            and existing["status"] in ("active", "duplicate")
            and existing["acl"] == acl
        ):
            if url:
                repo.set_url_status(db, url, "fetched", "content_unchanged", doc_id=doc_id)
            self._follow_links(doc.links, src, p)
            return

        cls = self.c.classifier.classify_document(doc, norm_hash)
        allowed_locales = src.get("locales")
        if allowed_locales and cls.locale not in allowed_locales and cls.locale.split("-")[0] not in allowed_locales:
            if url:
                repo.set_url_status(db, url, "excluded", f"locale:{cls.locale}")
            return
        quarantined = False
        if src.get("kind") == "file" or int(src.get("authority_tier", 2)) >= 3:
            signals = injection_signals(markdown)
            quarantined = bool(signals)
            if signals:
                log.warning("prompt-injection signals; quarantined", extra={"doc_id": doc_id, "signals": signals})

        tier = int(src.get("authority_tier", 2))
        version = (existing["version"] + 1) if existing else 1
        rec = DocumentRecord(
            doc_id=doc_id,
            origin=origin,
            source_id=source_id,
            title=doc.title,
            norm_hash=norm_hash,
            version=version,
            authority_tier=tier,
            acl=acl,
            classification=cls,
            pipeline_version=self.s.pipeline_version,
            quarantined=quarantined,
        )

        # ---- near-duplicate handling (only among same-ACL public docs; private files never dedupe across owners)
        mh = minhash_for(markdown)
        if acl == ["public"]:
            dup_id = self.c.dedup.check_and_reserve(doc_id, mh)
            dup = repo.get_document(db, dup_id) if dup_id else None
            if dup_id and dup is None:
                # the canonical is reserved by a concurrent worker but not saved yet: retry shortly
                raise RuntimeError(f"near-duplicate {dup_id} still in progress; retrying")
            if dup and dup["acl"] == acl:
                if prefer_new(tier, cls.locale, dup["authority_tier"], dup["classification"].locale):
                    repo.delete_document_content(db, dup["doc_id"])
                    self.c.indexer.remove_doc(dup["doc_id"])
                    self.c.dedup.remove(dup["doc_id"])
                    self.c.dedup.add(doc_id, mh)
                    repo.set_document_status(db, dup["doc_id"], "duplicate", canonical_of=doc_id)
                    repo.add_relationship(db, dup["doc_id"], doc_id, "duplicate_of")
                else:
                    rec.canonical_of = dup["doc_id"]
                    repo.save_document(db, rec, p["raw_hash"], None, "duplicate", None)
                    removed = repo.delete_document_content(db, doc_id)
                    if removed:
                        self.c.indexer.remove_doc(doc_id)
                    repo.add_relationship(db, doc_id, dup["doc_id"], "duplicate_of")
                    if url:
                        repo.set_url_status(db, url, "fetched", f"duplicate_of:{dup['doc_id']}", doc_id=doc_id)
                    self._follow_links(doc.links, src, p)
                    return

        sections, chunks = self.c.chunker.chunk(doc, doc_id)
        texts: dict[str, str] = {}
        for ch in chunks:
            ch.classification = self.c.rules.classify_chunk(ch, cls, origin, doc.title)
            texts[ch.chunk_id] = embed_text(ch, ch.classification, doc.title, self.c.taxonomy)
        npath = self.c.normalized.put(
            doc_id,
            version,
            markdown,
            {
                "doc": rec.model_dump(mode="json"),
                "breadcrumb": doc.breadcrumb,
                "meta": doc.meta,
                "blocks": [b.model_dump(mode="json") for b in doc.blocks],
            },
        )
        repo.save_document(db, rec, p["raw_hash"], to_bytes(mh) if acl == ["public"] else None, "active", str(npath))
        removed = repo.replace_sections_and_chunks(db, doc_id, sections, chunks, texts)
        n = self.c.indexer.index_doc(doc_id, removed)
        if acl == ["public"]:
            self.c.dedup.add(doc_id, mh)
        db.execute("DELETE FROM answer_cache")
        if url:
            repo.set_url_status(db, url, "fetched", None, doc_id=doc_id)
        if not src.get("retain_raw", True):
            self.c.raw.delete(p["raw_hash"])
        log.info(
            "document indexed",
            extra={
                "doc_id": doc_id,
                "origin": origin,
                "version": version,
                "sections": len(sections),
                "chunks": n,
                "products": cls.products,
                "product_conf": cls.product_confidence,
            },
        )
        self._follow_links(doc.links, src, p)

    def _follow_links(self, links: list[str], src: dict[str, Any], p: dict[str, Any]) -> None:
        if src.get("kind") != "web" or not self.s.fetch.follow_links or not src.get("follow_links", True):
            return
        depth = int(p.get("depth", 0)) + 1
        if depth > int(src.get("max_link_depth", self.s.fetch.max_link_depth)):
            return
        scope = Scope.from_config(src)
        for link in dict.fromkeys(links):
            self._register(src, scope, link, depth, p.get("url") or p["origin"], None)

    # ================================================================== worker
    def handle(self, job: Job) -> None:
        handler = self.handlers.get(job.kind)
        if handler is None:
            raise NonRetryable(f"unknown job kind {job.kind}")
        handler(job.payload)

    def run_worker(
        self,
        concurrency: int = 2,
        stop: threading.Event | None = None,
        drain: bool = False,
        kinds: list[str] | None = None,
    ) -> dict[str, int]:
        """Run worker threads. drain=True exits once the queue has no ready jobs."""
        stop = stop or threading.Event()
        stats = {"done": 0, "failed": 0, "dlq": 0}
        lock = threading.Lock()

        def loop(idx: int) -> None:
            owner = self.c.queue.worker_id(str(idx))
            idle_since: float | None = None
            while not stop.is_set():
                job = self.c.queue.lease(owner, kinds)
                if job is None:
                    if drain:
                        idle_since = idle_since or time.time()
                        if time.time() - idle_since > 2.0 and not self._leased_by_others(owner):
                            return
                    time.sleep(self.s.queue.poll_interval_seconds)
                    continue
                idle_since = None
                hb_stop = threading.Event()
                hb = threading.Thread(target=self._heartbeat, args=(job, hb_stop), daemon=True)
                hb.start()
                try:
                    self.handle(job)
                    self.c.queue.complete(job)
                    with lock:
                        stats["done"] += 1
                except NonRetryable as e:
                    self.c.queue.fail(job, f"{e}", retryable=False)
                    with lock:
                        stats["dlq"] += 1
                    log.warning("job failed permanently", extra={"job_id": job.job_id, "kind": job.kind, "err": str(e)})
                except Exception as e:
                    state = self.c.queue.fail(job, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
                    with lock:
                        stats["failed" if state == "pending" else "dlq"] += 1
                    log.exception("job failed", extra={"job_id": job.job_id, "kind": job.kind, "state": state})
                finally:
                    hb_stop.set()

        threads = [
            threading.Thread(target=loop, args=(i,), daemon=True, name=f"worker-{i}")
            for i in range(max(1, concurrency))
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            stop.set()
            for t in threads:
                t.join(timeout=30)
        return stats

    def _leased_by_others(self, owner: str) -> bool:
        row = self.c.db.one(
            "SELECT COUNT(*) FROM jobs WHERE state='leased' AND lease_owner<>? AND lease_expires>?",
            (owner, time.time()),
        )
        ready = self.c.db.one("SELECT COUNT(*) FROM jobs WHERE state='pending' AND run_after<=?", (time.time(),))
        return bool((row and row[0]) or (ready and ready[0]))

    def _heartbeat(self, job: Job, stop: threading.Event) -> None:
        interval = max(5.0, self.s.queue.lease_seconds / 3)
        while not stop.wait(interval):
            if not self.c.queue.heartbeat(job):
                log.warning("lease lost", extra={"job_id": job.job_id})
                return
