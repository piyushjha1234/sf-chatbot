"""Watch `incoming/` and enqueue ingestion jobs (the watcher never processes files itself).

Folder convention sets the ACL of uploaded files:
  incoming/public/...          -> ["public"]
  incoming/users/<name>/...    -> ["user:<name>"]
  incoming/groups/<name>/...   -> ["group:<name>"]
  anything else                -> ["group:internal"]
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchfiles import Change, watch

from sfrag.core.ids import sha256_bytes
from sfrag.pipeline.queue import JobQueue

log = logging.getLogger(__name__)
_IGNORE_PREFIXES = (".", "~$")


def acl_for(path: Path, root: Path) -> tuple[list[str], str]:
    rel = path.relative_to(root).parts
    if rel and rel[0] == "public":
        return ["public"], "public"
    if len(rel) >= 2 and rel[0] == "users":
        return [f"user:{rel[1]}"], f"user:{rel[1]}"
    if len(rel) >= 2 and rel[0] == "groups":
        return [f"group:{rel[1]}"], f"group:{rel[1]}"
    return ["group:internal"], "group:internal"


def _stable(path: Path, wait: float = 1.0) -> bool:
    try:
        s1 = path.stat().st_size
        time.sleep(wait)
        s2 = path.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False


def enqueue_file(queue: JobQueue, path: Path, root: Path) -> int | None:
    if path.name.startswith(_IGNORE_PREFIXES) or not path.is_file():
        return None
    acl, owner = acl_for(path, root)
    digest = sha256_bytes(path.read_bytes())
    rel = path.relative_to(root).as_posix()
    return queue.enqueue(
        "ingest_file",
        {"path": str(path), "rel": rel, "acl": acl, "owner": owner},
        idempotency_key=f"file:{rel}:{digest}",
        priority=50,
        revive_dead=False,
    )


def scan(queue: JobQueue, root: Path) -> int:
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and enqueue_file(queue, p, root):
            n += 1
    return n


def run_watcher(queue: JobQueue, root: Path, stop: threading.Event | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    log.info("initial scan", extra={"enqueued": scan(queue, root), "root": str(root)})
    for changes in watch(root, debounce=2000, step=500, stop_event=stop, recursive=True):
        for change, p in changes:
            path = Path(p)
            if change == Change.deleted:
                rel = path.relative_to(root).as_posix()
                queue.enqueue("delete_file", {"rel": rel}, idempotency_key=f"delete:{rel}:{time.time():.0f}")
                continue
            if _stable(path):
                job = enqueue_file(queue, path, root)
                if job:
                    log.info("file enqueued", extra={"path": str(path), "job_id": job})
