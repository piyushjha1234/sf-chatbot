from __future__ import annotations

import time

from sfrag.pipeline.queue import JobQueue
from sfrag.storage.db import Database


def test_idempotent_enqueue_and_complete(tmp_path) -> None:
    q = JobQueue(Database(tmp_path / "q.db"), lease_seconds=30, max_attempts=3)
    a = q.enqueue("fetch_url", {"url": "u"}, idempotency_key="k1")
    assert a and q.enqueue("fetch_url", {"url": "u"}, idempotency_key="k1") is None
    job = q.lease("w1")
    assert job and job.payload == {"url": "u"} and job.attempts == 1
    assert q.lease("w2") is None  # leased jobs are exclusive
    q.complete(job)
    assert q.stats() == {"done": 1}
    assert q.enqueue("fetch_url", {"url": "u"}, idempotency_key="k1") is None  # done stays done


def test_crashed_worker_lease_expires_and_job_is_recovered(tmp_path) -> None:
    q = JobQueue(Database(tmp_path / "q.db"), lease_seconds=1, max_attempts=3)
    q.enqueue("process_raw", {"x": 1}, idempotency_key="k")
    first = q.lease("crashed-worker")
    assert first
    time.sleep(1.1)  # worker died without completing
    second = q.lease("healthy-worker")
    assert second and second.job_id == first.job_id and second.attempts == 2
    q.complete(first)  # stale owner cannot complete
    assert q.stats() == {"leased": 1}
    q.complete(second)
    assert q.stats() == {"done": 1}


def test_retry_backoff_then_dlq_and_requeue(tmp_path) -> None:
    db = Database(tmp_path / "q.db")
    q = JobQueue(db, lease_seconds=30, max_attempts=2)
    q.enqueue("fetch_url", {}, idempotency_key="k")
    job = q.lease("w")
    assert job and q.fail(job, "boom") == "pending"
    db.execute("UPDATE jobs SET run_after=0")
    job = q.lease("w")
    assert job and q.fail(job, "boom again") == "dlq"
    assert q.dlq()[0]["last_error"] == "boom again"
    # deliberate re-enqueue of a dead-lettered key resets it
    assert q.enqueue("fetch_url", {}, idempotency_key="k") is not None
    assert q.stats() == {"pending": 1}


def test_non_retryable_goes_straight_to_dlq(tmp_path) -> None:
    q = JobQueue(Database(tmp_path / "q.db"))
    q.enqueue("process_raw", {})
    job = q.lease("w")
    assert job and q.fail(job, "unsupported", retryable=False) == "dlq"
