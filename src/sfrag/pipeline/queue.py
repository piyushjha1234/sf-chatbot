"""Durable, lease-based job queue on SQLite.

State machine:  pending -> leased -> done
                              \\-> pending (retry with backoff) -> ... -> dlq (max attempts)
A crashed worker's lease expires and the job becomes leasable again, so no job is lost.
Idempotency keys make enqueueing the same logical work a no-op while it is pending or done.
"""

from __future__ import annotations

import json
import os
import random
import socket
from dataclasses import dataclass
from typing import Any

from sfrag.storage.db import Database, now


@dataclass
class Job:
    job_id: int
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    lease_owner: str


class JobQueue:
    def __init__(self, db: Database, lease_seconds: int = 300, max_attempts: int = 5) -> None:
        self.db = db
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts

    @staticmethod
    def worker_id(suffix: str = "") -> str:
        return f"{socket.gethostname()}:{os.getpid()}{':' + suffix if suffix else ''}"

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        priority: int = 100,
        delay_seconds: float = 0.0,
        revive_dead: bool = True,
    ) -> int | None:
        """Returns the job id, or None when an equivalent job already exists (pending/leased/done).

        A job with the same key that ended in `dlq`/`failed` is reset so it can be retried deliberately,
        unless `revive_dead=False` (automatic re-scans must not resurrect known-bad, unchanged inputs).
        """
        t = now()
        with self.db.tx() as c:
            if idempotency_key:
                row = c.execute("SELECT job_id, state FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row:
                    if row["state"] in ("dlq", "failed") and revive_dead:
                        c.execute(
                            "UPDATE jobs SET state='pending', attempts=0, run_after=?, last_error=NULL, "
                            "payload_json=?, updated_at=? WHERE job_id=?",
                            (t + delay_seconds, json.dumps(payload), t, row["job_id"]),
                        )
                        return int(row["job_id"])
                    return None
            cur = c.execute(
                "INSERT INTO jobs(kind, payload_json, idempotency_key, state, priority, attempts, max_attempts,"
                " run_after, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    kind,
                    json.dumps(payload),
                    idempotency_key,
                    "pending",
                    priority,
                    0,
                    self.max_attempts,
                    t + delay_seconds,
                    t,
                    t,
                ),
            )
            return int(cur.lastrowid or 0)

    def lease(self, owner: str, kinds: list[str] | None = None) -> Job | None:
        t = now()
        kind_clause = ""
        params: list[Any] = [t, t]
        if kinds:
            kind_clause = f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        with self.db.tx() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE ((state='pending' AND run_after<=?) OR (state='leased' AND lease_expires<?))"
                f"{kind_clause} ORDER BY priority ASC, job_id ASC LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            attempts = row["attempts"] + 1
            c.execute(
                "UPDATE jobs SET state='leased', lease_owner=?, lease_expires=?, attempts=?, updated_at=? "
                "WHERE job_id=?",
                (owner, t + self.lease_seconds, attempts, t, row["job_id"]),
            )
        return Job(
            job_id=row["job_id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            attempts=attempts,
            max_attempts=row["max_attempts"],
            lease_owner=owner,
        )

    def heartbeat(self, job: Job) -> bool:
        """Extend the lease. Returns False if the lease was lost (another worker took over)."""
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE jobs SET lease_expires=?, updated_at=? WHERE job_id=? AND lease_owner=? AND state='leased'",
                (now() + self.lease_seconds, now(), job.job_id, job.lease_owner),
            )
            return cur.rowcount == 1

    def complete(self, job: Job) -> None:
        with self.db.tx() as c:
            c.execute(
                "UPDATE jobs SET state='done', lease_owner=NULL, lease_expires=NULL, updated_at=? "
                "WHERE job_id=? AND lease_owner=?",
                (now(), job.job_id, job.lease_owner),
            )

    def fail(self, job: Job, error: str, retryable: bool = True) -> str:
        """Record a failure. Returns the resulting state ('pending' for retry, or 'dlq')."""
        t = now()
        if retryable and job.attempts < job.max_attempts:
            backoff = min(3600.0, (2**job.attempts) * 5.0) * (0.5 + random.random())
            state, run_after = "pending", t + backoff
        else:
            state, run_after = "dlq", t
        with self.db.tx() as c:
            c.execute(
                "UPDATE jobs SET state=?, run_after=?, last_error=?, lease_owner=NULL, lease_expires=NULL, "
                "updated_at=? WHERE job_id=? AND lease_owner=?",
                (state, run_after, error[:4000], t, job.job_id, job.lease_owner),
            )
        return state

    def stats(self) -> dict[str, int]:
        rows = self.db.query("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {r["state"]: r["n"] for r in rows}

    def dlq(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT job_id, kind, payload_json, attempts, last_error, updated_at FROM jobs "
            "WHERE state='dlq' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def retry_dlq(self) -> int:
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE jobs SET state='pending', attempts=0, run_after=?, updated_at=? WHERE state='dlq'",
                (now(), now()),
            )
            return cur.rowcount
