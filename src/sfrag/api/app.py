"""HTTP API: streaming chat (SSE), ingestion, sources, feedback, coverage, health."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sfrag.container import Container
from sfrag.core.models import Depth, Persona
from sfrag.ops.coverage import coverage_report
from sfrag.sources.discovery import Scope, normalize_url
from sfrag.storage import repo
from sfrag.storage.db import dumps, now

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    persona: Persona = Persona.AUTO
    depth: Depth = Depth.STANDARD
    stream: bool = True


class UrlIngestRequest(BaseModel):
    url: str
    source_id: str | None = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str | None = None
    rating: int = Field(ge=-1, le=1)
    reason: str | None = None
    kind: str = Field(default="rating", pattern="^(rating|outdated)$")
    citations: list[dict[str, Any]] = Field(default_factory=list)


class Principal(BaseModel):
    id: str
    groups: list[str]

    @property
    def principals(self) -> list[str]:
        return [self.id, *self.groups]


class _RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> bool:
        t = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < t - 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(t)
            return True


def create_app(container: Container | None = None, with_worker: bool = False, worker_concurrency: int = 2) -> FastAPI:
    c = container or Container()
    s = c.settings
    limiter = _RateLimiter(s.api.rate_limit_per_minute)
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threads: list[threading.Thread] = []
        if with_worker:
            from sfrag.pipeline.orchestrator import Pipeline

            pipe = Pipeline(c)
            t = threading.Thread(
                target=pipe.run_worker,
                kwargs={"concurrency": worker_concurrency, "stop": stop},
                daemon=True,
                name="api-worker",
            )
            t.start()
            threads.append(t)
        yield
        stop.set()
        for t in threads:
            t.join(timeout=10)

    app = FastAPI(title="sfrag", version="0.1.0", lifespan=lifespan)

    def auth(
        x_api_key: Annotated[str | None, Header()] = None, authorization: Annotated[str | None, Header()] = None
    ) -> Principal:
        key = x_api_key or (
            authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
        )
        if not s.api.api_keys:  # local dev: auth disabled
            pid = "user:anonymous"
        else:
            if not key or key not in s.api.api_keys:
                raise HTTPException(401, "invalid or missing API key")
            pid = s.api.api_keys[key]
        groups = list(dict.fromkeys([*s.api.groups.get(pid, []), "group:internal"]))
        return Principal(id=pid, groups=groups)

    def rate_limited(p: Principal = Depends(auth)) -> Principal:
        if not limiter.check(p.id):
            raise HTTPException(429, "rate limit exceeded")
        return p

    @app.get("/health")
    def health() -> dict[str, Any]:
        live = c.store.live_collection()
        return {
            "status": "ok",
            "generation": live,
            "points": c.store.count(live) if live else 0,
            "queue": c.queue.stats(),
            "llm_enabled": s.llm.enabled,
            "embedding": s.embedding_model_version,
            "pipeline_version": s.pipeline_version,
        }

    @app.post("/chat")
    def chat(req: ChatRequest, p: Principal = Depends(rate_limited)) -> Any:
        if len(req.question) > s.api.max_question_chars:
            raise HTTPException(413, f"question exceeds {s.api.max_question_chars} characters")
        history = [t.model_dump() for t in req.history]
        events = c.engine.stream(
            req.question, history=history, persona=req.persona, depth=req.depth, principals=p.principals
        )
        if not req.stream:
            final = None
            for ev in events:
                if ev["type"] == "done":
                    final = ev
            return JSONResponse(final)

        def sse() -> Iterator[bytes]:
            try:
                for ev in events:
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
            except Exception as e:
                err = json.dumps({"error": type(e).__name__, "detail": str(e)[:500]})
                yield f"event: error\ndata: {err}\n\n".encode()

        return StreamingResponse(
            sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.post("/ingest/url")
    def ingest_url(req: UrlIngestRequest, p: Principal = Depends(rate_limited)) -> dict[str, Any]:
        url = normalize_url(req.url)
        candidates = [
            x
            for x in repo.list_sources(c.db)
            if x.get("kind") == "web" and x.get("permitted", True) and (req.source_id in (None, x["source_id"]))
        ]
        for src in candidates:
            ok, _ = Scope.from_config(src).check(url)
            if ok:
                repo.register_url(c.db, url, src["source_id"], status="queued", discovered_from=f"api:{p.id}")
                job = c.queue.enqueue(
                    "fetch_url",
                    {"url": url, "source_id": src["source_id"], "depth": 0},
                    idempotency_key=f"fetch:{url}:manual:{int(now())}",
                    priority=20,
                )
                return {"queued": True, "job_id": job, "source_id": src["source_id"], "url": url}
        raise HTTPException(400, "URL is not within any approved, permitted source scope")

    @app.post("/ingest/file")
    def ingest_file(file: Annotated[UploadFile, File()], p: Principal = Depends(rate_limited)) -> dict[str, Any]:
        name = _SAFE_NAME.sub("_", Path(file.filename or "upload").name)[:200] or "upload"
        user = _SAFE_NAME.sub("_", p.id.split(":", 1)[-1])
        dest = s.incoming_dir / "users" / user / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = file.file.read()
        if len(data) > s.fetch.max_bytes:
            raise HTTPException(413, "file too large")
        dest.write_bytes(data)
        from sfrag.sources.watcher import enqueue_file

        job = enqueue_file(c.queue, dest, s.incoming_dir)
        return {
            "queued": job is not None,
            "job_id": job,
            "path": dest.relative_to(s.incoming_dir).as_posix(),
            "acl": [f"user:{user}"],
        }

    @app.get("/sources")
    def sources(p: Principal = Depends(auth)) -> list[dict[str, Any]]:
        return repo.list_sources(c.db)

    @app.post("/feedback")
    def feedback(req: FeedbackRequest, p: Principal = Depends(rate_limited)) -> dict[str, Any]:
        c.db.execute(
            "INSERT INTO feedback(principal, question, answer, rating, reason, citations_json, kind, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (p.id, req.question, req.answer, req.rating, req.reason, dumps(req.citations), req.kind, now()),
        )
        return {"ok": True}

    @app.get("/coverage")
    def coverage(p: Principal = Depends(auth)) -> dict[str, Any]:
        return coverage_report(c.db)

    return app
