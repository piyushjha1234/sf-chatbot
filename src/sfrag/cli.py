"""`sfrag` command line."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Annotated

import typer

from sfrag.container import Container
from sfrag.core.models import Depth, Persona

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Salesforce documentation RAG platform")
index_app = typer.Typer(no_args_is_help=True, help="Index generations (blue/green)")
queue_app = typer.Typer(no_args_is_help=True, help="Job queue operations")
app.add_typer(index_app, name="index")
app.add_typer(queue_app, name="queue")


def _c() -> Container:
    return Container()


def _echo(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def init() -> None:
    """Create data dirs, migrate the DB, sync sources.yaml into the registry."""
    from sfrag.sources.registry import sync_sources

    c = _c()
    ids = sync_sources(c.db, c.settings.sources_path)
    live = c.indexer.live()  # create the first generation up front (avoids races between processes)
    _echo({"data_dir": str(c.settings.data_dir), "sources": ids, "live_index": live})


@app.command()
def discover(source: Annotated[list[str] | None, typer.Option("--source", "-s")] = None) -> None:
    """Enqueue discovery for sources (all permitted web sources by default)."""
    from sfrag.sources.registry import sync_sources
    from sfrag.storage import repo

    c = _c()
    sync_sources(c.db, c.settings.sources_path)
    targets = source or [
        x["source_id"] for x in repo.list_sources(c.db) if x.get("kind") == "web" and x.get("permitted", True)
    ]
    import time as _t

    jobs = {
        sid: c.queue.enqueue(
            "discover_source", {"source_id": sid}, idempotency_key=f"discover:{sid}:{int(_t.time())}", priority=10
        )
        for sid in targets
    }
    _echo({"enqueued": jobs})


@app.command("ingest-url")
def ingest_url(url: str, source: Annotated[str | None, typer.Option("--source", "-s")] = None) -> None:
    """Queue one URL (must fall inside an approved source scope)."""
    from fastapi.testclient import TestClient

    from sfrag.api.app import create_app

    with TestClient(create_app(_c())) as client:
        r = client.post("/ingest/url", json={"url": url, "source_id": source})
        _echo(r.json())


@app.command("ingest-path")
def ingest_path(path: Path) -> None:
    """Queue a file or directory that already lives under data/incoming/."""
    from sfrag.sources.watcher import enqueue_file, scan

    c = _c()
    root = c.settings.incoming_dir.resolve()
    p = path.resolve()
    if root not in (p, *p.parents):
        raise typer.BadParameter(f"path must be inside {root} (the folder decides the ACL)")
    n = scan(c.queue, p) if p.is_dir() else int(bool(enqueue_file(c.queue, p, root)))
    _echo({"enqueued": n})


@app.command()
def worker(concurrency: int = 2, drain: bool = typer.Option(False, help="exit when the queue is empty")) -> None:
    """Run ingestion workers."""
    from sfrag.pipeline.orchestrator import Pipeline

    _echo(Pipeline(_c()).run_worker(concurrency=concurrency, drain=drain))


@app.command()
def watch() -> None:
    """Watch data/incoming/ and enqueue new/changed files."""
    from sfrag.sources.watcher import run_watcher

    c = _c()
    run_watcher(c.queue, c.settings.incoming_dir)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    with_worker: bool = typer.Option(True, help="run ingestion workers in-process (needed for embedded Qdrant)"),
    with_watcher: bool = typer.Option(True, help="watch data/incoming/ in-process"),
) -> None:
    """Start the HTTP API."""
    import uvicorn

    from sfrag.api.app import create_app

    c = _c()
    if with_watcher:
        from sfrag.sources.watcher import run_watcher

        threading.Thread(target=run_watcher, args=(c.queue, c.settings.incoming_dir), daemon=True).start()
    uvicorn.run(create_app(c, with_worker=with_worker), host=host, port=port, log_config=None)


@app.command()
def ask(
    question: str, persona: Persona = Persona.AUTO, depth: Depth = Depth.STANDARD, show_sources: bool = True
) -> None:
    """Ask a question from the terminal (streams the answer)."""
    c = _c()
    final = None
    for ev in c.engine.stream(question, persona=persona, depth=depth):
        if ev["type"] == "delta":
            typer.echo(ev["text"], nl=False)
        elif ev["type"] == "done":
            final = ev["answer"]
    typer.echo("")
    if final and show_sources:
        typer.echo("\nSources:")
        for ci in final["citations"]:
            typer.echo(f"  [{ci['sid']}] {ci['title']} — {ci['heading']}  {ci['url']}")
        typer.echo(
            f"\nconfidence={final['confidence']} abstained={final['abstained']} "
            f"invalid_citations={final['invalid_citations']} uncited_ratio={final['uncited_sentence_ratio']}"
        )


@app.command()
def coverage() -> None:
    """Write and print the coverage / reconciliation report."""
    from sfrag.ops.coverage import coverage_report, write_report

    c = _c()
    path = write_report(c.db, c.settings.reports_dir)
    rep = coverage_report(c.db)
    rep["report_path"] = str(path)
    _echo(rep)


@app.command("eval")
def eval_cmd(
    dataset: Path,
    split: Annotated[str | None, typer.Option()] = "dev",
    mode: Annotated[str, typer.Option(help="retrieval | answer")] = "retrieval",
    confirm_holdout: bool = False,
    baseline: Annotated[Path | None, typer.Option(help="fail if metrics regress vs this summary")] = None,
) -> None:
    """Run the evaluation harness."""
    from sfrag.eval.runner import gate, run_eval

    c = _c()
    summary = run_eval(c, dataset, split, mode, confirm_holdout)
    _echo(summary)
    if baseline:
        failures = gate(summary, baseline)
        if failures:
            typer.echo("REGRESSION:\n" + "\n".join(failures), err=True)
            raise typer.Exit(1)


@index_app.command("rebuild")
def index_rebuild() -> None:
    """Build a new generation from stored chunks, validate, swap the live alias (keeps N-1)."""
    _echo(_c().indexer.rebuild())


@index_app.command("rollback")
def index_rollback() -> None:
    """Point the live alias back at the previous generation."""
    _echo({"live": _c().indexer.rollback()})


@index_app.command("status")
def index_status() -> None:
    c = _c()
    _echo(
        {
            "live": c.store.live_collection(),
            "generations": [dict(r) for r in c.db.query("SELECT * FROM index_generations ORDER BY created_at")],
        }
    )


@queue_app.command("stats")
def queue_stats() -> None:
    _echo(_c().queue.stats())


@queue_app.command("dlq")
def queue_dlq(retry: bool = False) -> None:
    c = _c()
    _echo({"retried": c.queue.retry_dlq()} if retry else c.queue.dlq())


if __name__ == "__main__":
    app()
