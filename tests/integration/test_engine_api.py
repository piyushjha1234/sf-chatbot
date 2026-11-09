from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

from sfrag.api.app import create_app
from sfrag.container import Container
from sfrag.core.models import Depth, Persona
from sfrag.llm.client import StreamResult
from sfrag.pipeline.orchestrator import Pipeline
from sfrag.query.prompts import ABSTAIN_MARKER
from tests.conftest import D360


class FakeLLM:
    """Stands in for the Anthropic client: returns a canned grounded answer with one bogus citation."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def parse(self, schema, system, user, **kw):  # type: ignore[no-untyped-def]
        return None  # planner/condense degrade to heuristics

    def stream_answer(self, system: str, messages: list[dict[str, Any]], result: StreamResult) -> Iterator[str]:
        self.calls.append({"system": system, "messages": messages})
        for part in self.text.split(" "):
            result.text += part + " "
            yield part + " "
        result.stop_reason = "end_turn"
        result.model = "fake"


def _indexed(c: Container) -> Container:
    c.queue.enqueue("discover_source", {"source_id": "sf_dev_data360"}, idempotency_key="d")
    Pipeline(c).run_worker(concurrency=2, drain=True)
    return c


def test_engine_grounded_answer_with_citation_verification(web_container: Container) -> None:
    c = _indexed(web_container)
    c.settings.llm.enabled = True
    fake = FakeLLM(
        "Reconciliation rules decide which value wins when sources disagree [S1]. "
        "This is supported by an invented source [S99]."
    )
    c.__dict__["llm"] = fake
    c.__dict__.pop("planner", None)
    c.__dict__.pop("engine", None)
    ans = c.engine.ask(
        "How do reconciliation rules work in identity resolution?", persona=Persona.DATA, depth=Depth.STANDARD
    )
    assert not ans.abstained
    assert ans.invalid_citations == ["S99"] and "[S99]" not in ans.answer
    assert ans.citations and ans.citations[0].url.startswith(D360)
    user_turn = fake.calls[0]["messages"][-1]["content"]
    assert '<source id="S1"' in user_turn and "Persona: data_engineer" in user_turn
    assert 'trust="official"' in user_turn
    # answers with invalid citations are never cached
    c.engine.ask("How do reconciliation rules work in identity resolution?", persona=Persona.DATA)
    assert len(fake.calls) == 2
    # clean answers are cached: the repeat question makes no new LLM call
    fake.text = "A data stream ingests source data into a Data Lake Object [S1]."
    c.engine.ask("How do I create a data stream?")
    c.engine.ask("How do I create a data stream?")
    assert len(fake.calls) == 3


def test_engine_abstains_without_llm_call_when_nothing_relevant(web_container: Container) -> None:
    c = _indexed(web_container)
    c.settings.llm.enabled = True
    fake = FakeLLM("should not be called")
    c.__dict__["llm"] = fake
    c.__dict__.pop("planner", None)
    c.__dict__.pop("engine", None)
    c.settings.retrieval.abstain_threshold = 0.99
    ans = c.engine.ask("What is the airspeed velocity of an unladen swallow?")
    assert ans.abstained and ans.answer.startswith(ABSTAIN_MARKER)
    assert fake.calls == []


def test_api_endpoints(web_container: Container) -> None:
    c = _indexed(web_container)
    with TestClient(create_app(c)) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["points"] > 0
        r = client.post("/chat", json={"question": "How do I create a data stream?", "stream": False})
        assert r.status_code == 200
        body = r.json()["answer"]
        assert body["answer"].startswith("LLM disabled")  # test config: retrieval-only
        with client.stream("POST", "/chat", json={"question": "Query API errors"}) as s:
            events = [line[7:] for line in s.iter_lines() if line.startswith("event: ")]
        assert events[0] == "plan" and "sources" in events and events[-1] == "done"
        assert client.post("/ingest/url", json={"url": "https://evil.example.com/x"}).status_code == 400
        ok = client.post("/ingest/url", json={"url": f"{D360}/new-page.html"}).json()
        assert ok["queued"] and ok["source_id"] == "sf_dev_data360"
        up = client.post("/ingest/file", files={"file": ("../../etc/passwd.md", b"# Hi\n\nThere")}).json()
        assert up["path"] == "users/anonymous/passwd.md"
        assert client.post("/feedback", json={"question": "q", "rating": -1, "reason": "outdated"}).json()["ok"]
        cov = client.get("/coverage").json()
        assert "reconciliation" in cov


def test_api_auth_and_rate_limit(web_container: Container) -> None:
    c = web_container
    c.settings.api.api_keys = {"k-alice": "user:alice"}
    c.settings.api.rate_limit_per_minute = 2
    with TestClient(create_app(c)) as client:
        assert client.get("/sources").status_code == 401
        assert client.get("/sources", headers={"X-API-Key": "k-alice"}).status_code == 200
        codes = [
            client.post(
                "/feedback", headers={"Authorization": "Bearer k-alice"}, json={"question": "q", "rating": 1}
            ).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 429]


def test_eval_harness_retrieval_mode(web_container: Container, tmp_path) -> None:
    from sfrag.eval.runner import run_eval

    c = _indexed(web_container)
    ds = tmp_path / "ds.jsonl"
    items = [
        {
            "id": "1",
            "split": "dev",
            "question": "How do I create a data stream in Data Cloud?",
            "gold_urls": [f"{D360}/data-streams.html"],
        },
        {
            "id": "2",
            "split": "dev",
            "question": "What does INVALID_SQL_SYNTAX mean?",
            "gold_urls": [f"{D360}/query-api.html"],
        },
        {"id": "3", "split": "test", "question": "held out", "gold_urls": []},
    ]
    ds.write_text("\n".join(json.dumps(i) for i in items))
    summary = run_eval(c, ds, "dev", "retrieval")
    assert summary["n"] == 2 and summary["recall@10"] == 1.0 and summary["mrr"] > 0.4
    import pytest

    with pytest.raises(SystemExit):
        run_eval(c, ds, "test", "retrieval")
