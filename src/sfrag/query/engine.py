"""Answer engine: condense -> (plan ‖ raw recall) -> retrieve/rerank -> sufficiency -> expand/pack ->
grounded streaming generation -> citation verification."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sfrag.core.ids import sha256_text
from sfrag.core.models import Answer, Candidate, Depth, EvidenceBlock, Persona, QueryPlan
from sfrag.core.settings import Settings
from sfrag.core.tokenizer import Tokenizer
from sfrag.index.store import VectorStore
from sfrag.llm.client import LLMClient, StreamResult
from sfrag.query.context import expand, pack
from sfrag.query.planner import QueryPlanner
from sfrag.query.prompts import ABSTAIN_MARKER, SYSTEM_PROMPT, user_message
from sfrag.query.retriever import HybridRetriever
from sfrag.query.verify import verify_citations
from sfrag.storage.db import Database, now

log = logging.getLogger(__name__)

Event = dict[str, Any]


class AnswerEngine:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        store: VectorStore,
        retriever: HybridRetriever,
        planner: QueryPlanner,
        llm: LLMClient,
        tokenizer: Tokenizer,
    ) -> None:
        self.s = settings
        self.db = db
        self.store = store
        self.retriever = retriever
        self.planner = planner
        self.llm = llm
        self.tok = tokenizer
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="engine")

    # ------------------------------------------------------------------ retrieval stage
    def gather(
        self, question: str, history: list[dict[str, str]], persona: Persona, depth: Depth, principals: list[str]
    ) -> tuple[str, QueryPlan, list[Candidate], dict[str, float]]:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        standalone = self.planner.condense(question, history) if self.s.llm.enabled else question
        t["condense_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        f_plan = self._pool.submit(self.planner.plan, standalone, persona, depth)
        f_raw = self._pool.submit(self.retriever.recall_raw, standalone, principals)
        plan = f_plan.result()
        raw = f_raw.result()
        t["plan_and_recall_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        cands = self.retriever.retrieve(standalone, plan, principals, base=raw)
        top = cands[0].final_score if cands else 0.0
        if top < self.s.retrieval.sufficiency_threshold:
            broadened = self.retriever.retrieve(standalone, plan, principals, base=None, broaden=True)
            if broadened and (not cands or broadened[0].final_score > top):
                cands = broadened
        t["retrieve_rerank_ms"] = (time.perf_counter() - t2) * 1000
        return standalone, plan, cands, t

    def evidence(self, cands: list[Candidate]) -> list[EvidenceBlock]:
        selected = cands[: self.s.retrieval.final_k]
        blocks = expand(self.db, selected, self.tok, parent_budget=self.s.chunking.parent_max_tokens)
        return pack(blocks, self.tok, self.s.retrieval.context_token_budget, self.s.retrieval.final_k)

    # ------------------------------------------------------------------ cache
    def _cache_key(self, standalone: str, persona: Persona, depth: Depth, principals: list[str]) -> str:
        gen = self.store.live_collection() or "none"
        return sha256_text(
            "|".join(
                [
                    " ".join(standalone.lower().split()),
                    persona.value,
                    depth.value,
                    ",".join(sorted(principals)),
                    gen,
                    self.s.llm.answer_model,
                ]
            )
        )

    def _cache_get(self, key: str) -> Answer | None:
        row = self.db.one("SELECT answer_json FROM answer_cache WHERE cache_key=?", (key,))
        return Answer.model_validate_json(row[0]) if row else None

    def _cache_put(self, key: str, ans: Answer) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO answer_cache(cache_key, generation, answer_json, created_at) VALUES (?,?,?,?)",
            (key, self.store.live_collection() or "none", ans.model_dump_json(), now()),
        )

    # ------------------------------------------------------------------ public API
    def stream(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        persona: Persona = Persona.AUTO,
        depth: Depth = Depth.STANDARD,
        principals: list[str] | None = None,
        use_cache: bool = True,
    ) -> Iterator[Event]:
        history = history or []
        principals = principals or []
        started = time.perf_counter()
        standalone, plan, cands, timings = self.gather(question, history, persona, depth, principals)
        yield {"type": "plan", "standalone_question": standalone, "plan": plan.model_dump(mode="json")}

        persona_eff = plan.persona if persona == Persona.AUTO else persona
        depth_eff = plan.depth if depth == Depth.STANDARD else depth
        top = cands[0].final_score if cands else 0.0

        key = self._cache_key(standalone, persona_eff, depth_eff, principals)
        if use_cache and not history:
            cached = self._cache_get(key)
            if cached:
                yield {"type": "delta", "text": cached.answer}
                yield {"type": "done", "answer": cached.model_dump(mode="json"), "cached": True}
                return

        if top < self.s.retrieval.abstain_threshold:
            closest = [{"title": c.payload.get("title"), "url": c.payload.get("url")} for c in cands[:3]]
            text = f"{ABSTAIN_MARKER} No indexed source matched the question closely enough (best relevance {top:.2f})."
            ans = Answer(
                question=question,
                answer=text,
                citations=[],
                abstained=True,
                confidence=round(top, 3),
                plan=plan,
                timings_ms=timings,
            )
            yield {"type": "delta", "text": text}
            yield {"type": "done", "answer": ans.model_dump(mode="json"), "closest": closest}
            return

        blocks = self.evidence(cands)
        yield {
            "type": "sources",
            "sources": [
                {
                    "sid": b.sid,
                    "title": b.title,
                    "url": b.url,
                    "heading": b.heading,
                    "page": b.page,
                    "release": b.release,
                    "score": round(b.score, 3),
                }
                for b in blocks
            ],
        }

        if not self.s.llm.enabled:
            text = "LLM disabled: retrieval-only result.\n\n" + "\n".join(
                f"[{b.sid}] {b.title} — {b.heading} ({b.url})" for b in blocks
            )
            ans = Answer(
                question=question,
                answer=text,
                citations=[],
                abstained=False,
                confidence=round(top, 3),
                plan=plan,
                timings_ms=timings,
            )
            yield {"type": "delta", "text": text}
            yield {"type": "done", "answer": ans.model_dump(mode="json")}
            return

        messages: list[dict[str, Any]] = [
            *({"role": m["role"], "content": m["content"]} for m in history[-6:]),
            {"role": "user", "content": user_message(standalone, plan, persona_eff, depth_eff, blocks)},
        ]
        result = StreamResult()
        t_gen = time.perf_counter()
        first_token_ms: float | None = None
        for delta in self.llm.stream_answer(SYSTEM_PROMPT, messages, result):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            yield {"type": "delta", "text": delta}
        timings["generate_ms"] = (time.perf_counter() - t_gen) * 1000
        timings["first_token_ms"] = first_token_ms or 0.0
        timings["total_ms"] = (time.perf_counter() - started) * 1000

        clean, citations, invalid, uncited = verify_citations(result.text, blocks)
        abstained = clean.strip().startswith(ABSTAIN_MARKER) or result.refused
        ans = Answer(
            question=question,
            answer=clean,
            citations=citations,
            abstained=abstained,
            confidence=round(top, 3),
            plan=plan,
            invalid_citations=invalid,
            uncited_sentence_ratio=uncited,
            stop_reason=result.stop_reason,
            model=result.model,
            timings_ms=timings,
        )
        if use_cache and not history and not abstained and result.stop_reason == "end_turn" and not invalid:
            self._cache_put(key, ans)
        log.info(
            "answer",
            extra={
                "q_hash": sha256_text(question)[:12],
                "top": round(top, 3),
                "sources": len(blocks),
                "citations": len(citations),
                "invalid": invalid,
                "uncited_ratio": uncited,
                "stop_reason": result.stop_reason,
                "usage": json.dumps(result.usage),
                **{k: round(v) for k, v in timings.items()},
            },
        )
        yield {"type": "done", "answer": ans.model_dump(mode="json")}

    def ask(self, question: str, **kwargs: Any) -> Answer:
        final: Answer | None = None
        for ev in self.stream(question, **kwargs):
            if ev["type"] == "done":
                final = Answer.model_validate(ev["answer"])
        assert final is not None
        return final
