"""Hybrid retrieval -> dedupe/collapse -> cross-encoder rerank + metadata boosts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from qdrant_client import models as qm

from sfrag.core.models import Candidate, QueryPlan
from sfrag.core.settings import RetrievalSettings
from sfrag.enrich.classify import extract_identifiers
from sfrag.index.embedder import Embedder
from sfrag.index.reranker import Reranker
from sfrag.index.store import VectorStore

RRF_K = 60


def acl_filter(principals: list[str]) -> qm.Filter:
    allowed = list(dict.fromkeys(["public", *principals]))
    return qm.Filter(must=[qm.FieldCondition(key="acl", match=qm.MatchAny(any=allowed))])


def build_filter(principals: list[str], plan: QueryPlan | None, cfg: RetrievalSettings, hard: bool) -> qm.Filter:
    flt = acl_filter(principals)
    must: list[Any] = list(flt.must or [])
    if hard and plan and plan.products:
        must.append(qm.FieldCondition(key="products", match=qm.MatchAny(any=plan.products)))
    return qm.Filter(must=must, must_not=[qm.FieldCondition(key="lifecycle", match=qm.MatchValue(value="archived"))])


class HybridRetriever:
    def __init__(
        self, store: VectorStore, embedder: Embedder, reranker: Reranker, cfg: RetrievalSettings, current_release: str
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        self.current_release = current_release

    # ---------------------------------------------------------------- recall
    def recall(self, queries: list[str], flt: qm.Filter, k_multiplier: float = 1.0) -> list[Candidate]:
        """Hybrid search per query (server-side RRF), then client-side RRF across queries."""
        fused: dict[str, float] = defaultdict(float)
        cands: dict[str, Candidate] = {}
        for q in queries:
            emb = self.embedder.embed_query(q)
            points = self.store.hybrid_search(
                emb,
                extract_identifiers(q),
                flt,
                dense_k=int(self.cfg.dense_k * k_multiplier),
                sparse_k=int(self.cfg.sparse_k * k_multiplier),
                identifier_k=self.cfg.identifier_k,
                limit=int(self.cfg.fused_k * k_multiplier),
            )
            for rank, p in enumerate(points):
                pl = p.payload or {}
                cid = pl["chunk_id"]
                fused[cid] += 1.0 / (RRF_K + rank + 1)
                if cid not in cands:
                    cands[cid] = Candidate(
                        chunk_id=cid,
                        doc_id=pl["doc_id"],
                        parent_id=pl["parent_id"],
                        text=pl.get("text", ""),
                        payload=pl,
                    )
        for cid, s in fused.items():
            cands[cid].fused_score = s
        return sorted(cands.values(), key=lambda c: c.fused_score, reverse=True)

    def collapse(self, cands: list[Candidate]) -> list[Candidate]:
        """Cap chunks per document so one long page cannot crowd out diverse evidence."""
        per_doc: dict[str, int] = defaultdict(int)
        out = []
        for c in cands:
            if per_doc[c.doc_id] < self.cfg.max_per_doc:
                out.append(c)
                per_doc[c.doc_id] += 1
        return out

    # ---------------------------------------------------------------- precision
    def _boost(self, c: Candidate, plan: QueryPlan | None) -> float:
        cfg = self.cfg
        pl = c.payload
        b = 0.0
        lifecycle = pl.get("lifecycle")
        if lifecycle == "deprecated":
            b -= cfg.penalty_deprecated
        elif lifecycle == "retired":
            b -= cfg.penalty_retired
        elif lifecycle == "preview":
            b -= cfg.penalty_preview
        if pl.get("authority_tier") == 1:
            b += cfg.boost_authority_tier1
        release = pl.get("release")
        wanted_release = plan.release_hint if plan and plan.release_hint else self.current_release
        if release and release == wanted_release:
            b += cfg.boost_current_release
        if plan:
            if plan.products and set(plan.products) & set(pl.get("products", [])):
                b += cfg.boost_product_match * max(plan.confidence, 0.5)
            if plan.categories and {x.value for x in plan.categories} & set(pl.get("categories", [])):
                b += cfg.boost_category_match
            if plan.persona.value != "auto" and plan.persona.value in pl.get("personas", []):
                b += cfg.boost_persona_match
        return b

    def rerank(self, question: str, cands: list[Candidate], plan: QueryPlan | None) -> list[Candidate]:
        top = cands[: self.cfg.rerank_k]
        if not top:
            return []
        passages = [f"{c.payload.get('title', '')} › {c.payload.get('heading', '')}\n{c.text}" for c in top]
        scores = self.reranker.score(question, passages)
        for c, s in zip(top, scores, strict=True):
            c.rerank_score = float(s)
            c.final_score = c.rerank_score + self._boost(c, plan)
        return sorted(top, key=lambda c: c.final_score, reverse=True)

    # ---------------------------------------------------------------- full pass
    def retrieve(
        self,
        question: str,
        plan: QueryPlan | None,
        principals: list[str],
        base: list[Candidate] | None = None,
        broaden: bool = False,
    ) -> list[Candidate]:
        """`base` = results of the plan-free recall for `question` (reused unless a hard filter applies)."""
        hard = bool(
            not broaden
            and plan
            and plan.explicit_product
            and plan.confidence >= self.cfg.plan_confidence_for_hard_filter
        )
        flt = build_filter(principals, plan, self.cfg, hard)
        queries: list[str] = []
        if base is None or hard or broaden:
            queries.append(question)
        if plan:
            if plan.standalone_question and plan.standalone_question != question:
                queries.append(plan.standalone_question)
            queries += [q for q in plan.sub_questions if q != question]
        cands = self.recall(queries, flt, k_multiplier=1.5 if broaden else 1.0) if queries else []
        if base and not hard:
            merged: dict[str, Candidate] = {c.chunk_id: c for c in cands}
            for c in base:
                if c.chunk_id in merged:
                    merged[c.chunk_id].fused_score += c.fused_score
                else:
                    merged[c.chunk_id] = c.model_copy()
            cands = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
        return self.rerank(question, self.collapse(cands), plan)

    def recall_raw(self, question: str, principals: list[str]) -> list[Candidate]:
        """Plan-free recall that runs in parallel with query planning."""
        return self.recall([question], build_filter(principals, None, self.cfg, False))
