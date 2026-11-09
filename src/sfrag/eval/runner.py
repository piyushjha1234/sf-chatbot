"""Evaluation harness.

Dataset: JSONL, one item per line:
  {"id": "d360-001", "split": "dev|test|smoke", "question": "...", "persona": "administrator",
   "depth": "standard", "products": ["data360"], "intents": ["configure"], "categories": ["configuration"],
   "gold_urls": ["https://..."], "answer_points": ["..."], "answerable": true, "difficulty": "medium",
   "cross_product": false}

Modes: `retrieval` (no LLM cost: Recall@5/10, Precision@5, MRR, nDCG@10, latency) and `answer`
(adds LLM-judged correctness/faithfulness/completeness, citation precision, abstention accuracy).
The held-out `test` split requires explicit confirmation and every touch is logged.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sfrag.container import Container
from sfrag.core.models import Depth, Persona
from sfrag.eval.metrics import dedupe_preserve, mrr, ndcg_at_k, precision_at_k, recall_at_k
from sfrag.sources.discovery import normalize_url


class EvalItem(BaseModel):
    id: str
    split: str = "dev"
    question: str
    persona: Persona = Persona.AUTO
    depth: Depth = Depth.STANDARD
    products: list[str] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    gold_urls: list[str] = Field(default_factory=list)
    answer_points: list[str] = Field(default_factory=list)
    answerable: bool = True
    difficulty: str = "medium"
    cross_product: bool = False


class _Judgement(BaseModel):
    correctness: float = Field(ge=0, le=1, description="share of answer points correctly covered")
    faithfulness: float = Field(ge=0, le=1, description="share of claims supported by the cited sources")
    completeness: float = Field(ge=0, le=1)
    persona_alignment: float = Field(ge=0, le=1)
    notes: str


JUDGE_SYSTEM = (
    "You grade answers from a Salesforce documentation assistant. Score strictly from the supplied answer "
    "points and sources; do not reward fluent but unsupported content. Return scores in [0,1]."
)


def load_items(path: Path, split: str | None) -> list[EvalItem]:
    items = [EvalItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    return [i for i in items if split is None or i.split == split]


def _log_holdout(path: Path, dataset: Path, n: int) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "dataset": str(dataset), "items": n}) + "\n")


def run_eval(
    c: Container,
    dataset: Path,
    split: str | None,
    mode: str = "retrieval",
    confirm_holdout: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if split == "test" and not confirm_holdout:
        raise SystemExit("Refusing to touch the held-out test split without --confirm-holdout.")
    items = load_items(dataset, split)
    if split == "test":
        _log_holdout(c.settings.reports_dir / "holdout_touches.jsonl", dataset, len(items))
    engine = c.engine
    rows: list[dict[str, Any]] = []
    for it in items:
        gold = {normalize_url(u) for u in it.gold_urls}
        t0 = time.perf_counter()
        _, plan, cands, _ = engine.gather(it.question, [], it.persona, it.depth, [])
        latency = (time.perf_counter() - t0) * 1000
        ranked = dedupe_preserve([normalize_url(x.payload.get("url", "")) for x in cands])
        row: dict[str, Any] = {
            "id": it.id,
            "split": it.split,
            "answerable": it.answerable,
            "latency_ms": round(latency),
            "plan_products": plan.products,
            "plan_confidence": plan.confidence,
            "top_score": round(cands[0].final_score, 3) if cands else 0.0,
        }
        if gold:
            row.update(
                recall5=recall_at_k(ranked, gold, 5),
                recall10=recall_at_k(ranked, gold, 10),
                precision5=precision_at_k(ranked, gold, 5),
                mrr=mrr(ranked, gold),
                ndcg10=ndcg_at_k(ranked, gold, 10),
            )
        if mode == "answer":
            t1 = time.perf_counter()
            ans = engine.ask(it.question, persona=it.persona, depth=it.depth, use_cache=False)
            row["answer_latency_ms"] = round((time.perf_counter() - t1) * 1000)
            row["abstained"] = ans.abstained
            row["abstention_correct"] = ans.abstained == (not it.answerable)
            cited = {normalize_url(ci.url) for ci in ans.citations}
            row["citation_precision"] = (len(cited & gold) / len(cited)) if (cited and gold) else None
            row["citation_recall"] = (len(cited & gold) / len(gold)) if gold else None
            row["invalid_citations"] = len(ans.invalid_citations)
            row["uncited_ratio"] = ans.uncited_sentence_ratio
            if it.answerable and not ans.abstained:
                j = c.llm.parse(
                    _Judgement,
                    JUDGE_SYSTEM,
                    json.dumps(
                        {
                            "question": it.question,
                            "persona": it.persona.value,
                            "answer_points": it.answer_points,
                            "answer": ans.answer,
                            "citations": [ci.model_dump() for ci in ans.citations],
                        }
                    ),
                    model=c.settings.llm.judge_model,
                    effort="medium",
                )
                if j:
                    row.update(
                        correctness=j.correctness,
                        faithfulness=j.faithfulness,
                        completeness=j.completeness,
                        persona_alignment=j.persona_alignment,
                        judge_notes=j.notes,
                    )
        rows.append(row)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    lat = sorted(r["latency_ms"] for r in rows)
    summary = {
        "dataset": str(dataset),
        "split": split,
        "mode": mode,
        "n": len(rows),
        "recall@5": mean("recall5"),
        "recall@10": mean("recall10"),
        "precision@5": mean("precision5"),
        "mrr": mean("mrr"),
        "ndcg@10": mean("ndcg10"),
        "retrieval_p50_ms": lat[len(lat) // 2] if lat else None,
        "retrieval_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "pipeline_version": c.settings.pipeline_version,
        "embedding_model_version": c.settings.embedding_model_version,
        "generation": c.store.live_collection(),
    }
    if mode == "answer":
        summary.update(
            {
                k: mean(k)
                for k in (
                    "correctness",
                    "faithfulness",
                    "completeness",
                    "persona_alignment",
                    "citation_precision",
                    "citation_recall",
                    "uncited_ratio",
                )
            }
        )
        ab = [r["abstention_correct"] for r in rows if "abstention_correct" in r]
        summary["abstention_accuracy"] = round(sum(ab) / len(ab), 4) if ab else None
    out_dir = out_dir or c.settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{split or 'all'}-{mode}-{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    return summary


def gate(summary: dict[str, Any], baseline_path: Path, tolerance: float = 0.02) -> list[str]:
    """CI regression gate: returns failed metrics (current < baseline - tolerance)."""
    if not baseline_path.exists():
        return []
    base = json.loads(baseline_path.read_text())
    failures = []
    for k in ("recall@10", "mrr", "ndcg@10", "faithfulness", "citation_precision", "abstention_accuracy"):
        if base.get(k) is not None and summary.get(k) is not None and summary[k] < base[k] - tolerance:
            failures.append(f"{k}: {summary[k]} < baseline {base[k]} - {tolerance}")
    return failures
