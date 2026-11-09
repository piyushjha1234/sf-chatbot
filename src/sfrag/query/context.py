"""Parent/sibling expansion and token-budget context packing."""

from __future__ import annotations

from sfrag.core.models import Candidate, EvidenceBlock
from sfrag.core.tokenizer import Tokenizer
from sfrag.storage import repo
from sfrag.storage.db import Database


def expand(db: Database, cands: list[Candidate], tok: Tokenizer, parent_budget: int) -> list[EvidenceBlock]:
    """Children -> parent context (deduped per parent). Oversized parents fall back to child ± siblings,
    and split code/table groups are reassembled."""
    by_parent: dict[str, list[Candidate]] = {}
    order: list[str] = []
    for c in cands:
        if c.parent_id not in by_parent:
            order.append(c.parent_id)
            by_parent[c.parent_id] = []
        by_parent[c.parent_id].append(c)

    blocks: list[EvidenceBlock] = []
    for pid in order:
        group = by_parent[pid]
        best = max(group, key=lambda c: c.final_score)
        pl = best.payload
        section = repo.get_section(db, pid)
        text: str
        chunk_ids = [c.chunk_id for c in group]
        if section and section["token_count"] <= parent_budget:
            text = section["text"]
        else:
            siblings = repo.get_chunks_for_parent(db, pid)
            hit_ords = {c.payload.get("ordinal") for c in group}
            hit_groups = {c.payload.get("group_id") for c in group if c.payload.get("group_id")}
            keep = [
                s
                for s in siblings
                if any(abs(s["ordinal"] - o) <= 1 for o in hit_ords if o is not None)
                or (s["group_id"] and s["group_id"] in hit_groups)
            ]
            parts: list[str] = []
            used = 0
            for s in keep:
                piece = (s["context_prefix"] or "") + s["body"]
                t = s["token_count"]
                if used + t > parent_budget and parts:
                    break
                parts.append(piece)
                used += t
            text = "\n\n".join(parts) if parts else best.text
        blocks.append(
            EvidenceBlock(
                sid="",
                doc_id=best.doc_id,
                parent_id=pid,
                chunk_ids=chunk_ids,
                title=pl.get("title", ""),
                url=pl.get("url", ""),
                heading=pl.get("heading", ""),
                page=pl.get("page"),
                release=pl.get("release"),
                text=text,
                score=best.final_score,
                untrusted=bool(pl.get("quarantined")),
            )
        )
    return blocks


def pack(blocks: list[EvidenceBlock], tok: Tokenizer, budget: int, max_blocks: int) -> list[EvidenceBlock]:
    """Greedy by score under a token budget, then 'sandwich' ordering (best at both ends) to limit
    lost-in-the-middle effects. Source ids S1..Sn follow final order."""
    chosen: list[EvidenceBlock] = []
    used = 0
    for b in sorted(blocks, key=lambda x: x.score, reverse=True):
        if len(chosen) >= max_blocks:
            break
        t = tok.count(b.text)
        if used + t > budget:
            if not chosen:
                # always keep the best block, truncated by paragraphs to fit
                paras = b.text.split("\n\n")
                acc: list[str] = []
                for p in paras:
                    if tok.count("\n\n".join([*acc, p])) > budget:
                        break
                    acc.append(p)
                b = b.model_copy(update={"text": "\n\n".join(acc) or b.text[: budget * 3]})
                chosen.append(b)
                used = tok.count(b.text)
            continue
        chosen.append(b)
        used += t
    front: list[EvidenceBlock] = []
    back: list[EvidenceBlock] = []
    for i, b in enumerate(chosen):
        (front if i % 2 == 0 else back).append(b)
    ordered = front + list(reversed(back))
    for i, b in enumerate(ordered, start=1):
        b.sid = f"S{i}"
    return ordered
