"""Post-generation citation verification."""

from __future__ import annotations

import re

from sfrag.core.models import Citation, EvidenceBlock

_CIT = re.compile(r"\[(S\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*])")


def verify_citations(answer: str, blocks: list[EvidenceBlock]) -> tuple[str, list[Citation], list[str], float]:
    """Strip citation ids that were never supplied; return (clean_answer, used_citations, invalid_ids,
    uncited_factual_sentence_ratio)."""
    valid = {b.sid: b for b in blocks}
    invalid = sorted({m for m in _CIT.findall(answer) if m not in valid})
    clean = _CIT.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer) if invalid else answer
    used = list(dict.fromkeys(m for m in _CIT.findall(clean) if m in valid))
    citations = [
        Citation(
            sid=s,
            title=valid[s].title,
            url=valid[s].url,
            heading=valid[s].heading,
            page=valid[s].page,
            release=valid[s].release,
        )
        for s in used
    ]
    # factual sentence heuristic: prose lines, not headings/code/very short connectors
    prose: list[str] = []
    in_code = False
    for line in clean.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip() or line.lstrip().startswith("#"):
            continue
        prose.extend(s for s in _SENT.split(line.strip()) if len(s.split()) >= 6)
    uncited = sum(1 for s in prose if not _CIT.search(s))
    ratio = round(uncited / len(prose), 3) if prose else 0.0
    return clean, citations, invalid, ratio
