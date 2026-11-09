"""Prompt-injection heuristics for untrusted content.

Flagged content is still indexed (coverage-first) but marked `quarantined`; at answer time it is wrapped as
untrusted data and the model is told never to follow instructions inside it.
"""

from __future__ import annotations

import re

_INJECTION = re.compile(
    r"(ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts)|disregard (the|all) (above|previous)|"
    r"you are now (?:an?|the) |system prompt|developer mode|jailbreak|do anything now|"
    r"<\s*/?\s*(system|assistant)\s*>|BEGIN (SYSTEM|ADMIN) (PROMPT|INSTRUCTIONS))",
    re.IGNORECASE,
)


def injection_signals(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).lower() for m in _INJECTION.finditer(text)))[:10]
