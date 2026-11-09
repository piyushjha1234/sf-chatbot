"""LLM tier of the document classifier (fast model, structured output)."""

from __future__ import annotations

from sfrag.enrich.classify import _LLMDocClass
from sfrag.llm.client import LLMClient

_SYSTEM = (
    "You classify Salesforce documentation pages. Choose product keys only from the allowed list; return an "
    "empty list if none fits (unknown products are acceptable). Confidence reflects certainty about the products."
)


class LLMDocClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def classify(self, title: str, text: str, allowed_products: dict[str, str]) -> _LLMDocClass | None:
        allowed = "\n".join(f"- {k}: {v}" for k, v in allowed_products.items())
        user = f'Allowed products:\n{allowed}\n\n<page title="{title}">\n{text}\n</page>'
        return self.llm.parse(_LLMDocClass, _SYSTEM, user, max_tokens=1500)
