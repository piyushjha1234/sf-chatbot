"""Query understanding: multi-turn condensing + schema-validated query plans (fast model)."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict

from pydantic import BaseModel, Field

from sfrag.core.ids import sha256_text
from sfrag.core.models import Depth, KnowledgeCategory, Persona, QueryIntent, QueryPlan
from sfrag.core.taxonomy import Taxonomy
from sfrag.enrich.classify import RuleClassifier, extract_identifiers
from sfrag.llm.client import LLMClient

CONDENSE_SYSTEM = (
    "You rewrite the user's latest message in a Salesforce documentation chat into one standalone question. "
    "Resolve pronouns and references using the conversation. Keep product names, identifiers, error codes and "
    "versions verbatim. If the latest message is already standalone, return it unchanged. Do not answer it."
)


class _Condensed(BaseModel):
    standalone_question: str


class _LLMPlan(BaseModel):
    products: list[str] = Field(description="product keys from the allowed list only")
    features: list[str]
    categories: list[KnowledgeCategory]
    intents: list[QueryIntent]
    persona: Persona
    depth: Depth
    release_hint: str | None = Field(description="Salesforce release or API version named or implied, else null")
    explicit_product: bool = Field(description="true only if the user literally named the product")
    cross_product: bool
    sub_questions: list[str] = Field(description="2-4 self-contained sub-questions if multi-part, else empty")
    expected_sections: list[str] = Field(description="short headings the ideal answer would contain")
    confidence: float = Field(ge=0, le=1)


def _plan_system(taxonomy: Taxonomy) -> str:
    products = "\n".join(
        f"- {k}: {p.label} (aliases: {', '.join(p.aliases[1:8])})" for k, p in taxonomy.products.items()
    )
    return (
        "You analyse questions for a retrieval system over official Salesforce documentation. Produce a query "
        "plan; never answer the question.\n\nAllowed product keys:\n" + products + "\n\n"
        "Rules: use only the product keys above (empty list if unsure; unknown products are fine). "
        "Persona is who appears to be asking (auto if unclear). Depth: beginner, standard or deep_technical. "
        "Decompose only genuinely multi-part questions. Confidence reflects how sure you are of the products "
        "and intent; low confidence is expected for vague questions."
    )


def _heuristic_plan(question: str, rules: RuleClassifier, persona: Persona, depth: Depth) -> QueryPlan:
    scores = rules.score_products("", question, question, question)
    products, conf = rules._top(scores, k=2)
    cats = rules.categories(question, question, question)
    explicit = bool(products) and any(
        p.alias_regex and p.alias_regex.search(question) for k, p in rules.tx.products.items() if k in products
    )
    _, intents = rules.personas_intents(cats)
    return QueryPlan(
        standalone_question=question,
        products=products,
        categories=cats,
        intents=intents[:3],
        persona=persona,
        depth=depth,
        explicit_product=explicit,
        confidence=min(conf, 0.6),
        source="heuristic",
    )


_TRIVIAL = re.compile(r"^\s*[\w.$/{}:-]+\s*\??\s*$")


class QueryPlanner:
    def __init__(self, llm: LLMClient, taxonomy: Taxonomy, rules: RuleClassifier, cache_size: int = 2048) -> None:
        self.llm = llm
        self.tx = taxonomy
        self.rules = rules
        self._system = _plan_system(taxonomy)
        self._cache: OrderedDict[str, QueryPlan] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def condense(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        convo = "\n".join(f"{m['role'].upper()}: {m['content'][:2000]}" for m in history[-8:])
        out = self.llm.parse(
            _Condensed,
            CONDENSE_SYSTEM,
            f"<conversation>\n{convo}\n</conversation>\n\nLatest message: {question}",
            max_tokens=1000,
        )
        return out.standalone_question.strip() if out and out.standalone_question.strip() else question

    def is_trivial(self, question: str) -> bool:
        """Identifier-only lookups (error codes, API names, endpoints) skip the LLM plan."""
        return bool(_TRIVIAL.match(question)) or (len(question.split()) <= 3 and bool(extract_identifiers(question)))

    def plan(self, question: str, persona: Persona = Persona.AUTO, depth: Depth = Depth.STANDARD) -> QueryPlan:
        if self.is_trivial(question):
            p = _heuristic_plan(question, self.rules, persona, depth)
            p.source = "skipped"
            return p
        key = sha256_text(f"{' '.join(question.lower().split())}|{persona}|{depth}")
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key].model_copy(deep=True)
        raw = self.llm.parse(
            _LLMPlan,
            self._system,
            f"Selected persona: {persona.value}\nSelected depth: {depth.value}\n\nQuestion: {question}",
            max_tokens=2000,
        )
        if raw is None:
            plan = _heuristic_plan(question, self.rules, persona, depth)
        else:
            products = [
                k for p in raw.products if (k := (p if p in self.tx.products else self.tx.normalize_product(p)))
            ]
            plan = QueryPlan(
                standalone_question=question,
                products=list(dict.fromkeys(products)),
                features=raw.features,
                categories=raw.categories,
                intents=raw.intents,
                persona=persona if persona != Persona.AUTO else raw.persona,
                depth=depth if depth != Depth.STANDARD else raw.depth,
                release_hint=raw.release_hint,
                explicit_product=raw.explicit_product,
                cross_product=raw.cross_product,
                sub_questions=[s for s in raw.sub_questions if s.strip()][:4],
                expected_sections=raw.expected_sections[:8],
                confidence=raw.confidence,
                source="llm",
            )
        with self._lock:
            self._cache[key] = plan
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return plan.model_copy(deep=True)
