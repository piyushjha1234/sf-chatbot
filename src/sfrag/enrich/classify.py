"""Tiered document/chunk classification.

Tier 1 (deterministic): URL patterns, breadcrumb, title/heading/body alias matches, identifier regexes,
                        category keyword cues, release/API-version/lifecycle detection.
Tier 2 (LLM, fast model): only when tier-1 product confidence is below threshold. Cached by
                        content hash + taxonomy version. Unknown/low-confidence never blocks indexing.
(The embedding-kNN tier from the plan plugs in between once labelled exemplars exist.)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Protocol

from pydantic import BaseModel, Field

from sfrag.core.ids import sha256_text
from sfrag.core.models import (
    Chunk,
    Classification,
    ContentType,
    KnowledgeCategory,
    Lifecycle,
    NormalizedDocument,
    Persona,
    QueryIntent,
)
from sfrag.core.taxonomy import Taxonomy
from sfrag.storage.db import Database, now

_RELEASE = re.compile(r"\b(Spring|Summer|Winter)\s*[’']?\s*(\d{2})\b", re.IGNORECASE)
_API_VERSION = re.compile(r"\b(?:API\s+version\s+|/v|\bv)(\d{2,3}\.0)\b", re.IGNORECASE)
_IDENTIFIERS = [
    re.compile(r"\b[A-Za-z][A-Za-z0-9_]*__(?:c|e|dlm|dll|cio|mdt|x|b|r|kav|share|history)\b"),
    re.compile(r"(?:/services/(?:data|apexrest|oauth2|async|Soap)[\w./{}\-]*)"),
    re.compile(
        r"(?:/(?:api|platform|interaction|contacts|messaging|hub|asset|automation|data|journey)/v\d+[\w./{}\-]*)"
    ),
    re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,}\b"),  # INVALID_FIELD, REQUEST_LIMIT_EXCEEDED
    re.compile(r"\bssot__[A-Za-z0-9_]+\b"),
]
_LIFECYCLE_CUES = [
    (Lifecycle.RETIRED, re.compile(r"\b(has been retired|is retired|no longer available|end of life)\b", re.I)),
    (Lifecycle.DEPRECATED, re.compile(r"\b(deprecated|will be retired|scheduled for retirement)\b", re.I)),
    (
        Lifecycle.PREVIEW,
        re.compile(r"\b(pilot|beta (?:feature|service|release)|developer preview|preview feature)\b", re.I),
    ),
]
_IDENT_STOPWORDS = {"HTTP_GET", "TODO_ITEM", "NOTE_THAT", "API_VERSION"}


def extract_identifiers(text: str, limit: int = 40) -> list[str]:
    found: list[str] = []
    for pat in _IDENTIFIERS:
        for m in pat.finditer(text):
            v = m.group(0).rstrip(".,;:)")
            if v not in _IDENT_STOPWORDS and len(v) >= 4:
                found.append(v)
    return list(dict.fromkeys(found))[:limit]


def detect_release(text: str) -> str | None:
    hits = Counter(f"{m.group(1).title()} '{m.group(2)}" for m in _RELEASE.finditer(text))
    return hits.most_common(1)[0][0] if hits else None


def detect_api_version(text: str) -> str | None:
    hits = Counter(m.group(1) for m in _API_VERSION.finditer(text))
    return max(hits, key=lambda v: float(v)) if hits else None


def detect_lifecycle(text: str) -> Lifecycle:
    head = text[:3000]
    for state, pat in _LIFECYCLE_CUES:
        if pat.search(head):
            return state
    return Lifecycle.CURRENT


class _LLMDocClass(BaseModel):
    products: list[str] = Field(description="product keys from the allowed list, most relevant first")
    categories: list[KnowledgeCategory]
    content_type: ContentType
    confidence: float = Field(ge=0, le=1)


class LLMClassifier(Protocol):
    def classify(self, title: str, text: str, allowed_products: dict[str, str]) -> _LLMDocClass | None: ...


class RuleClassifier:
    def __init__(self, taxonomy: Taxonomy) -> None:
        self.tx = taxonomy

    # ---- products ---------------------------------------------------------------
    def score_products(self, url: str, title: str, headings: str, body: str, breadcrumb: str = "") -> dict[str, float]:
        url_l = url.lower()
        scores: dict[str, float] = defaultdict(float)
        for key, p in self.tx.products.items():
            if any(pat in url_l for pat in p.url_patterns):
                scores[key] += 5.0
            if p.alias_regex is not None:
                scores[key] += 3.0 * len(p.alias_regex.findall(title))
                scores[key] += 2.0 * len(p.alias_regex.findall(breadcrumb))
                scores[key] += 1.0 * min(10, len(p.alias_regex.findall(headings)))
                scores[key] += 0.3 * min(30, len(p.alias_regex.findall(body)))
            for pat in p.identifier_patterns:
                scores[key] += 0.5 * min(10, len(pat.findall(body)))
        return dict(scores)

    @staticmethod
    def _top(scores: dict[str, float], k: int = 3) -> tuple[list[str], float]:
        ranked = sorted(((v, k_) for k_, v in scores.items() if v > 0), reverse=True)
        if not ranked:
            return [], 0.0
        total = sum(v for v, _ in ranked)
        top_v = ranked[0][0]
        conf = (top_v / total) * min(1.0, top_v / 6.0)
        keep = [k_ for v, k_ in ranked[:k] if v >= top_v * 0.35]
        return keep, round(conf, 3)

    # ---- categories ---------------------------------------------------------------
    def categories(self, title: str, headings: str, body: str) -> list[KnowledgeCategory]:
        t, h, b = title.lower(), headings.lower(), body.lower()[:20000]
        scores: dict[str, float] = defaultdict(float)
        for cat, cues in self.tx.categories.items():
            for cue in cues:
                scores[cat] += 4 * t.count(cue) + 2 * h.count(cue) + min(5, b.count(cue)) * 0.5
        ranked = [c for c, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if s >= 1.0][:3]
        return [KnowledgeCategory(c) for c in ranked] or [KnowledgeCategory.CONCEPT]

    def content_type(
        self, url: str, title: str, cats: list[KnowledgeCategory], blocks_kinds: Counter[str], ordered_lists: int
    ) -> ContentType:
        u, t = url.lower(), title.lower()
        if "release" in u or "release notes" in t or (cats[0] if cats else None) == KnowledgeCategory.RELEASE:
            return ContentType.RELEASE_NOTES
        if KnowledgeCategory.TROUBLESHOOTING in cats[:1]:
            return ContentType.TROUBLESHOOTING
        if KnowledgeCategory.API in cats[:1] or "/reference" in u or "api reference" in t:
            return ContentType.API_REFERENCE
        if KnowledgeCategory.ARCHITECTURE in cats[:1]:
            return ContentType.ARCHITECTURE
        total = sum(blocks_kinds.values()) or 1
        if blocks_kinds["code"] / total > 0.4:
            return ContentType.CODE_SAMPLE
        if blocks_kinds["table"] / total > 0.5:
            return ContentType.REFERENCE
        if ordered_lists:
            return ContentType.PROCEDURE
        return ContentType.CONCEPT

    def personas_intents(self, cats: list[KnowledgeCategory]) -> tuple[list[Persona], list[QueryIntent]]:
        personas: list[Persona] = []
        intents: list[QueryIntent] = []
        for c in cats:
            personas += [Persona(p) for p in self.tx.category_personas.get(c.value, ())]
            intents += [QueryIntent(i) for i in self.tx.category_intents.get(c.value, ())]
        return list(dict.fromkeys(personas)), list(dict.fromkeys(intents))

    # ---- entry points ---------------------------------------------------------------
    def classify_document(self, doc: NormalizedDocument) -> Classification:
        headings = "\n".join(b.text for b in doc.blocks if b.type == "heading")
        body = doc.plain_text()
        scores = self.score_products(doc.origin, doc.title, headings, body, " ".join(doc.breadcrumb))
        products, conf = self._top(scores)
        cats = self.categories(doc.title, headings, body)
        kinds: Counter[str] = Counter(str(b.type) for b in doc.blocks)
        ordered = sum(1 for b in doc.blocks if b.type == "list" and b.ordered)
        personas, intents = self.personas_intents(cats)
        lang = (doc.meta.get("lang") or "en-US").replace("_", "-")
        code_ratio = kinds["code"] / (sum(kinds.values()) or 1)
        complexity = (
            "advanced"
            if code_ratio > 0.25 or KnowledgeCategory.API in cats
            else ("beginner" if KnowledgeCategory.CONCEPT in cats[:1] else "intermediate")
        )
        return Classification(
            products=products,
            product_confidence=conf,
            personas=personas,
            categories=cats,
            intents=intents,
            complexity=complexity,  # type: ignore[arg-type]
            content_type=self.content_type(doc.origin, doc.title, cats, kinds, ordered),
            identifiers=extract_identifiers(body, limit=80),
            release=detect_release(doc.title + "\n" + body[:20000]),
            api_version=detect_api_version(body[:50000]),
            lifecycle=detect_lifecycle(doc.title + "\n" + body),
            locale=lang,
            method="rules",
            tags=doc.breadcrumb[:5],
        )

    def classify_chunk(self, chunk: Chunk, doc_cls: Classification, url: str, title: str) -> Classification:
        """Chunk metadata: specific where the chunk has evidence, inherits doc-level otherwise."""
        headings = " > ".join(chunk.heading_path)
        scores = self.score_products("", "", headings, chunk.body)
        for p in doc_cls.products:  # prior from the document
            scores[p] = scores.get(p, 0.0) + 3.0
        products, conf = self._top(scores)
        cats = self.categories("", headings, chunk.body)
        if cats == [KnowledgeCategory.CONCEPT] and doc_cls.categories:
            cats = doc_cls.categories[:2]
        personas, intents = self.personas_intents(cats)
        ctype = chunk.content_type if chunk.content_type != ContentType.OTHER else doc_cls.content_type
        lifecycle = detect_lifecycle(chunk.body)
        if lifecycle == Lifecycle.CURRENT:
            lifecycle = doc_cls.lifecycle
        return Classification(
            products=products or doc_cls.products,
            product_confidence=max(conf, doc_cls.product_confidence * 0.8),
            features=doc_cls.features,
            personas=personas,
            categories=cats,
            intents=intents,
            complexity=doc_cls.complexity,
            content_type=ctype,
            identifiers=extract_identifiers(chunk.body),
            release=detect_release(chunk.body) or doc_cls.release,
            api_version=detect_api_version(chunk.body) or doc_cls.api_version,
            lifecycle=lifecycle,
            locale=doc_cls.locale,
            method="rules",
            tags=doc_cls.tags,
        )


class TieredClassifier:
    def __init__(
        self,
        rules: RuleClassifier,
        db: Database | None = None,
        llm: LLMClassifier | None = None,
        threshold: float = 0.55,
    ) -> None:
        self.rules = rules
        self.db = db
        self.llm = llm
        self.threshold = threshold

    def classify_document(self, doc: NormalizedDocument, norm_hash: str) -> Classification:
        cls = self.rules.classify_document(doc)
        if self.llm is None or cls.product_confidence >= self.threshold:
            return cls
        key = sha256_text(f"doc:{norm_hash}:tx{self.rules.tx.version}")
        cached = self._cache_get(key)
        result = _LLMDocClass.model_validate(cached) if cached else None
        if result is None:
            allowed = {k: p.label for k, p in self.rules.tx.products.items()}
            result = self.llm.classify(doc.title, doc.markdown()[:12000], allowed)
            if result is None:
                return cls
            self._cache_put(key, result.model_dump(mode="json"))
        valid = [p for p in result.products if p in self.rules.tx.products]
        if valid and result.confidence > cls.product_confidence:
            cls.products = valid[:3]
            cls.product_confidence = round(result.confidence, 3)
            cls.categories = result.categories[:3] or cls.categories
            cls.content_type = result.content_type
            cls.personas, cls.intents = self.rules.personas_intents(cls.categories)
            cls.method = "llm"
        return cls

    def _cache_get(self, key: str) -> dict | None:
        if self.db is None:
            return None
        row = self.db.one("SELECT result_json FROM classification_cache WHERE cache_key=?", (key,))
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, value: dict) -> None:
        if self.db is not None:
            self.db.execute(
                "INSERT OR REPLACE INTO classification_cache(cache_key, result_json, created_at) VALUES (?,?,?)",
                (key, json.dumps(value), now()),
            )
