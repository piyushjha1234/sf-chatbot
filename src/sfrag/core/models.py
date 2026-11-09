"""Canonical domain models shared by ingestion and inference."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------------------


class Persona(StrEnum):
    AUTO = "auto"
    BUSINESS = "business_marketer"
    FUNCTIONAL = "functional_consultant"
    ADMIN = "administrator"
    DEVELOPER = "developer"
    ARCHITECT = "solution_architect"
    DATA = "data_engineer"
    SUPPORT = "support"


class Depth(StrEnum):
    BEGINNER = "beginner"
    STANDARD = "standard"
    DEEP = "deep_technical"


class KnowledgeCategory(StrEnum):
    CONCEPT = "concept"
    CONFIGURATION = "configuration"
    IMPLEMENTATION = "implementation"
    API = "api"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    GOVERNANCE = "governance"
    LIMITS = "limits"
    TROUBLESHOOTING = "troubleshooting"
    MIGRATION = "migration"
    RELEASE = "release_change"


class QueryIntent(StrEnum):
    EXPLAIN = "explain"
    CONFIGURE = "configure"
    IMPLEMENT = "implement"
    DEVELOP = "develop"
    TROUBLESHOOT = "troubleshoot"
    AUDIT = "audit"
    DESIGN = "design"
    COMPARE = "compare"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    EVALUATE = "evaluate"
    ADMINISTER = "administer"


class ContentType(StrEnum):
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    REFERENCE = "reference"
    API_REFERENCE = "api_reference"
    CODE_SAMPLE = "code_sample"
    TROUBLESHOOTING = "troubleshooting"
    RELEASE_NOTES = "release_notes"
    ARCHITECTURE = "architecture"
    TABLE_DATA = "table_data"
    OTHER = "other"


class Lifecycle(StrEnum):
    CURRENT = "current"
    PREVIEW = "preview"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    ARCHIVED = "archived"


BlockType = Literal["heading", "paragraph", "list", "table", "code", "admonition", "image"]


# --------------------------------------------------------------------------------------
# Normalised document model
# --------------------------------------------------------------------------------------


class Block(BaseModel):
    """One structural unit of a normalised document. `text` is Markdown."""

    type: BlockType
    text: str
    level: int = 0  # heading level (1-6); list nesting for lists
    page: int | None = None  # PDF page reference (1-based)
    language: str | None = None  # code language
    ordered: bool = False  # ordered list (procedure steps)
    table_header: str | None = None  # Markdown header row(s) for tables
    table_rows: list[str] = Field(default_factory=list)
    admonition_kind: str | None = None  # note / warning / important / tip


class NormalizedDocument(BaseModel):
    title: str
    blocks: list[Block]
    origin: str  # canonical URL or file:// URI
    mime: str
    links: list[str] = Field(default_factory=list)
    breadcrumb: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)  # parser-provided page metadata

    def markdown(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


class Classification(BaseModel):
    """Metadata shared by documents (broad) and chunks (specific)."""

    products: list[str] = Field(default_factory=list)
    product_confidence: float = 0.0
    features: list[str] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    categories: list[KnowledgeCategory] = Field(default_factory=list)
    intents: list[QueryIntent] = Field(default_factory=list)
    complexity: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    content_type: ContentType = ContentType.OTHER
    tags: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)  # API names, __c fields, endpoints, error codes
    release: str | None = None
    api_version: str | None = None
    lifecycle: Lifecycle = Lifecycle.CURRENT
    locale: str = "en-US"
    method: Literal["rules", "knn", "llm", "none"] = "rules"


class DocumentRecord(BaseModel):
    doc_id: str
    origin: str
    source_id: str
    title: str
    norm_hash: str
    version: int
    authority_tier: int
    acl: list[str]  # ["public"] or principals/groups
    classification: Classification
    pipeline_version: str
    canonical_of: str | None = None  # set when this doc is a near-duplicate of another
    quarantined: bool = False


class Section(BaseModel):
    """Parent unit: a structurally coherent span (≈1.6k tokens) used for context expansion."""

    parent_id: str
    doc_id: str
    ordinal: int
    heading_path: list[str]
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None


class Chunk(BaseModel):
    """Child unit: the retrieval granule (≈600 tokens)."""

    chunk_id: str
    doc_id: str
    parent_id: str
    ordinal: int  # order within document
    heading_path: list[str]
    body: str  # unique content of this chunk (lossless across chunks)
    context_prefix: str = ""  # repeated context (overlap tail or table header); not unique content
    token_count: int
    content_type: ContentType = ContentType.OTHER
    group_id: str | None = None  # links split code/table/API blocks
    page: int | None = None
    classification: Classification | None = None

    @property
    def text(self) -> str:
        return f"{self.context_prefix}{self.body}" if self.context_prefix else self.body


# --------------------------------------------------------------------------------------
# Query-time models
# --------------------------------------------------------------------------------------


class QueryPlan(BaseModel):
    standalone_question: str
    products: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    categories: list[KnowledgeCategory] = Field(default_factory=list)
    intents: list[QueryIntent] = Field(default_factory=list)
    persona: Persona = Persona.AUTO
    depth: Depth = Depth.STANDARD
    release_hint: str | None = None
    explicit_product: bool = False  # user literally named the product
    cross_product: bool = False
    sub_questions: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: Literal["llm", "heuristic", "skipped"] = "heuristic"


class Candidate(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    payload: dict[str, Any]
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class EvidenceBlock(BaseModel):
    sid: str  # "S1"...
    doc_id: str
    parent_id: str
    chunk_ids: list[str]
    title: str
    url: str
    heading: str
    page: int | None
    release: str | None
    text: str
    score: float
    untrusted: bool = False


class Citation(BaseModel):
    sid: str
    title: str
    url: str
    heading: str
    page: int | None = None
    release: str | None = None


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    abstained: bool
    confidence: float
    plan: QueryPlan | None = None
    invalid_citations: list[str] = Field(default_factory=list)
    uncited_sentence_ratio: float = 0.0
    stop_reason: str | None = None
    model: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
