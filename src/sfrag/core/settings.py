"""Typed, YAML-backed configuration.

Resolution order (later wins): defaults -> config/base.yaml -> config/{SFRAG_ENV}.yaml -> SFRAG_* env vars.
Nested env vars use a double underscore, e.g. SFRAG_LLM__ANSWER_MODEL=claude-sonnet-5-5.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ChunkingSettings(BaseModel):
    child_target_tokens: int = 600
    child_max_tokens: int = 800
    parent_target_tokens: int = 1600
    parent_max_tokens: int = 2200
    overlap_tokens: int = 60
    min_child_tokens: int = 150


class EmbeddingSettings(BaseModel):
    # "bge-m3" = production (dense + sparse from one model). "hashing" = deterministic test/dev backend.
    backend: Literal["bge-m3", "hashing"] = "bge-m3"
    model_name: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    batch_size: int = 16
    max_length: int = 8192
    use_fp16: bool = True
    device: str | None = None  # None = auto (cuda / mps / cpu)


class RerankerSettings(BaseModel):
    backend: Literal["bge-reranker", "lexical", "none"] = "bge-reranker"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = True
    batch_size: int = 16
    max_length: int = 1024


class TokenizerSettings(BaseModel):
    # Pinned to the embedding model's tokenizer so chunk limits are exact.
    backend: Literal["hf", "simple"] = "hf"
    name: str = "BAAI/bge-m3"


class IndexSettings(BaseModel):
    qdrant_url: str | None = None  # set to use a Qdrant server; otherwise embedded on-disk mode
    qdrant_path: str = "qdrant"  # relative to data_dir
    alias: str = "chunks_live"
    collection_prefix: str = "chunks_v"
    quantization: bool = True  # int8 scalar quantisation (server mode)


class RetrievalSettings(BaseModel):
    dense_k: int = 50
    sparse_k: int = 50
    identifier_k: int = 20
    fused_k: int = 50
    rerank_k: int = 40
    max_per_doc: int = 3
    final_k: int = 12
    context_token_budget: int = 12000
    sufficiency_threshold: float = 0.30
    abstain_threshold: float = 0.15
    plan_confidence_for_hard_filter: float = 0.85
    # Score adjustments added to the reranker score (reranker scores are normalised to 0..1).
    boost_product_match: float = 0.08
    boost_category_match: float = 0.03
    boost_persona_match: float = 0.03
    boost_current_release: float = 0.04
    boost_authority_tier1: float = 0.04
    penalty_deprecated: float = 0.15
    penalty_retired: float = 0.30
    penalty_preview: float = 0.03


class LLMSettings(BaseModel):
    answer_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"  # query plans, condense, low-confidence classification
    judge_model: str = "claude-sonnet-5-5"
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    fast_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    answer_max_tokens: int = 8000
    server_side_fallback: bool = True  # fallbacks="default" on refusals (Claude API only)
    timeout_seconds: float = 120.0
    enabled: bool = True  # False = no LLM calls (planner/condense/classifier skip; answers disabled)


class FetchSettings(BaseModel):
    user_agent: str = "sfrag-docs-indexer/0.1 (+internal documentation research)"
    requests_per_second_per_host: float = 1.0
    timeout_seconds: float = 30.0
    max_retries: int = 4
    max_bytes: int = 50_000_000
    follow_links: bool = True
    max_link_depth: int = 3
    respect_robots: bool = True


class QueueSettings(BaseModel):
    lease_seconds: int = 300
    max_attempts: int = 5
    poll_interval_seconds: float = 1.0


class ClassificationSettings(BaseModel):
    llm_fallback: bool = True
    llm_confidence_threshold: float = 0.55
    taxonomy_path: str = "taxonomy/taxonomy.yaml"


class ApiSettings(BaseModel):
    # api key -> principal id. Empty mapping = auth disabled (local dev only).
    api_keys: dict[str, str] = Field(default_factory=dict)
    # principal -> groups (for ACLs on user-uploaded files)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    max_question_chars: int = 4000
    rate_limit_per_minute: int = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SFRAG_", env_nested_delimiter="__", extra="ignore")

    env: str = "dev"
    data_dir: Path = PROJECT_ROOT / "data"
    sources_path: Path = PROJECT_ROOT / "config" / "sources.yaml"
    pipeline_version: str = "p1"  # bump when extraction/chunking/classification logic changes
    current_release: str = "Winter '27"
    log_level: str = "INFO"

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    # ---- derived paths -------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "sqlite" / "sfrag.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def incoming_dir(self) -> Path:
        return self.data_dir / "incoming"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def qdrant_location(self) -> Path:
        return self.data_dir / self.index.qdrant_path

    @property
    def taxonomy_file(self) -> Path:
        p = Path(self.classification.taxonomy_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def embedding_model_version(self) -> str:
        return f"{self.embedding.backend}:{self.embedding.model_name}"

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.raw_dir, self.normalized_dir, self.incoming_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _env_overrides() -> dict[str, Any]:
    """Collect SFRAG_* env vars as a nested dict so they win over YAML values."""
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("SFRAG_") or key in ("SFRAG_ENV", "SFRAG_CONFIG_DIR"):
            continue
        parts = key[len("SFRAG_") :].lower().split("__")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_settings(env: str | None = None, config_dir: Path | None = None, **overrides: Any) -> Settings:
    env = env or os.environ.get("SFRAG_ENV", "dev")
    config_dir = config_dir or Path(os.environ.get("SFRAG_CONFIG_DIR", PROJECT_ROOT / "config"))
    merged = _deep_merge(_load_yaml(config_dir / "base.yaml"), _load_yaml(config_dir / f"{env}.yaml"))
    merged = _deep_merge(merged, _env_overrides())
    merged = _deep_merge(merged, overrides)
    merged["env"] = env
    return Settings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
