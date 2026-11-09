"""Dependency wiring with lazy construction (heavy models load only when first needed)."""

from __future__ import annotations

import threading
from functools import cached_property

from sfrag.core.logging import configure_logging
from sfrag.core.settings import Settings, get_settings
from sfrag.core.taxonomy import Taxonomy, get_taxonomy
from sfrag.core.tokenizer import Tokenizer, get_tokenizer
from sfrag.enrich.chunker import HierarchicalChunker
from sfrag.enrich.classify import RuleClassifier, TieredClassifier
from sfrag.index.embedder import Embedder, build_embedder
from sfrag.index.indexer import Indexer
from sfrag.index.reranker import Reranker, build_reranker
from sfrag.index.store import VectorStore
from sfrag.llm.client import LLMClient
from sfrag.pipeline.dedup import DedupIndex
from sfrag.pipeline.queue import JobQueue
from sfrag.query.engine import AnswerEngine
from sfrag.query.planner import QueryPlanner
from sfrag.query.retriever import HybridRetriever
from sfrag.sources.fetch import Fetcher
from sfrag.storage.blobs import NormalizedStore, RawStore
from sfrag.storage.db import Database


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        configure_logging(self.settings.log_level)
        self._lock = threading.RLock()

    @cached_property
    def db(self) -> Database:
        return Database(self.settings.db_path)

    @cached_property
    def queue(self) -> JobQueue:
        q = self.settings.queue
        return JobQueue(self.db, lease_seconds=q.lease_seconds, max_attempts=q.max_attempts)

    @cached_property
    def raw(self) -> RawStore:
        return RawStore(self.settings.raw_dir)

    @cached_property
    def normalized(self) -> NormalizedStore:
        return NormalizedStore(self.settings.normalized_dir)

    @cached_property
    def taxonomy(self) -> Taxonomy:
        return get_taxonomy(str(self.settings.taxonomy_file))

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.settings.tokenizer)

    @cached_property
    def rules(self) -> RuleClassifier:
        return RuleClassifier(self.taxonomy)

    @cached_property
    def llm(self) -> LLMClient:
        return LLMClient(self.settings.llm)

    @cached_property
    def classifier(self) -> TieredClassifier:
        llm_cls = None
        if self.settings.llm.enabled and self.settings.classification.llm_fallback:
            from sfrag.llm.classifier import LLMDocClassifier

            llm_cls = LLMDocClassifier(self.llm)
        return TieredClassifier(self.rules, self.db, llm_cls, self.settings.classification.llm_confidence_threshold)

    @cached_property
    def chunker(self) -> HierarchicalChunker:
        return HierarchicalChunker(self.settings.chunking, self.tokenizer)

    @cached_property
    def dedup(self) -> DedupIndex:
        return DedupIndex(self.db)

    @cached_property
    def fetcher(self) -> Fetcher:
        return Fetcher(self.settings.fetch)

    @cached_property
    def embedder(self) -> Embedder:
        with self._lock:
            return build_embedder(self.settings.embedding)

    @cached_property
    def reranker(self) -> Reranker:
        with self._lock:
            return build_reranker(self.settings.reranker)

    @cached_property
    def store(self) -> VectorStore:
        return VectorStore(self.settings)

    @cached_property
    def indexer(self) -> Indexer:
        return Indexer(self.settings, self.db, self.store, self.embedder)

    @cached_property
    def planner(self) -> QueryPlanner:
        return QueryPlanner(self.llm, self.taxonomy, self.rules)

    @cached_property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(
            self.store, self.embedder, self.reranker, self.settings.retrieval, self.settings.current_release
        )

    @cached_property
    def engine(self) -> AnswerEngine:
        self.indexer.live()  # make sure the live alias exists
        return AnswerEngine(self.settings, self.db, self.store, self.retriever, self.planner, self.llm, self.tokenizer)
