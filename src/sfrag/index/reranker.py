"""Cross-encoder reranking. Production: BAAI/bge-reranker-v2-m3 (scores normalised to 0..1)."""

from __future__ import annotations

import math
import re
import threading
from typing import Protocol

from sfrag.core.settings import RerankerSettings


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class BGEReranker:
    def __init__(self, cfg: RerankerSettings, device: str | None = None) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("bge-reranker backend needs the 'models' extra: uv sync --extra models") from e
        from sfrag.index.embedder import _auto_device

        dev = device or _auto_device()
        use_fp16 = cfg.use_fp16 and dev != "cpu"
        try:
            self._model = FlagReranker(cfg.model_name, use_fp16=use_fp16, devices=dev)
        except TypeError:
            self._model = FlagReranker(cfg.model_name, use_fp16=use_fp16, device=dev)
        self.cfg = cfg
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        with self._lock:
            s = self._model.compute_score(
                pairs, normalize=True, batch_size=self.cfg.batch_size, max_length=self.cfg.max_length
            )
        return [float(x) for x in (s if isinstance(s, list) else [s])]


_W = re.compile(r"\w+")


class LexicalReranker:
    """Term-overlap scorer for tests/offline dev only."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = {w.lower() for w in _W.findall(query) if len(w) > 2}
        out = []
        for p in passages:
            words = [w.lower() for w in _W.findall(p)]
            if not q or not words:
                out.append(0.0)
                continue
            hits = sum(1 for w in words if w in q)
            coverage = len(q & set(words)) / len(q)
            raw = coverage * 3 + math.log1p(hits) * 0.5
            out.append(raw / (raw + 1.5))
        return out


class NoReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.5] * len(passages)


def build_reranker(cfg: RerankerSettings) -> Reranker:
    if cfg.backend == "lexical":
        return LexicalReranker()
    if cfg.backend == "none":
        return NoReranker()
    return BGEReranker(cfg)
