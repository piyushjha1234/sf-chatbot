"""Embedding backends. Production: BAAI/bge-m3 (dense + sparse lexical weights from one forward pass)."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from sfrag.core.settings import EmbeddingSettings


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass
class Embedding:
    dense: list[float]
    sparse: SparseVec


class Embedder(Protocol):
    dim: int
    sparse_uses_idf: bool

    def embed_documents(self, texts: list[str]) -> list[Embedding]: ...
    def embed_query(self, text: str) -> Embedding: ...


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class BGEM3Embedder:
    """bge-m3 via FlagEmbedding. Install with `uv sync --extra models`."""

    sparse_uses_idf = False  # lexical weights are already learned term importances

    def __init__(self, cfg: EmbeddingSettings) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError("bge-m3 backend needs the 'models' extra: uv sync --extra models") from e
        device = cfg.device or _auto_device()
        use_fp16 = cfg.use_fp16 and device != "cpu"
        try:
            self._model = BGEM3FlagModel(cfg.model_name, use_fp16=use_fp16, devices=device)
        except TypeError:  # older FlagEmbedding signature
            self._model = BGEM3FlagModel(cfg.model_name, use_fp16=use_fp16, device=device)
        self.cfg = cfg
        self.dim = cfg.dense_dim
        self._lock = threading.Lock()

    def _encode(self, texts: list[str], max_length: int) -> list[Embedding]:
        with self._lock:
            out = self._model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        result = []
        for dense, lex in zip(out["dense_vecs"], out["lexical_weights"], strict=True):
            items = sorted((int(k), float(v)) for k, v in lex.items() if float(v) > 0)
            result.append(
                Embedding(
                    dense=[float(x) for x in dense], sparse=SparseVec([i for i, _ in items], [v for _, v in items])
                )
            )
        return result

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return self._encode(texts, self.cfg.max_length) if texts else []

    def embed_query(self, text: str) -> Embedding:
        return self._encode([text], 512)[0]


_TOKEN = re.compile(r"[a-z0-9_]+(?:__[a-z]+)?|[^\sa-z0-9]", re.IGNORECASE)


def _h(s: str, mod: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % mod


class HashingEmbedder:
    """Deterministic, dependency-free backend for tests and offline pipeline dev. NOT for production quality."""

    sparse_uses_idf = True

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _one(self, text: str) -> Embedding:
        toks = [t.lower() for t in _TOKEN.findall(text) if t.strip() and (t.isalnum() or "_" in t)]
        vec = [0.0] * self.dim
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        for t in feats:
            idx = _h(t, self.dim)
            vec[idx] += 1.0 if _h("s" + t, 2) else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        tf = Counter(_h(t, 1 << 24) for t in toks)
        items = sorted(tf.items())
        return Embedding(
            dense=[x / norm for x in vec],
            sparse=SparseVec([i for i, _ in items], [1.0 + math.log(c) for _, c in items]),
        )

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._one(text)


def build_embedder(cfg: EmbeddingSettings) -> Embedder:
    if cfg.backend == "hashing":
        return HashingEmbedder(cfg.dense_dim)
    return BGEM3Embedder(cfg)
