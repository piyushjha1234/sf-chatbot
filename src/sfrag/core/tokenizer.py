"""Token counting pinned to the embedding model's tokenizer."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from sfrag.core.settings import TokenizerSettings


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class SimpleTokenizer:
    """Deterministic approximation (words + punctuation). For tests and offline dev only."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._pattern.findall(text))


class HFTokenizer:
    """Exact counts using the Hugging Face `tokenizers` fast tokenizer (e.g. BAAI/bge-m3)."""

    def __init__(self, name: str) -> None:
        from tokenizers import Tokenizer as _HF

        self._tok = _HF.from_pretrained(name)
        self._tok.no_truncation()
        self._tok.no_padding()

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False).ids)


@lru_cache(maxsize=4)
def _build(backend: str, name: str) -> Tokenizer:
    if backend == "simple":
        return SimpleTokenizer()
    return HFTokenizer(name)


def get_tokenizer(cfg: TokenizerSettings) -> Tokenizer:
    return _build(cfg.backend, cfg.name)
