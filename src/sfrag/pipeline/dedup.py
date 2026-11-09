"""Near-duplicate detection (MinHash LSH) and canonical selection."""

from __future__ import annotations

import re
import threading

import numpy as np
from datasketch import MinHash, MinHashLSH

from sfrag.storage.db import Database

NUM_PERM = 128
_WORD = re.compile(r"\w+")


def minhash_for(text: str, shingle: int = 5) -> MinHash:
    words = _WORD.findall(text.lower())
    m = MinHash(num_perm=NUM_PERM)
    if len(words) < shingle:
        m.update(" ".join(words).encode())
        return m
    for i in range(len(words) - shingle + 1):
        m.update(" ".join(words[i : i + shingle]).encode())
    return m


def to_bytes(m: MinHash) -> bytes:
    return np.asarray(m.hashvalues, dtype=np.uint64).tobytes()


def from_bytes(b: bytes) -> MinHash:
    return MinHash(num_perm=NUM_PERM, hashvalues=np.frombuffer(b, dtype=np.uint64))


class DedupIndex:
    """Process-local LSH over canonical documents, lazily built from the DB."""

    def __init__(self, db: Database, threshold: float = 0.9) -> None:
        self.db = db
        self.threshold = threshold
        self._lsh: MinHashLSH | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> MinHashLSH:
        if self._lsh is None:
            lsh = MinHashLSH(threshold=self.threshold, num_perm=NUM_PERM)
            for r in self.db.query(
                "SELECT doc_id, minhash FROM documents WHERE status='active' AND minhash IS NOT NULL"
            ):
                lsh.insert(r["doc_id"], from_bytes(r["minhash"]))
            self._lsh = lsh
        return self._lsh

    def find_duplicate(self, doc_id: str, m: MinHash) -> str | None:
        with self._lock:
            lsh = self._ensure()
            for other in lsh.query(m):
                if other != doc_id:
                    return str(other)
        return None

    def check_and_reserve(self, doc_id: str, m: MinHash) -> str | None:
        """Atomic: return an existing near-duplicate, or register `doc_id` as canonical and return None.
        Prevents two workers processing mirror copies concurrently from both becoming canonical."""
        with self._lock:
            lsh = self._ensure()
            for other in lsh.query(m):
                if other != doc_id:
                    return str(other)
            if doc_id in lsh:
                lsh.remove(doc_id)
            lsh.insert(doc_id, m)
        return None

    def add(self, doc_id: str, m: MinHash) -> None:
        with self._lock:
            lsh = self._ensure()
            if doc_id in lsh:
                lsh.remove(doc_id)
            lsh.insert(doc_id, m)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            lsh = self._ensure()
            if doc_id in lsh:
                lsh.remove(doc_id)


def prefer_new(new_tier: int, new_locale: str, old_tier: int, old_locale: str) -> bool:
    """Canonical selection: higher authority (lower tier number) first, then en-US, else keep incumbent."""
    if new_tier != old_tier:
        return new_tier < old_tier
    new_en, old_en = new_locale.lower().startswith("en"), old_locale.lower().startswith("en")
    return new_en and not old_en
