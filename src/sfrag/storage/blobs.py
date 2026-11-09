"""Content-addressed file stores for raw bytes and normalised documents."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sfrag.core.ids import sha256_bytes


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put(self, data: bytes, meta: dict[str, Any] | None = None) -> str:
        digest = sha256_bytes(data)
        p = self.path_for(digest)
        if not p.exists():
            _atomic_write(p, data)
        if meta is not None:
            _atomic_write(p.with_suffix(".meta.json"), json.dumps(meta, sort_keys=True, default=str).encode())
        return digest

    def get(self, digest: str) -> bytes:
        return self.path_for(digest).read_bytes()

    def meta(self, digest: str) -> dict[str, Any]:
        p = self.path_for(digest).with_suffix(".meta.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def delete(self, digest: str) -> None:
        self.path_for(digest).unlink(missing_ok=True)


class NormalizedStore:
    """Stores the normalised Markdown + JSON sidecar per document version."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, doc_id: str, version: int, markdown: str, sidecar: dict[str, Any]) -> Path:
        base = self.root / doc_id[:4] / doc_id / f"v{version}"
        _atomic_write(base.with_suffix(".md"), markdown.encode("utf-8"))
        _atomic_write(base.with_suffix(".json"), json.dumps(sidecar, indent=1, default=str).encode("utf-8"))
        return base.with_suffix(".md")
