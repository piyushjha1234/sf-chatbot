"""Parser adapter interface and format detection."""

from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath
from typing import Protocol

from sfrag.core.models import NormalizedDocument


class ExtractionError(Exception):
    """Malformed input; reported explicitly, never silently dropped."""


class UnsupportedFormat(ExtractionError):
    pass


class NeedsRender(ExtractionError):
    """The HTML is a JavaScript shell without server-rendered content (needs a headless browser)."""


class Parser(Protocol):
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument: ...


CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cls": "apex",
    ".trigger": "apex",
    ".apex": "apex",
    ".soql": "sql",
    ".sql": "sql",
    ".sh": "bash",
    ".go": "go",
    ".rb": "ruby",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".php": "php",
    ".css": "css",
    ".scss": "scss",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".amp": "ampscript",
    ".ssjs": "javascript",
    ".graphql": "graphql",
    ".gql": "graphql",
}

_EXT_MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".xml": "application/xml",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def detect_mime(name: str, data: bytes, declared: str | None = None) -> str:
    """Best-effort format detection: declared content-type, extension, then magic bytes."""
    declared = (declared or "").split(";")[0].strip().lower()
    ext = PurePosixPath(name.split("?")[0].lower()).suffix
    if ext in CODE_EXTENSIONS:
        return f"text/x-code;{CODE_EXTENSIONS[ext]}"
    if ext in _EXT_MIME and declared in ("", "application/octet-stream", "binary/octet-stream", "text/plain"):
        return _EXT_MIME[ext]
    if declared and declared not in ("application/octet-stream", "binary/octet-stream"):
        if declared in ("application/xhtml+xml",):
            return "text/html"
        if declared in ("text/xml",):
            return "application/xml"
        if declared in ("application/x-yaml", "text/yaml", "text/x-yaml"):
            return "application/yaml"
        return declared
    if ext in _EXT_MIME:
        return _EXT_MIME[ext]
    head = data[:512].lstrip().lower()
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if b"\x00" in data[:4096] and not data.startswith(b"PK\x03\x04"):
        return "application/octet-stream"  # binary we have no adapter for
    if data.startswith(b"PK\x03\x04"):
        guessed = mimetypes.guess_type(name)[0]
        return guessed or "application/zip"
    if head.startswith((b"<!doctype html", b"<html")) or b"<body" in head:
        return "text/html"
    if head.startswith(b"<?xml") or head.startswith(b"<"):
        return "application/xml"
    if head.startswith((b"{", b"[")):
        return "application/json"
    return "text/plain"
