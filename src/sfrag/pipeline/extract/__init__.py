"""Parser registry: one interface, adapters per format."""

from __future__ import annotations

from sfrag.core.models import NormalizedDocument
from sfrag.pipeline.extract.base import (
    ExtractionError,
    NeedsRender,
    Parser,
    UnsupportedFormat,
    detect_mime,
)
from sfrag.pipeline.extract.html import HtmlParser
from sfrag.pipeline.extract.markdown import MarkdownParser, TextParser
from sfrag.pipeline.extract.office import CsvParser, DocxParser, PptxParser, XlsxParser
from sfrag.pipeline.extract.pdf import PdfParser
from sfrag.pipeline.extract.structured import CodeParser, JsonYamlParser, XmlParser

_PARSERS: dict[str, Parser] = {
    "text/html": HtmlParser(),
    "text/markdown": MarkdownParser(),
    "text/plain": TextParser(),
    "application/pdf": PdfParser(),
    "application/json": JsonYamlParser(),
    "application/yaml": JsonYamlParser(),
    "application/xml": XmlParser(),
    "text/csv": CsvParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser(),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PptxParser(),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XlsxParser(),
}
_CODE = CodeParser()


def parse_document(data: bytes, origin: str, mime: str) -> NormalizedDocument:
    if mime.startswith("text/x-code"):
        return _CODE.parse(data, origin, mime)
    parser = _PARSERS.get(mime.split(";")[0])
    if parser is None:
        raise UnsupportedFormat(f"no parser for {mime}")
    doc = parser.parse(data, origin, mime.split(";")[0])
    if not any(b.text.strip() for b in doc.blocks):
        raise ExtractionError("document has no extractable content")
    return doc


__all__ = [
    "ExtractionError",
    "NeedsRender",
    "UnsupportedFormat",
    "detect_mime",
    "parse_document",
]
