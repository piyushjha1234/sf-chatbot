from __future__ import annotations

import io
import json

import pytest

from sfrag.pipeline.extract import (
    ExtractionError,
    NeedsRender,
    UnsupportedFormat,
    detect_mime,
    parse_document,
)
from tests.conftest import D360, PAGES


def test_html_structure_and_boilerplate() -> None:
    url = f"{D360}/data-streams.html"
    doc = parse_document(PAGES[url], url, "text/html")
    kinds = [b.type for b in doc.blocks]
    assert doc.title == "Create a Data Stream"
    assert "heading" in kinds and "list" in kinds and "admonition" in kinds
    ordered = [b for b in doc.blocks if b.type == "list" and b.ordered]
    assert ordered and ordered[0].text.startswith("1. From Data Cloud")
    text = doc.markdown()
    assert "Cookie preferences" not in text and "Login" not in text
    assert "`" not in text or "Individual__dlm" in text
    assert doc.meta.get("lang") == "en-US"


def test_html_tables_and_code() -> None:
    url = f"{D360}/query-api.html"
    doc = parse_document(PAGES[url], url, "text/html")
    table = next(b for b in doc.blocks if b.type == "table")
    assert table.table_header and table.table_header.startswith("| Code | Meaning |")
    assert any("INVALID_SQL_SYNTAX" in r for r in table.table_rows)
    code = next(b for b in doc.blocks if b.type == "code")
    assert code.language == "http" and "POST /api/v2/query" in code.text


def test_js_shell_needs_render() -> None:
    html = (
        b"<html><head><script src=a.js></script><script>window.__x=1</script><script></script></head>"
        b"<body><div id='app'>Loading...</div></body></html>"
    )
    with pytest.raises(NeedsRender):
        parse_document(html, "https://help.salesforce.com/s/articleView?id=x", "text/html")


def test_markdown_blocks() -> None:
    md = (
        b"# Title\n\nIntro text.\n\n## Steps\n\n1. one\n2. two\n\n```sql\nSELECT 1\n```\n\n"
        b"| a | b |\n|---|---|\n| 1 | 2 |\n\n> **Note:** careful\n"
    )
    doc = parse_document(md, "file://x.md", "text/markdown")
    kinds = [b.type for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "heading", "list", "code", "table", "admonition"]


def test_openapi() -> None:
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Query API", "version": "2"},
        "paths": {
            "/api/v2/query": {
                "post": {
                    "summary": "Run a query",
                    "parameters": [{"name": "limit", "in": "query"}],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    doc = parse_document(json.dumps(spec).encode(), "file://q.json", "application/json")
    texts = [b.text for b in doc.blocks]
    assert "## POST /api/v2/query" in texts
    assert any("`limit`" in t for t in texts)


def test_office_formats() -> None:
    import docx
    import openpyxl
    from pptx import Presentation

    d = docx.Document()
    d.add_heading("Setup Guide", level=1)
    d.add_paragraph("Assign permissions first.")
    d.add_paragraph("Open setup", style="List Number")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "Limit", "Value", "Streams", "100"
    buf = io.BytesIO()
    d.save(buf)
    doc = parse_document(buf.getvalue(), "file://g.docx", detect_mime("g.docx", buf.getvalue()))
    assert [b.type for b in doc.blocks] == ["heading", "paragraph", "list", "table"]

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Architecture"
    s.placeholders[1].text = "Ingest via data streams"
    buf = io.BytesIO()
    prs.save(buf)
    doc = parse_document(buf.getvalue(), "file://a.pptx", detect_mime("a.pptx", buf.getvalue()))
    assert any("Slide 1: Architecture" in b.text for b in doc.blocks)

    wb = openpyxl.Workbook()
    wb.active.append(["Object", "Type"])
    wb.active.append(["Individual__dlm", "DMO"])
    buf = io.BytesIO()
    wb.save(buf)
    doc = parse_document(buf.getvalue(), "file://o.xlsx", detect_mime("o.xlsx", buf.getvalue()))
    assert any(b.type == "table" and "Individual__dlm" in b.text for b in doc.blocks)


def test_pdf_headings_and_pages() -> None:
    import pymupdf

    pdf = pymupdf.open()
    for i in range(2):
        pg = pdf.new_page()
        pg.insert_text((72, 72), f"Chapter {i + 1}", fontsize=22)
        pg.insert_text((72, 110), "Body text describing data model objects in detail for page reference.", fontsize=11)
        pg.insert_text((72, 130), "More body text so the body size dominates the document overall.", fontsize=11)
    data = pdf.tobytes()
    doc = parse_document(data, "file://g.pdf", detect_mime("g.pdf", data))
    heads = [b for b in doc.blocks if b.type == "heading"]
    assert [h.page for h in heads] == [1, 2]
    assert heads[0].text.startswith("# Chapter 1")


def test_errors_are_explicit() -> None:
    with pytest.raises(ExtractionError):
        parse_document(b"{not json", "file://bad.json", "application/json")
    with pytest.raises(UnsupportedFormat):
        parse_document(b"\x00\x01", "file://x.bin", "application/octet-stream")


def test_detect_mime() -> None:
    assert detect_mime("a.cls", b"public class A {}") == "text/x-code;apex"
    assert detect_mime("x", b"%PDF-1.7") == "application/pdf"
    assert detect_mime("page", b"<!doctype html><html>") == "text/html"
    assert detect_mime("spec.yaml", b"openapi: 3.0.0", "application/octet-stream") == "application/yaml"
