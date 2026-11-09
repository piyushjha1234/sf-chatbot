"""DOCX, PPTX, XLSX and CSV -> Blocks."""

from __future__ import annotations

import csv
import io

from sfrag.core.models import Block, NormalizedDocument
from sfrag.pipeline.extract.base import ExtractionError


def _md_table(rows: list[list[str]]) -> Block | None:
    rows = [[(c or "").strip().replace("\n", " ").replace("|", "\\|") for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |\n|" + "|".join(["---"] * width) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return Block(type="table", text="\n".join([header, *body]), table_header=header, table_rows=body)


class DocxParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        try:
            d = docx.Document(io.BytesIO(data))
        except Exception as e:
            raise ExtractionError(f"cannot open DOCX: {e}") from e
        blocks: list[Block] = []
        list_buf: list[str] = []
        list_ordered = False

        def flush_list() -> None:
            nonlocal list_ordered
            if list_buf:
                blocks.append(Block(type="list", text="\n".join(list_buf), ordered=list_ordered))
                list_buf.clear()
                list_ordered = False

        for el in d.element.body.iterchildren():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(el, d)
                text = p.text.strip()
                style = (p.style.name if p.style is not None else "") or ""
                if not text:
                    continue
                if style.startswith("Heading") or style == "Title":
                    flush_list()
                    try:
                        lvl = 1 if style == "Title" else int(style.split()[-1])
                    except ValueError:
                        lvl = 2
                    blocks.append(Block(type="heading", text=f"{'#' * lvl} {text}", level=lvl))
                elif "List" in style:
                    ordered = "Number" in style
                    if list_buf and ordered != list_ordered:
                        flush_list()
                    list_ordered = ordered
                    marker = f"{len(list_buf) + 1}." if ordered else "-"
                    list_buf.append(f"{marker} {text}")
                elif "Code" in style or (
                    p.runs and all((r.font.name or "").lower().startswith(("courier", "consolas")) for r in p.runs)
                ):
                    flush_list()
                    if blocks and blocks[-1].type == "code":
                        prev = blocks[-1].text[:-4]
                        blocks[-1] = Block(type="code", text=f"{prev}\n{text}\n```")
                    else:
                        blocks.append(Block(type="code", text=f"```\n{text}\n```"))
                else:
                    flush_list()
                    low = text.lower()
                    if low.startswith(("note:", "warning:", "important:", "caution:", "tip:")):
                        blocks.append(Block(type="admonition", text=text, admonition_kind=low.split(":")[0]))
                    else:
                        blocks.append(Block(type="paragraph", text=text))
            elif tag == "tbl":
                flush_list()
                t = Table(el, d)
                b = _md_table([[c.text for c in row.cells] for row in t.rows])
                if b:
                    blocks.append(b)
        flush_list()
        title = (d.core_properties.title or "").strip()
        if not title:
            h = next((b for b in blocks if b.type == "heading"), None)
            title = h.text.lstrip("#").strip() if h else origin.rsplit("/", 1)[-1]
        return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime)


class PptxParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        from pptx import Presentation

        try:
            prs = Presentation(io.BytesIO(data))
        except Exception as e:
            raise ExtractionError(f"cannot open PPTX: {e}") from e
        blocks: list[Block] = []
        for i, slide in enumerate(prs.slides, start=1):
            title = ""
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
            blocks.append(Block(type="heading", text=f"## Slide {i}: {title}".rstrip(": "), level=2, page=i))
            for shape in slide.shapes:
                if shape == slide.shapes.title:
                    continue
                if getattr(shape, "has_table", False) and shape.has_table:
                    b = _md_table([[c.text for c in row.cells] for row in shape.table.rows])
                    if b:
                        b.page = i
                        blocks.append(b)
                elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                    lines = []
                    for para in shape.text_frame.paragraphs:
                        t = "".join(r.text for r in para.runs).strip()
                        if t:
                            lines.append("    " * para.level + f"- {t}")
                    if lines:
                        blocks.append(Block(type="list", text="\n".join(lines), page=i))
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
                if notes:
                    blocks.append(Block(type="paragraph", text=f"Speaker notes: {notes}", page=i))
        title = (prs.core_properties.title or "").strip() or origin.rsplit("/", 1)[-1]
        blocks.insert(0, Block(type="heading", text=f"# {title}", level=1))
        return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime)


class XlsxParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        import openpyxl

        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        except Exception as e:
            raise ExtractionError(f"cannot open XLSX: {e}") from e
        title = origin.rsplit("/", 1)[-1]
        blocks: list[Block] = [Block(type="heading", text=f"# {title}", level=1)]
        for ws in wb.worksheets:
            rows = [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
            b = _md_table(rows)
            if b:
                blocks.append(Block(type="heading", text=f"## {ws.title}", level=2))
                blocks.append(b)
        return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime)


class CsvParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        text = data.decode("utf-8-sig", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(text), dialect))
        title = origin.rsplit("/", 1)[-1]
        blocks = [Block(type="heading", text=f"# {title}", level=1)]
        b = _md_table(rows)
        if b:
            blocks.append(b)
        return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime)
