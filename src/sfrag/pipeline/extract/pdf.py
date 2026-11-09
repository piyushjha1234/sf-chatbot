"""PDF -> Blocks with heading detection (font size), tables, monospace code and page references."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sfrag.core.models import Block, NormalizedDocument
from sfrag.pipeline.extract.base import ExtractionError

_MONO = ("courier", "mono", "consolas", "menlo", "code", "inconsolata")


def _inside(bbox: tuple[float, float, float, float], rect: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    rx0, ry0, rx1, ry1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


class PdfParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        import pymupdf

        try:
            doc: Any = pymupdf.open(stream=data, filetype="pdf")
        except Exception as e:
            raise ExtractionError(f"cannot open PDF: {e}") from e

        # Pass 1: dominant body font size (weighted by characters).
        sizes: Counter[float] = Counter()
        for page in doc:
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip():
                            sizes[round(span["size"], 1)] += len(span["text"])
        if not sizes:
            raise ExtractionError("PDF has no extractable text (scanned? OCR not enabled)")
        body = sizes.most_common(1)[0][0]
        heading_sizes = sorted({s for s in sizes if s >= body * 1.15}, reverse=True)[:5]
        level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}

        blocks: list[Block] = []
        title = (doc.metadata or {}).get("title") or ""
        for pno, page in enumerate(doc, start=1):
            items: list[tuple[float, Block]] = []
            t: Any
            s: Any
            table_rects: list[tuple[float, float, float, float]] = []
            try:
                for t in page.find_tables().tables:
                    md = t.to_markdown().strip()
                    if md:
                        lines = md.split("\n")
                        header = "\n".join(lines[:2])
                        rows = lines[2:]
                        items.append(
                            (t.bbox[1], Block(type="table", text=md, table_header=header, table_rows=rows, page=pno))
                        )
                        table_rects.append(tuple(t.bbox))
            except Exception:
                pass
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                bbox: tuple[float, float, float, float] = tuple(b["bbox"])
                if any(_inside(bbox, r) for r in table_rects):
                    continue
                lines_text: list[str] = []
                max_size = 0.0
                mono_chars = total_chars = 0
                bold = True
                for line in b.get("lines", []):
                    t = "".join(s["text"] for s in line["spans"])
                    if t.strip():
                        lines_text.append(t)
                    for s in line["spans"]:
                        n = len(s["text"].strip())
                        total_chars += n
                        max_size = max(max_size, round(s["size"], 1))
                        if any(m in s["font"].lower() for m in _MONO):
                            mono_chars += n
                        if n and not (s["flags"] & 16 or "bold" in s["font"].lower()):
                            bold = False
                text = "\n".join(lines_text).strip()
                if not text:
                    continue
                y: float = b["bbox"][1]
                if total_chars and mono_chars / total_chars > 0.8:
                    items.append((y, Block(type="code", text=f"```\n{text}\n```", page=pno)))
                elif max_size in level_of and len(text) < 200:
                    lvl = level_of[max_size]
                    items.append(
                        (y, Block(type="heading", text=f"{'#' * lvl} {' '.join(text.split())}", level=lvl, page=pno))
                    )
                elif bold and len(text) < 120 and "\n" not in text and not text.endswith("."):
                    lvl = min(len(heading_sizes) + 1, 6)
                    items.append((y, Block(type="heading", text=f"{'#' * lvl} {text}", level=lvl, page=pno)))
                else:
                    joined = " ".join(ln.strip() for ln in lines_text)
                    if joined.lstrip().startswith(("•", "-", "–")) or joined[:3].rstrip(".").isdigit():
                        items.append((y, Block(type="list", text=text, page=pno)))
                    else:
                        items.append((y, Block(type="paragraph", text=joined, page=pno)))
            items.sort(key=lambda x: x[0])
            blocks.extend(b for _, b in items)

        # Drop repeated running headers/footers (identical short text on > 50% of pages).
        if doc.page_count >= 4:
            counts = Counter(b.text for b in blocks if len(b.text) < 100)
            repeated = {t for t, c in counts.items() if c > doc.page_count * 0.5}
            blocks = [b for b in blocks if b.text not in repeated]

        if not title:
            h = next((b for b in blocks if b.type == "heading"), None)
            title = h.text.lstrip("#").strip() if h else origin.rsplit("/", 1)[-1]
        return NormalizedDocument(
            title=title, blocks=blocks, origin=origin, mime=mime, meta={"page_count": doc.page_count}
        )
