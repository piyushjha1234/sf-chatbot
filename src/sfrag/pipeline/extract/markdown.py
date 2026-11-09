"""Markdown -> Blocks. Used for .md/.txt input and for the Trafilatura HTML fallback."""

from __future__ import annotations

import re

from sfrag.core.models import Block, NormalizedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+#.-]*)")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_ADMONITION = re.compile(
    r"^\s*(?:>\s*)?(?:\*\*|__)?\s*(note|warning|important|caution|tip|attention)\s*[:!]?\s*(?:\*\*|__)?\s*[:]?",
    re.IGNORECASE,
)


def parse_markdown_blocks(text: str) -> list[Block]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    i = 0
    n = len(lines)
    para: list[str] = []

    def flush_para() -> None:
        if para:
            body = "\n".join(para).strip()
            if body:
                m = _ADMONITION.match(body)
                if m:
                    blocks.append(Block(type="admonition", text=body, admonition_kind=m.group(1).lower()))
                else:
                    blocks.append(Block(type="paragraph", text=body))
            para.clear()

    while i < n:
        line = lines[i]
        # fenced code
        fm = _FENCE.match(line)
        if fm:
            flush_para()
            fence = fm.group(2)
            lang = fm.group(3) or None
            buf = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(fence[:3]):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            blocks.append(Block(type="code", text="\n".join(buf), language=lang))
            continue
        hm = _HEADING.match(line)
        if hm:
            flush_para()
            blocks.append(Block(type="heading", text=line.strip(), level=len(hm.group(1))))
            i += 1
            continue
        # table: header line followed by separator
        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush_para()
            header = line.rstrip() + "\n" + lines[i + 1].rstrip()
            rows: list[str] = []
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(type="table", text="\n".join([header, *rows]), table_header=header, table_rows=rows))
            continue
        lm = _LIST.match(line)
        if lm:
            flush_para()
            ordered = lm.group(2)[0].isdigit()
            buf = [line.rstrip()]
            i += 1
            while i < n:
                nxt = lines[i]
                if _LIST.match(nxt) or (nxt.startswith(("  ", "\t")) and nxt.strip()):
                    buf.append(nxt.rstrip())
                    i += 1
                elif not nxt.strip() and i + 1 < n and (_LIST.match(lines[i + 1]) or lines[i + 1].startswith("  ")):
                    buf.append("")
                    i += 1
                else:
                    break
            blocks.append(Block(type="list", text="\n".join(buf).strip("\n"), ordered=ordered))
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        para.append(line.rstrip())
        i += 1
    flush_para()
    return blocks


def title_from_blocks(blocks: list[Block], fallback: str) -> str:
    for b in blocks:
        if b.type == "heading":
            return b.text.lstrip("#").strip()
    return fallback


class MarkdownParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        text = data.decode("utf-8", errors="replace")
        blocks = parse_markdown_blocks(text)
        fallback = origin.rstrip("/").rsplit("/", 1)[-1]
        return NormalizedDocument(title=title_from_blocks(blocks, fallback), blocks=blocks, origin=origin, mime=mime)


class TextParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = [Block(type="paragraph", text=p) for p in paras]
        title = origin.rstrip("/").rsplit("/", 1)[-1]
        return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime)
