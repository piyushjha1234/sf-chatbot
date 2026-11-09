"""HTML -> structured Blocks, preserving headings, lists, tables, code and admonitions."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

from sfrag.core.models import Block, NormalizedDocument
from sfrag.pipeline.extract.base import NeedsRender
from sfrag.pipeline.extract.markdown import parse_markdown_blocks

_DROP_TAGS = ["script", "style", "noscript", "template", "svg", "iframe", "form", "button", "input", "select"]
_BOILERPLATE_TAGS = ["nav", "footer", "header", "aside"]
_BOILERPLATE_CLASS = re.compile(
    r"(?:^|[\s_-])(cookie|consent|feedback|survey|rating|share|social|breadcrumbs?|toc|table-of-contents|sidebar|"
    r"site-header|site-footer|global-nav|skip-link|banner|promo|newsletter|was-this-helpful)(?:$|[\s_-])",
    re.IGNORECASE,
)
_ADMONITION_CLASS = re.compile(r"(note|warning|important|caution|tip|alert|callout|admonition)", re.IGNORECASE)
_MAIN_SELECTORS = [
    "main article",
    "article",
    "main",
    "[role=main]",
    ".slds-text-longform",
    "#maincontent",
    "#main-content",
    ".body.conbody",
    ".content-body",
    "#content",
    ".content",
    ".doc-content",
]
_JS_SHELL_MARKERS = (
    "enable javascript",
    "requires javascript",
    "you need to enable javascript",
    "loading...",
    "aura.",
    "window.__",
    "lwr",
    "<c-",
    "lightning",
)
_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "table",
    "pre",
    "blockquote",
    "dl",
    "img",
    "figure",
}
_INLINE_WS = re.compile(r"[ \t\r\f\v]+")


def _inline_text(el: Tag | NavigableString) -> str:
    """Render inline content to Markdown-ish text (inline code as backticks)."""
    if isinstance(el, NavigableString):
        return str(el)
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in ("code", "kbd", "samp", "tt") and child.parent and child.parent.name != "pre":
                t = child.get_text()
                parts.append(f"`{t.strip()}`" if t.strip() else t)
            elif child.name in ("b", "strong"):
                t = _inline_text(child).strip()
                parts.append(f"**{t}**" if t else "")
            elif child.name == "br":
                parts.append("\n")
            elif child.name == "img":
                alt = child.get("alt")
                if alt:
                    parts.append(f"[image: {alt}]")
            else:
                parts.append(_inline_text(child))
    return "".join(parts)


def _clean(text: str) -> str:
    text = _INLINE_WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _render_list(el: Tag, depth: int = 0) -> str:
    ordered = el.name == "ol"
    lines: list[str] = []
    for idx, li in enumerate(el.find_all("li", recursive=False), start=1):
        marker = f"{idx}." if ordered else "-"
        inline_parts: list[str] = []
        nested: list[str] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested.append(_render_list(child, depth + 1))
            elif isinstance(child, Tag) and child.name == "pre":
                code = child.get_text().rstrip("\n")
                nested.append("\n".join("    " * (depth + 1) + ln for ln in ["```", *code.split("\n"), "```"]))
            elif isinstance(child, (Tag, NavigableString)):
                inline_parts.append(_inline_text(child))
        text = _clean(" ".join(inline_parts)).replace("\n", " ")
        lines.append("    " * depth + f"{marker} {text}")
        lines.extend(nested)
    return "\n".join(lines)


def _cell(c: Tag) -> str:
    return _clean(_inline_text(c)).replace("\n", " ").replace("|", "\\|")


def _render_table(el: Tag) -> Block | None:
    rows = el.find_all("tr")
    if not rows:
        return None
    matrix: list[list[str]] = []
    header_row: list[str] | None = None
    for tr in rows:
        cells = tr.find_all(["th", "td"], recursive=False)
        vals = [_cell(c) for c in cells]
        if not any(vals):
            continue
        if header_row is None and (all(c.name == "th" for c in cells) or (tr.parent and tr.parent.name == "thead")):
            header_row = vals
        else:
            matrix.append(vals)
    if header_row is None:
        if not matrix:
            return None
        header_row = matrix.pop(0)
    width = max([len(header_row), *[len(r) for r in matrix]])
    header_row += [""] * (width - len(header_row))
    header = "| " + " | ".join(header_row) + " |\n|" + "|".join(["---"] * width) + "|"
    body_rows = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in matrix]
    caption = el.find("caption")
    if caption and caption.get_text(strip=True):
        header = f"**{_clean(caption.get_text())}**\n\n{header}"
    return Block(type="table", text="\n".join([header, *body_rows]), table_header=header, table_rows=body_rows)


def _code_lang(el: Tag) -> str | None:
    for node in (el, el.find("code")):
        if isinstance(node, Tag):
            for cls in node.get("class") or []:
                m = re.match(r"(?:language|lang|brush)[-:]?(\w+)", cls)
                if m:
                    return m.group(1).lower()
    return None


class _Walker:
    def __init__(self, base_url: str) -> None:
        self.blocks: list[Block] = []
        self.base_url = base_url

    def walk(self, node: Tag) -> None:
        pending_inline: list[str] = []

        def flush_inline() -> None:
            text = _clean("".join(pending_inline))
            pending_inline.clear()
            if text:
                self.blocks.append(Block(type="paragraph", text=text))

        for child in node.children:
            if isinstance(child, NavigableString):
                if child.strip():
                    pending_inline.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            classes = " ".join(child.get("class") or [])
            if name in _BLOCK_TAGS or (name in ("div", "section", "aside") and _ADMONITION_CLASS.search(classes)):
                flush_inline()
                self.handle_block(child, name, classes)
            elif name in ("div", "section", "span", "article", "main", "details", "summary", "center", "body"):
                if name in ("span",) and not child.find(list(_BLOCK_TAGS)):
                    pending_inline.append(_inline_text(child))
                else:
                    flush_inline()
                    self.walk(child)
            else:
                pending_inline.append(_inline_text(child))
        flush_inline()

    def handle_block(self, el: Tag, name: str, classes: str) -> None:
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _clean(_inline_text(el)).replace("\n", " ")
            if text:
                level = int(name[1])
                self.blocks.append(Block(type="heading", text=f"{'#' * level} {text}", level=level))
        elif name == "p":
            text = _clean(_inline_text(el))
            if text:
                m = re.match(r"^(\*\*)?(note|warning|important|caution|tip)\b", text, re.IGNORECASE)
                if m:
                    self.blocks.append(Block(type="admonition", text=text, admonition_kind=m.group(2).lower()))
                else:
                    self.blocks.append(Block(type="paragraph", text=text))
        elif name in ("ul", "ol"):
            text = _render_list(el)
            if text.strip():
                self.blocks.append(Block(type="list", text=text, ordered=name == "ol"))
        elif name == "table":
            b = _render_table(el)
            if b:
                self.blocks.append(b)
        elif name == "pre":
            code = el.get_text().rstrip("\n")
            if code.strip():
                lang = _code_lang(el)
                self.blocks.append(Block(type="code", text=f"```{lang or ''}\n{code}\n```", language=lang))
        elif name == "blockquote":
            inner = _Walker(self.base_url)
            inner.walk(el)
            text = "\n".join(b.text for b in inner.blocks)
            if text.strip():
                self.blocks.append(Block(type="paragraph", text="> " + text.replace("\n", "\n> ")))
        elif name == "dl":
            lines = []
            for dt in el.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                lines.append(f"- **{_clean(_inline_text(dt))}**: {_clean(_inline_text(dd)) if dd else ''}")
            if lines:
                self.blocks.append(Block(type="list", text="\n".join(lines)))
        elif name in ("img", "figure"):
            img = el if name == "img" else el.find("img")
            alt = img.get("alt") if isinstance(img, Tag) else None
            cap = el.find("figcaption") if name == "figure" else None
            desc = " — ".join(str(x) for x in [alt, _clean(cap.get_text()) if cap else None] if x)
            if desc:
                src = img.get("src") if isinstance(img, Tag) else None
                ref = f" ({urljoin(self.base_url, str(src))})" if src else ""
                self.blocks.append(Block(type="image", text=f"[diagram: {desc}]{ref}"))
        else:  # admonition container
            inner = _Walker(self.base_url)
            inner.walk(el)
            text = "\n".join(b.text for b in inner.blocks).strip()
            if text:
                m = _ADMONITION_CLASS.search(classes)
                kind = m.group(1).lower() if m else "note"
                kind = (
                    "warning"
                    if kind in ("alert", "caution")
                    else ("note" if kind in ("callout", "admonition") else kind)
                )
                self.blocks.append(Block(type="admonition", text=text, admonition_kind=kind))


def _looks_like_js_shell(html: str, text_len: int) -> bool:
    if text_len > 400:
        return False
    low = html.lower()
    scripts = low.count("<script")
    return scripts >= 3 or any(m in low for m in _JS_SHELL_MARKERS)


class HtmlParser:
    def __init__(self, min_text_chars: int = 200) -> None:
        self.min_text_chars = min_text_chars

    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        html = data.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        meta: dict[str, str] = {}
        for m in soup.find_all("meta"):
            key = m.get("name") or m.get("property")
            if key and m.get("content"):
                meta[str(key).lower()] = str(m.get("content"))
        canonical = soup.find("link", rel="canonical")
        if isinstance(canonical, Tag) and canonical.get("href"):
            meta["canonical"] = urljoin(origin, str(canonical["href"]))
        html_tag = soup.find("html")
        if isinstance(html_tag, Tag) and html_tag.get("lang"):
            meta["lang"] = str(html_tag["lang"])

        breadcrumb: list[str] = []
        bc = soup.select_one("nav[aria-label*=readcrumb], .breadcrumb, .breadcrumbs, [class*=breadcrumb]")
        if bc:
            breadcrumb = [_clean(a.get_text()) for a in bc.find_all(["a", "span", "li"]) if _clean(a.get_text())]
            breadcrumb = list(dict.fromkeys(breadcrumb))[:8]

        links = []
        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if href and not href.startswith(("javascript:", "mailto:", "#", "tel:")):
                links.append(urljoin(origin, href))

        for t in soup(_DROP_TAGS):
            t.decompose()
        main: Tag | None = None
        for sel in _MAIN_SELECTORS:
            found = soup.select_one(sel)
            if isinstance(found, Tag) and len(found.get_text(strip=True)) >= self.min_text_chars:
                main = found
                break
        if main is None:
            main = soup.body if isinstance(soup.body, Tag) else soup
        assert main is not None
        for t in main.find_all(_BOILERPLATE_TAGS):
            if not _ADMONITION_CLASS.search(" ".join(t.get("class") or [])):
                t.decompose()
        for t in main.find_all(True):
            if getattr(t, "decomposed", False) or t.attrs is None:
                continue
            cls = " ".join(t.get("class") or []) + " " + str(t.get("id") or "")
            if _BOILERPLATE_CLASS.search(cls) and not t.find(["pre", "table"]):
                t.decompose()

        walker = _Walker(origin)
        walker.walk(main)
        blocks = walker.blocks
        text_len = sum(len(b.text) for b in blocks)

        if text_len < self.min_text_chars:
            import trafilatura

            md = trafilatura.extract(
                html,
                output_format="markdown",
                include_tables=True,
                include_formatting=True,
                include_links=False,
                favor_recall=True,
            )
            if md and len(md) > text_len:
                blocks = parse_markdown_blocks(md)
                text_len = sum(len(b.text) for b in blocks)
        if text_len < 50 and _looks_like_js_shell(html, text_len):
            raise NeedsRender("page content is rendered client-side")

        title = ""
        h1 = next((b for b in blocks if b.type == "heading" and b.level == 1), None)
        if h1:
            title = h1.text.lstrip("#").strip()
        elif soup.title and soup.title.string:
            title = _clean(soup.title.string)
        else:
            title = meta.get("og:title", origin)
        # If the page has no H1 inside the content, add the title as the root heading.
        if not h1 and title:
            blocks.insert(0, Block(type="heading", text=f"# {title}", level=1))

        return NormalizedDocument(
            title=title, blocks=blocks, origin=origin, mime=mime, links=links, breadcrumb=breadcrumb, meta=meta
        )
