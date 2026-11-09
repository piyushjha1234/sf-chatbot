"""Structure-aware hierarchical parent/child chunking.

Guarantees
----------
* Heading-first: sections split at headings; token-based splitting only inside oversized blocks.
* Sticky units: a heading binds to its first content block; a lead-in paragraph ending in ':' binds to the
  following list/code/table; an admonition binds to the block before it (warnings stay with their steps).
* Lossless: every block's text lands in exactly one chunk body. Pieces of a split block are exact
  substrings (they concatenate back to the block); repeated material (overlap tail, table header, code
  fence) lives in `context_prefix`, never in `body`. `reconstruct()` proves the invariant in tests.
* Bounded: no chunk exceeds `child_max_tokens` of body + prefix (except a single unsplittable token run).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sfrag.core.ids import chunk_id_for, parent_id_for
from sfrag.core.models import Block, Chunk, ContentType, NormalizedDocument, Section
from sfrag.core.settings import ChunkingSettings
from sfrag.core.tokenizer import Tokenizer

SEP = "\n\n"
_TEXT_SPLIT = re.compile(r"(?<=\n)|(?<=[.!?;:] )")
_WORD_SPLIT = re.compile(r"(?<= )")
_CODE_BOUNDARY = re.compile(
    r"^\s*(?:$|def |class |async def |function |public |private |protected |global |static |@|"
    r"export |const \w+ = (?:async )?\(|trigger |%%\[|<script|}\s*$)"
)


@dataclass
class _Unit:
    blocks: list[Block]
    heading_path: list[str]
    tokens: int

    @property
    def text(self) -> str:
        return SEP.join(b.text for b in self.blocks)


@dataclass
class _Leaf:
    heading_path: list[str]
    units: list[_Unit] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(u.tokens for u in self.units)


@dataclass
class _Piece:
    body: str
    prefix: str
    heading_path: list[str]
    group_id: str | None
    page: int | None
    ctype: ContentType


def _heading_text(b: Block) -> str:
    return b.text.lstrip("#").strip()


def _structural_type(blocks: list[Block]) -> ContentType:
    kinds = [b.type for b in blocks if b.type != "heading"]
    if not kinds:
        return ContentType.OTHER
    if all(k == "code" for k in kinds):
        return ContentType.CODE_SAMPLE
    if all(k == "table" for k in kinds):
        return ContentType.TABLE_DATA
    if any(b.type == "list" and b.ordered for b in blocks):
        return ContentType.PROCEDURE
    return ContentType.OTHER


class HierarchicalChunker:
    def __init__(self, cfg: ChunkingSettings, tokenizer: Tokenizer) -> None:
        self.cfg = cfg
        self.tok = tokenizer

    # ------------------------------------------------------------------ public
    def chunk(self, doc: NormalizedDocument, doc_id: str) -> tuple[list[Section], list[Chunk]]:
        blocks = [b for b in doc.blocks if b.text.strip()]
        leaves = self._leaves(blocks, doc.title)
        parents = self._parents(leaves)
        sections: list[Section] = []
        chunks: list[Chunk] = []
        group_counter = 0
        ordinal = 0
        for p_ord, units in enumerate(parents):
            parent_id = parent_id_for(doc_id, p_ord)
            p_blocks = [b for u in units for b in u.blocks]
            text = SEP.join(b.text for b in p_blocks)
            pages = [b.page for b in p_blocks if b.page is not None]
            sections.append(
                Section(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    ordinal=p_ord,
                    heading_path=units[0].heading_path,
                    text=text,
                    token_count=self.tok.count(text),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                )
            )
            pieces, group_counter = self._children(units, doc_id, group_counter)
            for pc in pieces:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_for(doc_id, pc.heading_path, ordinal),
                        doc_id=doc_id,
                        parent_id=parent_id,
                        ordinal=ordinal,
                        heading_path=pc.heading_path,
                        body=pc.body,
                        context_prefix=pc.prefix,
                        token_count=self.tok.count(pc.prefix + pc.body),
                        content_type=pc.ctype,
                        group_id=pc.group_id,
                        page=pc.page,
                    )
                )
                ordinal += 1
        return sections, chunks

    @staticmethod
    def reconstruct(chunks: list[Chunk]) -> str:
        """Inverse of chunking for the lossless invariant: pieces of one group join with '', others with SEP."""
        out: list[str] = []
        prev_group: str | None = None
        for ch in sorted(chunks, key=lambda c: c.ordinal):
            if out and ch.group_id is not None and ch.group_id == prev_group:
                out.append(ch.body)
            else:
                if out:
                    out.append(SEP)
                out.append(ch.body)
            prev_group = ch.group_id
        return "".join(out)

    # ------------------------------------------------------------------ structure
    def _leaves(self, blocks: list[Block], title: str) -> list[_Leaf]:
        stack: list[tuple[int, str]] = []
        leaves: list[_Leaf] = [_Leaf(heading_path=[title] if title else [])]
        for b in blocks:
            if b.type == "heading":
                lvl = b.level or 1
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, _heading_text(b)))
                leaves.append(_Leaf(heading_path=[h for _, h in stack]))
            leaves[-1].units.append(_Unit([b], leaves[-1].heading_path, self.tok.count(b.text)))
        leaves = [lf for lf in leaves if lf.units]
        for lf in leaves:
            lf.units = self._bind_units(lf.units)
        return leaves

    def _bind_units(self, units: list[_Unit]) -> list[_Unit]:
        """Merge sticky neighbours while the merged unit stays within child_max."""
        out: list[_Unit] = []
        for u in units:
            if out:
                prev = out[-1]
                last = prev.blocks[-1]
                cur = u.blocks[0]
                sticky = (
                    last.type == "heading"
                    or (
                        last.type == "paragraph"
                        and last.text.rstrip().endswith(":")
                        and cur.type in ("list", "code", "table")
                    )
                    or cur.type == "admonition"
                )
                if sticky and prev.tokens + u.tokens <= self.cfg.child_max_tokens:
                    out[-1] = _Unit(prev.blocks + u.blocks, prev.heading_path, prev.tokens + u.tokens)
                    continue
            out.append(u)
        return out

    def _parents(self, leaves: list[_Leaf]) -> list[list[_Unit]]:
        cfg = self.cfg
        parents: list[list[_Unit]] = []
        cur: list[_Unit] = []
        cur_tokens = 0
        cur_top: tuple[str, ...] = ()

        def flush() -> None:
            nonlocal cur, cur_tokens
            if cur:
                parents.append(cur)
            cur, cur_tokens = [], 0

        for leaf in leaves:
            top = tuple(leaf.heading_path[:2])
            topic_change = bool(cur) and top != cur_top and cur_tokens >= cfg.parent_target_tokens // 4
            if topic_change or (cur and cur_tokens + leaf.tokens > cfg.parent_target_tokens):
                flush()
            if not cur:
                cur_top = top
            for u in leaf.units:
                if cur and cur_tokens + u.tokens > cfg.parent_max_tokens:
                    flush()
                    cur_top = top
                cur.append(u)
                cur_tokens += u.tokens
        flush()
        return parents

    # ------------------------------------------------------------------ children
    def _children(self, units: list[_Unit], doc_id: str, group_counter: int) -> tuple[list[_Piece], int]:
        cfg = self.cfg
        pieces: list[_Piece] = []
        buf: list[_Unit] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if buf:
                blocks = [b for u in buf for b in u.blocks]
                pages = [b.page for b in blocks if b.page is not None]
                pieces.append(
                    _Piece(
                        SEP.join(u.text for u in buf),
                        "",
                        buf[0].heading_path,
                        None,
                        pages[0] if pages else None,
                        _structural_type(blocks),
                    )
                )
            buf, buf_tokens = [], 0

        for u in units:
            if u.tokens > cfg.child_max_tokens:
                flush()
                for b in u.blocks:
                    bt = self.tok.count(b.text)
                    if bt <= cfg.child_max_tokens:
                        pieces.append(_Piece(b.text, "", u.heading_path, None, b.page, _structural_type([b])))
                        continue
                    group_counter += 1
                    gid = f"{doc_id}:g{group_counter}"
                    for body, prefix in self._split_block(b):
                        pieces.append(_Piece(body, prefix, u.heading_path, gid, b.page, _structural_type([b])))
                # adjacent un-split blocks of an oversized unit must not share a group (they join with SEP)
                continue
            # flush past the target, allowing modest overshoot to keep a small unit with its neighbours
            if (
                buf
                and buf_tokens + u.tokens > cfg.child_target_tokens
                and (buf_tokens + u.tokens > cfg.child_max_tokens or buf_tokens >= cfg.min_child_tokens)
            ):
                flush()
            buf.append(u)
            buf_tokens += u.tokens
        flush()
        return self._merge_small(pieces), group_counter

    def _merge_small(self, pieces: list[_Piece]) -> list[_Piece]:
        """Merge an undersized ungrouped piece into its ungrouped predecessor when it fits."""
        out: list[_Piece] = []
        for pc in pieces:
            if (
                out
                and pc.group_id is None
                and out[-1].group_id is None
                and not pc.prefix
                and not out[-1].prefix
                and self.tok.count(pc.body) < self.cfg.min_child_tokens
                and self.tok.count(out[-1].body) + self.tok.count(pc.body) <= self.cfg.child_max_tokens
            ):
                prev = out[-1]
                out[-1] = _Piece(
                    prev.body + SEP + pc.body,
                    "",
                    prev.heading_path,
                    None,
                    prev.page,
                    prev.ctype if prev.ctype == pc.ctype else ContentType.OTHER,
                )
            else:
                out.append(pc)
        return out

    # ------------------------------------------------------------------ splitting
    def _split_block(self, b: Block) -> list[tuple[str, str]]:
        if b.type == "table":
            return self._split_table(b)
        if b.type == "code":
            return self._split_code(b)
        return self._split_text(b.text)

    def _greedy(self, segments: list[str], budget: int) -> list[str]:
        """Pack exact substrings into pieces of <= budget tokens; oversize segments are word-split."""
        expanded: list[str] = []
        for s in segments:
            if self.tok.count(s) > budget:
                words = [w for w in _WORD_SPLIT.split(s) if w]
                acc = ""
                for w in words:
                    if acc and self.tok.count(acc + w) > budget:
                        expanded.append(acc)
                        acc = ""
                    acc += w
                if acc:
                    expanded.append(acc)
            else:
                expanded.append(s)
        pieces: list[str] = []
        acc, acc_tokens = "", 0
        for s in expanded:
            st = self.tok.count(s)
            if acc and acc_tokens + st > budget:
                pieces.append(acc)
                acc, acc_tokens = "", 0
            acc += s
            acc_tokens += st
        if acc:
            pieces.append(acc)
        return pieces

    def _overlap_tail(self, text: str) -> str:
        words = text.split(" ")
        tail: list[str] = []
        for w in reversed(words):
            tail.insert(0, w)
            if self.tok.count(" ".join(tail)) >= self.cfg.overlap_tokens:
                break
        t = " ".join(tail).strip()
        return f"… {t} " if t else ""

    def _split_text(self, text: str) -> list[tuple[str, str]]:
        budget = self.cfg.child_target_tokens - self.cfg.overlap_tokens
        segs = [s for s in _TEXT_SPLIT.split(text) if s]
        bodies = self._greedy(segs, budget)
        out: list[tuple[str, str]] = []
        for i, body in enumerate(bodies):
            prefix = self._overlap_tail(bodies[i - 1]) if i > 0 else ""
            out.append((body, prefix))
        return out

    def _split_table(self, b: Block) -> list[tuple[str, str]]:
        lines = b.text.split("\n")
        header = b.table_header or "\n".join(lines[:2])
        header_lines = header.count("\n") + 1
        head_text = "\n".join(lines[:header_lines]) + "\n"
        rows = [ln + "\n" for ln in lines[header_lines:]]
        if rows:
            rows[-1] = rows[-1][:-1]  # last line has no trailing newline in the original
        budget = self.cfg.child_target_tokens - self.tok.count(head_text)
        groups = self._greedy(rows, max(budget, 50))
        out: list[tuple[str, str]] = []
        for i, g in enumerate(groups):
            if i == 0:
                out.append((head_text + g, ""))
            else:
                out.append((g, head_text))  # header repeated in every row group
        if not groups:
            out.append((b.text, ""))
        return out

    def _split_code(self, b: Block) -> list[tuple[str, str]]:
        lines = [ln + "\n" for ln in b.text.split("\n")]
        lines[-1] = lines[-1][:-1]
        fence = lines[0] if lines[0].startswith(("```", "~~~")) else ""
        budget = self.cfg.child_target_tokens
        pieces: list[str] = []
        acc: list[str] = []
        acc_tokens = 0
        for ln in lines:
            lt = self.tok.count(ln)
            if acc and acc_tokens + lt > budget:
                # back off to the nearest syntax boundary in the last 40% of the buffer
                cut = len(acc)
                for j in range(len(acc) - 1, int(len(acc) * 0.6), -1):
                    if _CODE_BOUNDARY.match(acc[j]):
                        cut = j
                        break
                pieces.append("".join(acc[:cut]))
                acc = acc[cut:]
                acc_tokens = sum(self.tok.count(x) for x in acc)
                if acc and acc_tokens + lt > self.cfg.child_max_tokens:
                    pieces.append("".join(acc))
                    acc, acc_tokens = [], 0
            acc.append(ln)
            acc_tokens += lt
        if acc:
            pieces.append("".join(acc))
        # any single piece still over max (one enormous line) -> hard split
        final: list[str] = []
        for p in pieces:
            final.extend(
                self._greedy([p], self.cfg.child_max_tokens) if self.tok.count(p) > self.cfg.child_max_tokens else [p]
            )
        return [(p, fence if i > 0 and fence else "") for i, p in enumerate(final)]
