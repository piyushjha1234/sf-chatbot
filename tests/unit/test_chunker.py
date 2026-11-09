from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sfrag.core.models import Block, NormalizedDocument
from sfrag.core.settings import ChunkingSettings
from sfrag.core.tokenizer import SimpleTokenizer
from sfrag.enrich.chunker import SEP, HierarchicalChunker

TOK = SimpleTokenizer()
CFG = ChunkingSettings(
    child_target_tokens=120,
    child_max_tokens=160,
    parent_target_tokens=320,
    parent_max_tokens=440,
    overlap_tokens=12,
    min_child_tokens=30,
)


def _chunker(cfg: ChunkingSettings = CFG) -> HierarchicalChunker:
    return HierarchicalChunker(cfg, TOK)


words = st.lists(
    st.sampled_from(
        [
            "data",
            "stream",
            "Identity",
            "rule.",
            "match",
            "API",
            "the",
            "field__c",
            "unified",
            "profile",
            "set",
            "up",
            "error:",
            "SQL;",
            "fast",
        ]
    ),
    min_size=1,
    max_size=400,
).map(" ".join)


@st.composite
def block_st(draw: st.DrawFn) -> Block:
    kind = draw(st.sampled_from(["heading", "paragraph", "list", "table", "code", "admonition"]))
    if kind == "heading":
        lvl = draw(st.integers(1, 4))
        return Block(type="heading", text="#" * lvl + " " + draw(words)[:60].strip() or "H", level=lvl)
    if kind == "table":
        rows = [f"| {draw(words)[:80]} | {i} |" for i in range(draw(st.integers(1, 60)))]
        header = "| Col | N |\n|---|---|"
        return Block(type="table", text="\n".join([header, *rows]), table_header=header, table_rows=rows)
    if kind == "code":
        lines = [f"x{i} = {draw(words)[:50]!r}" for i in range(draw(st.integers(1, 120)))]
        return Block(type="code", text="```python\n" + "\n".join(lines) + "\n```", language="python")
    if kind == "list":
        items = [f"{i + 1}. {draw(words)[:120]}" for i in range(draw(st.integers(1, 30)))]
        return Block(type="list", text="\n".join(items), ordered=True)
    text = draw(words)
    return Block(type=kind, text=text, admonition_kind="note" if kind == "admonition" else None)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(block_st(), min_size=1, max_size=25))
def test_lossless_and_bounded(blocks: list[Block]) -> None:
    doc = NormalizedDocument(title="Doc", blocks=blocks, origin="https://x/doc", mime="text/html")
    sections, chunks = _chunker().chunk(doc, "d_test")
    expected = SEP.join(b.text for b in blocks if b.text.strip())
    assert HierarchicalChunker.reconstruct(chunks) == expected
    for ch in chunks:
        assert ch.token_count <= CFG.child_max_tokens + 10, (ch.token_count, ch.text[:200])
    # every chunk belongs to an existing parent; parents cover all chunks
    pids = {s.parent_id for s in sections}
    assert {c.parent_id for c in chunks} <= pids
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_table_header_repeated_in_every_row_group() -> None:
    header = "| Code | Meaning |\n|---|---|"
    rows = [f"| ERR_{i} | something went wrong number {i} with a long explanation text here |" for i in range(80)]
    b = Block(type="table", text="\n".join([header, *rows]), table_header=header, table_rows=rows)
    doc = NormalizedDocument(
        title="Errors",
        blocks=[Block(type="heading", text="# Errors", level=1), b],
        origin="https://x/e",
        mime="text/html",
    )
    _, chunks = _chunker().chunk(doc, "d_t")
    table_chunks = [c for c in chunks if c.group_id]
    assert len(table_chunks) >= 2
    assert all(c.text.startswith("| Code | Meaning |") or c.text.startswith("# Errors") for c in table_chunks)
    assert all(c.context_prefix.startswith("| Code") for c in table_chunks[1:])


def test_code_split_keeps_group_and_fence() -> None:
    code_lines = []
    for f in range(30):
        code_lines += [f"def handler_{f}(event):", f"    value = compute_{f}(event)", "    return value", ""]
    b = Block(type="code", text="```python\n" + "\n".join(code_lines) + "\n```", language="python")
    doc = NormalizedDocument(title="Code", blocks=[b], origin="https://x/c", mime="text/html")
    _, chunks = _chunker().chunk(doc, "d_c")
    grouped = [c for c in chunks if c.group_id]
    assert len(grouped) >= 2 and len({c.group_id for c in grouped}) == 1
    assert all(c.context_prefix == "```python\n" for c in grouped[1:])
    # splits prefer function boundaries
    assert sum(1 for c in grouped[1:] if c.body.lstrip().startswith(("def ", "\n"))) >= len(grouped[1:]) // 2


def test_heading_paths_and_sticky_lead_in() -> None:
    blocks = [
        Block(type="heading", text="# Data Streams", level=1),
        Block(type="paragraph", text="Overview of streams. " * 5),
        Block(type="heading", text="## Prerequisites", level=2),
        Block(type="paragraph", text="Before you begin:"),
        Block(type="list", text="- Assign the permission set.\n- Configure a connector."),
        Block(type="heading", text="## Steps", level=2),
        Block(type="list", text="1. Open Data Streams.\n2. Click New.\n3. Deploy.", ordered=True),
        Block(
            type="admonition",
            text="Warning: changing the key requires recreating the stream.",
            admonition_kind="warning",
        ),
    ]
    doc = NormalizedDocument(title="Data Streams", blocks=blocks, origin="https://x/ds", mime="text/html")
    _, chunks = _chunker().chunk(doc, "d_ds")
    joined = [c.body for c in chunks]
    # lead-in stays with its list; warning stays with the steps
    assert any("Before you begin:" in t and "Assign the permission set" in t for t in joined)
    assert any("3. Deploy." in t and "Warning:" in t for t in joined)
    assert all(c.heading_path[0] == "Data Streams" for c in chunks)
