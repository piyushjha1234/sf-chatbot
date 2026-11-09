from __future__ import annotations

from sfrag.core.models import EvidenceBlock
from sfrag.core.tokenizer import SimpleTokenizer
from sfrag.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from sfrag.query.context import pack
from sfrag.query.verify import verify_citations
from sfrag.sources.discovery import Scope, normalize_url, parse_sitemap


def _block(i: int, score: float, words: int = 50) -> EvidenceBlock:
    return EvidenceBlock(
        sid="",
        doc_id=f"d{i}",
        parent_id=f"p{i}",
        chunk_ids=[f"c{i}"],
        title=f"T{i}",
        url=f"https://x/{i}",
        heading="H",
        page=None,
        release=None,
        text=" ".join(["word"] * words),
        score=score,
    )


def test_pack_budget_and_sandwich_order() -> None:
    blocks = [_block(i, s) for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])]
    out = pack(blocks, SimpleTokenizer(), budget=160, max_blocks=10)
    assert len(out) == 3  # 3 x 50 tokens fit in 160
    assert [b.doc_id for b in out] == ["d0", "d2", "d1"]  # best first, second-best last
    assert [b.sid for b in out] == ["S1", "S2", "S3"]


def test_pack_truncates_single_oversized_best_block() -> None:
    big = _block(0, 0.9, words=500)
    big.text = "\n\n".join(" ".join(["w"] * 50) for _ in range(10))
    out = pack([big], SimpleTokenizer(), budget=120, max_blocks=5)
    assert len(out) == 1 and SimpleTokenizer().count(out[0].text) <= 120


def test_verify_citations_strips_invalid_and_measures_uncited() -> None:
    blocks = [_block(1, 0.9)]
    blocks[0].sid = "S1"
    text = (
        "Match rules define how profiles are linked together across sources [S1]. "
        "Reconciliation rules always pick the newest value in every single case [S7]. "
        "This sentence has no citation at all and is long enough to count."
    )
    clean, cites, invalid, ratio = verify_citations(text, blocks)
    assert invalid == ["S7"] and "[S7]" not in clean
    assert [c.sid for c in cites] == ["S1"]
    assert ratio == round(2 / 3, 3)


def test_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    gold = {"b", "d"}
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert mrr(ranked, gold) == 0.5
    assert 0 < ndcg_at_k(ranked, gold, 10) < 1


def test_url_normalisation_and_scope() -> None:
    assert normalize_url("HTTPS://Help.Salesforce.com:443//s/a/?b=2&utm_source=x&a=1#frag") == (
        "https://help.salesforce.com/s/a?a=1&b=2"
    )
    scope = Scope.from_config(
        {
            "start_urls": ["https://developer.salesforce.com/docs/data/data-cloud-dev/guide"],
            "include": [r"/docs/data/data-cloud"],
            "exclude": [r"language=(?!en_US)"],
        }
    )
    assert scope.check("https://developer.salesforce.com/docs/data/data-cloud-dev/guide/x.html") == (True, None)
    assert scope.check("https://developer.salesforce.com/docs/other")[1] == "not_matching_include_rules"
    assert scope.check("https://developer.salesforce.com/docs/data/data-cloud-dev/a.png")[1] == "non_document_asset"
    assert scope.check("https://example.com/docs/data/data-cloud")[1] == "out_of_scope_host"


def test_sitemap_parsing_index_and_gzip() -> None:
    import gzip

    idx = b"<sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'><sitemap><loc>https://x/s1.xml</loc></sitemap></sitemapindex>"
    entries = parse_sitemap(gzip.compress(idx))
    assert entries[0].is_index and entries[0].loc == "https://x/s1.xml"
