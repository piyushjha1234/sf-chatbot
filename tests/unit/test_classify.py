from __future__ import annotations

from sfrag.core.models import KnowledgeCategory, Lifecycle
from sfrag.core.settings import PROJECT_ROOT
from sfrag.core.taxonomy import load_taxonomy
from sfrag.enrich.classify import (
    RuleClassifier,
    detect_api_version,
    detect_lifecycle,
    detect_release,
    extract_identifiers,
)
from sfrag.pipeline.extract import parse_document
from tests.conftest import D360, PAGES

TX = load_taxonomy(PROJECT_ROOT / "taxonomy" / "taxonomy.yaml")
RULES = RuleClassifier(TX)


def test_product_from_url_and_aliases() -> None:
    url = f"{D360}/identity-resolution.html"
    cls = RULES.classify_document(parse_document(PAGES[url], url, "text/html"))
    assert cls.products[0] == "data360"
    assert cls.product_confidence >= 0.55
    assert cls.release == "Winter '27"


def test_categories_and_content_type() -> None:
    url = f"{D360}/query-api.html"
    cls = RULES.classify_document(parse_document(PAGES[url], url, "text/html"))
    assert KnowledgeCategory.API in cls.categories or KnowledgeCategory.TROUBLESHOOTING in cls.categories
    assert "INVALID_SQL_SYNTAX" in cls.identifiers


def test_alias_normalisation() -> None:
    assert TX.normalize_product("Data Cloud") == "data360"
    assert TX.normalize_product("Datorama") == "mci"
    assert TX.normalize_product("Interaction Studio") == "mcp"
    assert TX.normalize_product("unknown thing") is None


def test_detectors() -> None:
    assert detect_release("New in Summer ’26 and Spring '25, summer '26") == "Summer '26"
    assert detect_api_version("Available in API version 62.0 and /services/data/v64.0/sobjects") == "64.0"
    assert detect_lifecycle("This feature is deprecated and will be retired.") == Lifecycle.DEPRECATED
    assert detect_lifecycle("This feature is a pilot program.") == Lifecycle.PREVIEW
    ids = extract_identifiers("Query Account__c via /services/data/v64.0/query; error REQUEST_LIMIT_EXCEEDED")
    assert {"Account__c", "/services/data/v64.0/query", "REQUEST_LIMIT_EXCEEDED"} <= set(ids)


def test_unknown_product_does_not_block() -> None:
    doc = parse_document(b"# Gardening\n\nTomatoes need sun.", "file://g.md", "text/markdown")
    cls = RULES.classify_document(doc)
    assert cls.products == [] and cls.product_confidence == 0.0
