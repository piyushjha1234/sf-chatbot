from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from sfrag.container import Container
from sfrag.core.settings import PROJECT_ROOT, Settings, load_settings
from sfrag.index.store import close_clients

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = load_settings("test", data_dir=tmp_path / "data", sources_path=tmp_path / "sources.yaml")
    s.ensure_dirs()
    return s


@pytest.fixture
def container(settings: Settings) -> Iterator[Container]:
    c = Container(settings)
    yield c
    close_clients()


# --------------------------------------------------------------------------- fake web
D360 = "https://developer.salesforce.com/docs/data/data-cloud-dev/guide"


def page(title: str, body: str, extra_head: str = "") -> bytes:
    return (
        f"<!doctype html><html lang='en-US'><head><title>{title} | Salesforce Developers</title>{extra_head}</head>"
        "<body><nav class='global-nav'><a href='/'>Home</a> Products Docs Login</nav>"
        f"<main><article><h1>{title}</h1>{body}</article></main>"
        "<footer class='site-footer'>© Salesforce, Inc. Privacy Cookie preferences</footer></body></html>"
    ).encode()


PAGES: dict[str, bytes] = {
    f"{D360}/identity-resolution.html": page(
        "Identity Resolution in Data Cloud",
        "<p>Identity resolution in Data 360 (formerly Data Cloud) unifies source profiles into unified individual "
        "profiles using match rules and reconciliation rules.</p>"
        "<h2>Match Rules</h2><p>Match rules define criteria such as exact email or fuzzy name matching. A ruleset "
        "can contain up to two match rules in Winter '27.</p>"
        "<h2>Reconciliation Rules</h2><p>Reconciliation rules decide which value wins when source records disagree, "
        "for example most recent or source priority.</p>"
        "<div class='note'><p>Note: Running identity resolution consumes Data Cloud credits.</p></div>"
        f"<p>See also <a href='{D360}/data-streams.html'>Data Streams</a>.</p>",
    ),
    f"{D360}/data-streams.html": page(
        "Create a Data Stream",
        "<p>A data stream ingests data from a source such as Amazon S3 or Marketing Cloud Engagement into a Data "
        "Lake Object (DLO).</p>"
        "<h2>Prerequisites</h2><p>Before you begin:</p><ul><li>Assign the Data Cloud Admin permission set.</li>"
        "<li>Configure a connector.</li></ul>"
        "<h2>Steps</h2><ol><li>From Data Cloud, open Data Streams.</li><li>Click New.</li>"
        "<li>Select the source and map fields to the Individual__dlm object.</li><li>Click Deploy.</li></ol>"
        "<div class='warning'><p>Warning: Changing the primary key after deployment requires recreating the "
        "stream.</p></div>",
    ),
    f"{D360}/query-api.html": page(
        "Query API Reference",
        "<p>Use the Data Cloud Query API to run ANSI SQL against data model objects.</p>"
        "<h2>Endpoint</h2><pre><code class='language-http'>POST /api/v2/query\nAuthorization: Bearer {token}"
        "</code></pre>"
        "<h2>Errors</h2><table><thead><tr><th>Code</th><th>Meaning</th></tr></thead><tbody>"
        "<tr><td>INVALID_SQL_SYNTAX</td><td>The SQL statement could not be parsed.</td></tr>"
        "<tr><td>QUERY_TIMEOUT</td><td>The query exceeded the time limit.</td></tr></tbody></table>",
    ),
    # Near-duplicate of identity-resolution (mirror path) to exercise canonical selection.
    f"{D360}/mirror/identity-resolution.html": page(
        "Identity Resolution in Data Cloud",
        "<p>Identity resolution in Data 360 (formerly Data Cloud) unifies source profiles into unified individual "
        "profiles using match rules and reconciliation rules.</p>"
        "<h2>Match Rules</h2><p>Match rules define criteria such as exact email or fuzzy name matching. A ruleset "
        "can contain up to two match rules in Winter '27.</p>"
        "<h2>Reconciliation Rules</h2><p>Reconciliation rules decide which value wins when source records disagree, "
        "for example most recent or source priority.</p>"
        "<div class='note'><p>Note: Running identity resolution consumes Data Cloud credits.</p></div>"
        f"<p>See also <a href='{D360}/data-streams.html'>Data Streams</a>.</p>",
    ),
}

SITEMAP_INDEX = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://developer.salesforce.com/sitemap-d360.xml</loc></sitemap>
</sitemapindex>"""

SITEMAP = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{D360}/identity-resolution.html</loc><lastmod>2026-10-01</lastmod></url>
  <url><loc>{D360}/query-api.html</loc></url>
  <url><loc>{D360}/mirror/identity-resolution.html</loc></url>
  <url><loc>{D360}/missing.html</loc></url>
  <url><loc>https://developer.salesforce.com/docs/other/unrelated.html</loc></url>
</urlset>""".encode()


class FakeWeb:
    def __init__(self) -> None:
        self.pages = dict(PAGES)
        self.requests: list[httpx.Request] = []
        self.etags: dict[str, str] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("/sitemap_index.xml"):
            return httpx.Response(200, content=SITEMAP_INDEX, headers={"content-type": "application/xml"})
        if url.endswith("/sitemap-d360.xml"):
            return httpx.Response(200, content=SITEMAP, headers={"content-type": "application/xml"})
        if url in self.pages:
            etag = '"' + str(abs(hash(self.pages[url]))) + '"'
            if request.headers.get("if-none-match") == etag:
                return httpx.Response(304)
            return httpx.Response(
                200, content=self.pages[url], headers={"content-type": "text/html; charset=utf-8", "etag": etag}
            )
        return httpx.Response(404, content=b"not found")


@pytest.fixture
def fake_web() -> FakeWeb:
    return FakeWeb()


@pytest.fixture
def web_container(container: Container, fake_web: FakeWeb) -> Container:
    from sfrag.sources.fetch import Fetcher

    container.__dict__["fetcher"] = Fetcher(container.settings.fetch, transport=httpx.MockTransport(fake_web.handler))
    from sfrag.storage import repo

    repo.upsert_source(
        container.db,
        "sf_dev_data360",
        {
            "name": "Data 360 dev docs",
            "kind": "web",
            "authority_tier": 1,
            "permitted": True,
            "retain_raw": True,
            "start_urls": [f"{D360}/data-streams.html"],
            "sitemaps": ["https://developer.salesforce.com/sitemap_index.xml"],
            "include": [r"^https://developer\.salesforce\.com/docs/data/data-cloud"],
            "exclude": [],
            "locales": ["en-US", "en"],
            "follow_links": True,
            "max_link_depth": 2,
            "reference_domains": ["salesforce.com"],
        },
    )
    return container


assert PROJECT_ROOT.exists()
