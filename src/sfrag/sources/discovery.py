"""URL normalisation, scope rules and sitemap discovery (robots Sitemap lines, nested indexes, gz)."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from lxml import etree

_TRACKING = re.compile(r"^(utm_\w+|gclid|fbclid|mkt_tok|mc_cid|mc_eid|_ga|_gl|ref|trk|cid)$", re.IGNORECASE)
_DOC_EXT_SKIP = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|woff2?|ttf|mp4|mov|zip|gz|tar|exe|dmg)$", re.I)


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if not _TRACKING.match(k)]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


@dataclass
class Scope:
    include: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]
    hosts: set[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Scope:
        inc = [re.compile(p) for p in cfg.get("include", [])]
        exc = [re.compile(p) for p in cfg.get("exclude", [])]
        hosts = {urlsplit(u).hostname or "" for u in cfg.get("start_urls", []) + cfg.get("sitemaps", [])}
        hosts |= set(cfg.get("hosts", []))
        return cls(inc, exc, {h for h in hosts if h})

    def check(self, url: str) -> tuple[bool, str | None]:
        """(in_scope, exclusion_reason)."""
        host = urlsplit(url).hostname or ""
        if self.hosts and host not in self.hosts:
            return False, "out_of_scope_host"
        if _DOC_EXT_SKIP.search(urlsplit(url).path):
            return False, "non_document_asset"
        for p in self.exclude:
            if p.search(url):
                return False, f"excluded_by_rule:{p.pattern}"
        if self.include and not any(p.search(url) for p in self.include):
            return False, "not_matching_include_rules"
        return True, None


@dataclass
class SitemapEntry:
    loc: str
    lastmod: str | None
    is_index: bool


def parse_sitemap(data: bytes) -> list[SitemapEntry]:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True, huge_tree=True)
    root = etree.fromstring(data, parser=parser)
    if root is None:
        return []
    is_index = etree.QName(root).localname == "sitemapindex"
    out: list[SitemapEntry] = []
    for el in root:
        if not isinstance(el.tag, str):
            continue
        loc = lastmod = None
        for child in el:
            if not isinstance(child.tag, str):
                continue
            name = etree.QName(child).localname
            if name == "loc" and child.text:
                loc = child.text.strip()
            elif name == "lastmod" and child.text:
                lastmod = child.text.strip()
        if loc:
            out.append(SitemapEntry(loc=loc, lastmod=lastmod, is_index=is_index))
    return out


def is_salesforce_reference(url: str, allowed_suffixes: list[str]) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == s or host.endswith("." + s) for s in allowed_suffixes)
