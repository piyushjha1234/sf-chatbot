"""Polite HTTP fetcher: per-host rate limits, robots.txt, conditional GET, retries with backoff + jitter."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from sfrag.core.settings import FetchSettings

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str  # final URL after redirects
    status: int
    content: bytes = b""
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False
    redirected: bool = False
    headers: dict[str, str] = field(default_factory=dict)


class FetchError(Exception):
    def __init__(self, msg: str, retryable: bool) -> None:
        super().__init__(msg)
        self.retryable = retryable


class _HostLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            t = time.monotonic()
            slot = max(t, self._next.get(host, 0.0))
            self._next[host] = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self, host: str, seconds: float) -> None:
        with self._lock:
            self._next[host] = max(self._next.get(host, 0.0), time.monotonic() + seconds)


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class Fetcher:
    def __init__(self, cfg: FetchSettings, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=cfg.timeout_seconds,
            transport=transport,
            headers={
                "User-Agent": cfg.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        self.limiter = _HostLimiter(cfg.requests_per_second_per_host)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()

    # ---- robots -------------------------------------------------------------------------
    def robots(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            if base in self._robots:
                return self._robots[base]
        rp: RobotFileParser | None = RobotFileParser()
        try:
            self.limiter.wait(parts.netloc)
            r = self.client.get(base + "/robots.txt")
            if r.status_code == 200:
                rp.parse(r.text.splitlines())  # type: ignore[union-attr]
            elif r.status_code in (401, 403):
                rp.disallow_all = True  # type: ignore[union-attr]
            else:
                rp = None  # no robots.txt -> allowed
        except httpx.HTTPError:
            rp = None
        with self._robots_lock:
            self._robots[base] = rp
        return rp

    def allowed(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        rp = self.robots(url)
        return True if rp is None else rp.can_fetch(self.cfg.user_agent, url)

    def sitemaps_from_robots(self, url: str) -> list[str]:
        rp = self.robots(url)
        return list(rp.site_maps() or []) if rp else []

    # ---- fetch -----------------------------------------------------------------------------
    def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        host = urlsplit(url).netloc
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        last_err = ""
        for attempt in range(self.cfg.max_retries + 1):
            self.limiter.wait(host)
            try:
                with self.client.stream("GET", url, headers=headers) as r:
                    if r.status_code == 304:
                        return FetchResult(
                            url=str(r.url), status=304, not_modified=True, etag=etag, last_modified=last_modified
                        )
                    if r.status_code in (429, 500, 502, 503, 504):
                        wait = _retry_after(r.headers.get("retry-after")) or (2**attempt) * 2.0
                        wait = min(wait, 300.0) * (0.75 + random.random() / 2)
                        self.limiter.penalize(host, wait)
                        last_err = f"HTTP {r.status_code}"
                        continue
                    if r.status_code >= 400:
                        return FetchResult(url=str(r.url), status=r.status_code)
                    chunks: list[bytes] = []
                    size = 0
                    for part in r.iter_bytes():
                        size += len(part)
                        if size > self.cfg.max_bytes:
                            raise FetchError(f"response exceeds {self.cfg.max_bytes} bytes", retryable=False)
                        chunks.append(part)
                    return FetchResult(
                        url=str(r.url),
                        status=r.status_code,
                        content=b"".join(chunks),
                        content_type=r.headers.get("content-type", ""),
                        etag=r.headers.get("etag"),
                        last_modified=r.headers.get("last-modified"),
                        redirected=str(r.url) != url,
                        headers={k.lower(): v for k, v in r.headers.items()},
                    )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(60.0, (2**attempt) * 1.5) * (0.75 + random.random() / 2))
        raise FetchError(f"giving up on {url}: {last_err}", retryable=True)

    def close(self) -> None:
        self.client.close()
