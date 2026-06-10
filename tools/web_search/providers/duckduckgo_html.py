"""DuckDuckGo HTML provider.

The Instant Answer API only returns encyclopedia-style abstracts, so it is
almost always empty for news, sports, and current events. This provider
scrapes the HTML endpoint instead, which returns real search results.
"""

from __future__ import annotations

import logging
import ssl
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from lxml import html

from ..models import SearchResult

logger = logging.getLogger(__name__)


class DuckDuckGoHtmlProvider:
    """Fetches search results from html.duckduckgo.com."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{params}"
        request = Request(
            url,
            headers={
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Mozilla/5.0",
            },
        )

        try:
            page = self._load_page(request)
        except Exception as exc:
            logger.info(
                "DuckDuckGo HTML search failed for query '%s': %s",
                query,
                exc,
            )
            return []

        return parse_results_page(page, limit)

    def _load_page(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except ssl.SSLCertVerificationError:
            return self._load_page_without_ssl_verification(request)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                return self._load_page_without_ssl_verification(request)
            raise

    def _load_page_without_ssl_verification(self, request: Request) -> bytes:
        logger.info(
            "DuckDuckGo HTML SSL verification failed, retrying without verification."
        )
        insecure_context = ssl._create_unverified_context()
        with urlopen(
            request,
            timeout=self.timeout,
            context=insecure_context,
        ) as response:
            return response.read()


def parse_results_page(page: bytes | str, limit: int = 5) -> list[SearchResult]:
    """Extract search results from a DuckDuckGo HTML results page."""
    try:
        document = html.fromstring(page)
    except Exception as exc:
        logger.info("DuckDuckGo HTML parsing failed: %s", exc)
        return []

    results: list[SearchResult] = []

    for item in document.xpath('//div[contains(@class, "result")]'):
        # Skip sponsored results.
        item_class = item.get("class", "")
        if "result--ad" in item_class:
            continue

        title = " ".join(
            text.strip()
            for text in item.xpath('.//a[contains(@class, "result__a")]//text()')
            if text.strip()
        )
        href = "".join(
            item.xpath('.//a[contains(@class, "result__a")]/@href')
        ).strip()
        snippet = " ".join(
            text.strip()
            for text in item.xpath(
                './/*[contains(@class, "result__snippet")]//text()'
            )
            if text.strip()
        )

        url = resolve_redirect_url(href)
        if not title or not url:
            continue

        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break

    return results


def resolve_redirect_url(href: str) -> str:
    """DuckDuckGo links point to a redirect like //duckduckgo.com/l/?uddg=<real url>."""
    if not href:
        return ""

    if href.startswith("//"):
        href = f"https:{href}"

    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        real_url = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(real_url)

    return href
