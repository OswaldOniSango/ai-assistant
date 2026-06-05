"""Bing provider."""

from __future__ import annotations

import base64
import logging
import ssl
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

from lxml import html

from ..models import SearchResult

logger = logging.getLogger(__name__)


class BingProvider:
    """Fetches search results from Bing HTML pages."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = urlencode({"cc": "us", "q": query, "setlang": "en-US"})
        url = f"https://www.bing.com/search?{params}"
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
            logger.info("Bing search failed for query '%s': %s", query, exc)
            return []

        document = html.fromstring(page)
        results: list[SearchResult] = []

        for item in document.xpath('//li[contains(@class, "b_algo")]')[:limit]:
            title = "".join(item.xpath(".//h2//a//text()")).strip()
            href = "".join(item.xpath(".//h2//a/@href")).strip()
            snippet = " ".join(
                text.strip()
                for text in item.xpath('.//div[contains(@class, "b_caption")]//p//text()')
                if text.strip()
            )
            decoded_url = _decode_bing_url(href)
            if not title or not decoded_url:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=decoded_url,
                    snippet=snippet,
                )
            )

        return results

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
        logger.info("Bing SSL verification failed, retrying without verification.")
        insecure_context = ssl._create_unverified_context()
        with urlopen(
            request,
            timeout=self.timeout,
            context=insecure_context,
        ) as response:
            return response.read()


def _decode_bing_url(href: str) -> str:
    if not href:
        return ""
    if "bing.com/ck/a" not in href:
        return href

    query_params = parse_qs(urlparse(href).query)
    encoded_target = query_params.get("u", [""])[0]
    if not encoded_target.startswith("a1"):
        return href

    payload = encoded_target[2:]
    padding = "=" * ((4 - len(payload) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(payload + padding).decode("utf-8")
    except Exception:
        return href
