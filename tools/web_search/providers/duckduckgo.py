"""DuckDuckGo provider."""

from __future__ import annotations

import json
import logging
import ssl
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..models import SearchResult

logger = logging.getLogger(__name__)


class DuckDuckGoProvider:
    """Fetches search context from DuckDuckGo Instant Answer."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = urlencode(
            {
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "q": query,
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            payload = self._load_payload(request)
        except Exception as exc:
            logger.info(
                "DuckDuckGo search failed for query '%s': %s",
                query,
                exc,
            )
            return []

        results: list[SearchResult] = []

        if payload.get("AbstractText"):
            results.append(
                SearchResult(
                    title=str(payload.get("Heading", "")).strip(),
                    url=str(payload.get("AbstractURL", "")).strip(),
                    snippet=str(payload.get("AbstractText", "")).strip(),
                )
            )

        for topic in payload.get("RelatedTopics", []):
            _collect_related_topics(topic, results, limit)
            if len(results) >= limit:
                break

        return results[:limit]

    def _load_payload(self, request: Request) -> dict[str, object]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except ssl.SSLCertVerificationError:
            return self._load_payload_without_ssl_verification(request)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                return self._load_payload_without_ssl_verification(request)
            raise

    def _load_payload_without_ssl_verification(
        self, request: Request
    ) -> dict[str, object]:
        logger.info("DuckDuckGo SSL verification failed, retrying without verification.")
        insecure_context = ssl._create_unverified_context()
        with urlopen(
            request,
            timeout=self.timeout,
            context=insecure_context,
        ) as response:
            return json.load(response)


def _collect_related_topics(
    topic: dict[str, object], results: list[SearchResult], limit: int
) -> None:
    nested_topics = topic.get("Topics")
    if isinstance(nested_topics, list):
        for nested_topic in nested_topics:
            _collect_related_topics(nested_topic, results, limit)
            if len(results) >= limit:
                return
        return

    text = str(topic.get("Text", "")).strip()
    url = str(topic.get("FirstURL", "")).strip()
    if not text or not url:
        return

    title, _, maybe_snippet = text.partition(" - ")
    results.append(
        SearchResult(
            title=title.strip(),
            url=url,
            snippet=(maybe_snippet or text).strip(),
        )
    )
