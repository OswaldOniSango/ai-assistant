"""DuckDuckGo search via the ddgs library.

The library rotates endpoints and handles the anti-bot measures that block
our hand-rolled HTML scraper, so it is the most reliable provider we have.
It is optional: if ddgs is not installed, the provider just returns nothing
and the service falls through to the next provider.
"""

from __future__ import annotations

import logging

from ..models import SearchResult

try:
    from ddgs import DDGS
except ImportError:
    try:
        # Older package name, same API.
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger(__name__)


class DdgsProvider:
    """Fetches search results using the ddgs library."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if DDGS is None:
            logger.info(
                "ddgs library is not installed; skipping DdgsProvider. "
                "Install it with: pip install ddgs"
            )
            return []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = list(ddgs.text(query, max_results=limit))
        except Exception as exc:
            logger.info("ddgs search failed for query '%s': %s", query, exc)
            return []

        return to_search_results(raw_results, limit)


def to_search_results(
    raw_results: list[dict],
    limit: int = 5,
) -> list[SearchResult]:
    """Map ddgs result dicts to our SearchResult model."""
    results: list[SearchResult] = []

    for raw in raw_results:
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("href", "") or raw.get("url", "")).strip()
        snippet = str(raw.get("body", "") or raw.get("snippet", "")).strip()

        if not title or not url:
            continue

        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break

    return results
