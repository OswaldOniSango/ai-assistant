"""Web search orchestration service."""

from __future__ import annotations

import logging

from .models import SearchResult
from .providers.bing import BingProvider
from .providers.ddgs_provider import DdgsProvider
from .providers.duckduckgo import DuckDuckGoProvider
from .providers.duckduckgo_html import DuckDuckGoHtmlProvider
from .relevance import filter_relevant_results

logger = logging.getLogger(__name__)


class WebSearchService:
    """Coordinates providers and result filtering."""

    def __init__(self, providers: list[object] | None = None) -> None:
        # Order matters: most reliable first, instant answers last.
        self.providers = providers or [
            DdgsProvider(),
            DuckDuckGoHtmlProvider(),
            BingProvider(),
            DuckDuckGoProvider(),
        ]

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        for provider in self.providers:
            provider_name = type(provider).__name__
            results = provider.search(query, limit=limit)
            relevant_results = filter_relevant_results(query, results)
            logger.info(
                "%s: %d results, %d relevant for '%s'",
                provider_name,
                len(results),
                len(relevant_results),
                query,
            )
            if relevant_results:
                return relevant_results[:limit]

        return []


def search_web(query: str, limit: int = 5) -> list[SearchResult]:
    return WebSearchService().search(query, limit=limit)
