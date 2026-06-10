"""Web search orchestration service."""

from __future__ import annotations

from .models import SearchResult
from .providers.bing import BingProvider
from .providers.duckduckgo import DuckDuckGoProvider
from .providers.duckduckgo_html import DuckDuckGoHtmlProvider
from .relevance import filter_relevant_results


class WebSearchService:
    """Coordinates providers and result filtering."""

    def __init__(self, providers: list[object] | None = None) -> None:
        # Order matters: real search results first, instant answers last.
        self.providers = providers or [
            DuckDuckGoHtmlProvider(),
            BingProvider(),
            DuckDuckGoProvider(),
        ]

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        for provider in self.providers:
            results = provider.search(query, limit=limit)
            relevant_results = filter_relevant_results(query, results)
            if relevant_results:
                return relevant_results[:limit]

        return []


def search_web(query: str, limit: int = 5) -> list[SearchResult]:
    return WebSearchService().search(query, limit=limit)
