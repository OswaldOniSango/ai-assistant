"""Web search orchestration service."""

from __future__ import annotations

from .models import SearchResult
from .providers.bing import BingProvider
from .providers.duckduckgo import DuckDuckGoProvider
from .relevance import filter_relevant_results


class WebSearchService:
    """Coordinates providers and result filtering."""

    def __init__(self, providers: list[object] | None = None) -> None:
        self.providers = providers or [DuckDuckGoProvider(), BingProvider()]

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        allow_fallback: bool = True,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        fallback_results: list[SearchResult] = []

        for provider in self.providers:
            results = provider.search(query, limit=limit)
            if results and not fallback_results:
                fallback_results = results[:limit]

            relevant_results = filter_relevant_results(query, results)
            if relevant_results:
                return relevant_results[:limit]

        return fallback_results if allow_fallback else []


def search_web(query: str, limit: int = 5) -> list[SearchResult]:
    return WebSearchService().search(query, limit=limit)


class WebSearchTool:
    """Small adapter for the assistant tool layer."""

    def __init__(self, service: WebSearchService | None = None) -> None:
        self.service = service or WebSearchService()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.service.search(query, limit=limit)
