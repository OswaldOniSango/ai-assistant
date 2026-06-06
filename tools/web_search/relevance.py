"""Relevance filtering for search results."""

from __future__ import annotations

import re

from .models import SearchResult


def filter_relevant_results(
    query: str, results: list[SearchResult]
) -> list[SearchResult]:
    return [result for result in results if is_relevant_result(query, result)]


def is_relevant_result(query: str, result: SearchResult) -> bool:
    query_terms = _extract_query_terms(query)
    if not query_terms:
        return False

    haystack = f"{result.title} {result.url} {result.snippet}".lower()

    matches = sum(1 for term in query_terms if term in haystack)
    if len(query_terms) == 1:
        return matches >= 1

    min_matches = min(2, len(query_terms))
    return matches >= min_matches


def _extract_query_terms(query: str) -> list[str]:
    normalized_query = (
        query.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    tokens = re.findall(r"[a-z0-9]+", normalized_query)
    return [token for token in tokens if len(token) >= 3]
