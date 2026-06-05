"""Relevance filtering for search results."""

from __future__ import annotations

from .models import SearchResult
from .query_builder import extract_entity_terms, extract_intent_terms, extract_query_terms


def filter_relevant_results(
    query: str, results: list[SearchResult]
) -> list[SearchResult]:
    return [result for result in results if is_relevant_result(query, result)]


def is_relevant_result(query: str, result: SearchResult) -> bool:
    query_terms = extract_query_terms(query)
    if not query_terms:
        return False

    entity_terms = extract_entity_terms(query)
    intent_terms = extract_intent_terms(query)
    haystack = f"{result.title} {result.url} {result.snippet}".lower()

    if entity_terms and not all(term in haystack for term in entity_terms):
        return False

    if intent_terms and not any(term in haystack for term in intent_terms):
        return False

    matches = sum(1 for term in query_terms if term in haystack)
    if len(query_terms) == 1:
        return matches >= 1

    min_matches = min(2, len(query_terms))
    return matches >= min_matches
