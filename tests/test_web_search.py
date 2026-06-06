from __future__ import annotations

import unittest

from tools.web_search.models import SearchResult
from tools.web_search.query_builder import finalize_queries
from tools.web_search.service import WebSearchService


class QueryBuilderTest(unittest.TestCase):
    def test_generated_queries_are_prioritized_before_fallback(self) -> None:
        queries = finalize_queries(
            ["Aaron Judge injury latest"],
            fallback_query="de que esta lesionado Aaron Judge?",
        )

        self.assertEqual(
            queries,
            ["Aaron Judge injury latest", "de que esta lesionado Aaron Judge?"],
        )

    def test_duplicate_fallback_query_appears_once(self) -> None:
        queries = finalize_queries(
            ["Aaron Judge injury latest"],
            fallback_query="Aaron Judge injury latest",
        )

        self.assertEqual(queries, ["Aaron Judge injury latest"])


class FakeProvider:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.results[:limit]


class WebSearchServiceTest(unittest.TestCase):
    def test_irrelevant_provider_results_are_not_returned(self) -> None:
        provider = FakeProvider(
            [
                SearchResult(
                    title="De",
                    url="https://de.gov/",
                    snippet="Delaware government information.",
                )
            ]
        )
        service = WebSearchService(providers=[provider])

        results = service.search("Aaron Judge injury latest")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
