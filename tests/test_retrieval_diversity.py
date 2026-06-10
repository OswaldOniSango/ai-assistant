"""Tests for result interleaving and extraction limits."""

from llm.web_rag import WebRagPipeline
from tools.web_search.content_extractor import WebContentExtractor
from tools.web_search.models import RetrievedDocument, SearchResult


def _result(url: str) -> SearchResult:
    return SearchResult(title=url, url=url, snippet="snippet")


class FakeSearchService:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.results_by_query.get(query, [])[:limit]


def test_results_are_interleaved_across_queries():
    service = FakeSearchService(
        {
            "q1": [_result("http://a1.com"), _result("http://a2.com")],
            "q2": [_result("http://b1.com"), _result("http://b2.com")],
        }
    )
    pipeline = WebRagPipeline(search_service=service, model=lambda p: "")

    collected = pipeline._collect_results(["q1", "q2"], results_per_query=3)
    urls = [result.url for result in collected]

    assert urls == ["http://a1.com", "http://b1.com", "http://a2.com", "http://b2.com"]


def test_interleaving_deduplicates_normalized_urls():
    service = FakeSearchService(
        {
            "q1": [_result("http://www.nba.com/")],
            "q2": [_result("https://nba.com/?language=us")],
        }
    )
    pipeline = WebRagPipeline(search_service=service, model=lambda p: "")

    collected = pipeline._collect_results(["q1", "q2"], results_per_query=3)
    assert len(collected) == 1


class FlakyExtractor(WebContentExtractor):
    """Fails on URLs containing 'bad' instead of fetching the network."""

    def extract_document(self, result: SearchResult) -> RetrievedDocument | None:
        if "bad" in result.url:
            return None
        return RetrievedDocument(
            title=result.title, url=result.url, content=f"content of {result.url}"
        )


def test_extraction_failures_do_not_consume_the_limit():
    extractor = FlakyExtractor()
    results = [
        _result("http://bad1.com"),
        _result("http://bad2.com"),
        _result("http://good1.com"),
        _result("http://good2.com"),
    ]

    documents = extractor.extract_documents(results, limit=2)
    assert [doc.url for doc in documents] == ["http://good1.com", "http://good2.com"]
