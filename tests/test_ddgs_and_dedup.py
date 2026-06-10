"""Tests for the ddgs result mapping and document deduplication."""

from llm.web_rag import _deduplicate_documents, _normalize_url
from tools.web_search.models import RetrievedDocument
from tools.web_search.providers.ddgs_provider import to_search_results


def test_maps_ddgs_dicts_to_search_results():
    raw = [
        {"title": "NBA Finals", "href": "https://espn.com/nba", "body": "Thunder vs Pacers"},
        {"title": "Other", "url": "https://nba.com", "snippet": "official"},
    ]
    results = to_search_results(raw)

    assert results[0].url == "https://espn.com/nba"
    assert results[0].snippet == "Thunder vs Pacers"
    assert results[1].url == "https://nba.com"
    assert results[1].snippet == "official"


def test_skips_entries_without_title_or_url():
    raw = [
        {"title": "", "href": "https://example.com"},
        {"title": "No url", "href": ""},
        {"title": "Good", "href": "https://good.com", "body": "ok"},
    ]
    results = to_search_results(raw)
    assert len(results) == 1
    assert results[0].url == "https://good.com"


def test_normalize_url_merges_scheme_www_and_trailing_slash():
    variants = [
        "http://www.nba.com/",
        "https://nba.com",
        "https://www.nba.com/?language=us",
    ]
    keys = {_normalize_url(url) for url in variants}
    assert keys == {"nba.com"}


def test_normalize_url_keeps_different_paths_apart():
    assert _normalize_url("https://nba.com/finals") != _normalize_url(
        "https://nba.com/standings"
    )


def test_deduplicates_identical_document_content():
    documents = [
        RetrievedDocument(title="A", url="http://a.com", content="Same   text here"),
        RetrievedDocument(title="B", url="http://b.com", content="same text  here"),
        RetrievedDocument(title="C", url="http://c.com", content="different text"),
    ]
    unique = _deduplicate_documents(documents)
    assert [doc.title for doc in unique] == ["A", "C"]
