"""Tests for the DuckDuckGo HTML results parser."""

from tools.web_search.providers.duckduckgo_html import (
    parse_results_page,
    resolve_redirect_url,
)

SAMPLE_PAGE = """
<html><body>
  <div class="result results_links results_links_deep web-result">
    <div class="links_main links_deep result__body">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.espn.com%2Fnba%2Fstory&amp;rut=abc">
          NBA Finals 2026: Thunder vs Pacers
        </a>
      </h2>
      <a class="result__snippet" href="#">Oklahoma City Thunder face the Indiana Pacers.</a>
    </div>
  </div>
  <div class="result result--ad">
    <h2 class="result__title">
      <a class="result__a" href="https://ads.example.com">Buy NBA tickets</a>
    </h2>
  </div>
  <div class="result web-result">
    <div class="result__body">
      <h2 class="result__title">
        <a class="result__a" href="https://www.nba.com/finals">NBA Finals official</a>
      </h2>
      <a class="result__snippet" href="#">Official NBA Finals page.</a>
    </div>
  </div>
</body></html>
"""


def test_parses_results_and_decodes_redirects():
    results = parse_results_page(SAMPLE_PAGE, limit=5)
    urls = [result.url for result in results]

    assert "https://www.espn.com/nba/story" in urls
    assert "https://www.nba.com/finals" in urls


def test_skips_sponsored_results():
    results = parse_results_page(SAMPLE_PAGE, limit=5)
    assert all("ads.example.com" not in result.url for result in results)


def test_respects_limit():
    results = parse_results_page(SAMPLE_PAGE, limit=1)
    assert len(results) == 1


def test_keeps_titles_and_snippets():
    results = parse_results_page(SAMPLE_PAGE, limit=5)
    first = results[0]
    assert "Thunder" in first.title
    assert "Pacers" in first.snippet


def test_resolve_redirect_url_decodes_uddg():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=xyz"
    assert resolve_redirect_url(href) == "https://example.com/page"


def test_resolve_redirect_url_passes_through_direct_links():
    assert resolve_redirect_url("https://example.com") == "https://example.com"


def test_resolve_redirect_url_empty():
    assert resolve_redirect_url("") == ""


def test_garbage_page_returns_empty_list():
    assert parse_results_page("", limit=5) == []
