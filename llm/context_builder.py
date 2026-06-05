"""Helpers for converting retrieved search results into model context."""

from __future__ import annotations

from tools.web_search import SearchResult


def build_search_context(results: list[SearchResult]) -> str:
    if not results:
        return "No search results were found."

    context_blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.title.strip() or "Untitled"
        url = result.url.strip() or "No URL"
        snippet = result.snippet.strip() or "No summary available."
        context_blocks.append(
            f"[Source {index}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}"
        )

    return "\n\n".join(context_blocks)
