"""Helpers for converting retrieved search results into model context."""

from __future__ import annotations

from tools.web_search import RetrievedDocument


def build_search_context(documents: list[RetrievedDocument]) -> str:
    if not documents:
        return "No search results were found."

    context_blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        title = document.title.strip() or "Untitled"
        url = document.url.strip() or "No URL"
        content = document.content.strip() or "No content available."
        context_blocks.append(
            f"[Source {index}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(context_blocks)
