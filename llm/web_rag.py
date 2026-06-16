"""Retrieval-augmented answering with web search."""

from __future__ import annotations

import logging
from datetime import date
from itertools import zip_longest
from typing import Callable

from llm.context_builder import build_search_context
from llm.qwen_runner import ask_model, build_language_instruction
from llm.query_planner import generate_search_queries
from tools.web_search import RetrievedDocument, SearchResult, WebSearchService
from tools.web_search.content_extractor import WebContentExtractor

# Internal control token. The RAG prompt asks the model to reply with this
# exact token when the web context is not enough, so callers can fall back
# without matching human phrases in multiple languages.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

logger = logging.getLogger(__name__)


def is_insufficient_context(answer: str) -> bool:
    return INSUFFICIENT_CONTEXT in answer.strip().upper()


def _deduplicate_documents(
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    """Drop documents whose text is identical to one already kept."""
    unique_documents: list[RetrievedDocument] = []
    seen_contents: set[str] = set()

    for document in documents:
        content_key = " ".join(document.content.split()).lower()
        if content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        unique_documents.append(document)

    return unique_documents


def _normalize_url(url: str) -> str:
    """Treat http/https, with/without www, and trailing slashes as the same page."""
    from urllib.parse import urlparse

    parsed = urlparse(url.strip().lower())
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


class WebRagPipeline:
    """Plan queries, retrieve pages, and answer using extracted context."""

    def __init__(
        self,
        search_service: WebSearchService | None = None,
        content_extractor: WebContentExtractor | None = None,
        model: Callable[[str], str] = ask_model,
    ) -> None:
        self.search_service = search_service or WebSearchService()
        self.content_extractor = content_extractor or WebContentExtractor()
        self.model = model

    def answer_question(
        self,
        question: str,
        query_limit: int = 4,
        results_per_query: int = 3,
        document_limit: int = 4,
    ) -> tuple[str, list[SearchResult], list[RetrievedDocument]]:
        planned_queries = generate_search_queries(
            question,
            limit=query_limit,
            model=self.model,
        )
        logger.info("Planned queries: %s", planned_queries)

        search_results = self._collect_results(planned_queries, results_per_query)
        logger.info("Search results collected: %d", len(search_results))

        documents = self.content_extractor.extract_documents(
            search_results,
            limit=document_limit,
        )
        documents = _deduplicate_documents(documents)
        logger.info("Documents extracted: %d", len(documents))
        for document in documents:
            logger.info(
                "Document: %s (%s) | content starts: %.150s",
                document.title,
                document.url,
                document.content,
            )

        if not documents:
            logger.info("No documents extracted; web answer is empty.")
            return "", search_results, documents

        context = build_search_context(documents)
        prompt = self._build_answer_prompt(question, context)
        answer = self.model(prompt)

        if is_insufficient_context(answer):
            logger.info("Model reported insufficient web context.")

        return answer, search_results, documents

    def _collect_results(
        self,
        queries: list[str],
        results_per_query: int,
    ) -> list[SearchResult]:
        """Interleave results across queries so no single query dominates.

        With 4 queries x 3 results, the old code put all of query 1 first,
        so document extraction never reached the other queries' results.
        Round-robin keeps the retrieved documents diverse.
        """
        per_query_results = [
            self.search_service.search(query, limit=results_per_query)
            for query in queries
        ]

        collected_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for result_round in zip_longest(*per_query_results):
            for result in result_round:
                if result is None:
                    continue
                url_key = _normalize_url(result.url)
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)
                collected_results.append(result)

        return collected_results

    def _build_answer_prompt(self, question: str, context: str) -> str:
        return (
            "You are a local AI assistant.\n"
            f"Today's date is {date.today().isoformat()}. Use it to interpret "
            "phrases like this year, this season, or recently.\n"
            "Answer the user question using only the provided context.\n"
            "If the context does not contain enough information to answer, "
            f"reply with exactly this single word and nothing else: {INSUFFICIENT_CONTEXT}\n"
            "Do not invent facts.\n"
            f"{build_language_instruction('web context and sources')}"
            "Keep the answer brief and factual.\n\n"
            f"User question: {question}\n\n"
            "Web context:\n"
            f"{context}\n\n"
            "Answer:"
        )
