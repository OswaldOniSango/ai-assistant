"""Retrieval-augmented answering with web search."""

from __future__ import annotations

import logging

from llm.context_builder import build_search_context
from llm.qwen_runner import ask_model
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


class WebRagPipeline:
    """Plan queries, retrieve pages, and answer using extracted context."""

    def __init__(
        self,
        search_service: WebSearchService | None = None,
        content_extractor: WebContentExtractor | None = None,
    ) -> None:
        self.search_service = search_service or WebSearchService()
        self.content_extractor = content_extractor or WebContentExtractor()

    def answer_question(
        self,
        question: str,
        query_limit: int = 4,
        results_per_query: int = 3,
        document_limit: int = 2,
    ) -> tuple[str, list[SearchResult], list[RetrievedDocument]]:
        planned_queries = generate_search_queries(question, limit=query_limit)
        logger.info("Planned queries: %s", planned_queries)

        search_results = self._collect_results(planned_queries, results_per_query)
        logger.info("Search results collected: %d", len(search_results))

        documents = self.content_extractor.extract_documents(
            search_results,
            limit=document_limit,
        )
        logger.info("Documents extracted: %d", len(documents))

        if not documents:
            logger.info("No documents extracted; web answer is empty.")
            return "", search_results, documents

        context = build_search_context(documents)
        prompt = self._build_answer_prompt(question, context)
        answer = ask_model(prompt)

        if is_insufficient_context(answer):
            logger.info("Model reported insufficient web context.")

        return answer, search_results, documents

    def _collect_results(
        self,
        queries: list[str],
        results_per_query: int,
    ) -> list[SearchResult]:
        collected_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = self.search_service.search(query, limit=results_per_query)
            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                collected_results.append(result)

        return collected_results

    def _build_answer_prompt(self, question: str, context: str) -> str:
        return (
            "You are a local AI assistant.\n"
            "Answer the user question using only the provided context.\n"
            "If the context does not contain enough information to answer, "
            f"reply with exactly this single word and nothing else: {INSUFFICIENT_CONTEXT}\n"
            "Do not invent facts.\n"
            "First identify the language of the user's question. Do not mention this analysis.\n"
            "Reply entirely in the same language as the user's question.\n"
            "The language of the web context and sources must not change the reply language.\n"
            "If the question is in English, answer in English.\n"
            "If the question is in Spanish, answer in Spanish.\n"
            "Keep the answer brief and factual.\n\n"
            f"User question: {question}\n\n"
            "Web context:\n"
            f"{context}\n\n"
            "Answer:"
        )
