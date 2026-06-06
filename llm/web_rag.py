"""Retrieval-augmented answering with web search."""

from __future__ import annotations

import os

from llm.context_builder import build_search_context
from llm.qwen_runner import ask_model
from llm.query_planner import generate_search_queries
from tools.web_search import RetrievedDocument, SearchResult, WebSearchService
from tools.web_search.content_extractor import WebContentExtractor


class WebRagPipeline:
    """Plan queries, retrieve pages, and answer using extracted context."""

    def __init__(
        self,
        search_service: WebSearchService | None = None,
        content_extractor: WebContentExtractor | None = None,
        use_query_planner: bool | None = None,
    ) -> None:
        self.search_service = search_service or WebSearchService()
        self.content_extractor = content_extractor or WebContentExtractor()
        self.use_query_planner = (
            use_query_planner
            if use_query_planner is not None
            else os.getenv("LOCAL_AI_ENABLE_QUERY_PLANNER") == "1"
        )

    def answer_question(
        self,
        question: str,
        query_limit: int = 4,
        results_per_query: int = 3,
        document_limit: int = 2,
    ) -> tuple[str, list[SearchResult], list[RetrievedDocument]]:
        planned_queries = (
            generate_search_queries(question, limit=query_limit)
            if self.use_query_planner
            else [question]
        )
        search_results = self._collect_results(planned_queries, results_per_query)
        documents = self.content_extractor.extract_documents(
            search_results,
            limit=document_limit,
        )

        if not documents:
            return "", search_results, documents

        context = build_search_context(documents)
        prompt = self._build_answer_prompt(question, context)
        answer = ask_model(
            prompt,
            max_tokens=192,
            temperature=0.1,
        )
        return answer, search_results, documents

    def _collect_results(
        self,
        queries: list[str],
        results_per_query: int,
    ) -> list[SearchResult]:
        collected_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = self.search_service.search(
                query,
                limit=results_per_query,
                allow_fallback=False,
            )
            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                collected_results.append(result)

        return collected_results

    def _build_answer_prompt(self, question: str, context: str) -> str:
        return (
            "You are a local AI assistant.\n"
            "Reply in the same language as the user's question.\n"
            "If the question is in English, answer in English.\n"
            "If the question is in Spanish, answer in Spanish.\n"
            "Answer the user question using only the provided context.\n"
            "If the context does not contain enough information, say that you do "
            "not have enough information.\n"
            "Do not invent facts.\n"
            "Keep the answer brief and factual.\n\n"
            f"User question: {question}\n\n"
            "Web context:\n"
            f"{context}\n\n"
            "Answer:"
        )
