"""Core assistant logic shared by the CLI and the interactive mode."""

from __future__ import annotations

import logging
from typing import Callable

from llm.assistant_router import AssistantRouter
from llm.project_rag import ProjectRagPipeline
from llm.qwen_runner import ask_model, build_direct_answer_prompt
from llm.web_rag import WebRagPipeline, is_insufficient_context
from tools.project_reader import ProjectFile
from tools.web_search import SearchResult

logger = logging.getLogger(__name__)


class Assistant:
    """Answer questions by routing between web search and the local model."""

    def __init__(
        self,
        router: AssistantRouter | None = None,
        web_pipeline: WebRagPipeline | None = None,
        project_pipeline: ProjectRagPipeline | None = None,
        model: Callable[[str], str] = ask_model,
    ) -> None:
        self.model = model
        self.router = router or AssistantRouter(model=model)
        self.web_pipeline = web_pipeline or WebRagPipeline(model=model)
        self.project_pipeline = project_pipeline or ProjectRagPipeline(model=model)

    def answer(self, question: str) -> str:
        """Route the question and answer it. Logs Decision: WEB or LOCAL."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        routing = self.router.decide(question)
        print(f"Decision: {routing.decision}")
        print(f"Reason: {routing.reason}")

        if routing.decision == "WEB":
            return self.answer_with_web_context(question)

        return self.answer_locally(question)

    def answer_locally(self, question: str) -> str:
        return self.model(build_direct_answer_prompt(question))

    def answer_with_web_context(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        answer, search_results, documents = self.web_pipeline.answer_question(
            question
        )

        # If the web pipeline could not produce a grounded answer,
        # quietly answer with the local model instead.
        if not documents or is_insufficient_context(answer):
            logger.info(
                "Web context unavailable; falling back to the local model."
            )
            return self.answer_locally(question)

        return _format_grounded_answer(answer, search_results[:3])

    def answer_with_project_context(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        answer, files = self.project_pipeline.answer_question(question)
        if not files:
            return (
                "I do not have enough information in the project files to "
                "answer with certainty."
            )

        return _format_project_answer(answer, files)


def _format_grounded_answer(answer: str, sources: list[SearchResult]) -> str:
    if not sources:
        return answer

    formatted_sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(formatted_sources)


def _format_project_answer(answer: str, files: list[ProjectFile]) -> str:
    formatted_files = [
        f"- {getattr(file, 'path', 'Unknown path')}"
        for file in files[:5]
    ]
    return f"{answer}\n\nProject files:\n" + "\n".join(formatted_files)
