"""Entry point for the local AI assistant."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict

from llm.assistant_router import AssistantRouter
from llm.qwen_runner import QwenRunner, ask_model, build_direct_answer_prompt
from llm.web_rag import WebRagPipeline, is_insufficient_context
from tools.web_search import SearchResult, WebSearchTool, search_web


class LocalAIAssistant:
    """Coordinate chat, the local model, and tools."""

    def __init__(self) -> None:
        self.model = QwenRunner()
        self.web_search = WebSearchTool()

    def run_interactive(self) -> None:
        print("Local AI Assistant")
        print("Type 'exit' to finish.\n")

        while True:
            user_message = input("You: ").strip()
            if user_message.lower() in {"exit", "quit"}:
                print("Assistant: Goodbye.")
                break

            if not user_message:
                print("Assistant: Type a message to continue.")
                continue

            if user_message.lower().startswith("search "):
                command_length = user_message.find(" ") + 1
                query = user_message[command_length:].strip()
                tool_result = self.web_search.search(query)
                print(_format_search_results(tool_result))
                continue

            if user_message.lower().startswith("search-answer "):
                command_length = user_message.find(" ") + 1
                query = user_message[command_length:].strip()
                print(_answer_with_web_context(query))
                continue

            response = self.model.generate(user_message)
            print(f"Assistant: {response}")


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Enable diagnostics with: ASSISTANT_LOG_LEVEL=INFO python3 main.py ..."""
    logging.basicConfig(
        level=os.getenv("ASSISTANT_LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = argv if argv is not None else sys.argv

    if len(args) >= 3 and args[1] == "chat":
        prompt = " ".join(args[2:]).strip()
        print(ask_model(build_direct_answer_prompt(prompt)))
        return 0

    if len(args) >= 3 and args[1] == "search":
        query = " ".join(args[2:]).strip()
        print(_format_search_results(search_web(query)))
        return 0

    if len(args) >= 3 and args[1] == "search-answer":
        query = " ".join(args[2:]).strip()
        print(_answer_with_web_context(query))
        return 0

    if len(args) >= 3 and args[1] == "ask":
        question = " ".join(args[2:]).strip()
        print(_answer_with_routing(question))
        return 0

    app = LocalAIAssistant()
    app.run_interactive()
    return 0


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."

    return json.dumps(
        [asdict(result) for result in results],
        ensure_ascii=False,
        indent=2,
    )


def _answer_with_routing(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    router = AssistantRouter()
    routing = router.decide(question)
    print(f"Decision: {routing.decision}")
    print(f"Reason: {routing.reason}")

    if routing.decision == "WEB":
        return _answer_with_web_context(question)

    return ask_model(build_direct_answer_prompt(question))


def _answer_with_web_context(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    pipeline = WebRagPipeline()
    answer, search_results, documents = pipeline.answer_question(question)

    # If the web pipeline could not produce a grounded answer,
    # quietly answer with the local model instead.
    if not documents or is_insufficient_context(answer):
        logger.info("Web context unavailable; falling back to the local model.")
        return ask_model(build_direct_answer_prompt(question))

    return _format_grounded_answer(answer, search_results[:3])


def _format_grounded_answer(answer: str, sources: list[SearchResult]) -> str:
    if not sources:
        return answer

    formatted_sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(formatted_sources)


if __name__ == "__main__":
    raise SystemExit(main())
