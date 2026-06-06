"""Entry point for the local AI assistant."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from llm.qwen_runner import QwenRunner, ask_model, build_direct_answer_prompt
from llm.web_rag import WebRagPipeline
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


def main(argv: list[str] | None = None) -> int:
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


def _answer_with_web_context(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    pipeline = WebRagPipeline()
    answer, search_results, documents = pipeline.answer_question(question)

    if not documents:
        if not search_results:
            return _format_insufficient_information(search_results)
        fallback_answer = ask_model(build_direct_answer_prompt(question))
        return _format_model_fallback_answer(
            fallback_answer,
            search_results[:3],
        )

    if _looks_like_insufficient_information(answer):
        fallback_answer = ask_model(build_direct_answer_prompt(question))
        return _format_model_fallback_answer(
            fallback_answer,
            search_results[:3],
        )

    return _format_grounded_answer(answer, search_results[:3])


def _format_grounded_answer(answer: str, sources: list[SearchResult]) -> str:
    if not sources:
        return _format_insufficient_information(sources)

    sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(sources)


def _format_insufficient_information(sources: list[SearchResult]) -> str:
    message = (
        "I do not have enough information in the retrieved context to answer "
        "with certainty."
    )
    if not sources:
        return f"{message}\n\nSources:\n- No relevant URLs were found."

    sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{message}\n\nSources:\n" + "\n".join(sources)


def _format_model_fallback_answer(
    answer: str,
    sources: list[SearchResult],
) -> str:
    prefix = (
        "I could not retrieve enough reliable web context. "
        "Here is a general answer from the local model:\n\n"
    )
    if not sources:
        return f"{prefix}{answer}"

    formatted_sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{prefix}{answer}\n\nRetrieved sources:\n" + "\n".join(
        formatted_sources
    )


def _looks_like_insufficient_information(answer: str) -> bool:
    normalized_answer = answer.strip().lower()
    markers = (
        "do not have enough information",
        "don't have enough information",
        "insufficient information",
        "no tengo suficiente informacion",
        "no tengo suficiente información",
    )
    return any(marker in normalized_answer for marker in markers)


if __name__ == "__main__":
    raise SystemExit(main())
