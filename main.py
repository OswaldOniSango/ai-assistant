"""Entry point for the local AI assistant."""

from __future__ import annotations

import json
import re
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
            return _format_insufficient_information(question, search_results)
        fallback_answer = ask_model(build_direct_answer_prompt(question))
        return _format_model_fallback_answer(
            question,
            fallback_answer,
            search_results[:3],
        )

    if _looks_like_insufficient_information(answer):
        fallback_answer = ask_model(build_direct_answer_prompt(question))
        return _format_model_fallback_answer(
            question,
            fallback_answer,
            search_results[:3],
        )

    return _format_grounded_answer(question, answer, search_results[:3])


def _format_grounded_answer(
    question: str, answer: str, sources: list[SearchResult]
) -> str:
    if not sources:
        return _format_insufficient_information(question, sources)

    sources = [
        f"- {result.title or _text(question, 'Untitled', 'Sin título')}: "
        f"{result.url or _text(question, 'No URL', 'Sin URL')}"
        for result in sources[:3]
    ]
    return f"{answer}\n\n{_text(question, 'Sources', 'Fuentes')}:\n" + "\n".join(
        sources
    )


def _format_insufficient_information(
    question: str,
    sources: list[SearchResult],
) -> str:
    message = _text(
        question,
        "I do not have enough information in the retrieved context to answer with certainty.",
        "No tengo suficiente información en el contexto recuperado para responder con certeza.",
    )
    if not sources:
        no_sources = _text(
            question,
            "No relevant URLs were found.",
            "No se encontraron URLs relevantes.",
        )
        return f"{message}\n\n{_text(question, 'Sources', 'Fuentes')}:\n- {no_sources}"

    sources = [
        f"- {result.title or _text(question, 'Untitled', 'Sin título')}: "
        f"{result.url or _text(question, 'No URL', 'Sin URL')}"
        for result in sources[:3]
    ]
    return f"{message}\n\n{_text(question, 'Sources', 'Fuentes')}:\n" + "\n".join(
        sources
    )


def _format_model_fallback_answer(
    question: str,
    answer: str,
    sources: list[SearchResult],
) -> str:
    prefix = _text(
        question,
        "I could not retrieve enough reliable web context. Here is a general answer from the local model:\n\n",
        "No pude recuperar suficiente contexto web confiable. Comparto una respuesta general del modelo local:\n\n",
    )
    if not sources:
        return f"{prefix}{answer}"

    formatted_sources = [
        f"- {result.title or _text(question, 'Untitled', 'Sin título')}: "
        f"{result.url or _text(question, 'No URL', 'Sin URL')}"
        for result in sources[:3]
    ]
    sources_label = _text(question, "Retrieved sources", "Fuentes recuperadas")
    return f"{prefix}{answer}\n\n{sources_label}:\n" + "\n".join(formatted_sources)


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


def _text(question: str, english: str, spanish: str) -> str:
    return spanish if _is_probably_spanish(question) else english


def _is_probably_spanish(text: str) -> bool:
    normalized_text = text.lower()
    if re.search(r"[áéíóúñ¿¡]", normalized_text):
        return True

    spanish_markers = {
        "como",
        "cuál",
        "cual",
        "cuando",
        "donde",
        "esta",
        "está",
        "para",
        "puedo",
        "que",
        "quien",
        "quién",
        "sobre",
    }
    tokens = set(re.findall(r"[a-záéíóúñ]+", normalized_text))
    return bool(tokens & spanish_markers)


if __name__ == "__main__":
    raise SystemExit(main())
