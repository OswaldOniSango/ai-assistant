"""Entry point for the local assistant."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict

from llm.qwen_runner import QwenRunner, ask_model, build_direct_answer_prompt
from llm.web_rag import WebRagPipeline
from tools.web_search import SearchResult, WebSearchTool, search_web


class LocalAIAssistant:
    """Coordinates chat, the local model, and tools."""

    def __init__(self) -> None:
        self.model = QwenRunner()
        self.web_search = WebSearchTool()

    def run_interactive(self) -> None:
        print("Local AI Assistant")
        print("Type 'exit' to quit.\n")

        while True:
            user_message = input("You: ").strip()
            if user_message.lower() in {"salir", "exit", "quit"}:
                print("Assistant: Goodbye.")
                break

            if not user_message:
                print("Assistant: Enter a message to continue.")
                continue

            if user_message.lower().startswith("search "):
                query = user_message[7:].strip()
                tool_result = self.web_search.search(query)
                print(_format_search_results(tool_result))
                continue

            if user_message.lower().startswith("search-answer "):
                query = user_message[14:].strip()
                print(_answer_with_web_context(query))
                continue

            if user_message.lower().startswith("buscar "):
                query = user_message[7:].strip()
                tool_result = self.web_search.search(query)
                print(_format_search_results(tool_result))
                continue

            if user_message.lower().startswith("buscar-responder "):
                query = user_message[17:].strip()
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
        raise ValueError("The question cannot be empty.")

    pipeline = WebRagPipeline()
    answer, search_results, documents = pipeline.answer_question(question)

    if not documents:
        if not search_results:
            fallback_answer = ask_model(
                build_direct_answer_prompt(question),
                max_tokens=192,
            )
            return fallback_answer
        fallback_answer = ask_model(
            build_direct_answer_prompt(question),
            max_tokens=192,
        )
        return _format_grounded_answer(question, fallback_answer, search_results[:3])

    if _looks_like_insufficient_information(answer):
        return _format_insufficient_information(question, search_results[:3])

    return _format_grounded_answer(question, answer, search_results[:3])


def _format_grounded_answer(
    question: str, answer: str, sources: list[SearchResult]
) -> str:
    if not sources:
        return _format_insufficient_information(question, sources)

    sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{answer}\n\n{_sources_label(question)}:\n" + "\n".join(sources)


def _format_insufficient_information(
    question: str,
    sources: list[SearchResult],
) -> str:
    message = _localized_message(
        question,
        english="I do not have enough information in the retrieved context to answer confidently.",
        spanish="No tengo suficiente información en el contexto recuperado para responder con certeza.",
    )
    if not sources:
        missing_sources = _localized_message(
            question,
            english="No relevant URLs were found.",
            spanish="No se encontraron URLs relevantes.",
        )
        return f"{message}\n\n{_sources_label(question)}:\n- {missing_sources}"

    sources = [
        f"- {result.title or 'Untitled'}: {result.url or 'No URL'}"
        for result in sources[:3]
    ]
    return f"{message}\n\n{_sources_label(question)}:\n" + "\n".join(sources)


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


def _sources_label(question: str) -> str:
    return _localized_message(
        question,
        english="Sources",
        spanish="Fuentes",
    )


def _localized_message(question: str, english: str, spanish: str) -> str:
    return spanish if _is_probably_spanish(question) else english


def _is_probably_spanish(text: str) -> bool:
    normalized_text = text.lower()
    if re.search(r"[áéíóúñ¿¡]", normalized_text):
        return True

    spanish_markers = {
        "como",
        "consulta",
        "cuál",
        "cual",
        "debo",
        "donde",
        "esta",
        "está",
        "mejor",
        "para",
        "por",
        "puedo",
        "que",
        "qué",
        "sql",
        "una",
    }
    tokens = set(re.findall(r"[a-záéíóúñ]+", normalized_text))
    return len(tokens & spanish_markers) >= 2


if __name__ == "__main__":
    raise SystemExit(main())
