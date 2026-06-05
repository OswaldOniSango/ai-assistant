"""Punto de entrada del asistente local."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from llm.qwen_runner import QwenRunner, ask_model
from llm.web_rag import WebRagPipeline
from tools.web_search import SearchResult, WebSearchTool, search_web


class LocalAIAssistant:
    """Coordina el chat, el modelo local y las herramientas."""

    def __init__(self) -> None:
        self.model = QwenRunner()
        self.web_search = WebSearchTool()

    def run_interactive(self) -> None:
        print("Local AI Assistant")
        print("Escribe 'salir' para terminar.\n")

        while True:
            user_message = input("Tú: ").strip()
            if user_message.lower() in {"salir", "exit", "quit"}:
                print("Asistente: Hasta luego.")
                break

            if not user_message:
                print("Asistente: Escribe un mensaje para continuar.")
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
            print(f"Asistente: {response}")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv

    if len(args) >= 3 and args[1] == "chat":
        prompt = " ".join(args[2:]).strip()
        print(ask_model(prompt))
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
        return "No se encontraron resultados."

    return json.dumps(
        [asdict(result) for result in results],
        ensure_ascii=False,
        indent=2,
    )


def _answer_with_web_context(question: str) -> str:
    if not question.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    pipeline = WebRagPipeline()
    answer, search_results, documents = pipeline.answer_question(question)

    if not documents:
        fallback_sources = search_results[:3]
        return _format_insufficient_information(fallback_sources)

    return _format_grounded_answer(answer, search_results[:3])


def _format_grounded_answer(
    answer: str, sources: list[SearchResult]
) -> str:
    if not sources:
        return _format_insufficient_information(sources)

    sources = [
        f"- {result.title or 'Sin título'}: {result.url or 'Sin URL'}"
        for result in sources[:3]
    ]
    return f"{answer}\n\nFuentes:\n" + "\n".join(sources)


def _format_insufficient_information(sources: list[SearchResult]) -> str:
    message = (
        "No tengo suficiente información en el contexto recuperado para "
        "responder con certeza."
    )
    if not sources:
        return f"{message}\n\nFuentes:\n- No se encontraron URLs relevantes."

    sources = [
        f"- {result.title or 'Sin título'}: {result.url or 'Sin URL'}"
        for result in sources[:3]
    ]
    return f"{message}\n\nFuentes:\n" + "\n".join(sources)


if __name__ == "__main__":
    raise SystemExit(main())
