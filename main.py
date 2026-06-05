"""Punto de entrada del asistente local."""

from __future__ import annotations

import json
import sys

from llm.qwen_runner import QwenRunner, ask_model
from tools.web_search import WebSearchTool, search_web


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

    app = LocalAIAssistant()
    app.run_interactive()
    return 0


def _format_search_results(results: list[dict[str, str]]) -> str:
    if not results:
        return "No se encontraron resultados."

    return json.dumps(results, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
