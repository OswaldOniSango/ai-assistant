"""Punto de entrada del asistente local."""

from __future__ import annotations

import sys

from llm.qwen_runner import QwenRunner, ask_model
from tools.web_search import WebSearchTool


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
                print(f"Asistente: {tool_result}")
                continue

            response = self.model.generate(user_message)
            print(f"Asistente: {response}")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv

    if len(args) >= 3 and args[1] == "chat":
        prompt = " ".join(args[2:]).strip()
        print(ask_model(prompt))
        return 0

    app = LocalAIAssistant()
    app.run_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
