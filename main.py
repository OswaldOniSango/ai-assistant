"""Punto de entrada del asistente local."""

from llm.qwen_runner import QwenRunner
from tools.web_search import WebSearchTool


class LocalAIAssistant:
    """Coordina el chat, el modelo local y las herramientas."""

    def __init__(self) -> None:
        self.model = QwenRunner()
        self.web_search = WebSearchTool()

    def run(self) -> None:
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


if __name__ == "__main__":
    app = LocalAIAssistant()
    app.run()
