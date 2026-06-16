"""Entry point for the local AI assistant."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict

from assistant import Assistant
from tools.web_search import SearchResult, search_web


def _configure_logging() -> None:
    """Enable diagnostics with: ASSISTANT_LOG_LEVEL=INFO python3 main.py ..."""
    logging.basicConfig(
        level=os.getenv("ASSISTANT_LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = argv if argv is not None else sys.argv

    command = args[1] if len(args) >= 2 else ""
    argument = " ".join(args[2:]).strip() if len(args) >= 3 else ""

    if command and argument:
        assistant = Assistant()

        if command == "chat":
            print(assistant.answer_locally(argument))
            return 0

        if command == "search":
            print(_format_search_results(search_web(argument)))
            return 0

        if command == "search-answer":
            print(assistant.answer_with_web_context(argument))
            return 0

        if command == "project-answer":
            print(assistant.answer_with_project_context(argument))
            return 0

        if command == "ask":
            print(assistant.answer(argument))
            return 0

    run_interactive()
    return 0


def run_interactive() -> None:
    assistant = Assistant()

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
            query = user_message[len("search "):].strip()
            print(_format_search_results(assistant.web_pipeline.search_service.search(query)))
            continue

        print(f"Assistant: {assistant.answer(user_message)}")


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."

    return json.dumps(
        [asdict(result) for result in results],
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
