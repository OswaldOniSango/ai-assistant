"""Retrieval-augmented answering over local project files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from llm.qwen_runner import ask_model, build_language_instruction
from tools.project_reader import ProjectFile, ProjectReaderService


class ProjectRagPipeline:
    """Read relevant project files and answer using only that context."""

    def __init__(
        self,
        reader: ProjectReaderService | None = None,
        model: Callable[[str], str] = ask_model,
    ) -> None:
        self.reader = reader or ProjectReaderService(Path.cwd())
        self.model = model

    def answer_question(
        self,
        question: str,
        file_limit: int = 5,
    ) -> tuple[str, list[ProjectFile]]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        files = self.reader.search(question, limit=file_limit)
        if not files:
            return "", files

        prompt = self._build_answer_prompt(question, files)
        return self.model(prompt), files

    def _build_answer_prompt(
        self,
        question: str,
        files: list[ProjectFile],
    ) -> str:
        return (
            "You are a local AI assistant helping explain a software project.\n"
            "Answer the user question using only the provided project files.\n"
            "If the project files do not contain enough information, say that "
            "you do not have enough information.\n"
            "Do not invent files, functions, classes, behavior, or dependencies.\n"
            "Each project file is provided as a block with a Path and Content. "
            "If the question names a file, use the block whose Path matches that file.\n"
            "When the user asks for exact code details such as signatures, copy "
            "them from the provided Content instead of paraphrasing.\n"
            f"{build_language_instruction('project files')}"
            "Include the relevant file paths you used.\n\n"
            f"User question: {question}\n\n"
            "Project context:\n"
            f"{build_project_context(files)}\n\n"
            "Answer:"
        )


def build_project_context(files: list[ProjectFile]) -> str:
    if not files:
        return "No project files were found."

    context_blocks: list[str] = []
    for index, file in enumerate(files, start=1):
        context_blocks.append(
            f"[File {index}]\n"
            f"Path: {file.path}\n"
            f"Language: {file.language}\n"
            "Content:\n"
            f"{file.content}"
        )

    return "\n\n".join(context_blocks)
