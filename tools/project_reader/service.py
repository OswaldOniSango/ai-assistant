"""Local project file discovery and relevance scoring."""

from __future__ import annotations

import re
from pathlib import Path

from .models import ProjectFile

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "models",
    "node_modules",
    "venv",
}
DEFAULT_INCLUDED_EXTENSIONS = {
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
DEFAULT_MAX_FILE_BYTES = 80_000


class ProjectReaderService:
    """Find and read project files relevant to a question."""

    def __init__(
        self,
        root: Path | str | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.max_file_bytes = max_file_bytes

    def search(self, question: str, limit: int = 5) -> list[ProjectFile]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        scored_files: list[tuple[int, Path]] = []
        for path in self._iter_candidate_paths():
            score = self._score_path(question, path)
            if score > 0:
                scored_files.append((score, path))

        scored_files.sort(key=lambda item: (-item[0], str(item[1])))
        selected_paths = [path for _, path in scored_files[:limit]]

        files: list[ProjectFile] = []
        for path in selected_paths:
            project_file = self._read_project_file(path)
            if project_file is not None:
                files.append(project_file)

        return files

    def _iter_candidate_paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            if self._is_excluded(path):
                continue

            if path.suffix.lower() not in DEFAULT_INCLUDED_EXTENSIONS:
                continue

            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue

            paths.append(path)

        return paths

    def _is_excluded(self, path: Path) -> bool:
        relative_parts = path.relative_to(self.root).parts
        return any(part in DEFAULT_EXCLUDED_DIRS for part in relative_parts)

    def _score_path(self, question: str, path: Path) -> int:
        query_tokens = _tokens(question)
        if not query_tokens:
            return 0

        relative_path = path.relative_to(self.root).as_posix().lower()
        path_tokens = _tokens(relative_path)
        score = sum(5 for token in query_tokens if token in path_tokens)

        try:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return score

        for token in query_tokens:
            if token in content:
                score += 1

        return score

    def _read_project_file(self, path: Path) -> ProjectFile | None:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        relative_path = path.relative_to(self.root).as_posix()
        return ProjectFile(
            path=relative_path,
            language=_language_for_extension(path.suffix),
            content=content,
        )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) >= 3
    }


def _language_for_extension(extension: str) -> str:
    return {
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(extension.lower(), "text")
