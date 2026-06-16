"""Local project file discovery and relevance scoring."""

from __future__ import annotations

import re
from pathlib import Path

from .models import ProjectFile

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".gradle",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "models",
    "node_modules",
    "out",
    "target",
    "venv",
}
DEFAULT_INCLUDED_EXTENSIONS = {
    ".gradle",
    ".ini",
    ".java",
    ".json",
    ".md",
    ".properties",
    ".py",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
DEFAULT_MAX_FILE_BYTES = 80_000
OVERVIEW_FILENAMES = {
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "readme.md",
    "settings.gradle",
    "settings.gradle.kts",
}


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

        exact_paths = self._exact_filename_paths(question)
        if exact_paths:
            return self._read_project_files(exact_paths[:limit])

        scored_files: list[tuple[int, Path]] = []
        for path in self._iter_candidate_paths():
            score = self._score_path(question, path)
            if score > 0:
                scored_files.append((score, path))

        scored_files.sort(key=lambda item: (-item[0], str(item[1])))
        if scored_files:
            selected_paths = [path for _, path in scored_files[:limit]]
        else:
            selected_paths = self._overview_paths(limit)

        return self._read_project_files(selected_paths)

    def _exact_filename_paths(self, question: str) -> list[Path]:
        requested_filenames = _requested_filenames(question)
        if not requested_filenames:
            return []

        matches: list[Path] = []
        for path in self._iter_candidate_paths():
            if path.name.lower() in requested_filenames:
                matches.append(path)

        matches.sort(key=lambda path: path.relative_to(self.root).as_posix())
        return matches

    def _overview_paths(self, limit: int) -> list[Path]:
        candidates = self._iter_candidate_paths()
        candidates.sort(key=self._overview_sort_key)
        return candidates[:limit]

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

    def _read_project_files(self, paths: list[Path]) -> list[ProjectFile]:
        files: list[ProjectFile] = []
        for path in paths:
            project_file = self._read_project_file(path)
            if project_file is not None:
                files.append(project_file)

        return files

    def _overview_sort_key(self, path: Path) -> tuple[int, str]:
        relative_path = path.relative_to(self.root).as_posix()
        filename = path.name.lower()
        suffix = path.suffix.lower()

        if filename in OVERVIEW_FILENAMES:
            priority = 0
        elif "/src/main/" in f"/{relative_path}" and suffix in {".java", ".py"}:
            priority = 1
        elif "/src/" in f"/{relative_path}" and suffix in {".java", ".py"}:
            priority = 2
        else:
            priority = 3

        return priority, relative_path


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) >= 3
    }


def _requested_filenames(text: str) -> set[str]:
    return {
        match.lower()
        for match in re.findall(r"\b[\w.-]+\.[A-Za-z0-9]+\b", text)
    }


def _language_for_extension(extension: str) -> str:
    return {
        ".gradle": "gradle",
        ".java": "java",
        ".json": "json",
        ".md": "markdown",
        ".properties": "properties",
        ".py": "python",
        ".toml": "toml",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(extension.lower(), "text")
