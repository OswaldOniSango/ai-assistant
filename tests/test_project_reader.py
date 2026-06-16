from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm.project_rag import ProjectRagPipeline, build_project_context
from tools.project_reader import ProjectFile, ProjectReaderService


class ProjectReaderServiceTest(unittest.TestCase):
    def test_finds_relevant_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "assistant.py").write_text(
                "class Assistant:\n"
                "    def answer_with_web_context(self, question):\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text("Project docs", encoding="utf-8")

            service = ProjectReaderService(root)
            files = service.search("how does answer_with_web_context work?")

            self.assertEqual(files[0].path, "assistant.py")
            self.assertEqual(files[0].language, "python")

    def test_excludes_cache_and_model_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "assistant.py").write_text(
                "answer_with_web_context", encoding="utf-8"
            )
            (root / "models").mkdir()
            (root / "models" / "notes.py").write_text(
                "answer_with_web_context", encoding="utf-8"
            )

            service = ProjectReaderService(root)
            self.assertEqual(
                service.search("answer_with_web_context"),
                [],
            )


class ProjectRagPipelineTest(unittest.TestCase):
    def test_build_project_context_includes_file_metadata(self) -> None:
        context = build_project_context(
            [
                ProjectFile(
                    path="assistant.py",
                    language="python",
                    content="class Assistant: pass",
                )
            ]
        )

        self.assertIn("Path: assistant.py", context)
        self.assertIn("Language: python", context)
        self.assertIn("class Assistant", context)

    def test_pipeline_uses_project_context_prompt(self) -> None:
        captured_prompts: list[str] = []

        def fake_model(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "The assistant is defined in assistant.py."

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "assistant.py").write_text(
                "class Assistant: pass",
                encoding="utf-8",
            )

            pipeline = ProjectRagPipeline(
                reader=ProjectReaderService(root),
                model=fake_model,
            )
            answer, files = pipeline.answer_question("where is Assistant?")

        self.assertEqual(answer, "The assistant is defined in assistant.py.")
        self.assertEqual(files[0].path, "assistant.py")
        self.assertIn("using only the provided project files", captured_prompts[0])


if __name__ == "__main__":
    unittest.main()
