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

    def test_reads_java_project_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "src" / "main" / "java" / "com" / "example"
            source_dir.mkdir(parents=True)
            (source_dir / "PaymentService.java").write_text(
                "package com.example;\n"
                "public class PaymentService {\n"
                "    void processPayment() {}\n"
                "}\n",
                encoding="utf-8",
            )
            (root / "build.gradle").write_text(
                "plugins { id 'java' }",
                encoding="utf-8",
            )
            (root / "pom.xml").write_text(
                "<project><artifactId>payments</artifactId></project>",
                encoding="utf-8",
            )

            service = ProjectReaderService(root)
            files = service.search("PaymentService processPayment java")

            self.assertEqual(files[0].path, "src/main/java/com/example/PaymentService.java")
            self.assertEqual(files[0].language, "java")

    def test_excludes_java_build_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dirname in [".gradle", ".idea", "build", "out", "target"]:
                directory = root / dirname
                directory.mkdir()
                (directory / "PaymentService.java").write_text(
                    "class PaymentService {}",
                    encoding="utf-8",
                )

            service = ProjectReaderService(root)
            self.assertEqual(service.search("PaymentService"), [])

    def test_generic_question_falls_back_to_project_overview_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "src" / "main" / "java" / "org" / "example"
            source_dir.mkdir(parents=True)
            (source_dir / "AlgorithmApplication.java").write_text(
                "package org.example;\n"
                "public class AlgorithmApplication {}\n",
                encoding="utf-8",
            )
            (root / "pom.xml").write_text(
                "<project><artifactId>algorithm</artifactId></project>",
                encoding="utf-8",
            )

            service = ProjectReaderService(root)
            files = service.search("explica algun error en este proyecto")

            self.assertEqual(
                [file.path for file in files],
                ["pom.xml", "src/main/java/org/example/AlgorithmApplication.java"],
            )

    def test_exact_filename_query_reads_only_matching_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Dijkstra.java").write_text(
                "public class Dijkstra {\n"
                "    public static int shortestPath(int[][] graph) { return 0; }\n"
                "}\n",
                encoding="utf-8",
            )
            (root / "BinarySearch.java").write_text(
                "public class BinarySearch {}",
                encoding="utf-8",
            )

            service = ProjectReaderService(root)
            files = service.search(
                "muestrame las firmas exactas de todos los metodos de Dijkstra.java"
            )

            self.assertEqual([file.path for file in files], ["Dijkstra.java"])


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

    def test_prompt_tells_model_to_use_named_file_path(self) -> None:
        prompt = ProjectRagPipeline()._build_answer_prompt(
            "muestrame las firmas exactas de todos los metodos de Dijkstra.java",
            [
                ProjectFile(
                    path="Dijkstra.java",
                    language="java",
                    content="public class Dijkstra {}",
                )
            ],
        )

        self.assertIn("If the question names a file", prompt)
        self.assertIn("copy them from the provided Content", prompt)


if __name__ == "__main__":
    unittest.main()
