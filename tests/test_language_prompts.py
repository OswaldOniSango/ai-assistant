from __future__ import annotations

import unittest

from llm.project_rag import ProjectRagPipeline
from llm.qwen_runner import build_direct_answer_prompt, build_language_instruction
from llm.web_rag import WebRagPipeline


class LanguagePromptTest(unittest.TestCase):
    def test_shared_language_instruction_is_context_aware(self) -> None:
        instruction = build_language_instruction("project files")

        self.assertIn("Reply entirely in the same language", instruction)
        self.assertIn("project files must not change", instruction)

    def test_direct_answer_prompt_requires_user_question_language(self) -> None:
        prompt = build_direct_answer_prompt("de que esta lesionado Aaron Judge?")

        self.assertIn("Reply entirely in the same language", prompt)
        self.assertIn("supporting context must not change", prompt)

    def test_web_rag_prompt_requires_user_question_language(self) -> None:
        prompt = WebRagPipeline()._build_answer_prompt(
            "de que esta lesionado Aaron Judge?",
            "English web context about an injury.",
        )

        self.assertIn("Reply entirely in the same language", prompt)
        self.assertIn("web context and sources must not change", prompt)

    def test_project_rag_prompt_requires_user_question_language(self) -> None:
        prompt = ProjectRagPipeline()._build_answer_prompt(
            "muestrame las firmas exactas de Dijkstra.java",
            [],
        )

        self.assertIn("Reply entirely in the same language", prompt)
        self.assertIn("project files must not change", prompt)


if __name__ == "__main__":
    unittest.main()
