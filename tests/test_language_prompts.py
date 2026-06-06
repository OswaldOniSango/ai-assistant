from __future__ import annotations

import unittest

from llm.qwen_runner import build_direct_answer_prompt
from llm.web_rag import WebRagPipeline


class LanguagePromptTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
