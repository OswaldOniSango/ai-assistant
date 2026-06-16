from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from llm import qwen_runner
from llm.qwen_runner import DEFAULT_MAX_TOKENS, _max_tokens, ask_model


class FakeLlama:
    def __init__(self) -> None:
        self.prompt = ""

    def create_chat_completion(self, **kwargs):
        self.prompt = kwargs["messages"][0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ]
        }


class QwenRunnerConfigTest(unittest.TestCase):
    def test_max_tokens_uses_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_max_tokens(), DEFAULT_MAX_TOKENS)

    def test_max_tokens_uses_environment_value(self) -> None:
        with patch.dict(os.environ, {"QWEN_MAX_TOKENS": "1024"}):
            self.assertEqual(_max_tokens(), 1024)

    def test_max_tokens_rejects_invalid_value(self) -> None:
        with patch.dict(os.environ, {"QWEN_MAX_TOKENS": "many"}):
            with self.assertRaises(ValueError):
                _max_tokens()

    def test_max_tokens_rejects_zero(self) -> None:
        with patch.dict(os.environ, {"QWEN_MAX_TOKENS": "0"}):
            with self.assertRaises(ValueError):
                _max_tokens()

    def test_ask_model_sends_prompt_without_wrapping_it_again(self) -> None:
        fake_model = FakeLlama()
        prompt = "Project RAG prompt\nUser question: explicame Dijkstra.java\nAnswer:"

        with patch.object(qwen_runner, "_load_model", return_value=fake_model):
            ask_model(prompt)

        self.assertEqual(fake_model.prompt, prompt)


if __name__ == "__main__":
    unittest.main()
