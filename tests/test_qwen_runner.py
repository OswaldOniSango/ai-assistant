from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from llm.qwen_runner import DEFAULT_MAX_TOKENS, _max_tokens


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


if __name__ == "__main__":
    unittest.main()
