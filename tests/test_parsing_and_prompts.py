from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unmask.api.parsing import generation_params, parse_bool, requested_model, should_clean_tail
from unmask.api.prompts import coerce_text, messages_to_chatml
from unmask.config import DEFAULT_STEPS, DEFAULT_TEMP, DEFAULT_TOKENS


class ParsingAndPromptTests(unittest.TestCase):
    def test_generation_params_uses_options_aliases(self) -> None:
        tokens, steps, temp = generation_params(
            {
                "options": {
                    "num_predict": "64",
                    "num_steps": "16",
                    "temperature": "0.3",
                }
            }
        )
        self.assertEqual((tokens, steps, temp), (64, 16, 0.3))

    def test_generation_params_defaults(self) -> None:
        self.assertEqual(generation_params({}), (DEFAULT_TOKENS, DEFAULT_STEPS, DEFAULT_TEMP))

    def test_parse_bool_accepts_common_strings(self) -> None:
        self.assertTrue(parse_bool("yes"))
        self.assertFalse(parse_bool("off", default=True))

    def test_invalid_options_raise_http_error(self) -> None:
        with self.assertRaises(HTTPException):
            should_clean_tail({"options": ["bad"]})

    def test_requested_model_falls_back_to_active_model(self) -> None:
        self.assertEqual(requested_model({}, "dream:7b"), "dream:7b")
        self.assertEqual(requested_model({"model": "llada:8b"}, "dream:7b"), "llada:8b")

    def test_coerce_text_accepts_openai_content_parts(self) -> None:
        self.assertEqual(coerce_text([{"type": "text", "text": "a"}, {"content": "b"}]), "ab")

    def test_messages_to_chatml_sanitizes_unknown_roles(self) -> None:
        prompt = messages_to_chatml([{"role": "tool", "content": "hello"}])
        self.assertEqual(prompt, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")


if __name__ == "__main__":
    unittest.main()
