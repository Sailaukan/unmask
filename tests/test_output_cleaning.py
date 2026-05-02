from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unmask.inference.output_cleaning import clean_cli_output, extract_cli_output


class OutputCleaningTests(unittest.TestCase):
    def test_extract_prefers_stdout(self) -> None:
        self.assertEqual(extract_cli_output(" hello \n", "ignored"), "hello")

    def test_extracts_stderr_after_timing_marker(self) -> None:
        stderr = "loading\n" "total time: 1 ms\n" "answer\n" "ggml_debug\n"
        self.assertEqual(extract_cli_output("", stderr), "answer")

    def test_clean_removes_role_markers_and_repetitive_tail(self) -> None:
        text = "assistant:\nUseful answer\n2 2 2 2 2 2 2 2\n"
        self.assertEqual(clean_cli_output(text), "Useful answer")


if __name__ == "__main__":
    unittest.main()
