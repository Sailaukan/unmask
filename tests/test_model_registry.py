from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unmask.models import DEFAULT_MODEL, get_model_config, list_models


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_is_registered(self) -> None:
        self.assertIn(DEFAULT_MODEL, list_models())
        self.assertIsNotNone(get_model_config(DEFAULT_MODEL))

    def test_model_config_has_filename_and_flags(self) -> None:
        config = get_model_config("llada:8b")
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config["filename"], "llada-8b-q4_k_m.gguf")
        self.assertEqual(config["flags"], ["--diffusion-block-length", "32"])


if __name__ == "__main__":
    unittest.main()
