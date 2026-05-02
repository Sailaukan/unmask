"""Model registry for locally-run diffusion language models."""

from __future__ import annotations

from typing import TypedDict


class ModelConfig(TypedDict):
    filename: str
    flags: list[str]


DEFAULT_MODEL = "dream:7b"

MODELS: dict[str, ModelConfig] = {
    "dream:7b": {
        "filename": "Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf",
        "flags": ["--diffusion-eps", "0.001", "--diffusion-algorithm", "3"],
    },
    "dream-coder:7b": {
        "filename": "Dream-Coder-v0-Base-7B.i1-Q4_K_S.gguf",
        "flags": ["--diffusion-eps", "0.001", "--diffusion-algorithm", "3"],
    },
    "llada:8b": {
        "filename": "llada-8b-q4_k_m.gguf",
        "flags": ["--diffusion-block-length", "32"],
    },
}


def list_models() -> list[str]:
    return list(MODELS)


def get_model_config(model: str) -> ModelConfig | None:
    return MODELS.get(model)
