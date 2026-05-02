from __future__ import annotations

import os
from pathlib import Path

from unmask.config import CLI_PATH, DIFFUSION_SERVER_PATH, MODELS_DIR
from unmask.inference.errors import (
    CliNotFoundError,
    ModelFileNotFoundError,
    UnknownModelError,
    WorkerNotFoundError,
)
from unmask.models import get_model_config


def expand_path(path: str) -> Path:
    return Path(path).expanduser()


def validate_cli_path(cli_path: str = CLI_PATH) -> Path:
    resolved = expand_path(cli_path)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CliNotFoundError(resolved)
    return resolved


def validate_diffusion_server_path(worker_path: str = DIFFUSION_SERVER_PATH) -> Path:
    resolved = expand_path(worker_path)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise WorkerNotFoundError(resolved)
    return resolved


def resolve_model_path(model: str, models_dir: str = MODELS_DIR) -> Path:
    model_config = get_model_config(model)
    if model_config is None:
        raise UnknownModelError(model)

    model_path = expand_path(models_dir) / model_config["filename"]
    if not model_path.is_file():
        raise ModelFileNotFoundError(model, model_path)

    return model_path


def estimated_diffusion_sequence_length(prompt: str, n_tokens: int) -> int:
    estimated_prompt_tokens = max(16, min(512, len(prompt) // 4))
    return max(64, n_tokens + estimated_prompt_tokens)
