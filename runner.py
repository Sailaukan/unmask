"""Local diffusion runner wrappers for persistent worker and CLI fallback."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path

import httpx

from config import (
    CLI_PATH,
    CLI_TIMEOUT_SECONDS,
    DEFAULT_STEPS,
    DEFAULT_TEMP,
    DEFAULT_TOKENS,
    DIFFUSION_SERVER_PATH,
    DIFFUSION_WORKER_URL,
    FALLBACK_TO_CLI,
    GPU_LAYERS,
    MODELS_DIR,
    USE_PERSISTENT_WORKER,
    WORKER_REQUEST_TIMEOUT_SECONDS,
)
from registry import get_model_config, list_models


class RunnerError(Exception):
    """Base error for local diffusion runner failures."""


class UnknownModelError(RunnerError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Unknown model {model!r}. Available models: {', '.join(list_models())}")


class CliNotFoundError(RunnerError):
    def __init__(self, cli_path: Path) -> None:
        self.cli_path = cli_path
        super().__init__(f"llama-diffusion-cli was not found or is not executable at {cli_path}")


class ModelFileNotFoundError(RunnerError):
    def __init__(self, model: str, model_path: Path) -> None:
        self.model = model
        self.model_path = model_path
        super().__init__(f"Model file for {model!r} was not found at {model_path}")


class CliTimeoutError(RunnerError):
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        super().__init__(f"llama-diffusion-cli timed out after {timeout} seconds")


class CliProcessError(RunnerError):
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"llama-diffusion-cli exited with code {returncode}")


class WorkerNotFoundError(RunnerError):
    def __init__(self, worker_path: Path) -> None:
        self.worker_path = worker_path
        super().__init__(f"llama-diffusion-server was not found or is not executable at {worker_path}")


class WorkerUnavailableError(RunnerError):
    def __init__(self, worker_url: str, detail: str) -> None:
        self.worker_url = worker_url
        self.detail = detail
        super().__init__(f"llama-diffusion-server is unavailable at {worker_url}: {detail}")


class WorkerProcessError(RunnerError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"llama-diffusion-server returned HTTP {status_code}")


def extract_cli_output(stdout: str, stderr: str, *, clean_tail: bool = False) -> str:
    stdout_text = stdout.strip()
    if stdout_text:
        return clean_cli_output(stdout_text) if clean_tail else stdout_text

    lines = stderr.splitlines()
    output_lines: list[str] = []
    after_timing = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("total time:"):
            after_timing = True
            output_lines.clear()
            continue
        if not after_timing:
            continue
        if stripped.startswith("ggml_") or stripped.startswith("llama_"):
            continue
        output_lines.append(line)

    output = "\n".join(output_lines).strip()
    return clean_cli_output(output) if clean_tail else output


def is_repetitive_tail_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if set(stripped) <= {"0", "1", "2", " ", ","}:
        return True

    tokens = re.findall(r"[A-Za-z0-9_]+", stripped.lower())
    if len(tokens) < 12:
        return False

    counts = {token: tokens.count(token) for token in set(tokens)}
    most_common = max(counts.values())
    unique_ratio = len(counts) / len(tokens)
    common_ratio = most_common / len(tokens)

    return unique_ratio <= 0.25 and common_ratio >= 0.35


def trim_inline_repetitive_tail(text: str) -> str:
    return re.sub(
        r"\s+(?:(?:0|1|2|the|to)\s+){8,}(?:0|1|2|the|to|,|\s)*$",
        "",
        text,
        flags=re.IGNORECASE,
    )


def clean_cli_output(text: str) -> str:
    lines = text.replace("\r", "\n").splitlines()
    cleaned: list[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower in {"user:", "assistant:"}:
            continue
        if is_repetitive_tail_line(stripped):
            if cleaned:
                break
            continue
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            cleaned.append("")
            continue
        previous_blank = False
        line = trim_inline_repetitive_tail(line.rstrip())
        if line.strip():
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


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

    filename = str(model_config["filename"])
    model_path = expand_path(models_dir) / filename
    if not model_path.is_file():
        raise ModelFileNotFoundError(model, model_path)

    return model_path


def estimated_diffusion_sequence_length(prompt: str, n_tokens: int) -> int:
    estimated_prompt_tokens = max(16, min(512, len(prompt) // 4))
    return max(64, n_tokens + estimated_prompt_tokens)


def run_diffusion_cli(
    *,
    model: str,
    prompt: str,
    n_tokens: int = DEFAULT_TOKENS,
    steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMP,
    cli_path: str = CLI_PATH,
    models_dir: str = MODELS_DIR,
    timeout: int = CLI_TIMEOUT_SECONDS,
    clean_tail: bool = False,
) -> str:
    model_config = get_model_config(model)
    if model_config is None:
        raise UnknownModelError(model)

    model_path = resolve_model_path(model, models_dir)
    cli = validate_cli_path(cli_path)
    model_flags = list(model_config["flags"])
    cmd = [
        str(cli),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(n_tokens),
        "--diffusion-steps",
        str(steps),
        "--temp",
        str(temperature),
        "--ubatch-size",
        str(estimated_diffusion_sequence_length(prompt, n_tokens)),
        "-ngl",
        GPU_LAYERS,
        *model_flags,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CliTimeoutError(timeout) from exc

    if result.returncode != 0:
        raise CliProcessError(result.returncode, result.stdout, result.stderr)

    return extract_cli_output(result.stdout, result.stderr, clean_tail=clean_tail)


async def run_diffusion_worker(
    *,
    model: str,
    prompt: str,
    n_tokens: int = DEFAULT_TOKENS,
    steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMP,
    worker_url: str = DIFFUSION_WORKER_URL,
    models_dir: str = MODELS_DIR,
    clean_tail: bool = False,
) -> str:
    model_config = get_model_config(model)
    if model_config is None:
        raise UnknownModelError(model)

    model_path = resolve_model_path(model, models_dir)
    payload = {
        "model": model,
        "model_path": str(model_path),
        "prompt": prompt,
        "n_tokens": n_tokens,
        "steps": steps,
        "temperature": temperature,
        "use_chat_template": False,
        "model_flags": list(model_config["flags"]),
    }

    try:
        async with httpx.AsyncClient(timeout=WORKER_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{worker_url}/generate", json=payload)
    except httpx.HTTPError as exc:
        raise WorkerUnavailableError(worker_url, str(exc)) from exc

    if response.status_code >= 400:
        raise WorkerProcessError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise WorkerProcessError(response.status_code, response.text) from exc

    text = str(body.get("text") or "")
    return clean_cli_output(text) if clean_tail else text


async def run_diffusion(
    *,
    model: str,
    prompt: str,
    n_tokens: int = DEFAULT_TOKENS,
    steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMP,
    clean_tail: bool = False,
) -> str:
    if USE_PERSISTENT_WORKER:
        try:
            return await run_diffusion_worker(
                model=model,
                prompt=prompt,
                n_tokens=n_tokens,
                steps=steps,
                temperature=temperature,
                clean_tail=clean_tail,
            )
        except WorkerUnavailableError:
            if not FALLBACK_TO_CLI:
                raise
        except WorkerProcessError as exc:
            if not FALLBACK_TO_CLI or exc.status_code != 409:
                raise

    return await asyncio.to_thread(
        run_diffusion_cli,
        model=model,
        prompt=prompt,
        n_tokens=n_tokens,
        steps=steps,
        temperature=temperature,
        clean_tail=clean_tail,
    )
