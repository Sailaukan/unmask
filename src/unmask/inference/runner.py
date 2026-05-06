"""Local diffusion runner wrappers for persistent worker and CLI fallback."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import AsyncIterator

import httpx

from unmask.config import (
    CLI_PATH,
    CLI_TIMEOUT_SECONDS,
    DEFAULT_CLEAN_TAIL,
    DEFAULT_STEPS,
    DEFAULT_TEMP,
    DEFAULT_TOKENS,
    DIFFUSION_WORKER_URL,
    FALLBACK_TO_CLI,
    GPU_LAYERS,
    MODELS_DIR,
    USE_PERSISTENT_WORKER,
    WORKER_REQUEST_TIMEOUT_SECONDS,
)
from unmask.inference.errors import (
    CliProcessError,
    CliTimeoutError,
    StreamingUnavailableError,
    UnknownModelError,
    WorkerProcessError,
    WorkerUnavailableError,
)
from unmask.inference.output_cleaning import clean_cli_output, extract_cli_output
from unmask.inference.paths import (
    estimated_diffusion_sequence_length,
    resolve_model_path,
    validate_cli_path,
)
from unmask.models import get_model_config


def worker_payload(
    *,
    model: str,
    prompt: str,
    n_tokens: int,
    steps: int,
    temperature: float,
    models_dir: str = MODELS_DIR,
) -> dict[str, object]:
    model_config = get_model_config(model)
    if model_config is None:
        raise UnknownModelError(model)

    model_path = resolve_model_path(model, models_dir)
    return {
        "model": model,
        "model_path": str(model_path),
        "prompt": prompt,
        "n_tokens": n_tokens,
        "steps": steps,
        "temperature": temperature,
        "use_chat_template": False,
        "model_flags": list(model_config["flags"]),
    }


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
    clean_tail: bool = DEFAULT_CLEAN_TAIL,
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
    clean_tail: bool = DEFAULT_CLEAN_TAIL,
) -> str:
    payload = worker_payload(
        model=model,
        prompt=prompt,
        n_tokens=n_tokens,
        steps=steps,
        temperature=temperature,
        models_dir=models_dir,
    )

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


async def stream_diffusion_worker(
    *,
    model: str,
    prompt: str,
    n_tokens: int = DEFAULT_TOKENS,
    steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMP,
    worker_url: str = DIFFUSION_WORKER_URL,
    models_dir: str = MODELS_DIR,
    clean_tail: bool = DEFAULT_CLEAN_TAIL,
) -> AsyncIterator[dict[str, object]]:
    if not USE_PERSISTENT_WORKER:
        raise StreamingUnavailableError()

    payload = worker_payload(
        model=model,
        prompt=prompt,
        n_tokens=n_tokens,
        steps=steps,
        temperature=temperature,
        models_dir=models_dir,
    )

    try:
        async with httpx.AsyncClient(timeout=WORKER_REQUEST_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", f"{worker_url}/generate/stream", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise WorkerProcessError(response.status_code, body.decode("utf-8", errors="replace"))

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise WorkerProcessError(500, line) from exc
                    if not isinstance(chunk, dict):
                        raise WorkerProcessError(500, line)
                    if "error" in chunk:
                        raise WorkerProcessError(500, json.dumps(chunk))

                    text = str(chunk.get("text") or "")
                    if clean_tail:
                        text = clean_cli_output(text)
                    chunk["text"] = text
                    yield chunk
    except httpx.HTTPError as exc:
        raise WorkerUnavailableError(worker_url, str(exc)) from exc


async def run_diffusion(
    *,
    model: str,
    prompt: str,
    n_tokens: int = DEFAULT_TOKENS,
    steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMP,
    clean_tail: bool = DEFAULT_CLEAN_TAIL,
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
