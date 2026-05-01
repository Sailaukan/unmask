"""Lifecycle management for the persistent llama-diffusion-server worker."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from config import (
    DIFFUSION_WORKER_HOST,
    DIFFUSION_WORKER_PORT,
    DIFFUSION_WORKER_URL,
    GPU_LAYERS,
    WORKER_MAX_SEQUENCE_LENGTH,
    WORKER_STARTUP_TIMEOUT_SECONDS,
)
from registry import get_model_config
from runner import (
    ModelFileNotFoundError,
    UnknownModelError,
    WorkerNotFoundError,
    resolve_model_path,
    validate_diffusion_server_path,
)


class DiffusionWorkerManager:
    def __init__(self, worker_url: str = DIFFUSION_WORKER_URL) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.process: subprocess.Popen[str] | None = None
        self.managed_model: str | None = None

    async def start(self, model: str) -> None:
        model_config = get_model_config(model)
        if model_config is None:
            raise UnknownModelError(model)

        worker_path = validate_diffusion_server_path()
        model_path = resolve_model_path(model)

        existing = await self.health()
        if existing:
            loaded_path = str(existing.get("model_path") or "")
            if Path(loaded_path).expanduser() == model_path:
                print(f"unmask using existing llama-diffusion-server at {self.worker_url}", file=sys.stderr)
            else:
                print(
                    "WARNING: llama-diffusion-server is already running at "
                    f"{self.worker_url}, but it has a different model loaded: {loaded_path}. "
                    "Requests for the active model will fall back to llama-diffusion-cli.",
                    file=sys.stderr,
                )
            return

        cmd = [
            str(worker_path),
            "-m",
            str(model_path),
            "--host",
            DIFFUSION_WORKER_HOST,
            "--port",
            str(DIFFUSION_WORKER_PORT),
            "-ngl",
            GPU_LAYERS,
            "--ubatch-size",
            str(WORKER_MAX_SEQUENCE_LENGTH),
            *list(model_config["flags"]),
        ]

        print("unmask starting llama-diffusion-server worker", file=sys.stderr)
        print(" ".join(cmd), file=sys.stderr)

        self.process = subprocess.Popen(cmd, text=True)
        self.managed_model = model
        await self.wait_until_ready(model_path)

    async def stop(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 10)

        self.process = None
        self.managed_model = None

    async def health(self) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.worker_url}/health")
            if response.status_code != 200:
                return None
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        return body if isinstance(body, dict) else None

    async def wait_until_ready(self, model_path: Path) -> None:
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"llama-diffusion-server exited with code {self.process.returncode}")

            body = await self.health()
            if body and Path(str(body.get("model_path") or "")).expanduser() == model_path:
                print(f"unmask worker ready at {self.worker_url}", file=sys.stderr)
                return

            await asyncio.sleep(1.0)

        raise TimeoutError(
            f"llama-diffusion-server did not become ready after {WORKER_STARTUP_TIMEOUT_SECONDS} seconds"
        )


def describe_worker_startup_error(exc: Exception) -> str:
    if isinstance(exc, WorkerNotFoundError):
        return (
            f"llama-diffusion-server was not found at {exc.worker_path}. "
            "Build it with: cd ~/llama.cpp && cmake --build build --target llama-diffusion-server -j"
        )
    if isinstance(exc, ModelFileNotFoundError):
        return f"Model file was not found at {exc.model_path}"
    return str(exc)
