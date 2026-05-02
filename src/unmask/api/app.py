from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from unmask import __version__
from unmask.api.ollama import build_router as build_ollama_router
from unmask.api.openai import build_router as build_openai_router
from unmask.config import FALLBACK_TO_CLI, USE_PERSISTENT_WORKER
from unmask.inference.errors import CliNotFoundError
from unmask.inference.paths import validate_cli_path
from unmask.inference.worker import DiffusionWorkerManager, describe_worker_startup_error
from unmask.models import DEFAULT_MODEL


def print_cli_status() -> None:
    try:
        cli_path = validate_cli_path()
    except CliNotFoundError as exc:
        print(
            f"ERROR: llama-diffusion-cli was not found at {exc.cli_path}. "
            "Build llama.cpp with diffusion support or update CLI_PATH in src/unmask/config.py.",
            file=sys.stderr,
        )
        return

    print(f"unmask using llama-diffusion-cli at {cli_path}", file=sys.stderr)


def create_app(active_model: str = DEFAULT_MODEL) -> FastAPI:
    worker_manager = DiffusionWorkerManager()

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if USE_PERSISTENT_WORKER:
            try:
                await worker_manager.start(str(fastapi_app.state.active_model))
            except Exception as exc:
                message = describe_worker_startup_error(exc)
                if not FALLBACK_TO_CLI:
                    raise RuntimeError(message) from exc
                print(f"WARNING: persistent worker unavailable: {message}", file=sys.stderr)
                print("unmask will fall back to llama-diffusion-cli per request.", file=sys.stderr)

        if FALLBACK_TO_CLI:
            print_cli_status()

        try:
            yield
        finally:
            await worker_manager.stop()

    app = FastAPI(title="unmask", version=__version__, lifespan=lifespan)
    app.state.active_model = active_model
    app.state.worker_manager = worker_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_active_model() -> str:
        return str(app.state.active_model)

    app.include_router(build_ollama_router(get_active_model))
    app.include_router(build_openai_router(get_active_model))
    return app


app = create_app()
