from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DEFAULT_STEPS, DEFAULT_TEMP, DEFAULT_TOKENS, HOST, PORT
from registry import get_model_config, list_models
from runner import (
    CliNotFoundError,
    CliProcessError,
    CliTimeoutError,
    ModelFileNotFoundError,
    UnknownModelError,
    run_diffusion_cli,
    validate_cli_path,
)

DEFAULT_MODEL = "dream:7b"
ACTIVE_MODEL = DEFAULT_MODEL

def print_cli_status() -> None:
    try:
        cli_path = validate_cli_path()
    except CliNotFoundError as exc:
        print(
            f"ERROR: llama-diffusion-cli was not found at {exc.cli_path}. "
            "Build llama.cpp with diffusion support or update CLI_PATH in config.py.",
            file=sys.stderr,
        )
        return

    print(f"unmask using llama-diffusion-cli at {cli_path}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print_cli_status()
    yield


app = FastAPI(title="unmask", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    return body


def requested_model(body: dict[str, Any]) -> str:
    return str(body.get("model") or ACTIVE_MODEL)


def parse_options(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") or {}
    if not isinstance(options, dict):
        raise HTTPException(status_code=400, detail="Ollama 'options' must be a JSON object when provided.")
    return options


def first_present(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def parse_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise HTTPException(status_code=400, detail=f"{field_name} must be greater than zero.")
    return parsed


def parse_float(value: Any, default: float, field_name: str) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a number.") from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail=f"{field_name} must be non-negative.")
    return parsed


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    raise HTTPException(status_code=400, detail="clean_tail must be a boolean.")


def generation_params(body: dict[str, Any]) -> tuple[int, int, float]:
    options = parse_options(body)

    token_value = first_present(
        body.get("max_tokens"),
        body.get("num_predict"),
        options.get("num_predict"),
        options.get("max_tokens"),
        default=DEFAULT_TOKENS,
    )
    steps_value = first_present(
        body.get("diffusion_steps"),
        body.get("steps"),
        body.get("num_steps"),
        options.get("diffusion_steps"),
        options.get("steps"),
        options.get("num_steps"),
        default=DEFAULT_STEPS,
    )
    temp_value = first_present(
        body.get("temperature"),
        options.get("temperature"),
        default=DEFAULT_TEMP,
    )

    return (
        parse_int(token_value, DEFAULT_TOKENS, "tokens"),
        parse_int(steps_value, DEFAULT_STEPS, "steps"),
        parse_float(temp_value, DEFAULT_TEMP, "temperature"),
    )


def should_clean_tail(body: dict[str, Any]) -> bool:
    options = parse_options(body)
    return parse_bool(first_present(body.get("clean_tail"), options.get("clean_tail"), default=False))


def messages_to_chatml(messages: Any) -> str:
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list.")

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Each message must be a JSON object.")

        role = str(message.get("role") or "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"

        content = coerce_text(message.get("content")).strip()
        if not content:
            continue

        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def ollama_model_entry(name: str) -> dict[str, Any]:
    family = name.split(":", 1)[0]
    return {
        "name": name,
        "model": name,
        "modified_at": utc_now(),
        "size": 0,
        "digest": "local-diffusion",
        "details": {
            "format": "gguf",
            "family": family,
            "families": [family, "diffusion"],
            "parameter_size": name.split(":", 1)[1] if ":" in name else "",
            "quantization_level": "",
        },
    }


def openai_model_entry(name: str) -> dict[str, Any]:
    return {
        "id": name,
        "object": "model",
        "created": 0,
        "owned_by": "unmask",
    }


def runner_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownModelError):
        return HTTPException(
            status_code=404,
            detail={
                "error": str(exc),
                "available_models": list_models(),
            },
        )

    if isinstance(exc, ModelFileNotFoundError):
        return HTTPException(
            status_code=404,
            detail={
                "error": str(exc),
                "model": exc.model,
                "expected_path": str(exc.model_path),
            },
        )

    if isinstance(exc, CliNotFoundError):
        return HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "cli_path": str(exc.cli_path),
            },
        )

    if isinstance(exc, CliTimeoutError):
        return HTTPException(
            status_code=504,
            detail={
                "error": str(exc),
                "timeout_seconds": exc.timeout,
            },
        )

    if isinstance(exc, CliProcessError):
        return HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "returncode": exc.returncode,
                "stdout": exc.stdout[-2000:],
                "stderr": exc.stderr[-2000:],
            },
        )

    return HTTPException(status_code=500, detail=f"Unexpected runner error: {exc}")


async def run_model(
    model: str,
    prompt: str,
    n_tokens: int,
    steps: int,
    temperature: float,
    clean_tail: bool,
) -> str:
    try:
        return await asyncio.to_thread(
            run_diffusion_cli,
            model=model,
            prompt=prompt,
            n_tokens=n_tokens,
            steps=steps,
            temperature=temperature,
            clean_tail=clean_tail,
        )
    except Exception as exc:
        raise runner_error_to_http(exc) from exc


@app.get("/api/tags")
async def api_tags() -> dict[str, Any]:
    return {"models": [ollama_model_entry(name) for name in list_models()]}


@app.post("/api/generate")
async def api_generate(request: Request) -> JSONResponse:
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = coerce_text(body.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required.")

    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    output = await run_model(model, prompt, n_tokens, steps, temperature, clean_tail)

    return JSONResponse(
        {
            "model": model,
            "created_at": utc_now(),
            "response": output,
            "done": True,
            "done_reason": "stop",
        }
    )


@app.post("/api/chat")
async def api_chat(request: Request) -> JSONResponse:
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = messages_to_chatml(body.get("messages"))
    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    output = await run_model(model, prompt, n_tokens, steps, temperature, clean_tail)

    return JSONResponse(
        {
            "model": model,
            "created_at": utc_now(),
            "message": {
                "role": "assistant",
                "content": output,
            },
            "done": True,
            "done_reason": "stop",
        }
    )


@app.get("/v1/models")
async def v1_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [openai_model_entry(name) for name in list_models()],
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> JSONResponse:
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = messages_to_chatml(body.get("messages"))
    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    output = await run_model(model, prompt, n_tokens, steps, temperature, clean_tail)

    return JSONResponse(
        {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unmask local diffusion LLM server.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list_models(), help="Default model id.")
    parser.add_argument("--host", default=HOST, help="Server host.")
    parser.add_argument("--port", type=int, default=PORT, help="Server port.")
    return parser.parse_args()


def main() -> None:
    global ACTIVE_MODEL

    args = parse_args()
    ACTIVE_MODEL = args.model

    model_config = get_model_config(ACTIVE_MODEL)
    filename = model_config["filename"] if model_config else "(unknown)"
    print(f"unmask default model: {ACTIVE_MODEL} ({filename})", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
