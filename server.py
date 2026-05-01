from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from config import DEFAULT_STEPS, DEFAULT_TEMP, DEFAULT_TOKENS, FALLBACK_TO_CLI, HOST, PORT, USE_PERSISTENT_WORKER
from registry import get_model_config, list_models
from runner import (
    CliNotFoundError,
    CliProcessError,
    CliTimeoutError,
    ModelFileNotFoundError,
    StreamingUnavailableError,
    UnknownModelError,
    WorkerNotFoundError,
    WorkerProcessError,
    WorkerUnavailableError,
    run_diffusion,
    stream_diffusion_worker,
    validate_cli_path,
)
from worker import DiffusionWorkerManager, describe_worker_startup_error

DEFAULT_MODEL = "dream:7b"
ACTIVE_MODEL = DEFAULT_MODEL
WORKER_MANAGER = DiffusionWorkerManager()

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
    if USE_PERSISTENT_WORKER:
        try:
            await WORKER_MANAGER.start(ACTIVE_MODEL)
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
        await WORKER_MANAGER.stop()


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


def parse_bool(value: Any, default: bool = False, field_name: str = "boolean value") -> bool:
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
    raise HTTPException(status_code=400, detail=f"{field_name} must be a boolean.")


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
    return parse_bool(
        first_present(body.get("clean_tail"), options.get("clean_tail"), default=False),
        field_name="clean_tail",
    )


def should_stream(body: dict[str, Any]) -> bool:
    return parse_bool(body.get("stream"), default=False, field_name="stream")


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

    if isinstance(exc, WorkerNotFoundError):
        return HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "worker_path": str(exc.worker_path),
            },
        )

    if isinstance(exc, WorkerUnavailableError):
        return HTTPException(
            status_code=503,
            detail={
                "error": str(exc),
                "worker_url": exc.worker_url,
                "detail": exc.detail,
            },
        )

    if isinstance(exc, WorkerProcessError):
        return HTTPException(
            status_code=exc.status_code if 400 <= exc.status_code < 500 else 500,
            detail={
                "error": str(exc),
                "status_code": exc.status_code,
                "body": exc.body[-2000:],
            },
        )

    if isinstance(exc, StreamingUnavailableError):
        return HTTPException(
            status_code=501,
            detail={
                "error": str(exc),
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
        return await run_diffusion(
            model=model,
            prompt=prompt,
            n_tokens=n_tokens,
            steps=steps,
            temperature=temperature,
            clean_tail=clean_tail,
        )
    except Exception as exc:
        raise runner_error_to_http(exc) from exc


def streaming_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


def error_detail(exc: Exception) -> Any:
    http_exc = runner_error_to_http(exc)
    return http_exc.detail


async def ollama_generate_stream(
    *,
    model: str,
    prompt: str,
    n_tokens: int,
    steps: int,
    temperature: float,
    clean_tail: bool,
):
    try:
        async for chunk in stream_diffusion_worker(
            model=model,
            prompt=prompt,
            n_tokens=n_tokens,
            steps=steps,
            temperature=temperature,
            clean_tail=clean_tail,
        ):
            done = bool(chunk.get("done"))
            payload: dict[str, Any] = {
                "model": model,
                "created_at": utc_now(),
                "response": str(chunk.get("text") or ""),
                "done": done,
                "diffusion_step": int(chunk.get("step") or 0),
                "diffusion_steps": int(chunk.get("total_steps") or steps),
            }
            if done:
                payload["done_reason"] = "stop"
            yield json.dumps(payload) + "\n"
    except Exception as exc:
        yield json.dumps({"error": error_detail(exc), "done": True}) + "\n"


async def ollama_chat_stream(
    *,
    model: str,
    prompt: str,
    n_tokens: int,
    steps: int,
    temperature: float,
    clean_tail: bool,
):
    try:
        async for chunk in stream_diffusion_worker(
            model=model,
            prompt=prompt,
            n_tokens=n_tokens,
            steps=steps,
            temperature=temperature,
            clean_tail=clean_tail,
        ):
            done = bool(chunk.get("done"))
            payload: dict[str, Any] = {
                "model": model,
                "created_at": utc_now(),
                "message": {
                    "role": "assistant",
                    "content": str(chunk.get("text") or ""),
                },
                "done": done,
                "diffusion_step": int(chunk.get("step") or 0),
                "diffusion_steps": int(chunk.get("total_steps") or steps),
            }
            if done:
                payload["done_reason"] = "stop"
            yield json.dumps(payload) + "\n"
    except Exception as exc:
        yield json.dumps({"error": error_detail(exc), "done": True}) + "\n"


async def openai_chat_stream(
    *,
    model: str,
    prompt: str,
    n_tokens: int,
    steps: int,
    temperature: float,
    clean_tail: bool,
):
    stream_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    try:
        async for chunk in stream_diffusion_worker(
            model=model,
            prompt=prompt,
            n_tokens=n_tokens,
            steps=steps,
            temperature=temperature,
            clean_tail=clean_tail,
        ):
            done = bool(chunk.get("done"))
            payload = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": str(chunk.get("text") or ""),
                        },
                        "finish_reason": "stop" if done else None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': error_detail(exc)})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/api/tags")
async def api_tags() -> dict[str, Any]:
    return {"models": [ollama_model_entry(name) for name in list_models()]}


@app.post("/api/generate")
async def api_generate(request: Request):
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = coerce_text(body.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required.")

    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    if should_stream(body):
        return StreamingResponse(
            ollama_generate_stream(
                model=model,
                prompt=prompt,
                n_tokens=n_tokens,
                steps=steps,
                temperature=temperature,
                clean_tail=clean_tail,
            ),
            media_type="application/x-ndjson",
            headers=streaming_headers(),
        )

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
async def api_chat(request: Request):
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = messages_to_chatml(body.get("messages"))
    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    if should_stream(body):
        return StreamingResponse(
            ollama_chat_stream(
                model=model,
                prompt=prompt,
                n_tokens=n_tokens,
                steps=steps,
                temperature=temperature,
                clean_tail=clean_tail,
            ),
            media_type="application/x-ndjson",
            headers=streaming_headers(),
        )

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
async def v1_chat_completions(request: Request):
    body = await read_json_object(request)
    model = requested_model(body)
    prompt = messages_to_chatml(body.get("messages"))
    n_tokens, steps, temperature = generation_params(body)
    clean_tail = should_clean_tail(body)
    if should_stream(body):
        return StreamingResponse(
            openai_chat_stream(
                model=model,
                prompt=prompt,
                n_tokens=n_tokens,
                steps=steps,
                temperature=temperature,
                clean_tail=clean_tail,
            ),
            media_type="text/event-stream",
            headers=streaming_headers(),
        )

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
