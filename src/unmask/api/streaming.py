from __future__ import annotations

import json
import time
from typing import Any

from unmask.api.errors import error_detail
from unmask.api.responses import utc_now
from unmask.inference.runner import stream_diffusion_worker


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
