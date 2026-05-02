from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from unmask.api.execution import run_model
from unmask.api.parsing import (
    generation_params,
    read_json_object,
    requested_model,
    should_clean_tail,
    should_stream,
)
from unmask.api.prompts import messages_to_chatml
from unmask.api.responses import openai_model_entry, streaming_headers
from unmask.api.streaming import openai_chat_stream
from unmask.models import list_models


def build_router(get_active_model: Callable[[], str]) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def v1_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [openai_model_entry(name) for name in list_models()],
        }

    @router.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request):
        body = await read_json_object(request)
        model = requested_model(body, get_active_model())
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

    return router
