from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from unmask.api.execution import run_model
from unmask.api.parsing import (
    generation_params,
    read_json_object,
    requested_model,
    should_clean_tail,
    should_stream,
)
from unmask.api.prompts import coerce_text, messages_to_chatml
from unmask.api.responses import ollama_model_entry, streaming_headers, utc_now
from unmask.api.streaming import ollama_chat_stream, ollama_generate_stream
from unmask.models import list_models


def build_router(get_active_model: Callable[[], str]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tags")
    async def api_tags() -> dict[str, Any]:
        return {"models": [ollama_model_entry(name) for name in list_models()]}

    @router.post("/api/generate")
    async def api_generate(request: Request):
        body = await read_json_object(request)
        model = requested_model(body, get_active_model())
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

    @router.post("/api/chat")
    async def api_chat(request: Request):
        body = await read_json_object(request)
        model = requested_model(body, get_active_model())
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

    return router
