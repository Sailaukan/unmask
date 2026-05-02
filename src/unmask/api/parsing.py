from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, Request

from unmask.config import DEFAULT_STEPS, DEFAULT_TEMP, DEFAULT_TOKENS


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    return body


def requested_model(body: dict[str, Any], active_model: str) -> str:
    return str(body.get("model") or active_model)


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
