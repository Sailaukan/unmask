from __future__ import annotations

from typing import Any

from fastapi import HTTPException


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
