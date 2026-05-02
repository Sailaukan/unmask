from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def streaming_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
