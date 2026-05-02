from __future__ import annotations

from unmask.api.errors import runner_error_to_http
from unmask.inference.runner import run_diffusion


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
