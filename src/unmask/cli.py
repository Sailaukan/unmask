from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from unmask.api.app import create_app
from unmask.config import HOST, PORT
from unmask.models import DEFAULT_MODEL, get_model_config, list_models


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unmask local diffusion LLM server.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list_models(), help="Default model id.")
    parser.add_argument("--host", default=HOST, help="Server host.")
    parser.add_argument("--port", type=int, default=PORT, help="Server port.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    model_config = get_model_config(args.model)
    filename = model_config["filename"] if model_config else "(unknown)"
    print(f"unmask default model: {args.model} ({filename})", file=sys.stderr)

    uvicorn.run(create_app(active_model=args.model), host=args.host, port=args.port)
