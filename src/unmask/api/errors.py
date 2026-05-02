from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from unmask.inference.errors import (
    CliNotFoundError,
    CliProcessError,
    CliTimeoutError,
    ModelFileNotFoundError,
    StreamingUnavailableError,
    UnknownModelError,
    WorkerNotFoundError,
    WorkerProcessError,
    WorkerUnavailableError,
)
from unmask.models import list_models


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


def error_detail(exc: Exception) -> Any:
    return runner_error_to_http(exc).detail
