from __future__ import annotations

from pathlib import Path

from unmask.models import list_models


class RunnerError(Exception):
    """Base error for local diffusion runner failures."""


class UnknownModelError(RunnerError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Unknown model {model!r}. Available models: {', '.join(list_models())}")


class CliNotFoundError(RunnerError):
    def __init__(self, cli_path: Path) -> None:
        self.cli_path = cli_path
        super().__init__(f"llama-diffusion-cli was not found or is not executable at {cli_path}")


class ModelFileNotFoundError(RunnerError):
    def __init__(self, model: str, model_path: Path) -> None:
        self.model = model
        self.model_path = model_path
        super().__init__(f"Model file for {model!r} was not found at {model_path}")


class CliTimeoutError(RunnerError):
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        super().__init__(f"llama-diffusion-cli timed out after {timeout} seconds")


class CliProcessError(RunnerError):
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"llama-diffusion-cli exited with code {returncode}")


class WorkerNotFoundError(RunnerError):
    def __init__(self, worker_path: Path) -> None:
        self.worker_path = worker_path
        super().__init__(f"llama-diffusion-server was not found or is not executable at {worker_path}")


class WorkerUnavailableError(RunnerError):
    def __init__(self, worker_url: str, detail: str) -> None:
        self.worker_url = worker_url
        self.detail = detail
        super().__init__(f"llama-diffusion-server is unavailable at {worker_url}: {detail}")


class WorkerProcessError(RunnerError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"llama-diffusion-server returned HTTP {status_code}")


class StreamingUnavailableError(RunnerError):
    def __init__(self) -> None:
        super().__init__("Streaming requires the persistent llama-diffusion-server worker.")
