"""Configuration for the unmask local diffusion server."""

CLI_PATH = "~/llama.cpp/build/bin/llama-diffusion-cli"
MODELS_DIR = "~/unmask/models"

DEFAULT_STEPS = 128
DEFAULT_TOKENS = 512
DEFAULT_TEMP = 0.2

HOST = "0.0.0.0"
PORT = 11434
GPU_LAYERS = "99"
CLI_TIMEOUT_SECONDS = 120
