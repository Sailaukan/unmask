# unmask

`unmask` is a local HTTP server that exposes Ollama and OpenAI-compatible APIs
for diffusion language models. It does not use LM Studio and does not proxy to
any external API. Requests are translated into local `llama-diffusion-cli`
subprocess calls against Dream or LLaDA `.gguf` files.

```text
Continue / Open WebUI / OpenAI-compatible tool
          |
          v
unmask FastAPI server on port 11434
          |
          v
llama-diffusion-cli subprocess
          |
          v
Dream / LLaDA .gguf model file
```

## Prerequisites

- Python 3.10+
- `llama.cpp` built with Metal support on macOS
- `llama-diffusion-cli` available at:

```text
~/llama.cpp/build/bin/llama-diffusion-cli
```

- Model files downloaded under:

```text
~/unmask/models
```

Expected filenames:

```text
~/unmask/models/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
~/unmask/models/Dream-Coder-v0-Base-7B.i1-Q4_K_S.gguf
~/unmask/models/llada-8b-q4_k_m.gguf
```

Edit `config.py` if your CLI or model directory is somewhere else.

## Install

```bash
pip install -r requirements.txt
```

## Run

Dream:

```bash
python server.py --model dream:7b
```

LLaDA:

```bash
python server.py --model llada:8b
```

Dream Coder:

```bash
python server.py --model dream-coder:7b
```

The server listens on:

```text
http://localhost:11434
```

If native Ollama is already using port `11434`, stop Ollama first or run
`unmask` on another port:

```bash
python server.py --model dream:7b --port 11435
```

## API

Ollama-compatible:

- `GET /api/tags`
- `POST /api/generate`
- `POST /api/chat`

OpenAI-compatible:

- `GET /v1/models`
- `POST /v1/chat/completions`

Streaming is intentionally not implemented yet. The server buffers the full
`llama-diffusion-cli` output and returns one complete response.

By default, `unmask` returns raw model output. If a diffusion model produces a
degenerate tail such as repeated `2 2 2` or repeated role labels, opt into
post-processing per request with:

```json
{
  "clean_tail": true
}
```

For Ollama requests, this can also be placed inside `options`.

## Quick Checks

List Ollama-style models:

```bash
curl http://localhost:11434/api/tags
```

Generate:

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dream:7b",
    "prompt": "Write one sentence about diffusion language models.",
    "stream": false,
    "options": {
      "num_predict": 128,
      "num_steps": 128,
      "temperature": 0.2,
      "clean_tail": true
    }
  }'
```

Chat:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dream:7b",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Explain diffusion language models in one sentence."}
    ],
    "stream": false
  }'
```

OpenAI-compatible:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dream:7b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

## Continue.dev

Point Continue at the OpenAI-compatible server:

```yaml
models:
  - name: Dream 7B
    provider: openai
    model: dream:7b
    apiBase: http://localhost:11434/v1
    apiKey: local
```

You can also use `dream-coder:7b` for the Dream Coder base model or `llada:8b`
once the LLaDA model file is downloaded.

## Open WebUI

Use the Ollama-compatible connection:

```text
http://localhost:11434
```

The model names exposed by `/api/tags` are:

```text
dream:7b
dream-coder:7b
llada:8b
```

## Model Registry

Model-specific diffusion flags live in `registry.py`. Dream and LLaDA flags are
kept separate and must not be mixed:

```python
MODELS = {
    "dream:7b": {
        "filename": "Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf",
        "flags": ["--diffusion-eps", "0.001", "--diffusion-algorithm", "3"],
    },
    "dream-coder:7b": {
        "filename": "Dream-Coder-v0-Base-7B.i1-Q4_K_S.gguf",
        "flags": ["--diffusion-eps", "0.001", "--diffusion-algorithm", "3"],
    },
    "llada:8b": {
        "filename": "llada-8b-q4_k_m.gguf",
        "flags": ["--diffusion-block-length", "32"],
    },
}
```

## Errors

- Missing `llama-diffusion-cli`: printed clearly on startup and returned as HTTP `500`.
- Missing model file: HTTP `404` with the expected full path.
- CLI timeout: HTTP `504`.
- CLI nonzero exit: HTTP `500` with recent stdout/stderr.
