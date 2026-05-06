from __future__ import annotations

import re


STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")


def trim_stop_markers(text: str) -> str:
    first_stop = min((idx for marker in STOP_MARKERS if (idx := text.find(marker)) >= 0), default=-1)
    return text if first_stop < 0 else text[:first_stop]


def extract_cli_output(stdout: str, stderr: str, *, clean_tail: bool = False) -> str:
    stdout_text = stdout.strip()
    if stdout_text:
        return clean_cli_output(stdout_text) if clean_tail else stdout_text

    lines = stderr.splitlines()
    output_lines: list[str] = []
    after_timing = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("total time:"):
            after_timing = True
            output_lines.clear()
            continue
        if not after_timing:
            continue
        if stripped.startswith("ggml_") or stripped.startswith("llama_"):
            continue
        output_lines.append(line)

    output = "\n".join(output_lines).strip()
    return clean_cli_output(output) if clean_tail else output


def is_repetitive_tail_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if set(stripped) <= {"0", "1", "2", " ", ","}:
        return True

    tokens = re.findall(r"[A-Za-z0-9_]+", stripped.lower())
    if len(tokens) < 12:
        return False

    counts = {token: tokens.count(token) for token in set(tokens)}
    most_common = max(counts.values())
    unique_ratio = len(counts) / len(tokens)
    common_ratio = most_common / len(tokens)

    return unique_ratio <= 0.25 and common_ratio >= 0.35


def trim_inline_repetitive_tail(text: str) -> str:
    return re.sub(
        r"\s+(?:(?:0|1|2|the|to)\s+){8,}(?:0|1|2|the|to|,|\s)*$",
        "",
        text,
        flags=re.IGNORECASE,
    )


def is_role_marker_line(text: str) -> bool:
    normalized = text.strip().lower().removeprefix("<|im_start|>").strip()
    return normalized in {"system", "system:", "user", "user:", "assistant", "assistant:"}


def clean_cli_output(text: str) -> str:
    text = trim_stop_markers(text)
    lines = text.replace("\r", "\n").splitlines()
    cleaned: list[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()
        if is_role_marker_line(stripped):
            continue
        if is_repetitive_tail_line(stripped):
            if cleaned:
                break
            continue
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            cleaned.append("")
            continue
        previous_blank = False
        line = trim_inline_repetitive_tail(line.rstrip())
        if line.strip():
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return re.sub(r"\n{3,}", "\n\n", result)
