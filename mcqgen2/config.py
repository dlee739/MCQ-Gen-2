from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GenerationMode = Literal["high_volume", "high_quality"]


@dataclass(frozen=True)
class ModeConfig:
    label: str
    description: str
    reasoning_effort: Literal["low", "high"]


MODE_CONFIGS: dict[GenerationMode, ModeConfig] = {
    "high_volume": ModeConfig(
        label="High-Volume",
        description="Routine recall and straightforward application questions.",
        reasoning_effort="low",
    ),
    "high_quality": ModeConfig(
        label="High-Quality",
        description="More complex clinical reasoning with stronger distractors.",
        reasoning_effort="high",
    ),
}

ALLOWED_MODELS = (
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_MODE: GenerationMode = "high_volume"
DEFAULT_QUESTION_COUNT = 20
MIN_QUESTION_COUNT = 1
MAX_QUESTION_COUNT = 100
MAX_PDF_BYTES = 50 * 1024 * 1024
PROMPT_VERSION = "mcq-v1"


def max_output_tokens(question_count: int) -> int:
    return min(128_000, 2_000 + 1_200 * question_count)


def load_local_api_key(path: Path) -> None:
    """Load OPENAI_API_KEY from a small local .env file without overriding the environment."""
    if os.getenv("OPENAI_API_KEY") or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            cleaned = value.strip().strip('"').strip("'")
            if cleaned:
                os.environ["OPENAI_API_KEY"] = cleaned
            return
