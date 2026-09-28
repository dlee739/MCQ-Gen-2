from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GenerationMode = Literal["high_volume", "high_quality"]
QuestionType = Literal["mcq", "sata"]
InputMode = Literal["extracted_text", "pdf"]


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
        description="More complex application and reasoning with stronger distractors.",
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

DEFAULT_MODE: GenerationMode = "high_volume"
MODE_DEFAULT_MODELS: dict[GenerationMode, str] = {
    "high_volume": "gpt-6-luna",
    "high_quality": "gpt-6-sol",
}
MODE_DEFAULT_QUESTION_COUNTS: dict[GenerationMode, int] = {
    "high_volume": 15,
    "high_quality": 5,
}
DEFAULT_MODEL = MODE_DEFAULT_MODELS[DEFAULT_MODE]
DEFAULT_QUESTION_TYPE: QuestionType = "mcq"
DEFAULT_INPUT_MODE: InputMode = "extracted_text"
DEFAULT_QUESTION_COUNT = MODE_DEFAULT_QUESTION_COUNTS[DEFAULT_MODE]
MIN_QUESTION_COUNT = 1
MAX_QUESTION_COUNT = 100
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_INSTRUCTION_RULES_CHARS = 2_000
# Compatibility alias for older callers and stored data.
MAX_CUSTOM_INSTRUCTIONS_CHARS = MAX_INSTRUCTION_RULES_CHARS
PROMPT_VERSION = "mcq-v9"

QUESTION_TYPE_LABELS: dict[QuestionType, str] = {
    "mcq": "MCQ",
    "sata": "SATA",
}

INPUT_MODE_LABELS: dict[InputMode, str] = {
    "extracted_text": "Extract text (recommended)",
    "pdf": "Send original PDF",
}


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
