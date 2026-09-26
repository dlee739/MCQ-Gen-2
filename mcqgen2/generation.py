from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from mcqgen2.config import (
    ALLOWED_MODELS,
    MAX_PDF_BYTES,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    MODE_CONFIGS,
    GenerationMode,
    max_output_tokens,
)
from mcqgen2.pricing import (
    CostBreakdown,
    TokenUsage,
    calculate_cost,
    pricing_snapshot,
    usage_from_response,
)
from mcqgen2.schemas import GeneratedQuestion, question_bank_model


class GenerationError(RuntimeError):
    """Raised when a generation request cannot produce a complete question set."""


@dataclass(frozen=True)
class GenerationResult:
    questions: list[GeneratedQuestion]
    usage: TokenUsage
    cost: CostBreakdown
    pricing: dict[str, Any]


def validate_pdf(filename: str, pdf_bytes: bytes) -> None:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The PDF exceeds the 50 MB API file-input limit.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")


def source_sha256(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def _prompt(question_count: int, mode: GenerationMode) -> str:
    shared = f"""
Create exactly {question_count} medical multiple-choice questions using only facts supported by the attached PDF.

Requirements:
- Every question must be answerable from the PDF without outside knowledge.
- Each stem must be clear and self-contained.
- Provide exactly four distinct choices with stable IDs A, B, C, and D.
- Exactly one choice must be correct.
- Distractors must be plausible but demonstrably incorrect according to the PDF.
- Do not use "all of the above" or "none of the above".
- Avoid duplicate questions and repeated testing of the same fact when the document supports broader coverage.
- Write a concise explanation that justifies the correct answer and explains why the distractors are wrong.
- In explanations, refer to the answer text rather than choice letters so choices can be shuffled safely.
- Do not cite outside sources or add facts absent from the PDF.
""".strip()

    if mode == "high_volume":
        mode_rules = """
High-Volume mode:
- Favor reliable routine-practice questions.
- Use a practical mix of direct recall and straightforward application.
- Keep stems and explanations concise so a larger set remains useful and readable.
""".strip()
    else:
        mode_rules = """
High-Quality mode:
- Favor clinically realistic questions that require multi-step application of the source material.
- Use richer patient details only when those details make the tested reasoning more meaningful.
- Make distractors strongly plausible while preserving one unambiguously best answer.
""".strip()
    return f"{shared}\n\n{mode_rules}"


def generate_question_set(
    *,
    client: OpenAI,
    filename: str,
    pdf_bytes: bytes,
    model: str,
    mode: GenerationMode,
    question_count: int,
) -> GenerationResult:
    validate_pdf(filename, pdf_bytes)
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if not MIN_QUESTION_COUNT <= question_count <= MAX_QUESTION_COUNT:
        raise ValueError(
            f"question_count must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}"
        )

    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    safe_filename = Path(filename).name
    response_model = question_bank_model(question_count)

    try:
        response = client.responses.parse(
            model=model,
            instructions=(
                "You are an expert medical assessment writer. Follow the source-only "
                "constraint strictly and return the requested structured question bank."
            ),
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": safe_filename,
                            "file_data": f"data:application/pdf;base64,{encoded}",
                            "detail": "auto",
                        },
                        {"type": "input_text", "text": _prompt(question_count, mode)},
                    ],
                }
            ],
            text_format=response_model,
            reasoning={"effort": MODE_CONFIGS[mode].reasoning_effort},
            max_output_tokens=max_output_tokens(question_count),
            service_tier="default",
            truncation="disabled",
            store=False,
        )
    except Exception as exc:  # SDK exceptions differ by transport/status.
        raise GenerationError(f"OpenAI request failed: {exc}") from exc

    status = getattr(response, "status", "completed")
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        raise GenerationError(f"Generation did not complete ({status}): {details}")

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise GenerationError("The model returned no parsed question bank.")

    questions = list(parsed.questions)
    if len(questions) != question_count:
        raise GenerationError(
            f"Expected {question_count} questions but received {len(questions)}."
        )

    try:
        usage = usage_from_response(response)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc

    cost = calculate_cost(model, usage)
    return GenerationResult(
        questions=questions,
        usage=usage,
        cost=cost,
        pricing=pricing_snapshot(model),
    )
