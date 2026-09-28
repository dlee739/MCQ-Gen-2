from __future__ import annotations

import base64
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from mcqgen2.config import (
    ALLOWED_MODELS,
    INPUT_MODE_LABELS,
    MAX_PDF_BYTES,
    MAX_INSTRUCTION_RULES_CHARS,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    MODE_CONFIGS,
    QUESTION_TYPE_LABELS,
    GenerationMode,
    InputMode,
    QuestionType,
    max_output_tokens,
)
from mcqgen2.pdf_input import extract_pdf_text
from mcqgen2.prompts import SYSTEM_INSTRUCTIONS, build_generation_prompt
from mcqgen2.pricing import (
    CostBreakdown,
    TokenUsage,
    calculate_cost,
    pricing_snapshot,
    usage_from_response,
)
from mcqgen2.schemas import (
    GeneratedQuestion,
    ModelGeneratedQuestion,
    model_questions_from_bank,
    question_bank_model,
)


class GenerationError(RuntimeError):
    """Raised when a generation request cannot produce a complete question set."""


SATA_CORRECT_COUNT_WEIGHTS = {0: 5, 1: 20, 2: 40, 3: 30, 4: 5}


@dataclass(frozen=True)
class GenerationResult:
    questions: list[GeneratedQuestion]
    usage: TokenUsage
    cost: CostBreakdown
    pricing: dict[str, Any]
    sata_correct_counts: list[int]


def validate_pdf(filename: str, pdf_bytes: bytes) -> None:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The PDF exceeds the 50 MB API file-input limit.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")


def sata_correct_count_quotas(
    question_count: int,
    *,
    rng: random.Random | None = None,
) -> dict[int, int]:
    """Apportion a center-weighted SATA mix across zero through four answers."""
    if question_count < 1:
        raise ValueError("question_count must be positive")

    scaled = {
        correct_count: question_count * weight
        for correct_count, weight in SATA_CORRECT_COUNT_WEIGHTS.items()
    }
    quotas = {
        correct_count: weighted_count // 100
        for correct_count, weighted_count in scaled.items()
    }
    remaining = question_count - sum(quotas.values())
    tie_order = list(SATA_CORRECT_COUNT_WEIGHTS)
    (rng if rng is not None else random).shuffle(tie_order)
    ranked = sorted(
        tie_order,
        key=lambda correct_count: scaled[correct_count] % 100,
        reverse=True,
    )
    for correct_count in ranked[:remaining]:
        quotas[correct_count] += 1
    return quotas


def source_sha256(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def normalize_instruction_rules(value: str) -> str:
    normalized = value.strip()
    if len(normalized) > MAX_INSTRUCTION_RULES_CHARS:
        raise ValueError(
            "Instruction rules must be "
            f"{MAX_INSTRUCTION_RULES_CHARS:,} characters or fewer."
        )
    return normalized


# Compatibility alias for callers from the previous prompt UI.
normalize_custom_instructions = normalize_instruction_rules


def randomize_choice_positions(
    question: ModelGeneratedQuestion,
    *,
    question_type: QuestionType = "mcq",
    rng: random.Random | None = None,
) -> GeneratedQuestion:
    """Shuffle tagged choices, assign A-D, and append fixed choice E for SATA."""
    shuffler = rng if rng is not None else random
    shuffled = [
        (choice, True) for choice in question.correct_choices
    ] + [
        (choice, False) for choice in question.incorrect_choices
    ]
    shuffler.shuffle(shuffled)

    remapped_choices: list[dict[str, str]] = []
    remapped_correct_ids: list[str] = []
    explanation_parts: list[str] = []
    for new_id, (choice, is_correct) in zip(
        ("A", "B", "C", "D"), shuffled, strict=True
    ):
        rationale = choice.rationale.rstrip()
        if rationale[-1] not in ".!?":
            rationale += "."
        remapped_choices.append(
            {"id": new_id, "text": choice.text, "rationale": rationale}
        )
        if is_correct:
            remapped_correct_ids.append(new_id)
        explanation_parts.append(f"{choice.text}: {rationale}")

    if question_type == "sata":
        if not remapped_correct_ids:
            remapped_correct_ids = ["E"]
            none_rationale = "Correct because every supplied choice is incorrect."
            explanation_parts.append(
                f"None of the above: {none_rationale}"
            )
        else:
            none_rationale = "Incorrect because at least one supplied choice is correct."
            explanation_parts.append(
                f"None of the above: {none_rationale}"
            )
        remapped_choices.append(
            {
                "id": "E",
                "text": "None of the above",
                "rationale": none_rationale,
            }
        )

    return GeneratedQuestion.model_validate(
        {
            "stem": question.stem,
            "choices": remapped_choices,
            "correct_choice_ids": remapped_correct_ids,
            "explanation": " ".join(explanation_parts),
        }
    )


def _request_content(
    *,
    filename: str,
    pdf_bytes: bytes,
    input_mode: InputMode,
    prompt: str,
) -> list[dict[str, str]]:
    if input_mode == "extracted_text":
        source_text = extract_pdf_text(pdf_bytes)
        return [
            {
                "type": "input_text",
                "text": f"<source_material>\n{source_text}\n</source_material>",
            },
            {"type": "input_text", "text": prompt},
        ]

    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return [
        {
            "type": "input_file",
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{encoded}",
            "detail": "auto",
        },
        {"type": "input_text", "text": prompt},
    ]


def generate_question_set(
    *,
    client: OpenAI,
    filename: str,
    pdf_bytes: bytes,
    model: str,
    mode: GenerationMode,
    question_type: QuestionType,
    question_count: int,
    input_mode: InputMode = "extracted_text",
    instruction_rules: str = "",
    custom_instructions: str | None = None,
) -> GenerationResult:
    validate_pdf(filename, pdf_bytes)
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if question_type not in QUESTION_TYPE_LABELS:
        raise ValueError(f"Unsupported question type: {question_type}")
    if input_mode not in INPUT_MODE_LABELS:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    if not MIN_QUESTION_COUNT <= question_count <= MAX_QUESTION_COUNT:
        raise ValueError(
            f"question_count must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}"
        )

    if custom_instructions is not None:
        if instruction_rules:
            raise ValueError("Pass instruction_rules, not both instruction arguments.")
        instruction_rules = custom_instructions
    instruction_rules = normalize_instruction_rules(instruction_rules)
    safe_filename = Path(filename).name
    sata_quotas = (
        sata_correct_count_quotas(question_count)
        if question_type == "sata"
        else None
    )
    response_model = question_bank_model(
        question_count,
        question_type,
        sata_quotas=sata_quotas,
    )
    prompt = build_generation_prompt(
        question_count=question_count,
        mode=mode,
        question_type=question_type,
        instruction_rules=instruction_rules,
        sata_correct_count_quotas=sata_quotas,
    )
    content = _request_content(
        filename=safe_filename,
        pdf_bytes=pdf_bytes,
        input_mode=input_mode,
        prompt=prompt,
    )

    try:
        response = client.responses.parse(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": content,
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

    model_questions = model_questions_from_bank(parsed, question_type)
    if question_type == "sata":
        random.shuffle(model_questions)
    questions = [
        randomize_choice_positions(question, question_type=question_type)
        for question in model_questions
    ]
    if len(questions) != question_count:
        raise GenerationError(
            f"Expected {question_count} questions but received {len(questions)}."
        )
    sata_counts = (
        [
            0 if question.correct_choice_ids == ["E"] else len(question.correct_choice_ids)
            for question in questions
        ]
        if question_type == "sata"
        else []
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
        sata_correct_counts=sata_counts,
    )
