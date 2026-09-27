from __future__ import annotations

import pytest

from mcqgen2.prompts import SYSTEM_INSTRUCTIONS, build_generation_prompt


@pytest.mark.parametrize("mode", ["high_volume", "high_quality"])
@pytest.mark.parametrize("question_type", ["mcq", "sata"])
def test_generation_prompt_assembles_every_mode_and_type(
    mode: str,
    question_type: str,
) -> None:
    prompt = build_generation_prompt(
        question_count=8,
        mode=mode,  # type: ignore[arg-type]
        question_type=question_type,  # type: ignore[arg-type]
        instruction_rules="Use concise scenarios.",
    )

    assert f"Create exactly 8 {question_type.upper()} questions" in prompt
    assert f"{question_type.upper()} rules:" in prompt
    assert "standalone assessment content" in prompt
    assert '"the source states,"' in prompt
    assert "Do not label choices with letters or numbers" in prompt
    assert "rationale in the same object" in prompt
    assert "Every question must be answerable from the source material" not in prompt
    assert "Use concise scenarios." in prompt
    assert "question_writing_preferences" not in prompt


def test_generation_prompt_handles_no_custom_instructions() -> None:
    prompt = build_generation_prompt(
        question_count=1,
        mode="high_volume",
        question_type="mcq",
        instruction_rules="",
    )

    assert "High-Volume mode:" in prompt


def test_system_instructions_are_centralized_and_domain_neutral() -> None:
    assert "structured output contract" in SYSTEM_INSTRUCTIONS
    assert "source-only constraint" not in SYSTEM_INSTRUCTIONS
    assert "medical" not in SYSTEM_INSTRUCTIONS.casefold()
    assert "clinical" not in SYSTEM_INSTRUCTIONS.casefold()
