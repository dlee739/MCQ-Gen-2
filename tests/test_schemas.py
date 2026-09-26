from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcqgen2.config import load_local_api_key
from mcqgen2.schemas import GeneratedQuestion, question_bank_model


def question_data() -> dict:
    return {
        "stem": "Which statement is supported by the source?",
        "choices": [
            {"id": "A", "text": "First option"},
            {"id": "B", "text": "Second option"},
            {"id": "C", "text": "Third option"},
            {"id": "D", "text": "Fourth option"},
        ],
        "correct_choice_ids": ["B"],
        "explanation": "Second option is supported; the alternatives conflict with the source.",
    }


def test_question_requires_a_through_d_once() -> None:
    data = question_data()
    data["choices"][3]["id"] = "A"
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate(data)


def test_question_rejects_duplicate_choice_text() -> None:
    data = question_data()
    data["choices"][3]["text"] = "first OPTION"
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate(data)


def test_dynamic_bank_enforces_exact_requested_count() -> None:
    model = question_bank_model(2, "mcq")
    valid = model.model_validate({"questions": [question_data(), question_data()]})
    assert len(valid.questions) == 2
    with pytest.raises(ValidationError):
        model.model_validate({"questions": [question_data()]})


def test_mcq_requires_one_answer_but_sata_accepts_multiple() -> None:
    data = question_data()
    data["correct_choice_ids"] = ["A", "C"]
    with pytest.raises(ValidationError):
        question_bank_model(1, "mcq").model_validate({"questions": [data]})

    parsed = question_bank_model(1, "sata").model_validate({"questions": [data]})
    assert parsed.questions[0].correct_choice_ids == ["A", "C"]


def test_local_env_does_not_override_existing_key(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    load_local_api_key(env_file)
    assert __import__("os").environ["OPENAI_API_KEY"] == "environment-key"
