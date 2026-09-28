from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcqgen2.config import load_local_api_key
from mcqgen2.schemas import SATA_GROUP_FIELDS, GeneratedQuestion, question_bank_model


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


def model_question_data(correct_count: int = 1) -> dict:
    choices = [
        {"text": "First option", "rationale": "First rationale"},
        {"text": "Second option", "rationale": "Second rationale"},
        {"text": "Third option", "rationale": "Third rationale"},
        {"text": "Fourth option", "rationale": "Fourth rationale"},
    ]
    return {
        "stem": "Which statement is supported by the source?",
        "correct_choices": choices[:correct_count],
        "incorrect_choices": choices[correct_count:],
    }


def sata_payload(quotas: dict[int, int]) -> dict:
    return {
        field_name: [model_question_data(correct_count) for _ in range(quotas[correct_count])]
        for correct_count, field_name in SATA_GROUP_FIELDS.items()
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
    valid = model.model_validate(
        {"questions": [model_question_data(), model_question_data()]}
    )
    assert len(valid.questions) == 2
    with pytest.raises(ValidationError):
        model.model_validate({"questions": [model_question_data()]})


def test_mcq_requires_one_answer_but_sata_accepts_multiple() -> None:
    data = model_question_data(correct_count=2)
    with pytest.raises(ValidationError):
        question_bank_model(1, "mcq").model_validate({"questions": [data]})

    quotas = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    parsed = question_bank_model(1, "sata", sata_quotas=quotas).model_validate(
        sata_payload(quotas)
    )
    assert len(parsed.two_correct_questions[0].correct_choices) == 2


@pytest.mark.parametrize("correct_count", range(5))
def test_sata_accepts_zero_through_four_correct_buckets(correct_count: int) -> None:
    quotas = {value: int(value == correct_count) for value in range(5)}
    parsed = question_bank_model(1, "sata", sata_quotas=quotas).model_validate(
        sata_payload(quotas)
    )
    question = getattr(parsed, SATA_GROUP_FIELDS[correct_count])[0]
    assert len(question.correct_choices) == correct_count
    assert len(question.incorrect_choices) == 4 - correct_count


def test_sata_schema_rejects_question_in_wrong_count_group() -> None:
    quotas = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    payload = sata_payload(quotas)
    payload["two_correct_questions"] = [model_question_data(correct_count=3)]

    with pytest.raises(ValidationError):
        question_bank_model(1, "sata", sata_quotas=quotas).model_validate(payload)


def test_model_question_rejects_duplicate_text_across_buckets() -> None:
    data = model_question_data(correct_count=1)
    data["incorrect_choices"][0]["text"] = "FIRST OPTION"
    quotas = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0}
    payload = sata_payload(quotas)
    payload["one_correct_questions"] = [data]
    with pytest.raises(ValidationError, match="choice text must be unique"):
        question_bank_model(1, "sata", sata_quotas=quotas).model_validate(payload)


def test_model_question_requires_exactly_four_total_choices() -> None:
    data = model_question_data(correct_count=1)
    data["incorrect_choices"].pop()
    quotas = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0}
    payload = sata_payload(quotas)
    payload["one_correct_questions"] = [data]
    with pytest.raises(ValidationError):
        question_bank_model(1, "sata", sata_quotas=quotas).model_validate(payload)


def test_local_env_does_not_override_existing_key(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    load_local_api_key(env_file)
    assert __import__("os").environ["OPENAI_API_KEY"] == "environment-key"
