from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

ChoiceId = Literal["A", "B", "C", "D", "E"]


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: ChoiceId = Field(description="Stable internal choice identifier.")
    text: str = Field(min_length=1, description="Answer choice text.")
    rationale: str | None = Field(
        default=None,
        min_length=1,
        description="Why this displayed choice is correct or incorrect.",
    )


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, description="Answer choice text.")
    rationale: str = Field(
        min_length=1,
        description="Why this choice belongs in its correct or incorrect bucket.",
    )


class ModelGeneratedQuestion(BaseModel):
    """Four-choice structured response returned by the model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stem: str = Field(min_length=1, description="Self-contained question stem.")
    correct_choices: list[ModelChoice] = Field(min_length=0, max_length=4)
    incorrect_choices: list[ModelChoice] = Field(min_length=0, max_length=4)

    @model_validator(mode="after")
    def validate_choices(self) -> "ModelGeneratedQuestion":
        choices = self.correct_choices + self.incorrect_choices
        if len(choices) != 4:
            raise ValueError("correct_choices and incorrect_choices must total four")
        if len({choice.text.casefold() for choice in choices}) != 4:
            raise ValueError("choice text must be unique")
        return self


class ModelMCQGeneratedQuestion(ModelGeneratedQuestion):
    correct_choices: list[ModelChoice] = Field(min_length=1, max_length=1)
    incorrect_choices: list[ModelChoice] = Field(min_length=3, max_length=3)


SATA_GROUP_FIELDS = {
    0: "zero_correct_questions",
    1: "one_correct_questions",
    2: "two_correct_questions",
    3: "three_correct_questions",
    4: "four_correct_questions",
}


@lru_cache(maxsize=5)
def sata_question_model(correct_count: int) -> type[ModelGeneratedQuestion]:
    if correct_count not in SATA_GROUP_FIELDS:
        raise ValueError("SATA correct_count must be between zero and four")
    incorrect_count = 4 - correct_count
    exact_correct = Annotated[
        list[ModelChoice],
        Field(min_length=correct_count, max_length=correct_count),
    ]
    exact_incorrect = Annotated[
        list[ModelChoice],
        Field(min_length=incorrect_count, max_length=incorrect_count),
    ]
    return create_model(
        f"ModelSATAQuestion{correct_count}Correct",
        __base__=ModelGeneratedQuestion,
        correct_choices=(exact_correct, ...),
        incorrect_choices=(exact_incorrect, ...),
    )


class GeneratedQuestion(BaseModel):
    """Final stored question after application-side choice handling."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stem: str = Field(min_length=1)
    choices: list[Choice] = Field(min_length=4, max_length=5)
    correct_choice_ids: list[ChoiceId] = Field(min_length=1, max_length=4)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_final_choices(self) -> "GeneratedQuestion":
        ids = [choice.id for choice in self.choices]
        expected = (
            ["A", "B", "C", "D"]
            if len(ids) == 4
            else ["A", "B", "C", "D", "E"]
        )
        if ids != expected:
            raise ValueError("choices must be ordered with each stable ID exactly once")
        if len({choice.text.casefold() for choice in self.choices}) != len(self.choices):
            raise ValueError("choice text must be unique")
        if len(set(self.correct_choice_ids)) != len(self.correct_choice_ids):
            raise ValueError("correct_choice_ids must be unique")
        if any(choice_id not in ids for choice_id in self.correct_choice_ids):
            raise ValueError("correct_choice_ids must reference existing choices")
        if "E" in self.correct_choice_ids and self.correct_choice_ids != ["E"]:
            raise ValueError("None of the above must be the only correct choice")
        return self


class MCQGeneratedQuestion(GeneratedQuestion):
    choices: list[Choice] = Field(min_length=4, max_length=4)
    correct_choice_ids: list[ChoiceId] = Field(min_length=1, max_length=1)


def question_bank_model(
    question_count: int,
    question_type: str,
    *,
    sata_quotas: Mapping[int, int] | None = None,
) -> type[BaseModel]:
    if question_type not in {"mcq", "sata"}:
        raise ValueError(f"Unsupported question type: {question_type}")
    if question_type == "sata":
        if sata_quotas is None:
            raise ValueError("SATA question banks require correct-count quotas")
        if set(sata_quotas) != set(SATA_GROUP_FIELDS):
            raise ValueError("SATA quotas must contain counts zero through four")
        if any(type(value) is not int or value < 0 for value in sata_quotas.values()):
            raise ValueError("SATA quotas must be non-negative integers")
        if sum(sata_quotas.values()) != question_count:
            raise ValueError("SATA quotas must total question_count")

        fields: dict[str, tuple[object, object]] = {}
        for correct_count, field_name in SATA_GROUP_FIELDS.items():
            quota = sata_quotas[correct_count]
            grouped_questions = Annotated[
                list[sata_question_model(correct_count)],  # type: ignore[valid-type]
                Field(
                    min_length=quota,
                    max_length=quota,
                    description=(
                        f"Questions with exactly {correct_count} correct supplied choices."
                    ),
                ),
            ]
            fields[field_name] = (grouped_questions, ...)
        quota_fingerprint = "_".join(
            str(sata_quotas[correct_count]) for correct_count in SATA_GROUP_FIELDS
        )
        return create_model(
            f"QuestionBankSATA{question_count}_{quota_fingerprint}",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    question_model = ModelMCQGeneratedQuestion
    exact_questions = Annotated[
        list[question_model],  # type: ignore[valid-type]
        Field(min_length=question_count, max_length=question_count),
    ]
    return create_model(
        f"QuestionBank{question_type.upper()}{question_count}",
        __config__=ConfigDict(extra="forbid"),
        questions=(exact_questions, ...),
    )


def model_questions_from_bank(
    parsed: BaseModel,
    question_type: str,
) -> list[ModelGeneratedQuestion]:
    if question_type == "mcq":
        return list(parsed.questions)  # type: ignore[attr-defined, no-any-return]
    if question_type != "sata":
        raise ValueError(f"Unsupported question type: {question_type}")

    questions: list[ModelGeneratedQuestion] = []
    for field_name in SATA_GROUP_FIELDS.values():
        questions.extend(getattr(parsed, field_name))
    return questions
