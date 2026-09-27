from __future__ import annotations

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


def question_bank_model(question_count: int, question_type: str) -> type[BaseModel]:
    if question_type not in {"mcq", "sata"}:
        raise ValueError(f"Unsupported question type: {question_type}")
    question_model = (
        ModelMCQGeneratedQuestion if question_type == "mcq" else ModelGeneratedQuestion
    )
    exact_questions = Annotated[
        list[question_model],  # type: ignore[valid-type]
        Field(min_length=question_count, max_length=question_count),
    ]
    return create_model(
        f"QuestionBank{question_type.upper()}{question_count}",
        __config__=ConfigDict(extra="forbid"),
        questions=(exact_questions, ...),
    )
