from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

ChoiceId = Literal["A", "B", "C", "D"]


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: ChoiceId = Field(description="Stable internal choice identifier.")
    text: str = Field(min_length=1, description="Answer choice text.")


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stem: str = Field(min_length=1, description="Self-contained medical question stem.")
    choices: list[Choice] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four distinct answer choices with IDs A through D.",
    )
    correct_choice_ids: list[ChoiceId] = Field(
        min_length=1,
        max_length=4,
        description="One or more stable IDs for the correct choices.",
    )
    explanation: str = Field(
        min_length=1,
        description="Concise justification of the answer and distractors using choice text, not letters.",
    )

    @model_validator(mode="after")
    def validate_choices(self) -> "GeneratedQuestion":
        ids = [choice.id for choice in self.choices]
        if set(ids) != {"A", "B", "C", "D"}:
            raise ValueError("choices must contain each ID A, B, C, and D exactly once")
        normalized = [choice.text.casefold() for choice in self.choices]
        if len(set(normalized)) != 4:
            raise ValueError("choice text must be unique")
        if len(set(self.correct_choice_ids)) != len(self.correct_choice_ids):
            raise ValueError("correct_choice_ids must be unique")
        if any(choice_id not in ids for choice_id in self.correct_choice_ids):
            raise ValueError("correct_choice_ids must reference existing choices")
        return self


class MCQGeneratedQuestion(GeneratedQuestion):
    correct_choice_ids: list[ChoiceId] = Field(
        min_length=1,
        max_length=1,
        description="The single stable ID for the correct choice.",
    )


def question_bank_model(question_count: int, question_type: str) -> type[BaseModel]:
    if question_type not in {"mcq", "sata"}:
        raise ValueError(f"Unsupported question type: {question_type}")
    question_model = MCQGeneratedQuestion if question_type == "mcq" else GeneratedQuestion
    exact_questions = Annotated[
        list[question_model],  # type: ignore[valid-type]
        Field(min_length=question_count, max_length=question_count),
    ]
    return create_model(
        f"QuestionBank{question_type.upper()}{question_count}",
        __config__=ConfigDict(extra="forbid"),
        questions=(exact_questions, ...),
    )
