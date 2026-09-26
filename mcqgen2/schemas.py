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
    correct_choice_id: ChoiceId
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
        if self.correct_choice_id not in ids:
            raise ValueError("correct_choice_id must reference a choice")
        return self


def question_bank_model(question_count: int) -> type[BaseModel]:
    exact_questions = Annotated[
        list[GeneratedQuestion],
        Field(min_length=question_count, max_length=question_count),
    ]
    return create_model(
        f"QuestionBank{question_count}",
        __config__=ConfigDict(extra="forbid"),
        questions=(exact_questions, ...),
    )
