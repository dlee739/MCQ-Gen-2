from __future__ import annotations

from mcqgen2.config import (
    MODE_CONFIGS,
    QUESTION_TYPE_LABELS,
    GenerationMode,
    QuestionType,
)


# Runtime prompt wording lives here so it can be reviewed and edited without
# touching API request, validation, storage, or randomization logic.
SYSTEM_INSTRUCTIONS = """
You are an expert assessment writer. The application rules are authoritative. Follow the structured output contract strictly.
""".strip()


SHARED_RULES_TEMPLATE = """
Create exactly {question_count} {question_type_label} questions based on the provided material.

Requirements:
- Write every stem, choice, and rationale as standalone assessment content. Never mention or allude to a source, PDF, slides, document, notes, or external material, and never use meta-phrases such as "the source states," "the source lists," or "the source describes."
- Each stem must be clear and self-contained.
- Provide exactly four distinct choices, divided between correct_choices and incorrect_choices.
- Do not label choices with letters or numbers. The application shuffles and labels them after generation.
- For every choice, include a concise rationale in the same object that states the underlying factual reason why it belongs in its bucket.
- Do not use "all of the above".
- Avoid duplicate questions and repeated testing of the same fact when broader coverage is possible.
""".strip()


QUESTION_TYPE_RULES: dict[QuestionType, str] = {
    "mcq": """
MCQ rules:
- Return exactly one object in correct_choices and exactly three objects in incorrect_choices.
- Do not use "none of the above".
- Distractors must be plausible but demonstrably incorrect.
""".strip(),
    "sata": """
SATA rules:
- Return zero through four objects in correct_choices and put every remaining object in incorrect_choices, for exactly four objects total.
- Return an empty correct_choices list when all four supplied choices are incorrect.
- The application will append a fifth choice, "None of the above", and mark it correct only when correct_choices is empty. Do not supply that choice yourself.
- Correctness is based on selecting the exact complete set.
- Evaluate each choice independently; do not use combined choices such as "A and B".
- Do not reveal how many choices are correct in the stem.
""".strip(),
}


MODE_RULES: dict[GenerationMode, str] = {
    "high_volume": """
High-Volume mode:
- Favor reliable routine-practice questions.
- Use a practical mix of direct recall and straightforward application.
- Keep stems and explanations concise so a larger set remains useful and readable.
""".strip(),
    "high_quality": """
High-Quality mode:
- Require the learner to combine at least two relevant facts to reach the answer.
- Use realistic scenario details only when they materially affect the decision.
- Do not state the diagnosis, mechanism, or category when identifying it is part of the task.
- Keep distractors in the same decision space as the correct answer and make each one plausible.
- Avoid reusing stock distractors or the same incorrect intervention across the set.
- Before returning the final structured response, silently check that each answer set is unambiguous,
  no choice logically entails another choice, and every fact is accurate.
""".strip(),
}


def build_generation_prompt(
    *,
    question_count: int,
    mode: GenerationMode,
    question_type: QuestionType,
    instruction_rules: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Assemble the editable prompt blocks for one generation request."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if question_type not in QUESTION_TYPE_LABELS:
        raise ValueError(f"Unsupported question type: {question_type}")
    if custom_instructions is not None:
        if instruction_rules:
            raise ValueError("Pass instruction_rules, not both instruction arguments.")
        instruction_rules = custom_instructions

    shared = SHARED_RULES_TEMPLATE.format(
        question_count=question_count,
        question_type_label=QUESTION_TYPE_LABELS[question_type],
    )
    return "\n\n".join(
        (
            shared,
            QUESTION_TYPE_RULES[question_type],
            instruction_rules or MODE_RULES[mode],
        )
    )
