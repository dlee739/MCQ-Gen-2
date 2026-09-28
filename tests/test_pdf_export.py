from __future__ import annotations

import pymupdf

from mcqgen2.explanations import choice_rationales
from mcqgen2.pdf_export import build_results_pdf


def test_results_pdf_is_created() -> None:
    questions = [
        {
            "id": "q1",
            "stem": "Which answer is correct?",
            "choices": [
                {"id": "A", "text": "Answer A"},
                {"id": "B", "text": "Answer B"},
                {"id": "C", "text": "Answer C"},
                {"id": "D", "text": "Answer D"},
            ],
            "correct_choice_ids": ["A", "C"],
            "explanation": "Answer A is supported.",
        }
    ]
    result = build_results_pdf(
        title="Test results",
        questions=questions,
        answers={"q1": ["A", "C"]},
        score=1,
        total=1,
        metadata="gpt-6-luna",
    )
    assert result.startswith(b"%PDF-")
    assert len(result) > 1_000


def test_results_pdf_places_each_rationale_below_its_choice() -> None:
    question = {
        "id": "q1",
        "stem": "Which answer is correct?",
        "choices": [
            {"id": "C", "text": "Third displayed choice"},
            {"id": "A", "text": "First generated choice"},
            {"id": "B", "text": "Second generated choice"},
        ],
        "correct_choice_ids": ["A"],
        "explanation": (
            "First generated choice: Rationale for first. "
            "Second generated choice: Rationale for second. "
            "Third displayed choice: Rationale for third."
        ),
    }

    assert choice_rationales(question) == {
        "A": "Rationale for first.",
        "B": "Rationale for second.",
        "C": "Rationale for third.",
    }

    result = build_results_pdf(
        title="Test results",
        questions=[question],
        answers={"q1": ["A"]},
        score=1,
        total=1,
    )
    with pymupdf.open(stream=result, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)

    assert text.count("Why:") == 3
    assert text.index("A. Third displayed choice") < text.index("Rationale for third.")
    assert text.index("Rationale for third.") < text.index("B. First generated choice")
    assert text.index("B. First generated choice") < text.index("Rationale for first.")
    assert text.index("Rationale for first.") < text.index("C. Second generated choice")
    assert text.index("C. Second generated choice") < text.index("Rationale for second.")


def test_results_pdf_keeps_legacy_explanation_as_a_readable_fallback() -> None:
    question = {
        "id": "q1",
        "stem": "Which answer is correct?",
        "choices": [
            {"id": "A", "text": "Answer A"},
            {"id": "B", "text": "Answer B"},
        ],
        "correct_choice_ids": ["A"],
        "explanation": "A legacy free-form explanation that cannot be split safely.",
    }
    assert choice_rationales(question) is None

    result = build_results_pdf(
        title="Test results",
        questions=[question],
        answers={"q1": ["A"]},
        score=1,
        total=1,
    )
    with pymupdf.open(stream=result, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)

    assert "Explanation:" in text
    assert "A legacy free-form explanation" in text


def test_structured_choice_rationales_take_precedence_over_legacy_text() -> None:
    question = {
        "choices": [
            {"id": "A", "text": "Alpha", "rationale": "Structured alpha."},
            {"id": "B", "text": "Bravo", "rationale": "Structured bravo."},
        ],
        "explanation": "An older explanation that cannot be split.",
    }

    assert choice_rationales(question) == {
        "A": "Structured alpha.",
        "B": "Structured bravo.",
    }


def test_results_pdf_marks_flagged_and_skipped_questions() -> None:
    question = {
        "id": "q1",
        "stem": "Which answer is correct?",
        "choices": [
            {"id": "A", "text": "Alpha", "rationale": "Alpha is correct."},
            {"id": "B", "text": "Bravo", "rationale": "Bravo is incorrect."},
            {"id": "C", "text": "Charlie", "rationale": "Charlie is incorrect."},
            {"id": "D", "text": "Delta", "rationale": "Delta is incorrect."},
        ],
        "correct_choice_ids": ["A"],
        "explanation": "Alpha is correct.",
    }
    result = build_results_pdf(
        title="Test results",
        questions=[question],
        answers={"q1": []},
        score=0,
        total=1,
        flagged_question_ids={"q1"},
        skipped_question_ids={"q1"},
    )
    with pymupdf.open(stream=result, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)

    assert "Flagged: 1" in text
    assert "Skipped: 1" in text
    assert "Question 1 (Incorrect; Flagged; Skipped)" in text


def test_results_pdf_uses_answer_and_choice_labels_for_sata_review() -> None:
    question = {
        "id": "q1",
        "stem": "Which options apply?",
        "choices": [
            {"id": "A", "text": "Missed answer", "rationale": "A rationale."},
            {"id": "B", "text": "Selected answer", "rationale": "B rationale."},
            {"id": "C", "text": "Unselected distractor", "rationale": "C rationale."},
            {"id": "D", "text": "Selected distractor", "rationale": "D rationale."},
            {"id": "E", "text": "None of the above", "rationale": "E rationale."},
        ],
        "correct_choice_ids": ["A", "B"],
        "explanation": "Legacy explanation.",
    }
    result = build_results_pdf(
        title="Test results",
        questions=[question],
        answers={"q1": ["B", "D"]},
        score=0,
        total=1,
    )
    with pymupdf.open(stream=result, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)

    assert text.count("Answer") == 2
    assert text.count("Your choice") == 2
    assert "A. Missed answer Answer" in text
    assert "B. Selected answer Answer Your choice" in text
    assert "C. Unselected distractor" in text
    assert "D. Selected distractor Your choice" in text
    assert "E. None of the above" in text
