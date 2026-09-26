from __future__ import annotations

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
            "correct_choice_id": "A",
            "explanation": "Answer A is supported.",
        }
    ]
    result = build_results_pdf(
        title="Test results",
        questions=questions,
        answers={"q1": "A"},
        score=1,
        total=1,
        metadata="gpt-6-luna",
    )
    assert result.startswith(b"%PDF-")
    assert len(result) > 1_000
