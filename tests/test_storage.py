from __future__ import annotations

from mcqgen2.generation import GenerationResult
from mcqgen2.pricing import CostBreakdown, TokenUsage, pricing_snapshot
from mcqgen2.schemas import GeneratedQuestion
from mcqgen2.storage import Database


def generated_question() -> GeneratedQuestion:
    return GeneratedQuestion.model_validate(
        {
            "stem": "What is the best answer?",
            "choices": [
                {"id": "A", "text": "Correct answer"},
                {"id": "B", "text": "Distractor one"},
                {"id": "C", "text": "Distractor two"},
                {"id": "D", "text": "Distractor three"},
            ],
            "correct_choice_id": "A",
            "explanation": "Correct answer is supported by the source.",
        }
    )


def generation_result() -> GenerationResult:
    return GenerationResult(
        questions=[generated_question()],
        usage=TokenUsage(100, 0, 0, 50, 10, 150),
        cost=CostBreakdown("short", 0.1, 0.0, 0.0, 0.2, 0.3),
        pricing=pricing_snapshot("gpt-6-luna"),
    )


def test_save_quiz_and_retry_lifecycle(tmp_path) -> None:
    db = Database(tmp_path / "app.sqlite3")
    db.initialize()
    set_id = db.save_question_set(
        source_filename="lecture.pdf",
        source_sha256="abc123",
        mode="high_volume",
        model="gpt-6-luna",
        requested_count=1,
        result=generation_result(),
    )

    saved = db.get_question_set(set_id)
    assert saved is not None
    assert saved["usage"]["input_tokens"] == 100
    assert len(saved["questions"]) == 1
    question_id = saved["questions"][0]["id"]

    first = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: "B"},
        kind="full",
        question_set_id=set_id,
    )
    assert first["score"] == 0
    assert [item["id"] for item in db.get_retry_questions()] == [question_id]

    second = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: "A"},
        kind="retry",
        question_set_id=None,
    )
    assert second["score"] == 1
    assert db.get_retry_questions() == []
