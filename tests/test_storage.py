from __future__ import annotations

import json
import sqlite3

import pytest

from mcqgen2.generation import GenerationResult
from mcqgen2.pricing import CostBreakdown, TokenUsage, pricing_snapshot
from mcqgen2.schemas import GeneratedQuestion
from mcqgen2.storage import Database


def generated_question(correct_choice_ids: list[str] | None = None) -> GeneratedQuestion:
    return GeneratedQuestion.model_validate(
        {
            "stem": "What is the best answer?",
            "choices": [
                {"id": "A", "text": "Correct answer"},
                {"id": "B", "text": "Distractor one"},
                {"id": "C", "text": "Distractor two"},
                {"id": "D", "text": "Distractor three"},
            ],
            "correct_choice_ids": correct_choice_ids or ["A"],
            "explanation": "Correct answer is supported by the source.",
        }
    )


def generation_result(
    correct_choice_ids: list[str] | None = None,
    sata_correct_counts: list[int] | None = None,
) -> GenerationResult:
    return GenerationResult(
        questions=[generated_question(correct_choice_ids)],
        usage=TokenUsage(100, 0, 0, 50, 10, 150),
        cost=CostBreakdown("short", 0.1, 0.0, 0.0, 0.2, 0.3),
        pricing=pricing_snapshot("gpt-6-luna"),
        sata_correct_counts=sata_correct_counts or [],
    )


def test_save_quiz_and_retry_lifecycle(tmp_path) -> None:
    db = Database(tmp_path / "app.sqlite3")
    db.initialize()
    set_id = db.save_question_set(
        source_filename="lecture.pdf",
        source_sha256="abc123",
        mode="high_volume",
        model="gpt-6-luna",
        input_mode="extracted_text",
        question_type="mcq",
        custom_instructions="Use real-world scenarios.",
        requested_count=1,
        result=generation_result(),
    )

    saved = db.get_question_set(set_id)
    assert saved is not None
    assert saved["usage"]["input_tokens"] == 100
    assert saved["question_type"] == "mcq"
    assert saved["input_mode"] == "extracted_text"
    assert saved["custom_instructions"] == "Use real-world scenarios."
    assert len(saved["questions"]) == 1
    question_id = saved["questions"][0]["id"]

    first = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: ["B"]},
        kind="full",
        question_set_id=set_id,
    )
    assert first["score"] == 0
    assert [item["id"] for item in db.get_retry_questions()] == [question_id]

    second = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: ["A"]},
        kind="retry",
        question_set_id=None,
    )
    assert second["score"] == 1
    assert db.get_retry_questions() == []


def test_sata_uses_exact_set_grading(tmp_path) -> None:
    db = Database(tmp_path / "sata.sqlite3")
    db.initialize()
    set_id = db.save_question_set(
        source_filename="lecture.pdf",
        source_sha256="sata123",
        mode="high_quality",
        model="gpt-6-sol",
        input_mode="pdf",
        question_type="sata",
        custom_instructions="Use application scenarios.",
        requested_count=1,
        result=generation_result(["A", "C"], [2]),
    )
    saved = db.get_question_set(set_id)
    assert saved is not None
    assert saved["input_mode"] == "pdf"
    question_id = saved["questions"][0]["id"]

    partial = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: ["A"]},
        kind="full",
        question_set_id=set_id,
    )
    assert partial["score"] == 0

    exact = db.record_quiz(
        question_ids=[question_id],
        answers={question_id: ["C", "A"]},
        kind="retry",
        question_set_id=None,
    )
    assert exact["score"] == 1
    assert db.get_retry_questions() == []


def test_v1_database_is_migrated_without_losing_quiz_data(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE question_sets (
            id TEXT PRIMARY KEY, source_filename TEXT NOT NULL,
            source_sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
            mode TEXT NOT NULL, model TEXT NOT NULL, requested_count INTEGER NOT NULL,
            prompt_version TEXT NOT NULL, usage_json TEXT NOT NULL,
            pricing_json TEXT NOT NULL, cost_json TEXT NOT NULL
        );
        CREATE TABLE questions (
            id TEXT PRIMARY KEY, question_set_id TEXT NOT NULL REFERENCES question_sets(id),
            ordinal INTEGER NOT NULL, stem TEXT NOT NULL, choices_json TEXT NOT NULL,
            correct_choice_id TEXT NOT NULL, explanation TEXT NOT NULL,
            UNIQUE(question_set_id, ordinal)
        );
        CREATE TABLE quiz_sessions (
            id TEXT PRIMARY KEY, question_set_id TEXT REFERENCES question_sets(id),
            kind TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL,
            score INTEGER NOT NULL, total INTEGER NOT NULL,
            question_order_json TEXT NOT NULL
        );
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES quiz_sessions(id),
            question_id TEXT NOT NULL REFERENCES questions(id),
            selected_choice_id TEXT NOT NULL, is_correct INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    choices = [
        {"id": "A", "text": "Answer A"},
        {"id": "B", "text": "Answer B"},
        {"id": "C", "text": "Answer C"},
        {"id": "D", "text": "Answer D"},
    ]
    conn.execute(
        "INSERT INTO question_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "set1", "legacy.pdf", "hash", "2026-01-01T00:00:00+00:00",
            "high_volume", "gpt-6-luna", 1, "mcq-v1", "{}", "{}", "{}",
        ),
    )
    conn.execute(
        "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("q1", "set1", 1, "Legacy question?", json.dumps(choices), "A", "Because."),
    )
    conn.execute(
        "INSERT INTO quiz_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "session1", "set1", "full", "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00", 0, 1, '["q1"]',
        ),
    )
    conn.execute(
        "INSERT INTO attempts (session_id, question_id, selected_choice_id, is_correct, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("session1", "q1", "B", 0, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize()
    saved = db.get_question_set("set1")
    assert saved is not None
    assert saved["question_type"] == "mcq"
    assert saved["input_mode"] == "pdf"
    assert saved["questions"][0]["correct_choice_ids"] == ["A"]
    assert [item["id"] for item in db.get_retry_questions()] == ["q1"]

    verify = sqlite3.connect(path)
    assert verify.execute("PRAGMA user_version").fetchone()[0] == 4
    assert verify.execute("PRAGMA foreign_key_check").fetchall() == []
    verify.close()


def test_v2_database_adds_pdf_input_mode_without_losing_data(tmp_path) -> None:
    path = tmp_path / "v2.sqlite3"
    db = Database(path)
    db.initialize()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE question_sets DROP COLUMN input_mode")
        conn.execute("PRAGMA user_version = 2")

    db.initialize()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(question_sets)")}
        assert "input_mode" in columns
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_instruction_profiles_are_mode_specific_and_keep_one_default(tmp_path) -> None:
    db = Database(tmp_path / "profiles.sqlite3")
    db.initialize()

    volume_default = db.get_default_instruction_profile("high_volume")
    quality_default = db.get_default_instruction_profile("high_quality")
    assert volume_default["name"] == "Default"
    assert quality_default["name"] == "Default"
    assert volume_default["instructions"] != quality_default["instructions"]

    new_id = db.create_instruction_profile(
        mode="high_volume", name="Course style", instructions="Use concise scenarios."
    )
    db.set_default_instruction_profile(new_id)
    db.update_instruction_profile(
        new_id, name="Course style", instructions="Use applied scenarios."
    )
    assert db.get_default_instruction_profile("high_volume")["id"] == new_id
    assert db.get_default_instruction_profile("high_quality")["id"] == quality_default["id"]

    with pytest.raises(ValueError, match="default"):
        db.delete_instruction_profile(new_id)
    db.delete_instruction_profile(volume_default["id"])
    assert [item["name"] for item in db.list_instruction_profiles("high_volume")] == [
        "Course style"
    ]


def test_bookmarks_practice_and_bulk_cleanup_preserve_profiles(tmp_path) -> None:
    db = Database(tmp_path / "bookmarks.sqlite3")
    db.initialize()
    profile_count = sum(
        len(db.list_instruction_profiles(mode))
        for mode in ("high_volume", "high_quality")
    )
    set_id = db.save_question_set(
        source_filename="lecture.pdf",
        source_sha256="abc123",
        mode="high_volume",
        model="gpt-6-luna",
        input_mode="extracted_text",
        question_type="mcq",
        instruction_profile_name="Default",
        instruction_rules="Use routine questions.",
        requested_count=1,
        result=generation_result(),
    )
    saved = db.get_question_set(set_id)
    assert saved is not None
    question_id = saved["questions"][0]["id"]

    db.set_bookmark(question_id, True)
    assert db.is_bookmarked(question_id)
    assert [item["id"] for item in db.get_bookmarked_questions()] == [question_id]
    assert db.record_quiz(
        question_ids=[question_id],
        answers={question_id: ["A"]},
        kind="bookmarked",
        question_set_id=None,
    )["score"] == 1

    counts = db.delete_all_runs()
    assert counts["question_sets"] == 1
    assert counts["bookmarks"] == 1
    assert db.list_question_sets() == []
    assert db.get_bookmarked_questions() == []
    assert sum(
        len(db.list_instruction_profiles(mode))
        for mode in ("high_volume", "high_quality")
    ) == profile_count


def test_v3_sata_questions_gain_fixed_none_choice(tmp_path) -> None:
    db = Database(tmp_path / "v3.sqlite3")
    db.initialize()
    set_id = db.save_question_set(
        source_filename="lecture.pdf",
        source_sha256="sata123",
        mode="high_volume",
        model="gpt-6-luna",
        input_mode="extracted_text",
        question_type="sata",
        requested_count=1,
        result=generation_result(["A"], [1]),
    )
    with sqlite3.connect(db.path) as conn:
        conn.execute("PRAGMA user_version = 3")

    db.initialize()
    saved = db.get_question_set(set_id)
    assert saved is not None
    assert saved["questions"][0]["choices"][-1] == {
        "id": "E",
        "text": "None of the above",
    }
    assert saved["questions"][0]["correct_choice_ids"] == ["A"]
