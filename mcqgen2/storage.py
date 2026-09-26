from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcqgen2.config import PROMPT_VERSION
from mcqgen2.generation import GenerationResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS question_sets (
                    id TEXT PRIMARY KEY,
                    source_filename TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requested_count INTEGER NOT NULL,
                    prompt_version TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    pricing_json TEXT NOT NULL,
                    cost_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    question_set_id TEXT NOT NULL REFERENCES question_sets(id),
                    ordinal INTEGER NOT NULL,
                    stem TEXT NOT NULL,
                    choices_json TEXT NOT NULL,
                    correct_choice_id TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    UNIQUE(question_set_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    id TEXT PRIMARY KEY,
                    question_set_id TEXT REFERENCES question_sets(id),
                    kind TEXT NOT NULL CHECK(kind IN ('full', 'retry')),
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    question_order_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES quiz_sessions(id),
                    question_id TEXT NOT NULL REFERENCES questions(id),
                    selected_choice_id TEXT NOT NULL,
                    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_questions_set
                    ON questions(question_set_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_attempts_question
                    ON attempts(question_id, id DESC);
                """
            )

    def save_question_set(
        self,
        *,
        source_filename: str,
        source_sha256: str,
        mode: str,
        model: str,
        requested_count: int,
        result: GenerationResult,
    ) -> str:
        set_id = uuid.uuid4().hex
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO question_sets (
                    id, source_filename, source_sha256, created_at, mode, model,
                    requested_count, prompt_version, usage_json, pricing_json, cost_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    source_filename,
                    source_sha256,
                    created_at,
                    mode,
                    model,
                    requested_count,
                    PROMPT_VERSION,
                    json.dumps(result.usage.as_dict()),
                    json.dumps(result.pricing),
                    json.dumps(result.cost.as_dict()),
                ),
            )
            for ordinal, question in enumerate(result.questions, start=1):
                question_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO questions (
                        id, question_set_id, ordinal, stem, choices_json,
                        correct_choice_id, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question_id,
                        set_id,
                        ordinal,
                        question.stem,
                        json.dumps(
                            [choice.model_dump() for choice in question.choices],
                            ensure_ascii=False,
                        ),
                        question.correct_choice_id,
                        question.explanation,
                    ),
                )
        return set_id

    @staticmethod
    def _set_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["usage"] = json.loads(item.pop("usage_json"))
        item["pricing"] = json.loads(item.pop("pricing_json"))
        item["cost"] = json.loads(item.pop("cost_json"))
        return item

    @staticmethod
    def _question_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["choices"] = json.loads(item.pop("choices_json"))
        return item

    def list_question_sets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM question_sets ORDER BY created_at DESC"
            ).fetchall()
        return [self._set_row(row) for row in rows]

    def get_question_set(self, set_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM question_sets WHERE id = ?", (set_id,)
            ).fetchone()
            if row is None:
                return None
            question_rows = conn.execute(
                "SELECT * FROM questions WHERE question_set_id = ? ORDER BY ordinal",
                (set_id,),
            ).fetchall()
        result = self._set_row(row)
        result["questions"] = [self._question_row(q) for q in question_rows]
        return result

    def get_retry_questions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH latest AS (
                    SELECT
                        question_id,
                        is_correct,
                        ROW_NUMBER() OVER (PARTITION BY question_id ORDER BY id DESC) AS rn
                    FROM attempts
                )
                SELECT
                    q.*,
                    qs.source_filename,
                    qs.model,
                    qs.mode
                FROM latest l
                JOIN questions q ON q.id = l.question_id
                JOIN question_sets qs ON qs.id = q.question_set_id
                WHERE l.rn = 1 AND l.is_correct = 0
                ORDER BY qs.created_at DESC, q.ordinal
                """
            ).fetchall()
        return [self._question_row(row) for row in rows]

    def record_quiz(
        self,
        *,
        question_ids: Sequence[str],
        answers: Mapping[str, str],
        kind: str,
        question_set_id: str | None,
    ) -> dict[str, Any]:
        if kind not in {"full", "retry"}:
            raise ValueError("kind must be 'full' or 'retry'")
        if not question_ids or set(question_ids) != set(answers):
            raise ValueError("Every quiz question must have exactly one answer.")

        placeholders = ",".join("?" for _ in question_ids)
        session_id = uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, correct_choice_id, choices_json FROM questions WHERE id IN ({placeholders})",
                tuple(question_ids),
            ).fetchall()
            correct_by_id = {row["id"]: row["correct_choice_id"] for row in rows}
            if set(correct_by_id) != set(question_ids):
                raise ValueError("The quiz contains an unknown question.")
            valid_choices = {
                row["id"]: {choice["id"] for choice in json.loads(row["choices_json"])}
                for row in rows
            }
            if any(
                answers[question_id] not in valid_choices[question_id]
                for question_id in question_ids
            ):
                raise ValueError("A quiz answer does not reference a valid choice.")

            score = sum(
                answers[question_id] == correct_by_id[question_id]
                for question_id in question_ids
            )
            conn.execute(
                """
                INSERT INTO quiz_sessions (
                    id, question_set_id, kind, started_at, completed_at,
                    score, total, question_order_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question_set_id,
                    kind,
                    timestamp,
                    timestamp,
                    score,
                    len(question_ids),
                    json.dumps(list(question_ids)),
                ),
            )
            for question_id in question_ids:
                selected = answers[question_id]
                conn.execute(
                    """
                    INSERT INTO attempts (
                        session_id, question_id, selected_choice_id, is_correct, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        question_id,
                        selected,
                        int(selected == correct_by_id[question_id]),
                        timestamp,
                    ),
                )
        return {"session_id": session_id, "score": score, "total": len(question_ids)}
