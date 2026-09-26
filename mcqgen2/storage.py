from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcqgen2.config import PROMPT_VERSION
from mcqgen2.generation import GenerationResult

SCHEMA_VERSION = 2


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

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _create_v2_schema(conn: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS question_sets (
                id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                model TEXT NOT NULL,
                question_type TEXT NOT NULL CHECK(question_type IN ('mcq', 'sata')),
                custom_instructions TEXT NOT NULL DEFAULT '',
                sata_correct_counts_json TEXT NOT NULL DEFAULT '[]',
                requested_count INTEGER NOT NULL,
                prompt_version TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                pricing_json TEXT NOT NULL,
                cost_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                question_set_id TEXT NOT NULL REFERENCES question_sets(id),
                ordinal INTEGER NOT NULL,
                stem TEXT NOT NULL,
                choices_json TEXT NOT NULL,
                correct_choice_ids_json TEXT NOT NULL,
                explanation TEXT NOT NULL,
                UNIQUE(question_set_id, ordinal)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id TEXT PRIMARY KEY,
                question_set_id TEXT REFERENCES question_sets(id),
                kind TEXT NOT NULL CHECK(kind IN ('full', 'retry')),
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                question_order_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES quiz_sessions(id),
                question_id TEXT NOT NULL REFERENCES questions(id),
                selected_choice_ids_json TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_questions_set
                ON questions(question_set_id, ordinal)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_attempts_question
                ON attempts(question_id, id DESC)
            """,
        )
        for statement in statements:
            conn.execute(statement)

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            question_set_columns = self._table_columns(conn, "question_sets")
            if "question_type" not in question_set_columns:
                conn.execute(
                    "ALTER TABLE question_sets ADD COLUMN "
                    "question_type TEXT NOT NULL DEFAULT 'mcq' "
                    "CHECK(question_type IN ('mcq', 'sata'))"
                )
            if "custom_instructions" not in question_set_columns:
                conn.execute(
                    "ALTER TABLE question_sets ADD COLUMN "
                    "custom_instructions TEXT NOT NULL DEFAULT ''"
                )
            if "sata_correct_counts_json" not in question_set_columns:
                conn.execute(
                    "ALTER TABLE question_sets ADD COLUMN "
                    "sata_correct_counts_json TEXT NOT NULL DEFAULT '[]'"
                )

            old_questions = conn.execute("SELECT * FROM questions").fetchall()
            old_attempts = conn.execute("SELECT * FROM attempts").fetchall()
            conn.execute("DROP INDEX IF EXISTS idx_questions_set")
            conn.execute("DROP INDEX IF EXISTS idx_attempts_question")
            conn.execute("ALTER TABLE questions RENAME TO questions_v1")
            conn.execute("ALTER TABLE attempts RENAME TO attempts_v1")
            self._create_v2_schema(conn)

            for row in old_questions:
                conn.execute(
                    """
                    INSERT INTO questions (
                        id, question_set_id, ordinal, stem, choices_json,
                        correct_choice_ids_json, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["question_set_id"],
                        row["ordinal"],
                        row["stem"],
                        row["choices_json"],
                        json.dumps([row["correct_choice_id"]]),
                        row["explanation"],
                    ),
                )
            for row in old_attempts:
                conn.execute(
                    """
                    INSERT INTO attempts (
                        id, session_id, question_id, selected_choice_ids_json,
                        is_correct, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["session_id"],
                        row["question_id"],
                        json.dumps([row["selected_choice_id"]]),
                        row["is_correct"],
                        row["created_at"],
                    ),
                )
            conn.execute("DROP TABLE attempts_v1")
            conn.execute("DROP TABLE questions_v1")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("Database migration produced foreign-key violations.")

    def initialize(self) -> None:
        conn = self._connect()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'question_sets'"
            ).fetchone()
            if exists and "correct_choice_id" in self._table_columns(conn, "questions"):
                self._migrate_v1_to_v2(conn)
            else:
                with conn:
                    self._create_v2_schema(conn)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        finally:
            conn.close()

    def save_question_set(
        self,
        *,
        source_filename: str,
        source_sha256: str,
        mode: str,
        model: str,
        question_type: str,
        custom_instructions: str,
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
                    question_type, custom_instructions, sata_correct_counts_json,
                    requested_count, prompt_version, usage_json, pricing_json, cost_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    source_filename,
                    source_sha256,
                    created_at,
                    mode,
                    model,
                    question_type,
                    custom_instructions,
                    json.dumps(result.sata_correct_counts),
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
                        correct_choice_ids_json, explanation
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
                        json.dumps(question.correct_choice_ids),
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
        item["sata_correct_counts"] = json.loads(
            item.pop("sata_correct_counts_json")
        )
        return item

    @staticmethod
    def _question_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["choices"] = json.loads(item.pop("choices_json"))
        item["correct_choice_ids"] = json.loads(
            item.pop("correct_choice_ids_json")
        )
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
                """
                SELECT q.*, qs.question_type
                FROM questions q
                JOIN question_sets qs ON qs.id = q.question_set_id
                WHERE q.question_set_id = ?
                ORDER BY q.ordinal
                """,
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
                    qs.mode,
                    qs.question_type
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
        answers: Mapping[str, Sequence[str]],
        kind: str,
        question_set_id: str | None,
    ) -> dict[str, Any]:
        if kind not in {"full", "retry"}:
            raise ValueError("kind must be 'full' or 'retry'")
        if not question_ids or set(question_ids) != set(answers):
            raise ValueError("Every quiz question must have an answer.")

        placeholders = ",".join("?" for _ in question_ids)
        session_id = uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    q.id,
                    q.correct_choice_ids_json,
                    q.choices_json,
                    qs.question_type
                FROM questions q
                JOIN question_sets qs ON qs.id = q.question_set_id
                WHERE q.id IN ({placeholders})
                """,
                tuple(question_ids),
            ).fetchall()
            correct_by_id = {
                row["id"]: json.loads(row["correct_choice_ids_json"])
                for row in rows
            }
            if set(correct_by_id) != set(question_ids):
                raise ValueError("The quiz contains an unknown question.")
            valid_choices = {
                row["id"]: {choice["id"] for choice in json.loads(row["choices_json"])}
                for row in rows
            }
            question_types = {row["id"]: row["question_type"] for row in rows}

            normalized_answers: dict[str, list[str]] = {}
            for question_id in question_ids:
                selected = answers[question_id]
                if isinstance(selected, str):
                    raise ValueError("Quiz answers must be choice-ID sequences.")
                selected_ids = list(selected)
                if not selected_ids or len(selected_ids) != len(set(selected_ids)):
                    raise ValueError("Each quiz answer must contain unique choices.")
                if any(
                    choice_id not in valid_choices[question_id]
                    for choice_id in selected_ids
                ):
                    raise ValueError("A quiz answer does not reference a valid choice.")
                if question_types[question_id] == "mcq" and len(selected_ids) != 1:
                    raise ValueError("MCQ answers must contain exactly one choice.")
                normalized_answers[question_id] = selected_ids

            score = sum(
                set(normalized_answers[question_id])
                == set(correct_by_id[question_id])
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
                selected_ids = normalized_answers[question_id]
                is_correct = set(selected_ids) == set(correct_by_id[question_id])
                conn.execute(
                    """
                    INSERT INTO attempts (
                        session_id, question_id, selected_choice_ids_json,
                        is_correct, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        question_id,
                        json.dumps(selected_ids),
                        int(is_correct),
                        timestamp,
                    ),
                )
        return {"session_id": session_id, "score": score, "total": len(question_ids)}
