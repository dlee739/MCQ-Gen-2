from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcqgen2.config import MAX_INSTRUCTION_RULES_CHARS, MODE_CONFIGS, PROMPT_VERSION
from mcqgen2.generation import GenerationResult
from mcqgen2.prompts import MODE_RULES

SCHEMA_VERSION = 4


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
    def _create_schema(conn: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS question_sets (
                id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                model TEXT NOT NULL,
                input_mode TEXT NOT NULL DEFAULT 'pdf'
                    CHECK(input_mode IN ('extracted_text', 'pdf')),
                question_type TEXT NOT NULL CHECK(question_type IN ('mcq', 'sata')),
                instruction_profile_name TEXT NOT NULL DEFAULT 'Legacy/custom',
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
                kind TEXT NOT NULL CHECK(kind IN ('full', 'retry', 'bookmarked')),
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
            CREATE TABLE IF NOT EXISTS bookmarks (
                question_id TEXT PRIMARY KEY REFERENCES questions(id),
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS instruction_profiles (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('high_volume', 'high_quality')),
                name TEXT NOT NULL COLLATE NOCASE,
                instructions TEXT NOT NULL,
                is_default INTEGER NOT NULL CHECK(is_default IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(mode, name)
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
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_instruction_profiles_default
                ON instruction_profiles(mode) WHERE is_default = 1
            """,
        )
        for statement in statements:
            conn.execute(statement)

    @staticmethod
    def _ensure_question_set_columns(conn: sqlite3.Connection) -> None:
        columns = Database._table_columns(conn, "question_sets")
        if "question_type" not in columns:
            conn.execute(
                "ALTER TABLE question_sets ADD COLUMN "
                "question_type TEXT NOT NULL DEFAULT 'mcq' "
                "CHECK(question_type IN ('mcq', 'sata'))"
            )
        if "custom_instructions" not in columns:
            conn.execute(
                "ALTER TABLE question_sets ADD COLUMN "
                "custom_instructions TEXT NOT NULL DEFAULT ''"
            )
        if "instruction_profile_name" not in columns:
            conn.execute(
                "ALTER TABLE question_sets ADD COLUMN "
                "instruction_profile_name TEXT NOT NULL DEFAULT 'Legacy/custom'"
            )
        if "sata_correct_counts_json" not in columns:
            conn.execute(
                "ALTER TABLE question_sets ADD COLUMN "
                "sata_correct_counts_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "input_mode" not in columns:
            conn.execute(
                "ALTER TABLE question_sets ADD COLUMN "
                "input_mode TEXT NOT NULL DEFAULT 'pdf' "
                "CHECK(input_mode IN ('extracted_text', 'pdf'))"
            )

    def _migrate_v1_to_current(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_question_set_columns(conn)

            old_questions = conn.execute("SELECT * FROM questions").fetchall()
            old_attempts = conn.execute("SELECT * FROM attempts").fetchall()
            conn.execute("DROP INDEX IF EXISTS idx_questions_set")
            conn.execute("DROP INDEX IF EXISTS idx_attempts_question")
            conn.execute("ALTER TABLE questions RENAME TO questions_v1")
            conn.execute("ALTER TABLE attempts RENAME TO attempts_v1")
            self._create_schema(conn)

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
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("Database migration produced foreign-key violations.")

    @staticmethod
    def _seed_instruction_profiles(conn: sqlite3.Connection) -> None:
        timestamp = _now()
        for mode, instructions in MODE_RULES.items():
            exists = conn.execute(
                "SELECT 1 FROM instruction_profiles WHERE mode = ? LIMIT 1", (mode,)
            ).fetchone()
            if exists is None:
                conn.execute(
                    """
                    INSERT INTO instruction_profiles (
                        id, mode, name, instructions, is_default, created_at, updated_at
                    ) VALUES (?, ?, 'Default', ?, 1, ?, ?)
                    """,
                    (uuid.uuid4().hex, mode, instructions, timestamp, timestamp),
                )

    @staticmethod
    def _upgrade_sata_choices(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT q.id, q.choices_json
            FROM questions q
            JOIN question_sets qs ON qs.id = q.question_set_id
            WHERE qs.question_type = 'sata'
            """
        ).fetchall()
        for row in rows:
            choices = json.loads(row["choices_json"])
            if not any(choice.get("id") == "E" for choice in choices):
                choices.append({"id": "E", "text": "None of the above"})
                conn.execute(
                    "UPDATE questions SET choices_json = ? WHERE id = ?",
                    (json.dumps(choices, ensure_ascii=False), row["id"]),
                )

    def _migrate_to_v4(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._create_schema(conn)
            self._ensure_question_set_columns(conn)
            session_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'quiz_sessions'"
            ).fetchone()
            session_sql = session_sql_row["sql"] if session_sql_row else ""
            if session_sql and "bookmarked" not in session_sql:
                conn.execute("DROP INDEX IF EXISTS idx_attempts_question")
                conn.execute("ALTER TABLE attempts RENAME TO attempts_v3")
                conn.execute("ALTER TABLE quiz_sessions RENAME TO quiz_sessions_v3")
                self._create_schema(conn)
                conn.execute(
                    """
                    INSERT INTO quiz_sessions
                    SELECT * FROM quiz_sessions_v3
                    """
                )
                conn.execute(
                    """
                    INSERT INTO attempts (
                        id, session_id, question_id, selected_choice_ids_json,
                        is_correct, created_at
                    )
                    SELECT id, session_id, question_id, selected_choice_ids_json,
                           is_correct, created_at
                    FROM attempts_v3
                    """
                )
                conn.execute("DROP TABLE attempts_v3")
                conn.execute("DROP TABLE quiz_sessions_v3")
            else:
                self._create_schema(conn)
            self._upgrade_sata_choices(conn)
            self._seed_instruction_profiles(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Database migration produced foreign-key violations.")

    def initialize(self) -> None:
        conn = self._connect()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'question_sets'"
            ).fetchone()
            if exists and "correct_choice_id" in self._table_columns(conn, "questions"):
                self._migrate_v1_to_current(conn)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < SCHEMA_VERSION:
                self._migrate_to_v4(conn)
            else:
                with conn:
                    self._create_schema(conn)
                    self._ensure_question_set_columns(conn)
                    self._seed_instruction_profiles(conn)
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
        input_mode: str,
        question_type: str,
        requested_count: int,
        result: GenerationResult,
        instruction_profile_name: str = "Legacy/custom",
        instruction_rules: str = "",
        custom_instructions: str | None = None,
    ) -> str:
        if custom_instructions is not None:
            if instruction_rules:
                raise ValueError("Pass instruction_rules, not both instruction arguments.")
            instruction_rules = custom_instructions
        set_id = uuid.uuid4().hex
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO question_sets (
                    id, source_filename, source_sha256, created_at, mode, model,
                    input_mode, question_type, instruction_profile_name,
                    custom_instructions,
                    sata_correct_counts_json, requested_count, prompt_version,
                    usage_json, pricing_json, cost_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    source_filename,
                    source_sha256,
                    created_at,
                    mode,
                    model,
                    input_mode,
                    question_type,
                    instruction_profile_name,
                    instruction_rules,
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
                            [
                                choice.model_dump(exclude_none=True)
                                for choice in question.choices
                            ],
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

    def list_instruction_profiles(self, mode: str) -> list[dict[str, Any]]:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"Unsupported generation mode: {mode}")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM instruction_profiles
                WHERE mode = ?
                ORDER BY is_default DESC, name COLLATE NOCASE
                """,
                (mode,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_default_instruction_profile(self, mode: str) -> dict[str, Any]:
        profiles = self.list_instruction_profiles(mode)
        for profile in profiles:
            if profile["is_default"]:
                return profile
        if not profiles:
            raise RuntimeError(f"No instruction profiles exist for {mode}.")
        return profiles[0]

    def create_instruction_profile(
        self, *, mode: str, name: str, instructions: str
    ) -> str:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"Unsupported generation mode: {mode}")
        clean_name = name.strip()
        clean_instructions = instructions.strip()
        if not clean_name:
            raise ValueError("Profile name is required.")
        if not clean_instructions:
            raise ValueError("Instruction rules cannot be empty.")
        if len(clean_instructions) > MAX_INSTRUCTION_RULES_CHARS:
            raise ValueError(
                f"Instruction rules must be {MAX_INSTRUCTION_RULES_CHARS:,} characters or fewer."
            )
        profile_id = uuid.uuid4().hex
        timestamp = _now()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO instruction_profiles (
                        id, mode, name, instructions, is_default, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (profile_id, mode, clean_name, clean_instructions, timestamp, timestamp),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("A profile with that name already exists for this mode.") from exc
        return profile_id

    def update_instruction_profile(
        self, profile_id: str, *, name: str, instructions: str
    ) -> None:
        clean_name = name.strip()
        clean_instructions = instructions.strip()
        if not clean_name or not clean_instructions:
            raise ValueError("Profile name and instruction rules are required.")
        if len(clean_instructions) > MAX_INSTRUCTION_RULES_CHARS:
            raise ValueError(
                f"Instruction rules must be {MAX_INSTRUCTION_RULES_CHARS:,} characters or fewer."
            )
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE instruction_profiles
                    SET name = ?, instructions = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (clean_name, clean_instructions, _now(), profile_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Instruction profile not found.")
        except sqlite3.IntegrityError as exc:
            raise ValueError("A profile with that name already exists for this mode.") from exc

    def set_default_instruction_profile(self, profile_id: str) -> None:
        with self._connect() as conn:
            profile = conn.execute(
                "SELECT mode FROM instruction_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if profile is None:
                raise ValueError("Instruction profile not found.")
            conn.execute(
                "UPDATE instruction_profiles SET is_default = 0 WHERE mode = ?",
                (profile["mode"],),
            )
            conn.execute(
                "UPDATE instruction_profiles SET is_default = 1, updated_at = ? WHERE id = ?",
                (_now(), profile_id),
            )

    def delete_instruction_profile(self, profile_id: str) -> None:
        with self._connect() as conn:
            profile = conn.execute(
                "SELECT mode, is_default FROM instruction_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if profile is None:
                raise ValueError("Instruction profile not found.")
            count = conn.execute(
                "SELECT COUNT(*) FROM instruction_profiles WHERE mode = ?",
                (profile["mode"],),
            ).fetchone()[0]
            if profile["is_default"]:
                raise ValueError("Set another profile as the default before deleting this one.")
            if count <= 1:
                raise ValueError("Each generation mode must keep at least one profile.")
            conn.execute("DELETE FROM instruction_profiles WHERE id = ?", (profile_id,))

    def is_bookmarked(self, question_id: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM bookmarks WHERE question_id = ?", (question_id,)
                ).fetchone()
                is not None
            )

    def set_bookmark(self, question_id: str, bookmarked: bool) -> None:
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
            if exists is None:
                raise ValueError("Question not found.")
            if bookmarked:
                conn.execute(
                    "INSERT OR IGNORE INTO bookmarks (question_id, created_at) VALUES (?, ?)",
                    (question_id, _now()),
                )
            else:
                conn.execute("DELETE FROM bookmarks WHERE question_id = ?", (question_id,))

    def get_bookmarked_questions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT q.*, qs.source_filename, qs.model, qs.mode, qs.question_type,
                       b.created_at AS bookmarked_at
                FROM bookmarks b
                JOIN questions q ON q.id = b.question_id
                JOIN question_sets qs ON qs.id = q.question_set_id
                ORDER BY b.created_at DESC, q.ordinal
                """
            ).fetchall()
        return [self._question_row(row) for row in rows]

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
        if kind not in {"full", "retry", "bookmarked"}:
            raise ValueError("kind must be 'full', 'retry', or 'bookmarked'")
        if not question_ids or set(question_ids) != set(answers):
            raise ValueError("Every quiz question must have an answer or explicit skip.")

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
                if len(selected_ids) != len(set(selected_ids)):
                    raise ValueError("Each quiz answer must contain unique choices.")
                if any(
                    choice_id not in valid_choices[question_id]
                    for choice_id in selected_ids
                ):
                    raise ValueError("A quiz answer does not reference a valid choice.")
                if (
                    selected_ids
                    and question_types[question_id] == "mcq"
                    and len(selected_ids) != 1
                ):
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

    def delete_all_runs(self) -> dict[str, int]:
        """Delete generated/run data while preserving instruction profiles."""
        with self._connect() as conn:
            counts = {
                "question_sets": conn.execute(
                    "SELECT COUNT(*) FROM question_sets"
                ).fetchone()[0],
                "questions": conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
                "quiz_sessions": conn.execute(
                    "SELECT COUNT(*) FROM quiz_sessions"
                ).fetchone()[0],
                "bookmarks": conn.execute("SELECT COUNT(*) FROM bookmarks").fetchone()[0],
            }
            conn.execute("DELETE FROM attempts")
            conn.execute("DELETE FROM quiz_sessions")
            conn.execute("DELETE FROM bookmarks")
            conn.execute("DELETE FROM questions")
            conn.execute("DELETE FROM question_sets")
        return counts
