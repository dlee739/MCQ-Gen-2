from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


def test_generate_page_exposes_type_and_instruction_controls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not at.exception
    assert at.segmented_control[0].label == "Question type"
    assert at.segmented_control[0].value == "mcq"
    assert at.text_area[0].label == "Question-writing instructions (optional)"
    visible_copy = " ".join(
        element.value
        for collection in (at.title, at.header, at.caption, at.markdown)
        for element in collection
        if isinstance(element.value, str)
    ).casefold()
    domain_defaults = ("medi" "cal", "clini" "cal")
    assert all(term not in visible_copy for term in domain_defaults)


def test_sata_quiz_collects_multiple_answers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["quiz"] = {
        "nonce": 123,
        "questions": [
            {
                "id": "q1",
                "question_type": "sata",
                "stem": "Which statements are supported?",
                "choices": [
                    {"id": "A", "text": "Alpha"},
                    {"id": "B", "text": "Bravo"},
                    {"id": "C", "text": "Charlie"},
                    {"id": "D", "text": "Delta"},
                ],
                "correct_choice_ids": ["A", "C"],
                "explanation": "Alpha and Charlie are supported.",
            }
        ],
        "answers": {},
        "index": 0,
        "kind": "full",
        "question_set_id": None,
        "title": "SATA test",
        "metadata": "test",
    }
    at.run()
    assert not at.exception
    assert len(at.checkbox) == 4

    at.checkbox[0].check().run()
    at.checkbox[2].check().run()

    assert not at.exception
    assert set(at.session_state["quiz"]["answers"]["q1"]) == {"A", "C"}
