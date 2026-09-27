from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


def test_generate_page_exposes_input_type_and_instruction_controls(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not at.exception
    assert at.segmented_control[0].label == "Input processing"
    assert at.segmented_control[0].value == "extracted_text"
    assert at.segmented_control[1].label == "Question type"
    assert at.segmented_control[1].value == "mcq"
    assert at.selectbox(key="generation_mode").value == "high_volume"
    assert at.selectbox(key="api_model").value == "gpt-6-luna"
    assert at.number_input(key="question_count").value == 10
    assert at.selectbox[2].label == "Instruction profile"
    assert at.text_area[0].label == "Instruction rules"
    assert any(element.value == "Local beta 0.1.0b1" for element in at.caption)
    visible_copy = " ".join(
        element.value
        for collection in (at.title, at.header, at.caption, at.markdown)
        for element in collection
        if isinstance(element.value, str)
    ).casefold()
    domain_defaults = ("medi" "cal", "clini" "cal")
    assert all(term not in visible_copy for term in domain_defaults)


def test_generation_mode_updates_defaults_and_preserves_manual_overrides(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()

    at.selectbox(key="generation_mode").select("high_quality").run()
    assert not at.exception
    assert at.selectbox(key="api_model").value == "gpt-6-sol"
    assert at.number_input(key="question_count").value == 5

    at.selectbox(key="api_model").select("gpt-5.6-terra").run()
    at.number_input(key="question_count").set_value(7).run()
    assert at.selectbox(key="api_model").value == "gpt-5.6-terra"
    assert at.number_input(key="question_count").value == 7

    at.selectbox(key="generation_mode").select("high_volume").run()
    assert at.selectbox(key="api_model").value == "gpt-6-luna"
    assert at.number_input(key="question_count").value == 10


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
                    {"id": "E", "text": "None of the above"},
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
    assert len(at.checkbox) == 5
    assert all(element.value != "Which statements are supported?" for element in at.subheader)
    assert any(
        element.value == "**Which statements are supported?**" for element in at.markdown
    )

    at.checkbox[0].check().run()
    at.checkbox[2].check().run()

    assert not at.exception
    assert set(at.session_state["quiz"]["answers"]["q1"]) == {"A", "C"}

    at.checkbox[4].check().run()
    assert at.session_state["quiz"]["answers"]["q1"] == ["E"]
