from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from mcqgen2.storage import Database

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
    assert at.number_input(key="question_count").value == 15
    assert at.button[0].label == "Generate questions"
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
    assert at.number_input(key="question_count").value == 15


def test_empty_instruction_state_recovers_default_rules(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()

    at.session_state["instruction_rules_high_volume"] = ""
    at.run()

    assert not at.exception
    assert at.text_area[0].value


def test_generation_drafts_survive_mode_switches_and_navigation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()

    at.text_area[0].set_value("High-Volume unsaved draft").run()
    at.selectbox(key="generation_mode").select("high_quality").run()
    at.text_area[0].set_value("High-Quality unsaved draft").run()
    at.selectbox(key="generation_mode").select("high_volume").run()

    assert not at.exception
    assert at.text_area[0].value == "High-Volume unsaved draft"

    at.radio(key="navigation").set_value("Question Sets").run()
    at.radio(key="navigation").set_value("Generate").run()

    assert not at.exception
    assert at.text_area[0].value == "High-Volume unsaved draft"
    at.selectbox(key="generation_mode").select("high_quality").run()
    assert at.text_area[0].value == "High-Quality unsaved draft"


def _quiz_state() -> dict:
    return {
        "nonce": 123,
        "questions": [
            {
                "id": "q1",
                "question_type": "mcq",
                "stem": "Which answer is supported?",
                "choices": [
                    {"id": "A", "text": "Alpha"},
                    {"id": "B", "text": "Bravo"},
                    {"id": "C", "text": "Charlie"},
                    {"id": "D", "text": "Delta"},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Alpha is supported.",
            }
        ],
        "answers": {},
        "index": 0,
        "kind": "full",
        "question_set_id": None,
        "title": "Test",
        "metadata": "",
    }


def test_exiting_quiz_restores_complete_fresh_generation_state(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at.selectbox(key="generation_mode").select("high_quality").run()
    at.number_input(key="question_count").set_value(7).run()
    at.text_area[0].set_value("Unsaved test draft").run()
    at.session_state["last_generated_id"] = "stale-set"
    at.session_state["quiz"] = _quiz_state()
    at.run()

    next(button for button in at.button if button.label == "Exit").click().run()

    assert not at.exception
    assert at.radio(key="navigation").value == "Generate"
    assert at.selectbox(key="generation_mode").value == "high_volume"
    assert at.selectbox(key="api_model").value == "gpt-6-luna"
    assert at.number_input(key="question_count").value == 15
    assert at.segmented_control(key="input_mode").value == "extracted_text"
    assert at.segmented_control(key="question_type").value == "mcq"
    assert at.text_area[0].value
    assert at.text_area[0].value != "Unsaved test draft"
    assert "last_generated_id" not in at.session_state


def test_results_use_per_choice_feedback_and_done_resets_generator(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["result"] = {
        "questions": [
            {
                "id": "q1",
                "question_type": "mcq",
                "stem": "Correctly answered question?",
                "choices": [
                    {"id": "A", "text": "Alpha", "rationale": "Correct alpha rationale."},
                    {"id": "B", "text": "Bravo", "rationale": "Wrong bravo rationale."},
                    {"id": "C", "text": "Charlie", "rationale": "Wrong charlie rationale."},
                    {"id": "D", "text": "Delta", "rationale": "Wrong delta rationale."},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Legacy correct explanation.",
            },
            {
                "id": "q2",
                "question_type": "sata",
                "stem": "Incorrectly answered SATA question?",
                "choices": [
                    {"id": "A", "text": "One", "rationale": "One is incorrect."},
                    {"id": "B", "text": "Two", "rationale": "Two is incorrect."},
                    {"id": "C", "text": "Three", "rationale": "Three is incorrect."},
                    {"id": "D", "text": "Four", "rationale": "Four is incorrect."},
                    {
                        "id": "E",
                        "text": "None of the above",
                        "rationale": "Every supplied choice is incorrect.",
                    },
                ],
                "correct_choice_ids": ["E"],
                "explanation": "Legacy SATA explanation.",
            },
        ],
        "answers": {"q1": ["A"], "q2": ["A"]},
        "score": 1,
        "total": 2,
        "kind": "full",
        "title": "Review",
        "metadata": "",
        "session_id": "session-1",
    }
    at.run()

    assert not at.exception
    markdown = "\n".join(element.value for element in at.markdown)
    rationales = [
        element.value
        for element in at.caption
        if isinstance(element.value, str) and element.value.startswith("**Why:**")
    ]
    assert ":green-badge[Correct]" in markdown
    assert ":red-badge[Incorrect]" in markdown
    assert ":blue-badge[Your answer]" in markdown
    assert "✅ **A.** Alpha :green-badge[Correct] :blue-badge[Your answer]" in markdown
    assert "Bravo" not in markdown
    assert "❌ **A.** One :red-badge[Incorrect] :blue-badge[Your answer]" in markdown
    assert any(element.value == "**B.** Two" for element in at.markdown)
    assert any(
        element.value == "✅ **E.** None of the above :green-badge[Correct]"
        for element in at.markdown
    )
    assert len(rationales) == 5
    assert all("Correct alpha rationale" not in rationale for rationale in rationales)
    assert any("Every supplied choice is incorrect" in rationale for rationale in rationales)

    next(button for button in at.button if button.label == "Done").click().run()
    assert not at.exception
    assert at.selectbox(key="generation_mode").value == "high_volume"
    assert at.selectbox(key="api_model").value == "gpt-6-luna"
    assert at.number_input(key="question_count").value == 15
    assert at.text_area[0].value


def test_quiz_flags_forward_only_navigation_and_final_skip_submission(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    recorded: dict = {}

    def fake_record_quiz(
        self,
        *,
        question_ids,
        answers,
        kind,
        question_set_id,
    ) -> dict:
        recorded.update(
            {
                "question_ids": question_ids,
                "answers": answers,
                "kind": kind,
                "question_set_id": question_set_id,
            }
        )
        return {"session_id": "session-forward", "score": 1, "total": 2}

    monkeypatch.setattr(Database, "record_quiz", fake_record_quiz)
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["quiz"] = {
        "nonce": 999,
        "questions": [
            {
                "id": "q1",
                "question_type": "mcq",
                "stem": "First question?",
                "choices": [
                    {"id": "A", "text": "Alpha", "rationale": "Alpha is correct."},
                    {"id": "B", "text": "Bravo", "rationale": "Bravo is incorrect."},
                    {"id": "C", "text": "Charlie", "rationale": "Charlie is incorrect."},
                    {"id": "D", "text": "Delta", "rationale": "Delta is incorrect."},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Alpha is correct.",
            },
            {
                "id": "q2",
                "question_type": "mcq",
                "stem": "Second question?",
                "choices": [
                    {"id": "A", "text": "Echo", "rationale": "Echo is correct."},
                    {"id": "B", "text": "Foxtrot", "rationale": "Foxtrot is incorrect."},
                    {"id": "C", "text": "Golf", "rationale": "Golf is incorrect."},
                    {"id": "D", "text": "Hotel", "rationale": "Hotel is incorrect."},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Echo is correct.",
            },
        ],
        "answers": {},
        "flagged_question_ids": set(),
        "skipped_question_ids": set(),
        "index": 0,
        "kind": "full",
        "question_set_id": None,
        "title": "Forward-only test",
        "metadata": "test",
    }
    at.run()

    assert not at.exception
    assert all(button.label != "Previous" for button in at.button)
    assert next(button for button in at.button if button.label == "Next").disabled
    next(button for button in at.button if button.label == "Flag question").click().run()
    assert at.session_state["quiz"]["flagged_question_ids"] == {"q1"}
    next(button for button in at.button if button.label == "Remove flag").click().run()
    assert at.session_state["quiz"]["flagged_question_ids"] == set()
    next(button for button in at.button if button.label == "Flag question").click().run()
    assert at.session_state["quiz"]["flagged_question_ids"] == {"q1"}

    at.radio[0].set_value("A").run()
    next(button for button in at.button if button.label == "Next").click().run()
    assert not at.exception
    assert at.session_state["quiz"]["index"] == 1
    assert all(button.label != "Previous" for button in at.button)

    next(button for button in at.button if button.label == "Flag question").click().run()
    next(
        button for button in at.button if button.label == "Skip and submit test"
    ).click().run()

    assert not at.exception
    assert recorded["answers"] == {"q1": ["A"], "q2": []}
    assert at.session_state["result"]["skipped_question_ids"] == {"q2"}
    assert at.session_state["result"]["flagged_question_ids"] == {"q1", "q2"}
    assert any(metric.label == "Flagged" and metric.value == "2" for metric in at.metric)
    result_markdown = "\n".join(element.value for element in at.markdown)
    assert ":orange-badge[Flagged]" in result_markdown
    assert ":gray-badge[Skipped]" in result_markdown
    result_rationales = [
        element.value
        for element in at.caption
        if isinstance(element.value, str) and element.value.startswith("**Why:**")
    ]
    assert any("Bravo is incorrect" in rationale for rationale in result_rationales)
    assert any("Echo is correct" in rationale for rationale in result_rationales)


def test_nonfinal_skip_clears_selection_and_permanently_advances(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MCQGEN_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.session_state["quiz"] = {
        "nonce": 321,
        "questions": [
            {
                "id": "q1",
                "question_type": "mcq",
                "stem": "First question?",
                "choices": [
                    {"id": "A", "text": "Alpha"},
                    {"id": "B", "text": "Bravo"},
                    {"id": "C", "text": "Charlie"},
                    {"id": "D", "text": "Delta"},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Alpha is correct.",
            },
            {
                "id": "q2",
                "question_type": "mcq",
                "stem": "Second question?",
                "choices": [
                    {"id": "A", "text": "Echo"},
                    {"id": "B", "text": "Foxtrot"},
                    {"id": "C", "text": "Golf"},
                    {"id": "D", "text": "Hotel"},
                ],
                "correct_choice_ids": ["A"],
                "explanation": "Echo is correct.",
            },
        ],
        "answers": {},
        "index": 0,
        "kind": "full",
        "question_set_id": None,
        "title": "Skip test",
        "metadata": "test",
    }
    at.run()
    at.radio[0].set_value("B").run()
    next(button for button in at.button if button.label == "Skip question").click().run()

    assert not at.exception
    assert at.session_state["quiz"]["index"] == 1
    assert "q1" not in at.session_state["quiz"]["answers"]
    assert at.session_state["quiz"]["skipped_question_ids"] == {"q1"}
    assert all(button.label != "Previous" for button in at.button)


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
