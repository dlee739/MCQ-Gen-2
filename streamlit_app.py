from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI

from mcqgen2 import __version__
from mcqgen2.config import (
    ALLOWED_MODELS,
    DEFAULT_INPUT_MODE,
    DEFAULT_MODE,
    DEFAULT_QUESTION_TYPE,
    MAX_INSTRUCTION_RULES_CHARS,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    INPUT_MODE_LABELS,
    MODE_CONFIGS,
    MODE_DEFAULT_MODELS,
    MODE_DEFAULT_QUESTION_COUNTS,
    QUESTION_TYPE_LABELS,
    load_local_api_key,
)
from mcqgen2.generation import (
    GenerationError,
    generate_question_set,
    normalize_instruction_rules,
    source_sha256,
    validate_pdf,
)
from mcqgen2.explanations import choice_rationales
from mcqgen2.pdf_export import build_results_pdf
from mcqgen2.pricing import (
    MODEL_PRICING,
    PRICING_SOURCE,
    PRICING_VERIFIED_AT,
    pricing_rows,
)
from mcqgen2.storage import Database

ROOT = Path(__file__).resolve().parent
load_local_api_key(ROOT / ".env")

st.set_page_config(
    page_title="MCQ-Gen 2",
    page_icon=":material/quiz:",
    layout="centered",
)

GENERATION_MODE_KEY = "generation_mode"
API_MODEL_KEY = "api_model"
QUESTION_COUNT_KEY = "question_count"
INPUT_MODE_KEY = "input_mode"
QUESTION_TYPE_KEY = "question_type"
NAVIGATION_KEY = "navigation"


@st.cache_resource
def database(path: str) -> Database:
    db = Database(Path(path))
    db.initialize()
    return db


def money(value: float) -> str:
    if value == 0:
        return "$0.000000"
    return f"${value:.6f}"


def reset_generation_defaults() -> None:
    mode = st.session_state[GENERATION_MODE_KEY]
    st.session_state[API_MODEL_KEY] = MODE_DEFAULT_MODELS[mode]
    st.session_state[QUESTION_COUNT_KEY] = MODE_DEFAULT_QUESTION_COUNTS[mode]


def reset_generation_form(db: Database) -> None:
    """Return the generation page to a complete, usable fresh state."""
    st.session_state[GENERATION_MODE_KEY] = DEFAULT_MODE
    st.session_state[API_MODEL_KEY] = MODE_DEFAULT_MODELS[DEFAULT_MODE]
    st.session_state[QUESTION_COUNT_KEY] = MODE_DEFAULT_QUESTION_COUNTS[DEFAULT_MODE]
    st.session_state[INPUT_MODE_KEY] = DEFAULT_INPUT_MODE
    st.session_state[QUESTION_TYPE_KEY] = DEFAULT_QUESTION_TYPE
    st.session_state[NAVIGATION_KEY] = "Generate"
    st.session_state.pop("last_generated_id", None)

    for mode in MODE_CONFIGS:
        profile = db.get_default_instruction_profile(mode)
        st.session_state[f"instruction_profile_{mode}"] = profile["id"]
        st.session_state[f"instruction_profile_loaded_{mode}"] = profile["id"]
        st.session_state[f"instruction_rules_{mode}"] = profile["instructions"]


def prepare_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(questions)
    random.shuffle(prepared)
    for question in prepared:
        fixed_none = [choice for choice in question["choices"] if choice["id"] == "E"]
        other_choices = [choice for choice in question["choices"] if choice["id"] != "E"]
        random.shuffle(other_choices)
        question["choices"] = other_choices + fixed_none
    return prepared


def toggle_bookmark(db: Database, question_id: str) -> None:
    db.set_bookmark(question_id, not db.is_bookmarked(question_id))
    st.rerun()


def enforce_sata_exclusivity(widget_key: str, changed_id: str, choice_ids: list[str]) -> None:
    changed_key = f"{widget_key}_{changed_id}"
    if not st.session_state.get(changed_key, False):
        return
    if changed_id == "E":
        for choice_id in choice_ids:
            if choice_id != "E":
                st.session_state[f"{widget_key}_{choice_id}"] = False
    else:
        st.session_state[f"{widget_key}_E"] = False


def render_choice_review(
    question: dict[str, Any],
    *,
    selected_ids: set[str] | None,
    show_rationales: bool,
) -> None:
    correct_ids = set(question["correct_choice_ids"])
    rationales = choice_rationales(question) if show_rationales else None

    for position, choice in enumerate(question["choices"]):
        choice_id = str(choice["id"])
        letter = chr(ord("A") + position)
        if choice_id in correct_ids:
            icon = "✅"
            status = ":green-badge[Correct]"
        else:
            icon = "❌"
            status = ":red-badge[Incorrect]"
        selected = (
            " :blue-badge[Your answer]"
            if selected_ids is not None and choice_id in selected_ids
            else ""
        )
        st.markdown(f"{icon} **{letter}.** {choice['text']} {status}{selected}")
        if rationales is not None:
            st.caption(f"**Why:** {rationales[choice_id]}")

    if show_rationales and rationales is None:
        st.info(question["explanation"])


def start_quiz(
    *,
    questions: list[dict[str, Any]],
    kind: str,
    question_set_id: str | None,
    title: str,
    metadata: str,
) -> None:
    st.session_state.pop("result", None)
    st.session_state.quiz = {
        "nonce": random.randrange(1_000_000_000),
        "questions": prepare_questions(questions),
        "answers": {},
        "index": 0,
        "kind": kind,
        "question_set_id": question_set_id,
        "title": title,
        "metadata": metadata,
    }
    st.rerun()


def render_quiz(db: Database) -> None:
    quiz = st.session_state.quiz
    questions = quiz["questions"]
    index = quiz["index"]
    question = questions[index]
    question_id = question["id"]
    question_type = question.get("question_type", "mcq")

    st.title(quiz["title"])
    st.progress((index + 1) / len(questions))
    st.caption(f"Question {index + 1} of {len(questions)}")
    st.markdown(f"**{question['stem']}**")
    bookmarked = db.is_bookmarked(question_id)
    if st.button(
        "Remove bookmark" if bookmarked else "Bookmark question",
        icon=":material/bookmark_remove:" if bookmarked else ":material/bookmark_add:",
        key=f"bookmark_{quiz['nonce']}_{question_id}",
    ):
        toggle_bookmark(db, question_id)

    choice_by_id = {choice["id"]: choice["text"] for choice in question["choices"]}
    display_letter = {
        choice["id"]: chr(ord("A") + position)
        for position, choice in enumerate(question["choices"])
    }
    widget_key = f"quiz_{quiz['nonce']}_{question_id}"
    existing_answers = quiz["answers"].get(question_id, [])
    if question_type == "mcq":
        options = list(choice_by_id)
        selected = st.radio(
            "Choose one answer",
            options=options,
            index=options.index(existing_answers[0]) if existing_answers else None,
            key=widget_key,
            format_func=lambda choice_id: (
                f"{display_letter[choice_id]}. {choice_by_id[choice_id]}"
            ),
        )
        if selected is not None:
            quiz["answers"][question_id] = [selected]
    else:
        st.caption("Select every answer that applies.")
        selected_ids: list[str] = []
        choice_ids = list(choice_by_id)
        for choice_id in choice_by_id:
            checked = st.checkbox(
                f"{display_letter[choice_id]}. {choice_by_id[choice_id]}",
                value=choice_id in existing_answers,
                key=f"{widget_key}_{choice_id}",
                on_change=enforce_sata_exclusivity,
                args=(widget_key, choice_id, choice_ids),
            )
            if checked:
                selected_ids.append(choice_id)
        if selected_ids:
            quiz["answers"][question_id] = selected_ids
        else:
            quiz["answers"].pop(question_id, None)

    previous_col, next_col, submit_col, cancel_col = st.columns([1, 1, 1.4, 1])
    with previous_col:
        if st.button("Previous", disabled=index == 0, width="stretch"):
            quiz["index"] -= 1
            st.rerun()
    with next_col:
        current_answered = question_id in quiz["answers"]
        if st.button(
            "Next",
            disabled=index == len(questions) - 1 or not current_answered,
            width="stretch",
        ):
            quiz["index"] += 1
            st.rerun()
    with submit_col:
        ready = len(quiz["answers"]) == len(questions)
        if st.button(
            "Submit test",
            type="primary",
            disabled=not ready,
            width="stretch",
        ):
            recorded = db.record_quiz(
                question_ids=[item["id"] for item in questions],
                answers=quiz["answers"],
                kind=quiz["kind"],
                question_set_id=quiz["question_set_id"],
            )
            st.session_state.result = {
                **quiz,
                **recorded,
            }
            del st.session_state.quiz
            st.rerun()
    with cancel_col:
        if st.button("Exit", width="stretch"):
            reset_generation_form(db)
            del st.session_state.quiz
            st.rerun()

    answered = len(quiz["answers"])
    st.caption(f"Answered: {answered} / {len(questions)}. Results remain hidden until submission.")


def render_result(db: Database) -> None:
    result = st.session_state.result
    questions = result["questions"]
    answers = result["answers"]
    score = result["score"]
    total = result["total"]

    st.title("Results")
    score_col, percentage_col, missed_col = st.columns(3)
    score_col.metric("Score", f"{score} / {total}")
    percentage_col.metric("Percentage", f"{(score / total) * 100:.1f}%")
    missed_col.metric("Incorrect", total - score)
    if result["kind"] == "retry":
        st.info("Correct retries have been removed from the incorrect-question queue.")

    pdf_bytes = build_results_pdf(
        title=result["title"],
        questions=questions,
        answers=answers,
        score=score,
        total=total,
        metadata=result["metadata"],
    )
    action_col, done_col = st.columns([1, 1])
    with action_col:
        st.download_button(
            "Download results PDF",
            data=pdf_bytes,
            file_name="mcqgen-results.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with done_col:
        if st.button("Done", type="primary", width="stretch"):
            reset_generation_form(db)
            del st.session_state.result
            st.rerun()

    for number, question in enumerate(questions, start=1):
        selected_ids = set(answers[question["id"]])
        correct_ids = set(question["correct_choice_ids"])
        correct = selected_ids == correct_ids
        icon = "✅" if correct else "❌"
        st.markdown(f"### {icon} Question {number}")
        st.markdown(question["stem"])
        bookmarked = db.is_bookmarked(question["id"])
        if st.button(
            "Remove bookmark" if bookmarked else "Bookmark question",
            icon=":material/bookmark_remove:" if bookmarked else ":material/bookmark_add:",
            key=f"result_bookmark_{result['session_id']}_{question['id']}",
        ):
            toggle_bookmark(db, question["id"])

        render_choice_review(
            question,
            selected_ids=selected_ids,
            show_rationales=not correct,
        )
        st.divider()


def render_usage(question_set: dict[str, Any]) -> None:
    usage = question_set["usage"]
    cost = question_set["cost"]
    cols = st.columns(5)
    cols[0].metric("Input tokens", f"{usage['input_tokens']:,}")
    cols[1].metric("Output tokens", f"{usage['output_tokens']:,}")
    cols[2].metric("Reasoning tokens", f"{usage['reasoning_tokens']:,}")
    cols[3].metric("Total tokens", f"{usage['total_tokens']:,}")
    cols[4].metric("Estimated cost", money(cost["total_cost"]))
    with st.expander("Usage and cost details"):
        st.json({"usage": usage, "cost": cost, "pricing": question_set["pricing"]})
        st.caption("Estimated from the stored pricing snapshot; this is not an invoice.")


@st.dialog("Save as a new instruction profile")
def save_profile_dialog(db: Database, mode: str, instructions: str) -> None:
    name = st.text_input("Profile name", max_chars=80)
    if st.button("Save profile", type="primary", disabled=not name.strip()):
        try:
            profile_id = db.create_instruction_profile(
                mode=mode, name=name, instructions=instructions
            )
            st.session_state[f"instruction_profile_{mode}"] = profile_id
            st.session_state.pop(f"instruction_profile_loaded_{mode}", None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


@st.dialog("Delete instruction profile")
def delete_profile_dialog(db: Database, profile: dict[str, Any]) -> None:
    st.write(f"Delete **{profile['name']}**? Generated sets will keep their saved snapshot.")
    if st.button("Delete profile", type="primary", icon=":material/delete:"):
        try:
            db.delete_instruction_profile(profile["id"])
            st.session_state.pop(f"instruction_profile_{profile['mode']}", None)
            st.session_state.pop(f"instruction_profile_loaded_{profile['mode']}", None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def render_instruction_profile_editor(
    db: Database, mode: str
) -> tuple[str, str]:
    profiles = db.list_instruction_profiles(mode)
    profile_by_id = {profile["id"]: profile for profile in profiles}
    selected_key = f"instruction_profile_{mode}"
    editor_key = f"instruction_rules_{mode}"
    loaded_key = f"instruction_profile_loaded_{mode}"

    if st.session_state.get(selected_key) not in profile_by_id:
        st.session_state[selected_key] = db.get_default_instruction_profile(mode)["id"]

    selected_id = st.selectbox(
        "Instruction profile",
        options=list(profile_by_id),
        key=selected_key,
        format_func=lambda profile_id: (
            profile_by_id[profile_id]["name"]
            + (" (default)" if profile_by_id[profile_id]["is_default"] else "")
        ),
        help="Profiles are kept separately for each generation mode.",
        persist_state="session",
    )
    selected = profile_by_id[selected_id]
    current_instructions = st.session_state.get(editor_key)
    if (
        st.session_state.get(loaded_key) != selected_id
        or not isinstance(current_instructions, str)
        or not current_instructions.strip()
    ):
        st.session_state[editor_key] = selected["instructions"]
        st.session_state[loaded_key] = selected_id

    instructions = st.text_area(
        "Instruction rules",
        key=editor_key,
        max_chars=MAX_INSTRUCTION_RULES_CHARS,
        height=220,
        help=(
            "These editable rules control the writing style for this request. "
            "Question count, source grounding, answer structure, and output format remain fixed."
        ),
        persist_state="session",
    )
    save_col, new_col, default_col, delete_col = st.columns(4)
    with save_col:
        if st.button("Save changes", width="stretch"):
            try:
                db.update_instruction_profile(
                    selected_id, name=selected["name"], instructions=instructions
                )
                st.success("Profile saved.")
            except ValueError as exc:
                st.error(str(exc))
    with new_col:
        if st.button("Save as new", width="stretch"):
            save_profile_dialog(db, mode, instructions)
    with default_col:
        if st.button(
            "Set as default",
            width="stretch",
            disabled=bool(selected["is_default"]),
        ):
            try:
                db.update_instruction_profile(
                    selected_id, name=selected["name"], instructions=instructions
                )
                db.set_default_instruction_profile(selected_id)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with delete_col:
        if st.button(
            "Delete profile",
            width="stretch",
            disabled=bool(selected["is_default"]) or len(profiles) == 1,
        ):
            delete_profile_dialog(db, selected)

    return selected["name"], instructions


def render_generate(db: Database) -> None:
    st.header("Generate a question set")
    st.write("Upload one PDF and generate the complete question set in one model request.")

    api_key_available = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_available:
        st.warning(
            "Set OPENAI_API_KEY in your environment or in a local .env file before generating."
        )

    uploaded = st.file_uploader("Source PDF", type=["pdf"], accept_multiple_files=False)
    generate_button_slot = st.empty()

    if st.session_state.get(INPUT_MODE_KEY) not in INPUT_MODE_LABELS:
        st.session_state[INPUT_MODE_KEY] = DEFAULT_INPUT_MODE
    input_mode = st.segmented_control(
        "Input processing",
        options=list(INPUT_MODE_LABELS),
        required=True,
        format_func=lambda value: INPUT_MODE_LABELS[value],
        width="stretch",
        key=INPUT_MODE_KEY,
        persist_state="session",
        help=(
            "Extracted text is much less expensive. Send the original PDF when "
            "questions depend on diagrams or other page visuals."
        ),
    )
    if input_mode == "extracted_text":
        st.caption("All extracted pages are sent together; no chunking is used.")
    else:
        st.caption("Includes page images and can use substantially more input tokens.")

    if st.session_state.get(QUESTION_TYPE_KEY) not in QUESTION_TYPE_LABELS:
        st.session_state[QUESTION_TYPE_KEY] = DEFAULT_QUESTION_TYPE
    question_type = st.segmented_control(
        "Question type",
        options=list(QUESTION_TYPE_LABELS),
        required=True,
        format_func=lambda value: QUESTION_TYPE_LABELS[value],
        width="stretch",
        key=QUESTION_TYPE_KEY,
        persist_state="session",
        help="MCQ has one correct answer. SATA means Select All That Apply.",
    )
    if st.session_state.get(GENERATION_MODE_KEY) not in MODE_CONFIGS:
        st.session_state[GENERATION_MODE_KEY] = DEFAULT_MODE
    selected_mode = st.session_state[GENERATION_MODE_KEY]
    if st.session_state.get(API_MODEL_KEY) not in ALLOWED_MODELS:
        st.session_state[API_MODEL_KEY] = MODE_DEFAULT_MODELS[selected_mode]
    question_count_state = st.session_state.get(QUESTION_COUNT_KEY)
    if (
        not isinstance(question_count_state, int)
        or isinstance(question_count_state, bool)
        or not MIN_QUESTION_COUNT <= question_count_state <= MAX_QUESTION_COUNT
    ):
        st.session_state[QUESTION_COUNT_KEY] = MODE_DEFAULT_QUESTION_COUNTS[selected_mode]

    left, middle, right = st.columns([1.1, 1.2, 1])
    with left:
        mode = st.selectbox(
            "Generation mode",
            options=list(MODE_CONFIGS),
            format_func=lambda value: MODE_CONFIGS[value].label,
            key=GENERATION_MODE_KEY,
            on_change=reset_generation_defaults,
            persist_state="session",
        )
        st.caption(MODE_CONFIGS[mode].description)
    with middle:
        model = st.selectbox(
            "API model",
            options=list(ALLOWED_MODELS),
            key=API_MODEL_KEY,
            persist_state="session",
        )
        if MODEL_PRICING[model].note:
            st.caption(MODEL_PRICING[model].note)
    with right:
        question_count = st.number_input(
            "Number of questions",
            min_value=MIN_QUESTION_COUNT,
            max_value=MAX_QUESTION_COUNT,
            step=1,
            key=QUESTION_COUNT_KEY,
            persist_state="session",
        )

    profile_name, instruction_rules = render_instruction_profile_editor(db, mode)

    with st.expander("Compare model pricing"):
        st.dataframe(pricing_rows(), hide_index=True, width="stretch")
        st.caption(
            f"USD per 1M tokens, standard service tier. Verified {PRICING_VERIFIED_AT}. "
            "Long-context rates apply above 272K input tokens."
        )
        st.link_button("Official OpenAI pricing", PRICING_SOURCE)

    pdf_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_sha = source_sha256(pdf_bytes) if pdf_bytes is not None else None
    with generate_button_slot:
        generate_clicked = st.button(
            "Generate questions",
            type="primary",
            disabled=(
                pdf_bytes is None
                or not api_key_available
                or not instruction_rules.strip()
            ),
            width="stretch",
        )
    if generate_clicked and pdf_bytes is not None:
        assert uploaded_sha is not None
        try:
            validate_pdf(uploaded.name, pdf_bytes)
            normalized_instructions = normalize_instruction_rules(instruction_rules)
            with st.spinner("Generating the complete question set…"):
                result = generate_question_set(
                    client=OpenAI(),
                    filename=uploaded.name,
                    pdf_bytes=pdf_bytes,
                    model=model,
                    mode=mode,
                    question_type=question_type,
                    question_count=int(question_count),
                    input_mode=input_mode,
                    instruction_rules=normalized_instructions,
                )
                set_id = db.save_question_set(
                    source_filename=Path(uploaded.name).name,
                    source_sha256=uploaded_sha,
                    mode=mode,
                    model=model,
                    input_mode=input_mode,
                    question_type=question_type,
                    instruction_profile_name=profile_name,
                    instruction_rules=normalized_instructions,
                    requested_count=int(question_count),
                    result=result,
                )
            st.session_state.last_generated_id = set_id
            st.success(f"Generated and saved {len(result.questions)} questions.")
        except (ValueError, GenerationError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The question set could not be saved: {exc}")

    last_id = st.session_state.get("last_generated_id")
    if last_id:
        saved = db.get_question_set(last_id)
        if saved and uploaded_sha == saved["source_sha256"]:
            render_usage(saved)
            if st.button("Start this test", type="primary"):
                start_quiz(
                    questions=saved["questions"],
                    kind="full",
                    question_set_id=saved["id"],
                    title=f"{saved['source_filename']} — Practice Test",
                    metadata=(
                        f"{saved['model']} · {MODE_CONFIGS[saved['mode']].label} · "
                        f"{QUESTION_TYPE_LABELS[saved['question_type']]} · "
                        f"{INPUT_MODE_LABELS[saved['input_mode']]}"
                    ),
                )
        else:
            st.session_state.pop("last_generated_id", None)


@st.dialog("Delete all previous runs", icon=":material/delete:")
def delete_all_runs_dialog(db: Database) -> None:
    st.warning(
        "This permanently deletes every generated question set, test attempt, and "
        "bookmark. Your instruction profiles will be kept."
    )
    if st.button("Delete all runs", type="primary", icon=":material/delete:"):
        counts = db.delete_all_runs()
        st.session_state.pop("last_generated_id", None)
        st.session_state.pop("quiz", None)
        st.session_state.pop("result", None)
        st.session_state.runs_deleted = counts
        st.rerun()


def render_question_sets(db: Database) -> None:
    st.header("Question sets")
    deleted = st.session_state.pop("runs_deleted", None)
    if deleted:
        st.success(
            f"Deleted {deleted['question_sets']} question sets, "
            f"{deleted['quiz_sessions']} test sessions, and "
            f"{deleted['bookmarks']} bookmarks. Instruction profiles were kept."
        )
    question_sets = db.list_question_sets()
    if not question_sets:
        st.info("No question sets have been generated yet.")
        return

    for item in question_sets:
        label = (
            f"{item['source_filename']} · {QUESTION_TYPE_LABELS[item['question_type']]} · "
            f"{item['requested_count']} questions · "
            f"{item['created_at'][:16].replace('T', ' ')} UTC"
        )
        with st.expander(label):
            st.write(
                f"**Model:** `{item['model']}`  |  "
                f"**Mode:** {MODE_CONFIGS[item['mode']].label}  |  "
                f"**Input:** {INPUT_MODE_LABELS[item['input_mode']]}  |  "
                f"**Estimated cost:** {money(item['cost']['total_cost'])}"
            )
            st.markdown(f"**Instruction profile:** {item['instruction_profile_name']}")
            if item["custom_instructions"]:
                st.markdown("**Saved instruction rules**")
                st.code(item["custom_instructions"], language=None)
            if item["question_type"] == "sata":
                st.caption(
                    "Correct-answer counts: "
                    + ", ".join(str(value) for value in item["sata_correct_counts"])
                )
            render_usage(item)
            if st.button("Start test", key=f"start_{item['id']}", type="primary"):
                full = db.get_question_set(item["id"])
                if full:
                    start_quiz(
                        questions=full["questions"],
                        kind="full",
                        question_set_id=full["id"],
                        title=f"{full['source_filename']} — Practice Test",
                        metadata=(
                            f"{full['model']} · {MODE_CONFIGS[full['mode']].label} · "
                            f"{QUESTION_TYPE_LABELS[full['question_type']]} · "
                            f"{INPUT_MODE_LABELS[full['input_mode']]}"
                        ),
                    )

    st.divider()
    st.subheader("Danger zone")
    st.caption(
        "Delete every question set, attempt, and bookmark. Instruction profiles are preserved."
    )
    if st.button("Delete all previous runs", icon=":material/delete:"):
        delete_all_runs_dialog(db)


def render_incorrect(db: Database) -> None:
    st.header("Incorrect questions")
    questions = db.get_retry_questions()
    st.metric("Questions to retry", len(questions))
    if not questions:
        st.success("Nothing to retry. Incorrect questions will appear here after a test.")
        return

    sources = sorted({question["source_filename"] for question in questions})
    st.caption("Sources: " + ", ".join(sources))
    if st.button("Retry all incorrect questions", type="primary"):
        start_quiz(
            questions=questions,
            kind="retry",
            question_set_id=None,
            title="Incorrect Questions Retry",
            metadata="Mixed retry set" if len(sources) > 1 else sources[0],
        )


def render_bookmarks(db: Database) -> None:
    st.header("Bookmarks")
    questions = db.get_bookmarked_questions()
    st.metric("Saved questions", len(questions))
    if not questions:
        st.info("Bookmark a question during a test or while reviewing results.")
        return

    if st.button("Practice all bookmarks", type="primary"):
        sources = sorted({question["source_filename"] for question in questions})
        start_quiz(
            questions=questions,
            kind="bookmarked",
            question_set_id=None,
            title="Bookmarked Questions",
            metadata="Mixed bookmarked set" if len(sources) > 1 else sources[0],
        )

    for number, question in enumerate(questions, start=1):
        with st.expander(f"{number}. {question['stem']}"):
            st.caption(
                f"{question['source_filename']} · "
                f"{QUESTION_TYPE_LABELS[question['question_type']]}"
            )
            render_choice_review(
                question,
                selected_ids=None,
                show_rationales=True,
            )
            if st.button(
                "Remove bookmark",
                key=f"remove_bookmark_{question['id']}",
                icon=":material/bookmark_remove:",
            ):
                toggle_bookmark(db, question["id"])


database_path = os.getenv("MCQGEN_DATABASE_PATH", str(ROOT / "data" / "mcqgen.sqlite3"))
db = database(database_path)
st.title("MCQ-Gen 2")
st.caption("Generate source-grounded practice questions from a PDF.")

if "quiz" in st.session_state:
    render_quiz(db)
    st.stop()
if "result" in st.session_state:
    render_result(db)
    st.stop()

if st.session_state.get(NAVIGATION_KEY) not in {
    "Generate",
    "Question Sets",
    "Incorrect Questions",
    "Bookmarks",
}:
    st.session_state[NAVIGATION_KEY] = "Generate"
page = st.sidebar.radio(
    "Navigation",
    ["Generate", "Question Sets", "Incorrect Questions", "Bookmarks"],
    key=NAVIGATION_KEY,
    persist_state="session",
)
st.sidebar.divider()
st.sidebar.caption(f"Local beta {__version__}")
st.sidebar.caption(
    "Generated questions may contain errors; verify them against the source."
)

if page == "Generate":
    render_generate(db)
elif page == "Question Sets":
    render_question_sets(db)
elif page == "Incorrect Questions":
    render_incorrect(db)
else:
    render_bookmarks(db)
