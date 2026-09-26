from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI

from mcqgen2.config import (
    ALLOWED_MODELS,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_QUESTION_COUNT,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    MODE_CONFIGS,
    load_local_api_key,
)
from mcqgen2.generation import (
    GenerationError,
    generate_question_set,
    source_sha256,
    validate_pdf,
)
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

st.set_page_config(page_title="MCQ-Gen 2", page_icon="🩺", layout="wide")


@st.cache_resource
def database() -> Database:
    db = Database(ROOT / "data" / "mcqgen.sqlite3")
    db.initialize()
    return db


def money(value: float) -> str:
    if value == 0:
        return "$0.000000"
    return f"${value:.6f}"


def prepare_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(questions)
    random.shuffle(prepared)
    for question in prepared:
        random.shuffle(question["choices"])
    return prepared


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

    st.title(quiz["title"])
    st.progress((index + 1) / len(questions))
    st.caption(f"Question {index + 1} of {len(questions)}")
    st.subheader(question["stem"])

    choice_by_id = {choice["id"]: choice["text"] for choice in question["choices"]}
    display_letter = {
        choice["id"]: chr(ord("A") + position)
        for position, choice in enumerate(question["choices"])
    }
    widget_key = f"quiz_{quiz['nonce']}_{question_id}"
    selected = st.radio(
        "Choose one answer",
        options=list(choice_by_id),
        index=None,
        key=widget_key,
        format_func=lambda choice_id: (
            f"{display_letter[choice_id]}. {choice_by_id[choice_id]}"
        ),
    )
    if selected is not None:
        quiz["answers"][question_id] = selected

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
            del st.session_state.quiz
            st.rerun()

    answered = len(quiz["answers"])
    st.caption(f"Answered: {answered} / {len(questions)}. Results remain hidden until submission.")


def render_result() -> None:
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
            del st.session_state.result
            st.rerun()

    for number, question in enumerate(questions, start=1):
        selected_id = answers[question["id"]]
        correct_id = question["correct_choice_id"]
        correct = selected_id == correct_id
        icon = "✅" if correct else "❌"
        st.markdown(f"### {icon} Question {number}")
        st.markdown(question["stem"])

        for position, choice in enumerate(question["choices"]):
            letter = chr(ord("A") + position)
            markers: list[str] = []
            if choice["id"] == selected_id:
                markers.append("your answer")
            if choice["id"] == correct_id:
                markers.append("correct")
            marker = f" — **{', '.join(markers)}**" if markers else ""
            st.markdown(f"{letter}. {choice['text']}{marker}")
        st.info(question["explanation"])
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


def render_generate(db: Database) -> None:
    st.header("Generate a question set")
    st.write("Upload one PDF and generate the complete question set in one model request.")

    api_key_available = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_available:
        st.warning(
            "Set OPENAI_API_KEY in your environment or in a local .env file before generating."
        )

    uploaded = st.file_uploader("Source PDF", type=["pdf"], accept_multiple_files=False)
    left, middle, right = st.columns([1.1, 1.2, 1])
    with left:
        mode = st.selectbox(
            "Generation mode",
            options=list(MODE_CONFIGS),
            index=list(MODE_CONFIGS).index(DEFAULT_MODE),
            format_func=lambda value: MODE_CONFIGS[value].label,
        )
        st.caption(MODE_CONFIGS[mode].description)
    with middle:
        model = st.selectbox(
            "API model",
            options=list(ALLOWED_MODELS),
            index=list(ALLOWED_MODELS).index(DEFAULT_MODEL),
        )
        if MODEL_PRICING[model].note:
            st.caption(MODEL_PRICING[model].note)
    with right:
        question_count = st.number_input(
            "Number of questions",
            min_value=MIN_QUESTION_COUNT,
            max_value=MAX_QUESTION_COUNT,
            value=DEFAULT_QUESTION_COUNT,
            step=1,
        )

    with st.expander("Compare model pricing"):
        st.dataframe(pricing_rows(), hide_index=True, width="stretch")
        st.caption(
            f"USD per 1M tokens, standard service tier. Verified {PRICING_VERIFIED_AT}. "
            "Long-context rates apply above 272K input tokens."
        )
        st.link_button("Official OpenAI pricing", PRICING_SOURCE)

    generate_clicked = st.button(
        "Generate questions",
        type="primary",
        disabled=uploaded is None or not api_key_available,
    )
    if generate_clicked and uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        try:
            validate_pdf(uploaded.name, pdf_bytes)
            with st.spinner("Generating the complete question set…"):
                result = generate_question_set(
                    client=OpenAI(),
                    filename=uploaded.name,
                    pdf_bytes=pdf_bytes,
                    model=model,
                    mode=mode,
                    question_count=int(question_count),
                )
                set_id = db.save_question_set(
                    source_filename=Path(uploaded.name).name,
                    source_sha256=source_sha256(pdf_bytes),
                    mode=mode,
                    model=model,
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
        if saved:
            render_usage(saved)
            if st.button("Start this test", type="primary"):
                start_quiz(
                    questions=saved["questions"],
                    kind="full",
                    question_set_id=saved["id"],
                    title=f"{saved['source_filename']} — MCQ Test",
                    metadata=f"{saved['model']} · {MODE_CONFIGS[saved['mode']].label}",
                )


def render_question_sets(db: Database) -> None:
    st.header("Question sets")
    question_sets = db.list_question_sets()
    if not question_sets:
        st.info("No question sets have been generated yet.")
        return

    for item in question_sets:
        label = (
            f"{item['source_filename']} · {item['requested_count']} questions · "
            f"{item['created_at'][:16].replace('T', ' ')} UTC"
        )
        with st.expander(label):
            st.write(
                f"**Model:** `{item['model']}`  |  "
                f"**Mode:** {MODE_CONFIGS[item['mode']].label}  |  "
                f"**Estimated cost:** {money(item['cost']['total_cost'])}"
            )
            render_usage(item)
            if st.button("Start test", key=f"start_{item['id']}", type="primary"):
                full = db.get_question_set(item["id"])
                if full:
                    start_quiz(
                        questions=full["questions"],
                        kind="full",
                        question_set_id=full["id"],
                        title=f"{full['source_filename']} — MCQ Test",
                        metadata=f"{full['model']} · {MODE_CONFIGS[full['mode']].label}",
                    )


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


db = database()
st.title("MCQ-Gen 2")
st.caption("Generate source-grounded medical practice questions from a PDF.")

if "quiz" in st.session_state:
    render_quiz(db)
    st.stop()
if "result" in st.session_state:
    render_result()
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["Generate", "Question Sets", "Incorrect Questions"],
)
st.sidebar.divider()
st.sidebar.caption("For educational use. Generated medical questions should not guide patient care.")

if page == "Generate":
    render_generate(db)
elif page == "Question Sets":
    render_question_sets(db)
else:
    render_incorrect(db)
