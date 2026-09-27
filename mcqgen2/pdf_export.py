from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _choice_rationales(question: Mapping[str, Any]) -> dict[str, str] | None:
    """Map the current structured explanation back to choices when unambiguous."""
    explanation = str(question.get("explanation", ""))
    located: list[tuple[int, int, str]] = []
    for choice in question["choices"]:
        choice_id = str(choice["id"])
        marker = f"{choice['text']}:"
        marker_start = explanation.find(marker)
        if marker_start < 0 or explanation.find(marker, marker_start + len(marker)) >= 0:
            return None
        located.append((marker_start, marker_start + len(marker), choice_id))

    if len({start for start, _, _ in located}) != len(located):
        return None

    rationales: dict[str, str] = {}
    ordered = sorted(located)
    for index, (_, rationale_start, choice_id) in enumerate(ordered):
        rationale_end = (
            ordered[index + 1][0] if index + 1 < len(ordered) else len(explanation)
        )
        rationale = explanation[rationale_start:rationale_end].strip()
        if not rationale:
            return None
        rationales[choice_id] = rationale
    return rationales


def build_results_pdf(
    *,
    title: str,
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Sequence[str]],
    score: int,
    total: int,
    metadata: str = "",
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=title,
    )
    styles = getSampleStyleSheet()
    rationale_style = ParagraphStyle(
        "ChoiceRationale",
        parent=styles["BodyText"],
        leftIndent=18,
        rightIndent=6,
        leading=14,
        spaceAfter=6,
        textColor=colors.HexColor("#444444"),
    )
    explanation_style = ParagraphStyle(
        "FallbackExplanation",
        parent=styles["BodyText"],
        leading=14,
        spaceBefore=4,
        spaceAfter=8,
    )
    story: list[Any] = [Paragraph(escape(title), styles["Title"])]
    story.append(Paragraph(f"Score: {score} / {total}", styles["Heading2"]))
    if metadata:
        story.append(Paragraph(escape(metadata), styles["Normal"]))
    story.append(Spacer(1, 12))

    for number, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        selected_ids = set(answers.get(question_id, []))
        correct_ids = {str(value) for value in question["correct_choice_ids"]}
        status = "Correct" if selected_ids == correct_ids else "Incorrect"
        story.append(
            Paragraph(
                f"<b>Question {number} ({status})</b><br/>{escape(str(question['stem']))}",
                styles["BodyText"],
            )
        )
        choice_rationales = _choice_rationales(question)
        for index, choice in enumerate(question["choices"]):
            display_letter = chr(ord("A") + index)
            markers: list[str] = []
            if choice["id"] in selected_ids:
                markers.append("selected")
            if choice["id"] in correct_ids:
                markers.append("correct")
            suffix = f" ({', '.join(markers)})" if markers else ""
            story.append(
                Paragraph(
                    f"{display_letter}. {escape(str(choice['text']))}{escape(suffix)}",
                    styles["BodyText"],
                )
            )
            if choice_rationales is not None:
                story.append(
                    Paragraph(
                        f"<b>Why:</b> {escape(choice_rationales[str(choice['id'])])}",
                        rationale_style,
                    )
                )
        if choice_rationales is None:
            story.append(
                Paragraph(
                    f"<b>Explanation:</b> {escape(str(question['explanation']))}",
                    explanation_style,
                )
            )
        story.append(Spacer(1, 14))

    document.build(story)
    return buffer.getvalue()
