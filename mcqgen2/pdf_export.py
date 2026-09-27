from __future__ import annotations

from io import BytesIO
from typing import Any, Collection, Mapping, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from mcqgen2.explanations import choice_rationales


def build_results_pdf(
    *,
    title: str,
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Sequence[str]],
    score: int,
    total: int,
    metadata: str = "",
    flagged_question_ids: Collection[str] = (),
    skipped_question_ids: Collection[str] = (),
) -> bytes:
    flagged_ids = {str(value) for value in flagged_question_ids}
    skipped_ids = {str(value) for value in skipped_question_ids}
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
    story.append(
        Paragraph(
            f"Flagged: {len(flagged_ids)} &nbsp;&nbsp; Skipped: {len(skipped_ids)}",
            styles["Normal"],
        )
    )
    if metadata:
        story.append(Paragraph(escape(metadata), styles["Normal"]))
    story.append(Spacer(1, 12))

    for number, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        selected_ids = set(answers.get(question_id, []))
        correct_ids = {str(value) for value in question["correct_choice_ids"]}
        status_parts = ["Correct" if selected_ids == correct_ids else "Incorrect"]
        if question_id in flagged_ids:
            status_parts.append("Flagged")
        if question_id in skipped_ids:
            status_parts.append("Skipped")
        status = "; ".join(status_parts)
        story.append(
            Paragraph(
                f"<b>Question {number} ({status})</b><br/>{escape(str(question['stem']))}",
                styles["BodyText"],
            )
        )
        rationales = choice_rationales(question)
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
            if rationales is not None:
                story.append(
                    Paragraph(
                        f"<b>Why:</b> {escape(rationales[str(choice['id'])])}",
                        rationale_style,
                    )
                )
        if rationales is None:
            story.append(
                Paragraph(
                    f"<b>Explanation:</b> {escape(str(question['explanation']))}",
                    explanation_style,
                )
            )
        story.append(Spacer(1, 14))

    document.build(story)
    return buffer.getvalue()
