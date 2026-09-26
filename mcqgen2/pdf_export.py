from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_results_pdf(
    *,
    title: str,
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
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
    story: list[Any] = [Paragraph(escape(title), styles["Title"])]
    story.append(Paragraph(f"Score: {score} / {total}", styles["Heading2"]))
    if metadata:
        story.append(Paragraph(escape(metadata), styles["Normal"]))
    story.append(Spacer(1, 12))

    for number, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        selected_id = answers.get(question_id, "")
        correct_id = str(question["correct_choice_id"])
        status = "Correct" if selected_id == correct_id else "Incorrect"
        story.append(
            Paragraph(
                f"<b>Question {number} ({status})</b><br/>{escape(str(question['stem']))}",
                styles["BodyText"],
            )
        )
        for index, choice in enumerate(question["choices"]):
            display_letter = chr(ord("A") + index)
            markers: list[str] = []
            if choice["id"] == selected_id:
                markers.append("selected")
            if choice["id"] == correct_id:
                markers.append("correct")
            suffix = f" ({', '.join(markers)})" if markers else ""
            story.append(
                Paragraph(
                    f"{display_letter}. {escape(str(choice['text']))}{escape(suffix)}",
                    styles["BodyText"],
                )
            )
        story.append(
            Paragraph(
                f"<b>Explanation:</b> {escape(str(question['explanation']))}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 14))

    document.build(story)
    return buffer.getvalue()
