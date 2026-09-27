from __future__ import annotations

from typing import Any, Mapping


def choice_rationales(question: Mapping[str, Any]) -> dict[str, str] | None:
    """Return one rationale per choice, including a fallback for older saved sets."""
    structured: dict[str, str] = {}
    for choice in question["choices"]:
        rationale = choice.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            break
        structured[str(choice["id"])] = rationale.strip()
    else:
        return structured

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
