from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcqgen2.generation import GenerationError, generate_question_set, validate_pdf


def question_data(index: int) -> dict:
    return {
        "stem": f"Question {index}?",
        "choices": [
            {"id": "A", "text": f"Alpha {index}"},
            {"id": "B", "text": f"Bravo {index}"},
            {"id": "C", "text": f"Charlie {index}"},
            {"id": "D", "text": f"Delta {index}"},
        ],
        "correct_choice_id": "A",
        "explanation": f"Alpha {index} is correct; the other options are not supported.",
    }


class FakeResponses:
    def __init__(self, *, status: str = "completed") -> None:
        self.calls: list[dict] = []
        self.status = status

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        response_model = kwargs["text_format"]
        if self.status != "completed":
            return SimpleNamespace(
                status=self.status,
                incomplete_details={"reason": "max_output_tokens"},
                output_parsed=None,
                usage=None,
            )
        count = response_model.model_json_schema()["properties"]["questions"]["minItems"]
        parsed = response_model.model_validate(
            {"questions": [question_data(i) for i in range(count)]}
        )
        usage = SimpleNamespace(
            input_tokens=12_000,
            input_tokens_details=SimpleNamespace(cached_tokens=1_000, cache_write_tokens=0),
            output_tokens=2_000,
            output_tokens_details=SimpleNamespace(reasoning_tokens=300),
            total_tokens=14_000,
        )
        return SimpleNamespace(status="completed", output_parsed=parsed, usage=usage)


class FakeClient:
    def __init__(self, *, status: str = "completed") -> None:
        self.responses = FakeResponses(status=status)


def test_generation_is_one_full_pdf_request() -> None:
    client = FakeClient()
    result = generate_question_set(
        client=client,
        filename="lecture.pdf",
        pdf_bytes=b"%PDF-1.7\ncontent",
        model="gpt-6-luna",
        mode="high_volume",
        question_count=3,
    )

    assert len(result.questions) == 3
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["reasoning"] == {"effort": "low"}
    assert request["truncation"] == "disabled"
    assert request["store"] is False
    content = request["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["detail"] == "auto"
    assert content[0]["file_data"].startswith("data:application/pdf;base64,")
    assert "exactly 3" in content[1]["text"]
    assert result.usage.reasoning_tokens == 300


def test_high_quality_uses_high_reasoning() -> None:
    client = FakeClient()
    generate_question_set(
        client=client,
        filename="lecture.pdf",
        pdf_bytes=b"%PDF-1.7\ncontent",
        model="gpt-5.6-terra",
        mode="high_quality",
        question_count=1,
    )
    request = client.responses.calls[0]
    assert request["reasoning"] == {"effort": "high"}
    assert "multi-step" in request["input"][0]["content"][1]["text"]


def test_incomplete_generation_is_rejected() -> None:
    with pytest.raises(GenerationError, match="did not complete"):
        generate_question_set(
            client=FakeClient(status="incomplete"),
            filename="lecture.pdf",
            pdf_bytes=b"%PDF-1.7\ncontent",
            model="gpt-6-sol",
            mode="high_volume",
            question_count=1,
        )


@pytest.mark.parametrize(
    ("filename", "contents"),
    [("notes.txt", b"%PDF-1.7"), ("notes.pdf", b"not a pdf"), ("notes.pdf", b"")],
)
def test_invalid_pdf_is_rejected(filename: str, contents: bytes) -> None:
    with pytest.raises(ValueError):
        validate_pdf(filename, contents)
