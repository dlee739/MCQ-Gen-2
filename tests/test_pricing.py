from __future__ import annotations

import pytest

from mcqgen2.pricing import MODEL_PRICING, TokenUsage, calculate_cost, pricing_rows


def test_all_requested_models_have_pricing() -> None:
    assert set(MODEL_PRICING) == {
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }
    assert len(pricing_rows()) == 5


def test_cost_uses_each_usage_category() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000,
        cached_input_tokens=200_000,
        cache_write_tokens=100_000,
        output_tokens=50_000,
        reasoning_tokens=10_000,
        total_tokens=1_050_000,
    )
    cost = calculate_cost("gpt-6-sol", usage)

    # Long context: 700k * $4 + 200k * $0.40 + 100k * $5 + 50k * $15.
    assert cost.tier == "long"
    assert cost.ordinary_input_cost == pytest.approx(2.8)
    assert cost.cached_input_cost == pytest.approx(0.08)
    assert cost.cache_write_cost == pytest.approx(0.5)
    assert cost.output_cost == pytest.approx(0.75)
    assert cost.total_cost == pytest.approx(4.13)


@pytest.mark.parametrize(
    ("input_tokens", "expected_tier"),
    [(272_000, "short"), (272_001, "long")],
)
def test_long_context_boundary(input_tokens: int, expected_tier: str) -> None:
    usage = TokenUsage(input_tokens, 0, 0, 0, 0, input_tokens)
    assert calculate_cost("gpt-6-luna", usage).tier == expected_tier


def test_missing_detail_categories_can_be_zero() -> None:
    usage = TokenUsage(1_000, 0, 0, 200, 0, 1_200)
    cost = calculate_cost("gpt-6-luna", usage)
    assert cost.total_cost == pytest.approx(0.0002)
