from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRICING_VERIFIED_AT = "2026-09-25"
LONG_CONTEXT_THRESHOLD = 272_000
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


@dataclass(frozen=True)
class TokenRates:
    input: float
    cached_input: float
    cache_write: float
    output: float


@dataclass(frozen=True)
class ModelPricing:
    short: TokenRates
    long: TokenRates
    note: str = ""


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-6-sol": ModelPricing(
        short=TokenRates(2.00, 0.20, 2.50, 10.00),
        long=TokenRates(4.00, 0.40, 5.00, 15.00),
    ),
    "gpt-6-luna": ModelPricing(
        short=TokenRates(0.10, 0.01, 0.125, 0.50),
        long=TokenRates(0.20, 0.02, 0.25, 0.75),
    ),
    "gpt-5.6-sol": ModelPricing(
        short=TokenRates(4.00, 0.40, 5.00, 20.00),
        long=TokenRates(8.00, 0.80, 10.00, 30.00),
        note="Promotional pricing documented through at least 2026-11-21.",
    ),
    "gpt-5.6-terra": ModelPricing(
        short=TokenRates(2.00, 0.20, 2.50, 12.00),
        long=TokenRates(4.00, 0.40, 5.00, 18.00),
    ),
    "gpt-5.6-luna": ModelPricing(
        short=TokenRates(0.20, 0.02, 0.25, 1.20),
        long=TokenRates(0.40, 0.04, 0.50, 1.80),
    ),
}


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int

    @property
    def ordinary_input_tokens(self) -> int:
        return max(
            0,
            self.input_tokens - self.cached_input_tokens - self.cache_write_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CostBreakdown:
    tier: str
    ordinary_input_cost: float
    cached_input_cost: float
    cache_write_cost: float
    output_cost: float
    total_cost: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _value(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def usage_from_response(response: Any) -> TokenUsage:
    raw = _value(response, "usage", None)
    if raw is None:
        raise ValueError("The API response did not include token usage.")
    input_details = _value(raw, "input_tokens_details", None)
    output_details = _value(raw, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=int(_value(raw, "input_tokens", 0) or 0),
        cached_input_tokens=int(_value(input_details, "cached_tokens", 0) or 0),
        cache_write_tokens=int(_value(input_details, "cache_write_tokens", 0) or 0),
        output_tokens=int(_value(raw, "output_tokens", 0) or 0),
        reasoning_tokens=int(_value(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=int(_value(raw, "total_tokens", 0) or 0),
    )


def calculate_cost(model: str, usage: TokenUsage) -> CostBreakdown:
    pricing = MODEL_PRICING[model]
    is_long = usage.input_tokens > LONG_CONTEXT_THRESHOLD
    rates = pricing.long if is_long else pricing.short
    per_million = 1_000_000

    ordinary = usage.ordinary_input_tokens * rates.input / per_million
    cached = usage.cached_input_tokens * rates.cached_input / per_million
    cache_write = usage.cache_write_tokens * rates.cache_write / per_million
    output = usage.output_tokens * rates.output / per_million
    return CostBreakdown(
        tier="long" if is_long else "short",
        ordinary_input_cost=ordinary,
        cached_input_cost=cached,
        cache_write_cost=cache_write,
        output_cost=output,
        total_cost=ordinary + cached + cache_write + output,
    )


def pricing_snapshot(model: str) -> dict[str, Any]:
    pricing = MODEL_PRICING[model]
    return {
        "model": model,
        "verified_at": PRICING_VERIFIED_AT,
        "source": PRICING_SOURCE,
        "threshold": LONG_CONTEXT_THRESHOLD,
        "short": asdict(pricing.short),
        "long": asdict(pricing.long),
        "note": pricing.note,
    }


def pricing_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model, pricing in MODEL_PRICING.items():
        rows.append(
            {
                "Model": model,
                "Short input": f"${pricing.short.input:g}",
                "Short cached": f"${pricing.short.cached_input:g}",
                "Short cache write": f"${pricing.short.cache_write:g}",
                "Short output": f"${pricing.short.output:g}",
                "Long input": f"${pricing.long.input:g}",
                "Long cached": f"${pricing.long.cached_input:g}",
                "Long cache write": f"${pricing.long.cache_write:g}",
                "Long output": f"${pricing.long.output:g}",
            }
        )
    return rows
