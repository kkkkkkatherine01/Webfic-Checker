"""Per-model prices (USD per million tokens). Checked 2026-09; update when providers
change prices. DeepSeek figures are peak-hour rates, i.e. the conservative estimate."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from webfic.llm.base import Usage

log = logging.getLogger(__name__)

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    cached_input: Decimal
    output: Decimal


PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(Decimal("0.30"), Decimal("0.006"), Decimal("1.20")),
    "deepseek-v4-pro": ModelPrice(Decimal("1.32"), Decimal("0.044"), Decimal("3.96")),
    "claude-haiku-4-5": ModelPrice(Decimal("1.00"), Decimal("0.10"), Decimal("5.00")),
    "claude-sonnet-5": ModelPrice(Decimal("2.00"), Decimal("0.20"), Decimal("10.00")),
}


def cost_of(model: str, usage: Usage) -> Decimal:
    price = PRICES.get(model)
    if price is None:
        log.warning("No price configured for model %s; recording cost as 0", model)
        return Decimal(0)
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    total = (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    )
    return total / _MILLION
