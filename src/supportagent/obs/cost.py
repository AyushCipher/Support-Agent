from __future__ import annotations

from ..config import PRICING
from ..harness.types import Usage


def cost_usd(model: str, usage: Usage) -> float:
    pin, pout = PRICING.get(model, (3.0, 15.0))
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000
