"""Limits of one agent run. A run that reaches one stops without an answer: for the
verify agent that leaves the issue as reported (unverified), never dismissed.

The input limit is per turn, not summed over turns: every turn resends the conversation,
and providers serve the repeated prefix from their cache at a fraction of the price, so a
sum mostly counts the same text again. What costs money is capped by `max_cost_usd`; what
strains the model's attention is the size of one request.
"""

from decimal import Decimal

from pydantic import BaseModel

from webfic.llm.base import Usage


class Budget(BaseModel):
    max_turns: int = 10  # model calls
    max_turn_input_tokens: int = 60_000  # the largest single request
    max_cost_usd: Decimal = Decimal("0.05")


class Spend(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0  # summed over turns (for reports)
    output_tokens: int = 0
    last_input_tokens: int = 0
    cost_usd: Decimal = Decimal(0)

    def add_turn(self, usage: Usage, cost: Decimal) -> None:
        self.turns += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.last_input_tokens = usage.input_tokens
        self.cost_usd += cost

    def exceeded(self, budget: Budget) -> str | None:
        """Which limit is used up, checked before the next turn."""
        if self.turns >= budget.max_turns:
            return f"已用满 {budget.max_turns} 轮"
        if self.last_input_tokens >= budget.max_turn_input_tokens:
            return (
                f"单轮输入已达 {self.last_input_tokens} token"
                f"（上限 {budget.max_turn_input_tokens}）"
            )
        if self.cost_usd >= budget.max_cost_usd:
            return f"花费已达 ${self.cost_usd:.4f}（上限 ${budget.max_cost_usd}）"
        return None
