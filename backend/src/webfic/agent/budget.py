"""Limits of one agent run. A run that reaches one stops without an answer: for the
verify agent that leaves the issue as reported (unverified), never dismissed."""

from decimal import Decimal

from pydantic import BaseModel

from webfic.llm.base import Usage


class Budget(BaseModel):
    max_turns: int = 10  # model calls
    max_input_tokens: int = 80_000  # summed over turns, cached ones included
    max_cost_usd: Decimal = Decimal("0.05")


class Spend(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)

    def add_turn(self, usage: Usage, cost: Decimal) -> None:
        self.turns += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += cost

    def exceeded(self, budget: Budget) -> str | None:
        """Which limit is used up, checked before the next turn."""
        if self.turns >= budget.max_turns:
            return f"已用满 {budget.max_turns} 轮"
        if self.input_tokens >= budget.max_input_tokens:
            return f"输入 token 已达 {self.input_tokens}（上限 {budget.max_input_tokens}）"
        if self.cost_usd >= budget.max_cost_usd:
            return f"花费已达 ${self.cost_usd:.4f}（上限 ${budget.max_cost_usd}）"
        return None
