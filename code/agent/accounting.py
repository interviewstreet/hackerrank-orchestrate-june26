"""Run-level cost and token accounting.

Estimated list-price costs are derived from the pricing snapshot in
vision_client.py (PRICING_SNAPSHOT).  Free-quota usage is still reported
as estimated list cost, labelled with "estimated" to avoid claiming it
was actually charged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from code.agent.models import RowStats


@dataclass
class RunAccounting:
    """Aggregates per-row RowStats across an entire pipeline run."""
    total_rows: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    total_retries: int = 0
    total_errors: int = 0
    total_estimated_input_cost_usd: float = 0.0
    total_estimated_output_cost_usd: float = 0.0
    _has_cost: bool = field(default=False, repr=False)

    def add(self, stats: RowStats) -> None:
        self.total_rows += 1
        if stats.cache_hit:
            self.cache_hits += 1
        self.total_input_tokens += stats.input_tokens
        self.total_output_tokens += stats.output_tokens
        self.total_latency_ms += stats.latency_ms
        self.total_retries += stats.retries
        if stats.error:
            self.total_errors += 1
        if stats.estimated_input_cost_usd is not None:
            self.total_estimated_input_cost_usd += stats.estimated_input_cost_usd
            self._has_cost = True
        if stats.estimated_output_cost_usd is not None:
            self.total_estimated_output_cost_usd += stats.estimated_output_cost_usd
            self._has_cost = True

    def summary(self) -> str:
        total_cost = self.total_estimated_input_cost_usd + self.total_estimated_output_cost_usd
        lines = [
            "=== Run Accounting Summary ===",
            f"  Rows processed     : {self.total_rows}",
            f"  Cache hits         : {self.cache_hits}",
            f"  API calls          : {self.total_rows - self.cache_hits}",
            f"  Total input tokens : {self.total_input_tokens:,}",
            f"  Total output tokens: {self.total_output_tokens:,}",
            f"  Total latency      : {self.total_latency_ms / 1000:.1f}s",
            f"  Retries            : {self.total_retries}",
            f"  Errors             : {self.total_errors}",
        ]
        if self._has_cost:
            lines += [
                f"  Est. input cost    : ${self.total_estimated_input_cost_usd:.4f} USD (list price, not necessarily charged)",
                f"  Est. output cost   : ${self.total_estimated_output_cost_usd:.4f} USD (list price, not necessarily charged)",
                f"  Est. total cost    : ${total_cost:.4f} USD (list price, not necessarily charged)",
            ]
        else:
            lines.append("  Cost estimate      : not available for this model")
        lines.append("=" * 32)
        return "\n".join(lines)
