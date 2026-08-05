"""Small character-budget helpers for context construction."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ContextBudget


@dataclass(slots=True)
class BudgetResult:
    text: str
    truncated: bool = False


def truncate_text(text: str, max_chars: int) -> BudgetResult:
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    if len(text) <= max_chars:
        return BudgetResult(text=text)
    if max_chars == 0:
        return BudgetResult(text="", truncated=True)
    marker = f"\n...<truncated {len(text) - max_chars} chars>"
    keep = max(max_chars - len(marker), 0)
    return BudgetResult(text=text[:keep] + marker, truncated=True)


__all__ = ["BudgetResult", "ContextBudget", "truncate_text"]
