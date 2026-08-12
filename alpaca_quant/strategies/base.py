"""Shared interfaces for independently testable strategy modules."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence


class Signal(str, Enum):
    """Portfolio-state changes that a long-only strategy can request."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DecisionLike(Protocol):
    """The minimum decision shape required by the backtest engine."""

    signal: Signal
    reason: str


class BacktestStrategy(Protocol):
    """Interface implemented by every strategy available to the backtester."""

    name: str

    @property
    def minimum_history(self) -> int:
        """Return the minimum number of completed bars required for a signal."""

    def evaluate(
        self,
        closes: Sequence[float],
        currently_long: bool,
        entry_price: float | None = None,
    ) -> DecisionLike:
        """Evaluate completed closes and optional current-position entry price."""
