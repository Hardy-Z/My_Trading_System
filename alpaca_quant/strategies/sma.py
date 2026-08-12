"""Pure strategy logic for a long-only simple moving-average model."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Sequence

from alpaca_quant.strategies.base import Signal


@dataclass(frozen=True)
class StrategyDecision:
    """A strategy signal together with the values that produced it."""

    signal: Signal
    latest_close: float
    short_sma: float
    long_sma: float
    reason: str


@dataclass(frozen=True)
class SmaStrategy:
    """Configurable SMA strategy implementation used by the backtest registry."""

    short_window: int = 20
    long_window: int = 50
    name: str = "sma"

    def __post_init__(self) -> None:
        """Reject invalid windows when the strategy object is constructed."""

        if self.short_window < 2:
            raise ValueError("The short window must be at least 2.")
        if self.long_window <= self.short_window:
            raise ValueError("The long window must be greater than the short window.")

    @property
    def minimum_history(self) -> int:
        """Return the number of closes needed to calculate the long SMA."""

        return self.long_window

    def evaluate(
        self,
        closes: Sequence[float],
        currently_long: bool,
        entry_price: float | None = None,
    ) -> StrategyDecision:
        """Evaluate the configured SMA regime for the current position state."""

        # The SMA regime depends on position direction, not the position cost basis.
        del entry_price
        return evaluate_sma_regime(
            closes=closes,
            short_window=self.short_window,
            long_window=self.long_window,
            currently_long=currently_long,
        )


def simple_moving_average(values: Sequence[float], window: int) -> float:
    """Return the arithmetic mean of the final ``window`` observations."""

    if window < 1:
        raise ValueError("The moving-average window must be positive.")
    if len(values) < window:
        raise ValueError(
            f"At least {window} observations are required; received {len(values)}."
        )

    selected_values = [float(value) for value in values[-window:]]
    if not all(isfinite(value) and value > 0 for value in selected_values):
        raise ValueError("Close prices must be finite positive numbers.")

    # fsum provides a more accurate floating-point sum than the built-in sum.
    return fsum(selected_values) / window


def evaluate_sma_regime(
    closes: Sequence[float],
    short_window: int,
    long_window: int,
    currently_long: bool,
) -> StrategyDecision:
    """Map the current SMA regime and account position to an idempotent action.

    The desired state is long when the short SMA is above the long SMA and cash
    when it is below. The strategy acts only when the account state does not
    match that desired state. Equal averages produce HOLD to avoid unnecessary
    turnover around a flat boundary.
    """

    if short_window >= long_window:
        raise ValueError("The short window must be smaller than the long window.")

    short_sma = simple_moving_average(closes, short_window)
    long_sma = simple_moving_average(closes, long_window)
    latest_close = float(closes[-1])

    if short_sma > long_sma and not currently_long:
        return StrategyDecision(
            signal=Signal.BUY,
            latest_close=latest_close,
            short_sma=short_sma,
            long_sma=long_sma,
            reason="The short SMA is above the long SMA and the account is flat.",
        )

    if short_sma < long_sma and currently_long:
        return StrategyDecision(
            signal=Signal.SELL,
            latest_close=latest_close,
            short_sma=short_sma,
            long_sma=long_sma,
            reason="The short SMA is below the long SMA and a long position exists.",
        )

    if short_sma == long_sma:
        reason = "The moving averages are equal, so the strategy keeps its state."
    elif currently_long:
        reason = "The bullish regime is already represented by the long position."
    else:
        reason = "The bearish regime is already represented by a flat position."

    return StrategyDecision(
        signal=Signal.HOLD,
        latest_close=latest_close,
        short_sma=short_sma,
        long_sma=long_sma,
        reason=reason,
    )
