"""Central strategy registry used by command-line and backtest workflows."""

from __future__ import annotations

from decimal import Decimal

from alpaca_quant.strategies.base import BacktestStrategy
from alpaca_quant.strategies.sma import SmaStrategy
from alpaca_quant.strategies.sp500_candidate import (
    Sp500CandidateProfile,
    Sp500CandidateStrategy,
)


def available_strategy_names() -> tuple[str, ...]:
    """Return stable command-line names for all registered strategies."""

    return ("sma", "sp500-candidate")


def create_strategy(
    name: str,
    *,
    short_window: int = 20,
    long_window: int = 50,
    candidate_profile: Sp500CandidateProfile | None = None,
    take_profit_fraction: Decimal = Decimal("0.20"),
    trend_window: int = 50,
) -> BacktestStrategy:
    """Build a registered strategy from its command-line name and parameters."""

    normalized_name = name.strip().lower()
    if normalized_name == "sma":
        return SmaStrategy(
            short_window=short_window,
            long_window=long_window,
        )

    if normalized_name == "sp500-candidate":
        if candidate_profile is None:
            raise ValueError(
                "The sp500-candidate strategy requires a candidate profile."
            )
        return Sp500CandidateStrategy(
            profile=candidate_profile,
            take_profit_fraction=take_profit_fraction,
            trend_window=trend_window,
        )

    supported_names = ", ".join(available_strategy_names())
    raise ValueError(
        f"Unknown strategy '{name}'. Available strategies: {supported_names}."
    )
