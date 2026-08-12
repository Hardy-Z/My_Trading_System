"""Pure, independently testable trading-strategy implementations."""

from alpaca_quant.strategies.base import BacktestStrategy, Signal
from alpaca_quant.strategies.sma import (
    SmaStrategy,
    StrategyDecision,
    evaluate_sma_regime,
    simple_moving_average,
)
from alpaca_quant.strategies.sp500_candidate import (
    CandidateEligibility,
    Sp500CandidateDecision,
    Sp500CandidateProfile,
    Sp500CandidateStrategy,
    evaluate_candidate_eligibility,
)

__all__ = [
    "BacktestStrategy",
    "Signal",
    "SmaStrategy",
    "CandidateEligibility",
    "Sp500CandidateDecision",
    "Sp500CandidateProfile",
    "Sp500CandidateStrategy",
    "StrategyDecision",
    "evaluate_candidate_eligibility",
    "evaluate_sma_regime",
    "simple_moving_average",
]
