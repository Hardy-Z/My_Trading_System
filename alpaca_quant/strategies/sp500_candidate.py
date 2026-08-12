"""Potential S&P 500 entrant strategy with a fixed profit-taking target."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Sequence

from alpaca_quant.strategies.base import Signal
from alpaca_quant.strategies.sma import simple_moving_average


# This is the methodology threshold in the July 2026 S&P U.S. Indices document.
# S&P DJI reviews the guideline quarterly, so it remains configurable per profile.
DEFAULT_MARKET_CAP_THRESHOLD_USD = Decimal("22700000000")


@dataclass(frozen=True)
class Sp500CandidateProfile:
    """Point-in-time fundamentals used to approximate index eligibility.

    Sector balance and final committee judgment cannot be represented by a
    mechanical rule, so passing this profile is never a guarantee of inclusion.
    """

    company_name: str
    symbol: str
    as_of_date: date
    is_us_company: bool
    eligible_exchange: bool
    eligible_security_type: bool
    already_in_sp500: bool
    total_market_cap_usd: Decimal
    security_float_adjusted_market_cap_usd: Decimal
    investable_weight_factor: Decimal
    minimum_monthly_volume_last_six_months: Decimal
    float_adjusted_liquidity_ratio: Decimal
    latest_quarter_gaap_net_income: Decimal
    trailing_four_quarters_gaap_net_income: Decimal
    months_since_ipo: int
    market_cap_threshold_usd: Decimal = DEFAULT_MARKET_CAP_THRESHOLD_USD

    def __post_init__(self) -> None:
        """Reject incomplete or structurally impossible candidate information."""

        if not self.company_name.strip():
            raise ValueError("Candidate company_name is required.")
        if not self.symbol.strip():
            raise ValueError("Candidate symbol is required.")
        non_negative_values = (
            self.total_market_cap_usd,
            self.security_float_adjusted_market_cap_usd,
            self.investable_weight_factor,
            self.minimum_monthly_volume_last_six_months,
            self.float_adjusted_liquidity_ratio,
            self.market_cap_threshold_usd,
        )
        if not all(value.is_finite() and value >= 0 for value in non_negative_values):
            raise ValueError("Candidate size, float, and liquidity values must be finite and non-negative.")
        income_values = (
            self.latest_quarter_gaap_net_income,
            self.trailing_four_quarters_gaap_net_income,
        )
        if not all(value.is_finite() for value in income_values):
            raise ValueError("Candidate income values must be finite.")
        if self.investable_weight_factor > Decimal("1"):
            raise ValueError("investable_weight_factor cannot exceed 1.")
        if self.months_since_ipo < 0:
            raise ValueError("months_since_ipo cannot be negative.")
        if self.market_cap_threshold_usd <= 0:
            raise ValueError("market_cap_threshold_usd must be positive.")

    @classmethod
    def from_json_file(cls, profile_path: Path) -> "Sp500CandidateProfile":
        """Load and validate a candidate profile from a UTF-8 JSON document."""

        if not profile_path.is_file():
            raise FileNotFoundError(f"Candidate profile does not exist: {profile_path}")
        try:
            raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Candidate profile is not valid JSON: {exc}") from exc
        if not isinstance(raw_profile, dict):
            raise ValueError("Candidate profile JSON must contain one object.")

        required_fields = {
            "company_name",
            "symbol",
            "as_of_date",
            "is_us_company",
            "eligible_exchange",
            "eligible_security_type",
            "already_in_sp500",
            "total_market_cap_usd",
            "security_float_adjusted_market_cap_usd",
            "investable_weight_factor",
            "minimum_monthly_volume_last_six_months",
            "float_adjusted_liquidity_ratio",
            "latest_quarter_gaap_net_income",
            "trailing_four_quarters_gaap_net_income",
            "months_since_ipo",
        }
        allowed_fields = required_fields | {"market_cap_threshold_usd"}
        missing_fields = sorted(required_fields - raw_profile.keys())
        if missing_fields:
            raise ValueError(
                "Candidate profile is missing fields: " + ", ".join(missing_fields)
            )
        unexpected_fields = sorted(raw_profile.keys() - allowed_fields)
        if unexpected_fields:
            raise ValueError(
                "Candidate profile has unexpected fields: "
                + ", ".join(unexpected_fields)
            )

        boolean_fields = (
            "is_us_company",
            "eligible_exchange",
            "eligible_security_type",
            "already_in_sp500",
        )
        for field_name in boolean_fields:
            if not isinstance(raw_profile[field_name], bool):
                raise ValueError(f"Candidate field '{field_name}' must be true or false.")
        if (
            not isinstance(raw_profile["months_since_ipo"], int)
            or isinstance(raw_profile["months_since_ipo"], bool)
        ):
            raise ValueError("Candidate field 'months_since_ipo' must be an integer.")

        decimal_fields = (
            "total_market_cap_usd",
            "security_float_adjusted_market_cap_usd",
            "investable_weight_factor",
            "minimum_monthly_volume_last_six_months",
            "float_adjusted_liquidity_ratio",
            "latest_quarter_gaap_net_income",
            "trailing_four_quarters_gaap_net_income",
            "market_cap_threshold_usd",
        )
        converted_profile: dict[str, Any] = dict(raw_profile)
        for field_name in decimal_fields:
            if field_name not in converted_profile:
                continue
            try:
                converted_profile[field_name] = Decimal(str(converted_profile[field_name]))
            except InvalidOperation as exc:
                raise ValueError(
                    f"Candidate field '{field_name}' must be a decimal number."
                ) from exc
        try:
            converted_profile["as_of_date"] = date.fromisoformat(
                str(converted_profile["as_of_date"])
            )
        except ValueError as exc:
            raise ValueError("Candidate as_of_date must use YYYY-MM-DD format.") from exc
        converted_profile["symbol"] = str(converted_profile["symbol"]).strip().upper()
        return cls(**converted_profile)


@dataclass(frozen=True)
class CandidateEligibility:
    """Mechanical approximation of the published outside-addition requirements."""

    qualifies: bool
    passed_rules: tuple[str, ...]
    failed_rules: tuple[str, ...]


def evaluate_candidate_eligibility(
    profile: Sp500CandidateProfile,
) -> CandidateEligibility:
    """Evaluate measurable S&P 500 eligibility rules from one dated profile."""

    float_market_cap_threshold = profile.market_cap_threshold_usd * Decimal("0.50")
    rules = (
        (not profile.already_in_sp500, "not_already_in_sp500"),
        (profile.is_us_company, "us_company"),
        (profile.eligible_exchange, "eligible_us_exchange"),
        (profile.eligible_security_type, "eligible_security_type"),
        (
            profile.total_market_cap_usd >= profile.market_cap_threshold_usd,
            "total_market_cap",
        ),
        (
            profile.security_float_adjusted_market_cap_usd
            >= float_market_cap_threshold,
            "security_float_adjusted_market_cap",
        ),
        (profile.investable_weight_factor >= Decimal("0.10"), "public_float_iwf"),
        (
            profile.minimum_monthly_volume_last_six_months >= Decimal("250000"),
            "six_month_minimum_monthly_volume",
        ),
        (
            profile.float_adjusted_liquidity_ratio >= Decimal("0.75"),
            "float_adjusted_liquidity_ratio",
        ),
        (
            profile.latest_quarter_gaap_net_income > 0,
            "positive_latest_quarter_income",
        ),
        (
            profile.trailing_four_quarters_gaap_net_income > 0,
            "positive_trailing_four_quarter_income",
        ),
        (profile.months_since_ipo >= 12, "minimum_ipo_trading_history"),
    )
    passed_rules = tuple(rule_name for passed, rule_name in rules if passed)
    failed_rules = tuple(rule_name for passed, rule_name in rules if not passed)
    return CandidateEligibility(
        qualifies=not failed_rules,
        passed_rules=passed_rules,
        failed_rules=failed_rules,
    )


@dataclass(frozen=True)
class Sp500CandidateDecision:
    """Candidate signal plus the trend and profit-target values behind it."""

    signal: Signal
    latest_close: float
    trend_sma: float
    target_price: float | None
    eligibility: CandidateEligibility
    reason: str


@dataclass(frozen=True)
class Sp500CandidateStrategy:
    """Buy qualified candidates in an uptrend and sell at fixed appreciation."""

    profile: Sp500CandidateProfile
    take_profit_fraction: Decimal = Decimal("0.20")
    trend_window: int = 50
    name: str = "sp500-candidate"

    def __post_init__(self) -> None:
        """Validate the technical confirmation and fixed-exit parameters."""

        if self.trend_window < 2:
            raise ValueError("Candidate trend_window must be at least 2.")
        if not self.take_profit_fraction.is_finite() or self.take_profit_fraction <= 0:
            raise ValueError("Candidate take_profit_fraction must be positive.")

    @property
    def minimum_history(self) -> int:
        """Return the completed closes needed for trend confirmation."""

        return self.trend_window

    def evaluate(
        self,
        closes: Sequence[float],
        currently_long: bool,
        entry_price: float | None = None,
    ) -> Sp500CandidateDecision:
        """Create a long-only entry or fixed-profit exit signal.

        The profit target is calculated from actual simulated or paper-account
        entry cost. A close at or above the target creates a sell signal that the
        execution layer fills at the next available market open.
        """

        latest_close = float(closes[-1])
        trend_sma = simple_moving_average(closes, self.trend_window)
        eligibility = evaluate_candidate_eligibility(self.profile)

        if currently_long:
            if entry_price is None or entry_price <= 0:
                raise ValueError("A positive entry_price is required for an open position.")
            target_price = float(
                Decimal(str(entry_price)) * (Decimal("1") + self.take_profit_fraction)
            )
            if latest_close >= target_price:
                return Sp500CandidateDecision(
                    signal=Signal.SELL,
                    latest_close=latest_close,
                    trend_sma=trend_sma,
                    target_price=target_price,
                    eligibility=eligibility,
                    reason=(
                        f"Close {latest_close:.4f} reached the fixed target "
                        f"{target_price:.4f}."
                    ),
                )
            return Sp500CandidateDecision(
                signal=Signal.HOLD,
                latest_close=latest_close,
                trend_sma=trend_sma,
                target_price=target_price,
                eligibility=eligibility,
                reason=f"The fixed target {target_price:.4f} has not been reached.",
            )

        if not eligibility.qualifies:
            failed_text = ", ".join(eligibility.failed_rules)
            return Sp500CandidateDecision(
                signal=Signal.HOLD,
                latest_close=latest_close,
                trend_sma=trend_sma,
                target_price=None,
                eligibility=eligibility,
                reason=f"Candidate profile failed: {failed_text}.",
            )

        if latest_close <= trend_sma:
            return Sp500CandidateDecision(
                signal=Signal.HOLD,
                latest_close=latest_close,
                trend_sma=trend_sma,
                target_price=None,
                eligibility=eligibility,
                reason=(
                    f"Candidate passed the eligibility screen, but close "
                    f"{latest_close:.4f} is not above the {self.trend_window}-bar "
                    f"SMA {trend_sma:.4f}."
                ),
            )

        return Sp500CandidateDecision(
            signal=Signal.BUY,
            latest_close=latest_close,
            trend_sma=trend_sma,
            target_price=None,
            eligibility=eligibility,
            reason=(
                "Candidate passed the mechanical eligibility screen and the "
                "latest close is above the trend SMA."
            ),
        )
