"""Offline tests for the potential S&P 500 entrant strategy."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from alpaca_quant.strategies.base import Signal
from alpaca_quant.strategies.registry import create_strategy
from alpaca_quant.strategies.sp500_candidate import (
    Sp500CandidateProfile,
    Sp500CandidateStrategy,
    evaluate_candidate_eligibility,
)


def _qualifying_profile() -> Sp500CandidateProfile:
    """Return a fictional candidate that passes every mechanical rule."""

    return Sp500CandidateProfile(
        company_name="Test Candidate",
        symbol="TEST",
        as_of_date=date(2024, 1, 1),
        is_us_company=True,
        eligible_exchange=True,
        eligible_security_type=True,
        already_in_sp500=False,
        total_market_cap_usd=Decimal("30000000000"),
        security_float_adjusted_market_cap_usd=Decimal("20000000000"),
        investable_weight_factor=Decimal("0.70"),
        minimum_monthly_volume_last_six_months=Decimal("1000000"),
        float_adjusted_liquidity_ratio=Decimal("1.20"),
        latest_quarter_gaap_net_income=Decimal("250000000"),
        trailing_four_quarters_gaap_net_income=Decimal("900000000"),
        months_since_ipo=36,
    )


class CandidateEligibilityTests(unittest.TestCase):
    """Verify measurable eligibility rules and profile parsing."""

    def test_qualifying_profile_passes_all_rules(self) -> None:
        """A complete profile above every threshold should pass the screen."""

        result = evaluate_candidate_eligibility(_qualifying_profile())

        self.assertTrue(result.qualifies)
        self.assertEqual(result.failed_rules, ())

    def test_unprofitable_profile_fails_both_income_rules(self) -> None:
        """Both the latest quarter and trailing-four-quarter income must be positive."""

        profile = replace(
            _qualifying_profile(),
            latest_quarter_gaap_net_income=Decimal("-1"),
            trailing_four_quarters_gaap_net_income=Decimal("-2"),
        )
        result = evaluate_candidate_eligibility(profile)

        self.assertFalse(result.qualifies)
        self.assertIn("positive_latest_quarter_income", result.failed_rules)
        self.assertIn("positive_trailing_four_quarter_income", result.failed_rules)

    def test_loads_decimal_values_from_json_strings(self) -> None:
        """JSON string numbers should retain exact decimal semantics."""

        profile_text = """{
          "company_name": "Test Candidate",
          "symbol": "test",
          "as_of_date": "2024-01-01",
          "is_us_company": true,
          "eligible_exchange": true,
          "eligible_security_type": true,
          "already_in_sp500": false,
          "total_market_cap_usd": "30000000000",
          "security_float_adjusted_market_cap_usd": "20000000000",
          "investable_weight_factor": "0.70",
          "minimum_monthly_volume_last_six_months": "1000000",
          "float_adjusted_liquidity_ratio": "1.20",
          "latest_quarter_gaap_net_income": "250000000",
          "trailing_four_quarters_gaap_net_income": "900000000",
          "months_since_ipo": 36
        }"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(profile_text, encoding="utf-8")
            profile = Sp500CandidateProfile.from_json_file(profile_path)

        self.assertEqual(profile.symbol, "TEST")
        self.assertEqual(profile.investable_weight_factor, Decimal("0.70"))

    def test_rejects_non_integer_ipo_age(self) -> None:
        """A quoted IPO age must not pass as a structurally valid profile."""

        profile_text = """{
          "company_name": "Test Candidate",
          "symbol": "TEST",
          "as_of_date": "2024-01-01",
          "is_us_company": true,
          "eligible_exchange": true,
          "eligible_security_type": true,
          "already_in_sp500": false,
          "total_market_cap_usd": "30000000000",
          "security_float_adjusted_market_cap_usd": "20000000000",
          "investable_weight_factor": "0.70",
          "minimum_monthly_volume_last_six_months": "1000000",
          "float_adjusted_liquidity_ratio": "1.20",
          "latest_quarter_gaap_net_income": "250000000",
          "trailing_four_quarters_gaap_net_income": "900000000",
          "months_since_ipo": "36"
        }"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(profile_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                Sp500CandidateProfile.from_json_file(profile_path)


class Sp500CandidateStrategyTests(unittest.TestCase):
    """Verify entry confirmation and the fixed appreciation exit."""

    def test_buys_a_qualified_candidate_above_its_trend(self) -> None:
        """Eligibility and a close above the trend SMA should create a buy."""

        strategy = Sp500CandidateStrategy(
            profile=_qualifying_profile(),
            trend_window=3,
        )
        decision = strategy.evaluate(
            closes=[100, 102, 105],
            currently_long=False,
        )

        self.assertIs(decision.signal, Signal.BUY)

    def test_does_not_buy_a_candidate_that_fails_the_screen(self) -> None:
        """Technical momentum must not override failed fundamental eligibility."""

        profile = replace(
            _qualifying_profile(),
            total_market_cap_usd=Decimal("10000000000"),
        )
        strategy = Sp500CandidateStrategy(profile=profile, trend_window=3)
        decision = strategy.evaluate(
            closes=[100, 102, 105],
            currently_long=False,
        )

        self.assertIs(decision.signal, Signal.HOLD)
        self.assertIn("total_market_cap", decision.eligibility.failed_rules)

    def test_sells_when_the_fixed_target_is_reached(self) -> None:
        """A 20 percent strategy target should sell at a 120 close from a 100 entry."""

        strategy = Sp500CandidateStrategy(
            profile=_qualifying_profile(),
            take_profit_fraction=Decimal("0.20"),
            trend_window=3,
        )
        decision = strategy.evaluate(
            closes=[110, 115, 120],
            currently_long=True,
            entry_price=100,
        )

        self.assertIs(decision.signal, Signal.SELL)
        self.assertEqual(decision.target_price, 120.0)

    def test_registry_requires_a_candidate_profile(self) -> None:
        """The registry must not construct a candidate strategy without fundamentals."""

        with self.assertRaises(ValueError):
            create_strategy("sp500-candidate")


if __name__ == "__main__":
    unittest.main()
