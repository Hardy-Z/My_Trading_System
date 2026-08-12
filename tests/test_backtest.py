"""Offline tests for next-open execution and portfolio return accounting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import unittest

from alpaca_quant.backtesting.engine import run_backtest
from alpaca_quant.backtesting.models import BacktestConfig, MarketBar
from alpaca_quant.strategies.sma import SmaStrategy


def _bar(day: int, open_price: str, close_price: str) -> MarketBar:
    """Build a valid January 2024 test bar with a wide high-low range."""

    open_value = Decimal(open_price)
    close_value = Decimal(close_price)
    return MarketBar(
        timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
        open=open_value,
        high=max(open_value, close_value) + Decimal("1"),
        low=min(open_value, close_value) - Decimal("0.5"),
        close=close_value,
        volume=Decimal("1000"),
    )


class BacktestEngineTests(unittest.TestCase):
    """Verify signal timing, accounting, warm-up rules, and open positions."""

    def test_executes_signals_at_the_next_open(self) -> None:
        """A close signal must never fill on the same bar that created it."""

        bars = [
            _bar(1, "1", "1"),
            _bar(2, "2", "2"),
            _bar(3, "3", "3"),
            _bar(4, "4", "4"),
            _bar(5, "3", "3"),
            _bar(6, "2", "2"),
            _bar(7, "1", "1"),
        ]
        result = run_backtest(
            bars=bars,
            strategy=SmaStrategy(short_window=2, long_window=3),
            config=BacktestConfig(
                symbol="TEST",
                initial_cash=Decimal("1000"),
                allocation_fraction=Decimal("1"),
                slippage_bps=Decimal("0"),
            ),
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 7),
        )

        self.assertEqual([trade.action for trade in result.trades], ["BUY", "SELL"])
        self.assertEqual(result.trades[0].timestamp.date(), date(2024, 1, 4))
        self.assertEqual(result.trades[0].fill_price, Decimal("4"))
        self.assertEqual(result.trades[1].timestamp.date(), date(2024, 1, 7))
        self.assertEqual(result.summary["final_equity"], 250.0)
        self.assertEqual(result.summary["net_profit"], -750.0)
        self.assertEqual(result.summary["total_return_pct"], -75.0)

    def test_marks_an_open_position_to_market(self) -> None:
        """Final income must include unrealized profit from an unclosed position."""

        bars = [
            _bar(1, "10", "10"),
            _bar(2, "11", "11"),
            _bar(3, "12", "12"),
            _bar(4, "12", "13"),
            _bar(5, "13", "14"),
        ]
        result = run_backtest(
            bars=bars,
            strategy=SmaStrategy(short_window=2, long_window=3),
            config=BacktestConfig(
                symbol="TEST",
                initial_cash=Decimal("1200"),
                slippage_bps=Decimal("0"),
            ),
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 5),
        )

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].action, "BUY")
        self.assertEqual(result.summary["realized_pnl"], 0.0)
        self.assertEqual(result.summary["unrealized_pnl"], 200.0)
        self.assertEqual(result.summary["final_equity"], 1400.0)

    def test_requires_indicator_history_before_the_period(self) -> None:
        """Insufficient pre-period data must fail instead of using future bars."""

        bars = [
            _bar(1, "10", "10"),
            _bar(2, "11", "11"),
            _bar(3, "12", "12"),
        ]
        with self.assertRaises(ValueError):
            run_backtest(
                bars=bars,
                strategy=SmaStrategy(short_window=2, long_window=3),
                config=BacktestConfig(symbol="TEST"),
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 3),
            )

    def test_applies_adverse_slippage_and_fixed_commission(self) -> None:
        """Buy prices rise, sell prices fall, and both fills charge commission."""

        bars = [
            _bar(1, "1", "1"),
            _bar(2, "2", "2"),
            _bar(3, "3", "3"),
            _bar(4, "4", "4"),
            _bar(5, "3", "3"),
            _bar(6, "2", "2"),
            _bar(7, "1", "1"),
        ]
        result = run_backtest(
            bars=bars,
            strategy=SmaStrategy(short_window=2, long_window=3),
            config=BacktestConfig(
                symbol="TEST",
                initial_cash=Decimal("1000"),
                commission_per_order=Decimal("1"),
                slippage_bps=Decimal("100"),
            ),
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 7),
        )

        self.assertEqual(result.trades[0].fill_price, Decimal("4.04"))
        self.assertEqual(result.trades[1].fill_price, Decimal("0.99"))
        self.assertEqual(result.summary["total_commission"], 2.0)
        self.assertLess(result.summary["final_equity"], 250.0)


if __name__ == "__main__":
    unittest.main()
