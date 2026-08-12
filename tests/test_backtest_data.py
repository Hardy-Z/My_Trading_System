"""Offline tests for CSV ingestion and backtest result file exports."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from alpaca_quant.backtesting.data import load_csv_daily_bars
from alpaca_quant.backtesting.engine import run_backtest
from alpaca_quant.backtesting.models import BacktestConfig, MarketBar
from alpaca_quant.backtesting.reporting import export_result
from alpaca_quant.strategies.sma import SmaStrategy


class CsvDataTests(unittest.TestCase):
    """Verify CSV schema handling, warm-up retention, and row ordering."""

    def test_loads_sorts_and_retains_warmup_rows(self) -> None:
        """Rows before the performance period must remain available to indicators."""

        csv_text = (
            "timestamp,open,high,low,close,volume\n"
            "2024-01-03,3,4,2,3,100\n"
            "2024-01-01,1,2,0.5,1,100\n"
            "2024-01-02,2,3,1,2,100\n"
            "2024-01-05,5,6,4,5,100\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "bars.csv"
            csv_path.write_text(csv_text, encoding="utf-8")
            bars = load_csv_daily_bars(csv_path, end_date=date(2024, 1, 3))

        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[0].timestamp.date(), date(2024, 1, 1))
        self.assertEqual(bars[-1].timestamp.date(), date(2024, 1, 3))

    def test_rejects_duplicate_timestamps(self) -> None:
        """Duplicate bars must not silently create repeated trading sessions."""

        csv_text = (
            "timestamp,open,high,low,close\n"
            "2024-01-01,1,2,0.5,1\n"
            "2024-01-01,1,2,0.5,1\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "bars.csv"
            csv_path.write_text(csv_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_csv_daily_bars(csv_path, end_date=date(2024, 1, 1))


class BacktestReportingTests(unittest.TestCase):
    """Verify that each documented result artifact is generated."""

    def test_exports_summary_equity_and_trade_files(self) -> None:
        """A complete run must produce three machine-readable result files."""

        bars = []
        for day, close in enumerate((1, 2, 3, 4, 5), start=1):
            price = Decimal(close)
            bars.append(
                MarketBar(
                    timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
                    open=price,
                    high=price + 1,
                    low=max(price - 1, Decimal("0.1")),
                    close=price,
                    volume=Decimal("100"),
                )
            )
        result = run_backtest(
            bars=bars,
            strategy=SmaStrategy(short_window=2, long_window=3),
            config=BacktestConfig(symbol="TEST", slippage_bps=Decimal("0")),
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 5),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "results"
            export_result(result, output_directory)
            self.assertTrue((output_directory / "summary.json").is_file())
            self.assertTrue((output_directory / "equity_curve.csv").is_file())
            self.assertTrue((output_directory / "trades.csv").is_file())


if __name__ == "__main__":
    unittest.main()
