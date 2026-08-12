"""Command-line entry point for historical strategy simulations."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from alpaca_quant.backtesting.data import (
    load_alpaca_daily_bars,
    load_csv_daily_bars,
)
from alpaca_quant.backtesting.engine import run_backtest
from alpaca_quant.backtesting.models import BacktestConfig
from alpaca_quant.backtesting.reporting import export_result, print_summary
from alpaca_quant.strategies.registry import (
    available_strategy_names,
    create_strategy,
)
from alpaca_quant.strategies.sp500_candidate import Sp500CandidateProfile


def _date_argument(raw_value: str) -> date:
    """Parse a YYYY-MM-DD command-line date with a concise error message."""

    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format.") from exc


def _decimal_argument(raw_value: str) -> Decimal:
    """Parse a finite command-line decimal without binary rounding errors."""

    try:
        value = Decimal(raw_value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Value must be a decimal number.") from exc
    if not value.is_finite():
        raise argparse.ArgumentTypeError("Value must be finite.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the complete historical-backtest command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Backtest a registered long-only strategy with Alpaca or CSV daily "
            "bars. Signals are generated at close and filled at the next open."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=available_strategy_names(),
        default="sma",
        help="Registered strategy name (default: sma).",
    )
    parser.add_argument("--symbol", default="SPY", help="Asset symbol.")
    parser.add_argument(
        "--data-source",
        choices=("alpaca", "csv"),
        default="alpaca",
        help="Historical daily-bar source (default: alpaca).",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        help="OHLCV CSV path; required when --data-source csv is selected.",
    )
    parser.add_argument(
        "--start",
        type=_date_argument,
        required=True,
        help="First performance date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=_date_argument,
        required=True,
        help="Last performance date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=20,
        help="Short SMA window for the sma strategy (default: 20).",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=50,
        help="Long SMA window for the sma strategy (default: 50).",
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        help="Point-in-time fundamentals JSON required by sp500-candidate.",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=_decimal_argument,
        default=Decimal("20"),
        help="Fixed candidate gain target in percent (default: 20).",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=50,
        help="Candidate trend-confirmation SMA window (default: 50).",
    )
    parser.add_argument(
        "--initial-cash",
        type=_decimal_argument,
        default=Decimal("100000"),
        help="Starting portfolio cash (default: 100000).",
    )
    parser.add_argument(
        "--allocation",
        type=_decimal_argument,
        default=Decimal("1.0"),
        help="Fraction of cash allocated to each entry, from 0 to 1 (default: 1).",
    )
    parser.add_argument(
        "--commission-per-order",
        type=_decimal_argument,
        default=Decimal("0"),
        help="Fixed commission charged per simulated fill (default: 0).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=_decimal_argument,
        default=Decimal("5"),
        help="Adverse execution slippage in basis points (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Result directory. The default is a strategy/symbol/date-specific "
            "folder under backtest_results."
        ),
    )
    return parser


def main() -> int:
    """Load historical data, run one simulation, and export all result files."""

    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        if args.start > args.end:
            raise ValueError("--start must not be after --end.")

        symbol = args.symbol.strip().upper()
        candidate_profile = None
        if args.strategy == "sp500-candidate":
            if args.candidate_profile is None:
                raise ValueError(
                    "--candidate-profile is required by sp500-candidate."
                )
            candidate_profile = Sp500CandidateProfile.from_json_file(
                args.candidate_profile
            )
            if candidate_profile.symbol != symbol:
                raise ValueError(
                    f"Candidate profile symbol {candidate_profile.symbol} does not "
                    f"match --symbol {symbol}."
                )
            if candidate_profile.as_of_date > args.start:
                raise ValueError(
                    "Candidate profile as_of_date must be on or before --start "
                    "to prevent fundamental-data look-ahead bias."
                )
        strategy = create_strategy(
            args.strategy,
            short_window=args.short_window,
            long_window=args.long_window,
            candidate_profile=candidate_profile,
            take_profit_fraction=args.take_profit_pct / Decimal("100"),
            trend_window=args.trend_window,
        )
        config = BacktestConfig(
            symbol=symbol,
            initial_cash=args.initial_cash,
            allocation_fraction=args.allocation,
            commission_per_order=args.commission_per_order,
            slippage_bps=args.slippage_bps,
        )

        if args.data_source == "csv":
            if args.csv_file is None:
                raise ValueError("--csv-file is required for the CSV data source.")
            bars = load_csv_daily_bars(
                csv_path=args.csv_file,
                end_date=args.end,
            )
        else:
            bars = load_alpaca_daily_bars(
                api_key=os.getenv("ALPACA_API_KEY", ""),
                secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                warmup_bars=strategy.minimum_history,
            )

        result = run_backtest(
            bars=bars,
            strategy=strategy,
            config=config,
            start_date=args.start,
            end_date=args.end,
        )
        result.summary["data_source"] = args.data_source
        if args.data_source == "csv":
            result.summary["csv_file"] = str(args.csv_file.resolve())

        output_directory = args.output_dir or (
            project_root
            / "backtest_results"
            / f"{strategy.name}_{symbol}_{args.start}_{args.end}"
        )
        export_result(result, output_directory)
        print_summary(result)
        print(f"\nResult files: {output_directory.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("Backtest configuration or data error: %s", exc)
        return 2
    except Exception:
        logging.exception("The historical backtest failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
