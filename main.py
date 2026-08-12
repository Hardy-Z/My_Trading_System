"""Command-line entry point for the Alpaca paper-trading model."""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal, InvalidOperation
import logging
from pathlib import Path

from dotenv import load_dotenv

from alpaca_quant.config import ConfigurationError, Settings
from alpaca_quant.runner import run_once
from alpaca_quant.strategies.registry import (
    available_strategy_names,
    create_strategy,
)
from alpaca_quant.strategies.sp500_candidate import Sp500CandidateProfile


def _percentage_argument(raw_value: str) -> Decimal:
    """Parse a positive percentage and return its fractional representation."""

    try:
        percentage = Decimal(raw_value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Percentage must be a decimal number.") from exc
    if not percentage.is_finite() or percentage <= 0:
        raise argparse.ArgumentTypeError("Percentage must be finite and positive.")
    return percentage / Decimal("100")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for a single strategy evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a registered model once against an Alpaca paper account. "
            "The default mode previews the decision without submitting orders."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=available_strategy_names(),
        default="sma",
        help="Registered strategy name (default: sma).",
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        help="Candidate fundamentals JSON required by sp500-candidate.",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=_percentage_argument,
        default=Decimal("0.20"),
        help="Fixed candidate gain target in percent (default: 20).",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=50,
        help="Candidate trend-confirmation SMA window (default: 50).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the resulting order to Alpaca Paper Trading.",
    )
    return parser


def main() -> int:
    """Load configuration, run the model once, and return a process exit code."""

    project_root = Path(__file__).resolve().parent

    # Loading a local .env file keeps credentials out of the source code.
    load_dotenv(dotenv_path=project_root / ".env")
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        settings = Settings.from_environment()
        candidate_profile = None
        if args.strategy == "sp500-candidate":
            if args.candidate_profile is None:
                raise ValueError(
                    "--candidate-profile is required by sp500-candidate."
                )
            candidate_profile = Sp500CandidateProfile.from_json_file(
                args.candidate_profile
            )
            if candidate_profile.symbol != settings.symbol:
                raise ValueError(
                    f"Candidate profile symbol {candidate_profile.symbol} does not "
                    f"match TRADING_SYMBOL {settings.symbol}."
                )
            if candidate_profile.as_of_date > date.today():
                raise ValueError("Candidate profile as_of_date cannot be in the future.")
        strategy = create_strategy(
            args.strategy,
            short_window=settings.short_window,
            long_window=settings.long_window,
            candidate_profile=candidate_profile,
            take_profit_fraction=args.take_profit_pct,
            trend_window=args.trend_window,
        )
        result = run_once(
            settings=settings,
            execute_orders=args.execute,
            strategy=strategy,
        )
    except ConfigurationError as exc:
        logging.error("Configuration error: %s", exc)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Strategy configuration error: %s", exc)
        return 2
    except Exception:
        # The full traceback is useful for diagnosing API and connectivity errors.
        logging.exception("The paper-trading run failed.")
        return 1

    logging.info("Run completed with action=%s", result.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
