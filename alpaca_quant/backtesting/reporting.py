"""Terminal and file reporting for completed historical simulations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from alpaca_quant.backtesting.models import BacktestResult


def _display_value(value: Any) -> str:
    """Format optional numeric summary values for compact terminal output."""

    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def print_summary(result: BacktestResult) -> None:
    """Print the most useful return and risk fields to the terminal."""

    summary = result.summary
    rows = (
        ("Strategy", summary["strategy"]),
        ("Symbol", summary["symbol"]),
        ("Period", f"{summary['start_date']} to {summary['end_date']}"),
        ("Initial cash", summary["initial_cash"]),
        ("Final equity", summary["final_equity"]),
        ("Net profit", summary["net_profit"]),
        ("Total return (%)", summary["total_return_pct"]),
        ("Annualized return (%)", summary["annualized_return_pct"]),
        ("Benchmark return (%)", summary["benchmark_return_pct"]),
        ("Maximum drawdown (%)", summary["max_drawdown_pct"]),
        ("Sharpe ratio", summary["sharpe_ratio"]),
        ("Completed trades", summary["completed_trades"]),
        ("Win rate (%)", summary["win_rate_pct"]),
        ("Exposure (%)", summary["exposure_pct"]),
        ("Ending position quantity", summary["ending_position_quantity"]),
    )
    width = max(len(label) for label, _ in rows)
    print("\nBacktest summary")
    print("-" * (width + 24))
    for label, value in rows:
        print(f"{label:<{width}} : {_display_value(value)}")


def export_result(result: BacktestResult, output_directory: Path) -> None:
    """Write summary JSON, equity-curve CSV, and trade-detail CSV files."""

    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    equity_path = output_directory / "equity_curve.csv"
    with equity_path.open("w", encoding="utf-8", newline="") as equity_file:
        writer = csv.writer(equity_file)
        writer.writerow(
            (
                "timestamp",
                "close",
                "cash",
                "position_quantity",
                "position_market_value",
                "equity",
                "drawdown_pct",
            )
        )
        for point in result.equity_curve:
            writer.writerow(
                (
                    point.timestamp.isoformat(),
                    str(point.close),
                    str(point.cash),
                    str(point.position_quantity),
                    str(point.position_market_value),
                    str(point.equity),
                    str(point.drawdown * 100),
                )
            )

    trades_path = output_directory / "trades.csv"
    with trades_path.open("w", encoding="utf-8", newline="") as trades_file:
        writer = csv.writer(trades_file)
        writer.writerow(
            (
                "timestamp",
                "action",
                "quantity",
                "fill_price",
                "notional",
                "commission",
                "realized_pnl",
                "reason",
            )
        )
        for trade in result.trades:
            writer.writerow(
                (
                    trade.timestamp.isoformat(),
                    trade.action,
                    str(trade.quantity),
                    str(trade.fill_price),
                    str(trade.notional),
                    str(trade.commission),
                    "" if trade.realized_pnl is None else str(trade.realized_pnl),
                    trade.reason,
                )
            )
