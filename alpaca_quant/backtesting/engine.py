"""Event-driven, long-only historical simulator with next-open execution."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from math import sqrt
from statistics import mean, stdev
from typing import Any, Sequence

from alpaca_quant.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    EquityPoint,
    MarketBar,
    TradeRecord,
)
from alpaca_quant.strategies.base import BacktestStrategy, Signal


BASIS_POINTS_PER_UNIT = Decimal("10000")


def _strategy_parameters(strategy: BacktestStrategy) -> dict[str, Any]:
    """Return serializable public strategy parameters for the result summary."""

    if not is_dataclass(strategy):
        return {}
    parameters = {
        key: value
        for key, value in asdict(strategy).items()
        if key != "name"
    }
    return _make_json_serializable(parameters)


def _make_json_serializable(value: Any) -> Any:
    """Convert nested strategy parameters to JSON-safe primitive values."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _make_json_serializable(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(item) for item in value]
    return value


def _calculate_summary(
    *,
    config: BacktestConfig,
    strategy: BacktestStrategy,
    start_date: date,
    end_date: date,
    warmup_count: int,
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    ending_cost_basis: Decimal,
    pending_signal: tuple[Signal, str] | None,
) -> dict[str, Any]:
    """Calculate common return, risk, exposure, and trade statistics."""

    initial_cash = config.initial_cash
    final_point = equity_curve[-1]
    final_equity = final_point.equity
    net_profit = final_equity - initial_cash
    total_return = final_equity / initial_cash - Decimal("1")

    prior_equity = initial_cash
    daily_returns: list[float] = []
    for point in equity_curve:
        daily_returns.append(float(point.equity / prior_equity - Decimal("1")))
        prior_equity = point.equity

    period_days = max((end_date - start_date).days, 1)
    annualized_return = (
        float(final_equity / initial_cash) ** (365.25 / period_days) - 1.0
        if final_equity > 0
        else None
    )
    volatility = stdev(daily_returns) * sqrt(252) if len(daily_returns) > 1 else 0.0
    sharpe_ratio = (
        mean(daily_returns) / stdev(daily_returns) * sqrt(252)
        if len(daily_returns) > 1 and stdev(daily_returns) > 0
        else None
    )

    completed_trade_pnls = [
        trade.realized_pnl
        for trade in trades
        if trade.action == "SELL" and trade.realized_pnl is not None
    ]
    winning_trades = sum(pnl > 0 for pnl in completed_trade_pnls)
    gross_profit = sum(
        (pnl for pnl in completed_trade_pnls if pnl > 0), Decimal("0")
    )
    gross_loss = sum(
        (-pnl for pnl in completed_trade_pnls if pnl < 0), Decimal("0")
    )
    profit_factor = (
        float(gross_profit / gross_loss)
        if gross_loss > 0
        else None
    )
    realized_pnl = sum(completed_trade_pnls, Decimal("0"))
    unrealized_pnl = (
        final_point.position_market_value - ending_cost_basis
        if final_point.position_quantity > 0
        else Decimal("0")
    )
    benchmark_return = (
        equity_curve[-1].close / equity_curve[0].close - Decimal("1")
    )
    exposure = sum(
        point.position_quantity > 0 for point in equity_curve
    ) / len(equity_curve)
    total_commission = sum(
        (trade.commission for trade in trades), Decimal("0")
    )

    return {
        "strategy": strategy.name,
        "strategy_parameters": _strategy_parameters(strategy),
        "symbol": config.symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_bars": len(equity_curve),
        "warmup_bars": warmup_count,
        "initial_cash": float(initial_cash),
        "final_equity": float(final_equity),
        "net_profit": float(net_profit),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "total_return_pct": float(total_return * Decimal("100")),
        "annualized_return_pct": (
            annualized_return * 100 if annualized_return is not None else None
        ),
        "benchmark_return_pct": float(benchmark_return * Decimal("100")),
        "max_drawdown_pct": float(
            min(point.drawdown for point in equity_curve) * Decimal("100")
        ),
        "annualized_volatility_pct": volatility * 100,
        "sharpe_ratio": sharpe_ratio,
        "exposure_pct": exposure * 100,
        "orders": len(trades),
        "completed_trades": len(completed_trade_pnls),
        "winning_trades": winning_trades,
        "win_rate_pct": (
            winning_trades / len(completed_trade_pnls) * 100
            if completed_trade_pnls
            else None
        ),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": profit_factor,
        "average_completed_trade_pnl": (
            float(sum(completed_trade_pnls, Decimal("0")) / len(completed_trade_pnls))
            if completed_trade_pnls
            else None
        ),
        "total_commission": float(total_commission),
        "allocation_fraction": float(config.allocation_fraction),
        "commission_per_order": float(config.commission_per_order),
        "slippage_bps": float(config.slippage_bps),
        "ending_cash": float(final_point.cash),
        "ending_position_quantity": float(final_point.position_quantity),
        "ending_position_market_value": float(final_point.position_market_value),
        "pending_signal_after_end": (
            pending_signal[0].value if pending_signal is not None else None
        ),
    }


def run_backtest(
    *,
    bars: Sequence[MarketBar],
    strategy: BacktestStrategy,
    config: BacktestConfig,
    start_date: date,
    end_date: date,
) -> BacktestResult:
    """Run a historical simulation using close signals and next-open fills.

    Bars before ``start_date`` are indicator warm-up data only. A decision made
    from the final warm-up close may fill at the first test-period open. Every
    later decision uses a completed close and fills no earlier than the next
    bar, preventing look-ahead bias.
    """

    if start_date > end_date:
        raise ValueError("The backtest start date must not be after the end date.")
    if not bars:
        raise ValueError("At least one historical bar is required.")

    sorted_bars = sorted(bars, key=lambda bar: bar.timestamp)
    if any(
        current.timestamp == previous.timestamp
        for previous, current in zip(sorted_bars, sorted_bars[1:])
    ):
        raise ValueError("Historical bars must have unique timestamps.")

    evaluation_indexes = [
        index
        for index, bar in enumerate(sorted_bars)
        if start_date <= bar.timestamp.date() <= end_date
    ]
    if not evaluation_indexes:
        raise ValueError("No historical bars fall inside the requested time period.")

    first_evaluation_index = evaluation_indexes[0]
    if first_evaluation_index < strategy.minimum_history:
        raise ValueError(
            f"Strategy '{strategy.name}' requires at least "
            f"{strategy.minimum_history} bars before {start_date}; only "
            f"{first_evaluation_index} are available."
        )

    cash = config.initial_cash
    quantity = Decimal("0")
    entry_cost_basis = Decimal("0")
    entry_fill_price: Decimal | None = None
    trades: list[TradeRecord] = []
    equity_curve: list[EquityPoint] = []
    peak_equity = config.initial_cash
    slippage_rate = config.slippage_bps / BASIS_POINTS_PER_UNIT

    warmup_closes = [
        float(bar.close) for bar in sorted_bars[:first_evaluation_index]
    ]
    warmup_decision = strategy.evaluate(
        warmup_closes,
        currently_long=False,
        entry_price=None,
    )
    pending_signal: tuple[Signal, str] | None = None
    if warmup_decision.signal is not Signal.HOLD:
        pending_signal = (warmup_decision.signal, warmup_decision.reason)

    for index in evaluation_indexes:
        bar = sorted_bars[index]

        # Execute only the signal created from the previous completed close.
        if pending_signal is not None:
            signal, reason = pending_signal
            if signal is Signal.BUY and quantity == 0:
                fill_price = bar.open * (Decimal("1") + slippage_rate)
                available_cash = cash - config.commission_per_order
                if available_cash <= 0:
                    raise RuntimeError("Commission leaves no cash available to invest.")
                target_notional = cash * config.allocation_fraction
                notional = min(target_notional, available_cash)
                quantity = notional / fill_price
                entry_cost_basis = notional + config.commission_per_order
                entry_fill_price = fill_price
                cash -= entry_cost_basis
                trades.append(
                    TradeRecord(
                        timestamp=bar.timestamp,
                        action="BUY",
                        quantity=quantity,
                        fill_price=fill_price,
                        notional=notional,
                        commission=config.commission_per_order,
                        realized_pnl=None,
                        reason=reason,
                    )
                )
            elif signal is Signal.SELL and quantity > 0:
                fill_price = bar.open * (Decimal("1") - slippage_rate)
                notional = quantity * fill_price
                realized_pnl = (
                    notional - config.commission_per_order - entry_cost_basis
                )
                cash += notional - config.commission_per_order
                trades.append(
                    TradeRecord(
                        timestamp=bar.timestamp,
                        action="SELL",
                        quantity=quantity,
                        fill_price=fill_price,
                        notional=notional,
                        commission=config.commission_per_order,
                        realized_pnl=realized_pnl,
                        reason=reason,
                    )
                )
                quantity = Decimal("0")
                entry_cost_basis = Decimal("0")
                entry_fill_price = None
            pending_signal = None

        position_market_value = quantity * bar.close
        equity = cash + position_market_value
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - Decimal("1")
        equity_curve.append(
            EquityPoint(
                timestamp=bar.timestamp,
                close=bar.close,
                cash=cash,
                position_quantity=quantity,
                position_market_value=position_market_value,
                equity=equity,
                drawdown=drawdown,
            )
        )

        completed_closes = [
            float(historical_bar.close) for historical_bar in sorted_bars[: index + 1]
        ]
        decision = strategy.evaluate(
            completed_closes,
            currently_long=quantity > 0,
            entry_price=(
                float(entry_fill_price) if entry_fill_price is not None else None
            ),
        )
        if decision.signal is not Signal.HOLD:
            pending_signal = (decision.signal, decision.reason)

    summary = _calculate_summary(
        config=config,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        warmup_count=first_evaluation_index,
        equity_curve=equity_curve,
        trades=trades,
        ending_cost_basis=entry_cost_basis,
        pending_signal=pending_signal,
    )
    return BacktestResult(
        summary=summary,
        equity_curve=tuple(equity_curve),
        trades=tuple(trades),
    )
