"""Orchestration for one safe evaluation of the paper-trading strategy."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from alpaca_quant.alpaca_broker import AlpacaPaperBroker
from alpaca_quant.config import Settings
from alpaca_quant.strategies.base import BacktestStrategy, Signal
from alpaca_quant.strategies.sma import SmaStrategy


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """A compact record of what the single-run process decided to do."""

    action: str
    reason: str


def run_once(
    settings: Settings,
    execute_orders: bool,
    strategy: BacktestStrategy | None = None,
) -> RunResult:
    """Evaluate one registered model and optionally submit one paper order."""

    selected_strategy = strategy or SmaStrategy(
        short_window=settings.short_window,
        long_window=settings.long_window,
    )
    broker = AlpacaPaperBroker(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
    )
    closes = broker.get_completed_daily_closes(
        symbol=settings.symbol,
        minimum_bars=selected_strategy.minimum_history,
    )
    quantity, average_entry_price = broker.position_state(settings.symbol)

    # This long-only model refuses to manage a pre-existing short position.
    if quantity < 0:
        raise RuntimeError(
            f"A short {settings.symbol} position exists; manual review is required."
        )

    decision = selected_strategy.evaluate(
        closes=closes,
        currently_long=quantity > 0,
        entry_price=(
            float(average_entry_price) if average_entry_price is not None else None
        ),
    )
    LOGGER.info(
        "strategy=%s symbol=%s position_qty=%s signal=%s",
        selected_strategy.name,
        settings.symbol,
        quantity,
        decision.signal.value,
    )
    LOGGER.info("Strategy reason: %s", decision.reason)

    if decision.signal is Signal.HOLD:
        return RunResult(action="HOLD", reason=decision.reason)

    if not execute_orders:
        preview = f"DRY_RUN_{decision.signal.value}"
        LOGGER.info(
            "%s: pass --execute to submit this order to Paper Trading.", preview
        )
        return RunResult(action=preview, reason=decision.reason)

    market_clock = broker.market_clock()
    if not market_clock.is_open:
        reason = f"The market is closed; next open is {market_clock.next_open}."
        LOGGER.warning(reason)
        return RunResult(action="SKIPPED_MARKET_CLOSED", reason=reason)

    if broker.has_open_order(settings.symbol):
        reason = f"An open {settings.symbol} order already exists."
        LOGGER.warning(reason)
        return RunResult(action="SKIPPED_OPEN_ORDER", reason=reason)

    if decision.signal is Signal.BUY:
        notional = broker.new_position_notional(settings.allocation_fraction)
        order = broker.submit_market_buy(settings.symbol, notional)
        reason = f"Submitted paper buy order {order.id} for ${notional}."
        LOGGER.info(reason)
        return RunResult(action="SUBMITTED_BUY", reason=reason)

    order = broker.close_position(settings.symbol)
    reason = f"Submitted paper close order {order.id} for {settings.symbol}."
    LOGGER.info(reason)
    return RunResult(action="SUBMITTED_SELL", reason=reason)
