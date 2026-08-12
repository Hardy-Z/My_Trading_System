"""Data models shared by historical data loaders and the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class MarketBar:
    """One completed OHLCV market-data bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Normalize timestamps and reject structurally invalid market data."""

        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))

        prices = (self.open, self.high, self.low, self.close)
        if not all(price.is_finite() and price > 0 for price in prices):
            raise ValueError("OHLC prices must be finite positive decimal values.")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("A bar's high price cannot be below another OHLC price.")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("A bar's low price cannot be above another OHLC price.")
        if not self.volume.is_finite() or self.volume < 0:
            raise ValueError("Bar volume must be a finite non-negative value.")


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and portfolio assumptions used by a simulation."""

    symbol: str
    initial_cash: Decimal = Decimal("100000")
    allocation_fraction: Decimal = Decimal("1.0")
    commission_per_order: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("5")

    def __post_init__(self) -> None:
        """Validate assumptions before any historical simulation begins."""

        if not self.symbol.strip():
            raise ValueError("A backtest symbol is required.")
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("Initial cash must be a finite positive amount.")
        if not Decimal("0") < self.allocation_fraction <= Decimal("1"):
            raise ValueError("Allocation must be greater than 0 and at most 1.")
        if not self.commission_per_order.is_finite() or self.commission_per_order < 0:
            raise ValueError("Commission must be a finite non-negative amount.")
        if not self.slippage_bps.is_finite() or self.slippage_bps < 0:
            raise ValueError("Slippage must be a finite non-negative number of bps.")


@dataclass(frozen=True)
class TradeRecord:
    """One simulated order fill at the following bar's open price."""

    timestamp: datetime
    action: str
    quantity: Decimal
    fill_price: Decimal
    notional: Decimal
    commission: Decimal
    realized_pnl: Decimal | None
    reason: str


@dataclass(frozen=True)
class EquityPoint:
    """End-of-bar portfolio state used to build the equity curve."""

    timestamp: datetime
    close: Decimal
    cash: Decimal
    position_quantity: Decimal
    position_market_value: Decimal
    equity: Decimal
    drawdown: Decimal


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest output ready for terminal display or file export."""

    summary: dict[str, Any]
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[TradeRecord, ...]
