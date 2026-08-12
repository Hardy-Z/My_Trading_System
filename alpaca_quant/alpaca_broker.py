"""A narrow Alpaca adapter that is permanently restricted to paper trading."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from uuid import uuid4

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


class AlpacaPaperBroker:
    """Provide only the market-data and paper-order operations used by the model."""

    def __init__(self, api_key: str, secret_key: str) -> None:
        """Create authenticated market-data and paper-trading clients."""

        # paper=True is deliberately hard-coded; this project cannot select live mode.
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    def get_completed_daily_closes(
        self,
        symbol: str,
        minimum_bars: int,
    ) -> list[float]:
        """Return enough completed IEX daily closes to evaluate the strategy."""

        now = datetime.now(timezone.utc)

        # Calendar days exceed trading sessions, so a generous buffer covers
        # weekends, holidays, and occasional missing bars.
        history_days = max(minimum_bars * 4, 180)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=now - timedelta(days=history_days),
            end=now,
            feed=DataFeed.IEX,
        )
        bar_set = self.data_client.get_stock_bars(request)
        symbol_bars = list(getattr(bar_set, "data", {}).get(symbol, []))

        # Alpaca can expose an evolving current-day daily bar while the market is
        # open. Removing it prevents an incomplete bar from driving a decision.
        market_clock = self.trading_client.get_clock()
        if market_clock.is_open:
            market_date = market_clock.timestamp.date()
            symbol_bars = [
                bar
                for bar in symbol_bars
                if bar.timestamp.astimezone(market_clock.timestamp.tzinfo).date()
                < market_date
            ]

        closes = [float(bar.close) for bar in symbol_bars]
        if len(closes) < minimum_bars:
            raise RuntimeError(
                f"Alpaca returned {len(closes)} completed bars for {symbol}; "
                f"at least {minimum_bars} are required."
            )
        return closes

    def position_state(self, symbol: str) -> tuple[Decimal, Decimal | None]:
        """Return signed quantity and average entry price for one symbol."""

        for position in self.trading_client.get_all_positions():
            if position.symbol.upper() == symbol.upper():
                return (
                    Decimal(str(position.qty)),
                    Decimal(str(position.avg_entry_price)),
                )
        return Decimal("0"), None

    def position_quantity(self, symbol: str) -> Decimal:
        """Return only the signed quantity for compatibility with existing callers."""

        quantity, _ = self.position_state(symbol)
        return quantity

    def has_open_order(self, symbol: str) -> bool:
        """Return whether the paper account already has an open symbol order."""

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            limit=50,
        )
        return bool(self.trading_client.get_orders(filter=request))

    def market_clock(self):
        """Return Alpaca's current US equity market clock model."""

        return self.trading_client.get_clock()

    def new_position_notional(self, allocation_fraction: Decimal) -> Decimal:
        """Calculate a cent-rounded order value capped by available buying power."""

        account = self.trading_client.get_account()
        if bool(account.trading_blocked):
            raise RuntimeError("The Alpaca paper account is blocked from trading.")

        equity = Decimal(str(account.equity))
        buying_power = Decimal(str(account.buying_power))
        requested_notional = equity * allocation_fraction
        available_notional = min(requested_notional, buying_power)
        return available_notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def submit_market_buy(self, symbol: str, notional: Decimal):
        """Submit a notional market buy to the Alpaca paper endpoint."""

        if notional < Decimal("1.00"):
            raise RuntimeError("The calculated order notional is below $1.00.")

        # A recognizable client ID makes the order easy to identify in Alpaca.
        client_order_id = (
            f"quant-{symbol.lower()}-{datetime.now(timezone.utc):%Y%m%d}-"
            f"{uuid4().hex[:8]}"
        )
        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=float(notional),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        return self.trading_client.submit_order(order_data=order_request)

    def close_position(self, symbol: str):
        """Submit an order that liquidates the complete paper position."""

        return self.trading_client.close_position(symbol)
