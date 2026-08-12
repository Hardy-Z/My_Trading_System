"""Historical daily-bar loaders for Alpaca and user-provided CSV files."""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca_quant.backtesting.models import MarketBar


REQUIRED_CSV_COLUMNS = ("timestamp", "open", "high", "low", "close")


def _parse_decimal(raw_value: str, column: str, row_number: int) -> Decimal:
    """Parse one CSV decimal and attach row context to validation errors."""

    try:
        value = Decimal(raw_value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(
            f"CSV row {row_number}: '{column}' must be a decimal number."
        ) from exc
    if not value.is_finite():
        raise ValueError(f"CSV row {row_number}: '{column}' must be finite.")
    return value


def _parse_timestamp(raw_value: str, row_number: int) -> datetime:
    """Parse an ISO date or timestamp and normalize a trailing UTC marker."""

    normalized_value = raw_value.strip().replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(normalized_value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: timestamp must use ISO-8601 format."
        ) from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _validate_unique_sorted_bars(bars: list[MarketBar]) -> list[MarketBar]:
    """Sort bars chronologically and reject duplicate timestamps."""

    sorted_bars = sorted(bars, key=lambda bar: bar.timestamp)
    timestamps = [bar.timestamp for bar in sorted_bars]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Historical data contains duplicate timestamps.")
    return sorted_bars


def load_csv_daily_bars(csv_path: Path, end_date: date) -> list[MarketBar]:
    """Load standard OHLCV CSV rows up to and including the requested end date.

    Rows before the requested backtest start date are intentionally retained as
    warm-up history for indicators. The backtest engine excludes those rows from
    performance calculations.
    """

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV data file does not exist: {csv_path}")

    bars: list[MarketBar] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("The CSV file must contain a header row.")

        column_names = {name.strip().lower(): name for name in reader.fieldnames}
        missing_columns = [
            name for name in REQUIRED_CSV_COLUMNS if name not in column_names
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"CSV file is missing required columns: {missing_text}.")

        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_timestamp(
                row[column_names["timestamp"]], row_number
            )
            if timestamp.date() > end_date:
                continue

            volume_column = column_names.get("volume")
            raw_volume = row[volume_column] if volume_column else "0"
            bars.append(
                MarketBar(
                    timestamp=timestamp,
                    open=_parse_decimal(
                        row[column_names["open"]], "open", row_number
                    ),
                    high=_parse_decimal(
                        row[column_names["high"]], "high", row_number
                    ),
                    low=_parse_decimal(
                        row[column_names["low"]], "low", row_number
                    ),
                    close=_parse_decimal(
                        row[column_names["close"]], "close", row_number
                    ),
                    volume=_parse_decimal(raw_volume or "0", "volume", row_number),
                )
            )

    if not bars:
        raise ValueError("The CSV file contains no usable bars through the end date.")
    return _validate_unique_sorted_bars(bars)


def load_alpaca_daily_bars(
    *,
    api_key: str,
    secret_key: str,
    symbol: str,
    start_date: date,
    end_date: date,
    warmup_bars: int,
) -> list[MarketBar]:
    """Download adjusted daily IEX bars plus automatic indicator warm-up data."""

    if not api_key.strip() or not secret_key.strip():
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY are required for Alpaca data."
        )

    # Four calendar days per requested session safely spans weekends, holidays,
    # and other non-trading days for ordinary daily strategies.
    warmup_days = max(warmup_bars * 4, 180)
    request_start = datetime.combine(
        start_date - timedelta(days=warmup_days), time.min, tzinfo=timezone.utc
    )
    request_end = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    data_client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=request_start,
        end=request_end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    bar_set = data_client.get_stock_bars(request)
    alpaca_bars = list(getattr(bar_set, "data", {}).get(symbol, []))

    bars = [
        MarketBar(
            timestamp=bar.timestamp,
            open=Decimal(str(bar.open)),
            high=Decimal(str(bar.high)),
            low=Decimal(str(bar.low)),
            close=Decimal(str(bar.close)),
            volume=Decimal(str(bar.volume)),
        )
        for bar in alpaca_bars
        if bar.timestamp.date() <= end_date
    ]
    if not bars:
        raise RuntimeError(
            f"Alpaca returned no daily IEX bars for {symbol} through {end_date}."
        )
    return _validate_unique_sorted_bars(bars)
