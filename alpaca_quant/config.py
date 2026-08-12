"""Environment-based configuration for the paper-trading application."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import os
import re


class ConfigurationError(ValueError):
    """Raised when a required setting is missing or invalid."""


def _required_environment_value(name: str) -> str:
    """Return a required environment value without surrounding whitespace."""

    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required.")
    return value


def _integer_environment_value(name: str, default: int) -> int:
    """Parse an integer setting and provide a clear configuration error."""

    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _decimal_environment_value(name: str, default: str) -> Decimal:
    """Parse a decimal setting without introducing binary floating-point error."""

    raw_value = os.getenv(name, default).strip()
    try:
        return Decimal(raw_value)
    except InvalidOperation as exc:
        raise ConfigurationError(f"{name} must be a decimal number.") from exc


@dataclass(frozen=True)
class Settings:
    """Validated account, symbol, allocation, and default SMA settings."""

    # repr=False prevents accidental credential exposure in logs and tracebacks.
    api_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    symbol: str = "SPY"
    short_window: int = 20
    long_window: int = 50
    allocation_fraction: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        """Validate settings that depend on more than one field."""

        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", self.symbol):
            raise ConfigurationError(
                "TRADING_SYMBOL must be an uppercase US market symbol."
            )
        if self.short_window < 2:
            raise ConfigurationError("SHORT_WINDOW must be at least 2.")
        if self.long_window <= self.short_window:
            raise ConfigurationError(
                "LONG_WINDOW must be greater than SHORT_WINDOW."
            )
        if not Decimal("0") < self.allocation_fraction <= Decimal("1"):
            raise ConfigurationError(
                "ALLOCATION_FRACTION must be greater than 0 and at most 1."
            )

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build validated settings from environment variables."""

        return cls(
            api_key=_required_environment_value("ALPACA_API_KEY"),
            secret_key=_required_environment_value("ALPACA_SECRET_KEY"),
            symbol=os.getenv("TRADING_SYMBOL", "SPY").strip().upper(),
            short_window=_integer_environment_value("SHORT_WINDOW", 20),
            long_window=_integer_environment_value("LONG_WINDOW", 50),
            allocation_fraction=_decimal_environment_value(
                "ALLOCATION_FRACTION", "0.10"
            ),
        )
