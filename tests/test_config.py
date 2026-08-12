"""Offline unit tests for environment-based application configuration."""

from __future__ import annotations

from decimal import Decimal
import os
import unittest
from unittest.mock import patch

from alpaca_quant.config import ConfigurationError, Settings


class SettingsTests(unittest.TestCase):
    """Verify defaults, validation, and credential-safe representations."""

    def test_loads_defaults_with_required_credentials(self) -> None:
        """Only the two paper credentials should be required from a new user."""

        environment = {
            "ALPACA_API_KEY": "paper-key",
            "ALPACA_SECRET_KEY": "paper-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_environment()

        self.assertEqual(settings.symbol, "SPY")
        self.assertEqual(settings.short_window, 20)
        self.assertEqual(settings.long_window, 50)
        self.assertEqual(settings.allocation_fraction, Decimal("0.10"))

    def test_requires_both_credentials(self) -> None:
        """The runner must fail clearly before making an unauthenticated request."""

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()

    def test_rejects_invalid_window_order(self) -> None:
        """A long window cannot be shorter than the strategy's short window."""

        environment = {
            "ALPACA_API_KEY": "paper-key",
            "ALPACA_SECRET_KEY": "paper-secret",
            "SHORT_WINDOW": "50",
            "LONG_WINDOW": "20",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()

    def test_rejects_allocation_above_full_equity(self) -> None:
        """The model must not allocate more than the complete account equity."""

        environment = {
            "ALPACA_API_KEY": "paper-key",
            "ALPACA_SECRET_KEY": "paper-secret",
            "ALLOCATION_FRACTION": "1.01",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()

    def test_hides_credentials_from_repr(self) -> None:
        """Logging the settings object must not reveal API credentials."""

        settings = Settings(api_key="private-key", secret_key="private-secret")
        representation = repr(settings)

        self.assertNotIn("private-key", representation)
        self.assertNotIn("private-secret", representation)


if __name__ == "__main__":
    unittest.main()
