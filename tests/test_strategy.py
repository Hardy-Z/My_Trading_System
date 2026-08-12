"""Offline unit tests for the pure moving-average strategy logic."""

from __future__ import annotations

import unittest

from alpaca_quant.strategies.sma import (
    Signal,
    evaluate_sma_regime,
    simple_moving_average,
)


class SimpleMovingAverageTests(unittest.TestCase):
    """Verify the reusable simple moving-average calculation."""

    def test_uses_only_the_requested_trailing_window(self) -> None:
        """Older observations must not affect the trailing average."""

        self.assertEqual(simple_moving_average([100, 1, 2, 3], 3), 2.0)

    def test_rejects_insufficient_history(self) -> None:
        """A window cannot be calculated without enough observations."""

        with self.assertRaises(ValueError):
            simple_moving_average([1, 2], 3)

    def test_rejects_non_positive_close_prices(self) -> None:
        """Invalid market data must fail before it can create a signal."""

        with self.assertRaises(ValueError):
            simple_moving_average([1, 0, 3], 3)


class SmaRegimeTests(unittest.TestCase):
    """Verify signals for every strategy and position-state combination."""

    def test_buys_when_bullish_and_flat(self) -> None:
        """A rising market should request a long position from cash."""

        decision = evaluate_sma_regime(
            closes=[1, 2, 3, 4, 5],
            short_window=2,
            long_window=4,
            currently_long=False,
        )
        self.assertIs(decision.signal, Signal.BUY)

    def test_holds_when_bullish_and_already_long(self) -> None:
        """A matching bullish position should not generate duplicate buys."""

        decision = evaluate_sma_regime(
            closes=[1, 2, 3, 4, 5],
            short_window=2,
            long_window=4,
            currently_long=True,
        )
        self.assertIs(decision.signal, Signal.HOLD)

    def test_sells_when_bearish_and_long(self) -> None:
        """A falling market should request liquidation of a long position."""

        decision = evaluate_sma_regime(
            closes=[5, 4, 3, 2, 1],
            short_window=2,
            long_window=4,
            currently_long=True,
        )
        self.assertIs(decision.signal, Signal.SELL)

    def test_holds_when_bearish_and_already_flat(self) -> None:
        """A matching bearish state should not create a short position."""

        decision = evaluate_sma_regime(
            closes=[5, 4, 3, 2, 1],
            short_window=2,
            long_window=4,
            currently_long=False,
        )
        self.assertIs(decision.signal, Signal.HOLD)

    def test_holds_when_averages_are_equal(self) -> None:
        """An equal-average boundary should not create unnecessary turnover."""

        decision = evaluate_sma_regime(
            closes=[10, 10, 10, 10, 10],
            short_window=2,
            long_window=4,
            currently_long=False,
        )
        self.assertIs(decision.signal, Signal.HOLD)

    def test_rejects_reversed_windows(self) -> None:
        """The short window must remain smaller than the long window."""

        with self.assertRaises(ValueError):
            evaluate_sma_regime(
                closes=[1, 2, 3],
                short_window=3,
                long_window=2,
                currently_long=False,
            )


if __name__ == "__main__":
    unittest.main()
