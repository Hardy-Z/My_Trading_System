"""Timing and persistence checks for the online next-minute Transformer."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from online_transformer import (OnlineAgent, Settings, StreamingTrainer,
                                current_context, make_sessions, run_replay,
                                score_predictions)


def fixture():
    dates = pd.bdate_range("2026-01-01", periods=40)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, (40, 3)), axis=0))
    daily_prices = np.stack((close * .999, close), axis=-1)
    bars = []
    for j, day in enumerate(dates[10:22]):
        start = pd.Timestamp(day.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for k in range(5):
            stamp = (start + pd.Timedelta(minutes=k)).tz_convert("UTC")
            moves = np.array([k, -k, (-1)**k]) * .01
            for s, price in zip(("SPY", "QQQ", "IWM"), 100 + j + moves):
                bars.append((stamp, s, price-.005, price))
    raw = pd.DataFrame(bars, columns=["timestamp_utc", "symbol", "open", "close"])
    wide = raw.pivot(index="timestamp_utc", columns="symbol", values=["open", "close"])
    return dates, daily_prices, wide


class OnlineTransformerTests(unittest.TestCase):
    def test_streaming_updates_only_after_consecutive_next_bar(self):
        dates, prices, bars = fixture()
        day = bars.iloc[:5]
        cfg = Settings(width=8, heads=2, daily_window=6, minute_window=3)
        stream = StreamingTrainer(OnlineAgent(cfg))
        first = current_context(dates, prices, day.iloc[:1], cfg)
        probability, updated = stream.on_completed_bar(*first)
        self.assertEqual(probability.shape, (3,))
        self.assertFalse(updated)
        second = current_context(dates, prices, day.iloc[:2], cfg)
        _, updated = stream.on_completed_bar(*second)
        self.assertTrue(updated)
        self.assertEqual(stream.agent.updates, 1)
        gap = current_context(dates, prices, day.iloc[[0, 1, 3]], cfg)
        _, updated = stream.on_completed_bar(*gap)
        self.assertFalse(updated)
        self.assertEqual(stream.agent.updates, 1)

    def test_current_day_daily_bar_is_excluded_and_label_is_next_minute(self):
        dates, prices, bars = fixture()
        _, sessions = make_sessions(dates, prices, bars)
        first = sessions[0]
        self.assertEqual(len(sessions), 12)
        self.assertLess(dates[first.daily_end-1], first.date)
        self.assertEqual(dates[first.daily_end], first.date)
        self.assertEqual(len(first.decision_indices), 4)
        np.testing.assert_array_equal(first.labels[0], [1, 0, 0])
        changed = prices.copy()
        changed[first.daily_end] *= 5
        changed_tokens, changed_sessions = make_sessions(dates, changed, bars)
        original_tokens, _ = make_sessions(dates, prices, bars)
        np.testing.assert_array_equal(original_tokens[:first.daily_end], changed_tokens[:first.daily_end])
        np.testing.assert_array_equal(first.minute_tokens, changed_sessions[0].minute_tokens)

    def test_prequential_updates_and_checkpoint(self):
        dates, prices, bars = fixture()
        tokens, sessions = make_sessions(dates, prices, bars)
        cfg = Settings(width=8, heads=2, daily_window=6, minute_window=3)
        result = run_replay(tokens, sessions, cfg)
        self.assertEqual(result["split_sessions"], {"train": 8, "validation": 2, "test": 2})
        predictions = result["predictions"]
        self.assertEqual(len(predictions), (8+2+2*2)*4)
        self.assertEqual(result["frozen_agent"].updates, 8*4)
        self.assertEqual(result["online_agent"].updates, (8+2)*4)
        first = predictions.iloc[0]
        self.assertEqual(pd.Timestamp(first.decision_time_utc)-pd.Timestamp(first.input_bar_start_utc),
                         pd.Timedelta(minutes=1))
        self.assertEqual(pd.Timestamp(first.outcome_available_utc)-pd.Timestamp(first.decision_time_utc),
                         pd.Timedelta(minutes=1))
        self.assertEqual(len(score_predictions(predictions)), 12)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agent.pt"
            result["online_agent"].save(path)
            restored = OnlineAgent.load(path)
            self.assertEqual(restored.updates, result["online_agent"].updates)
            daily = torch.from_numpy(tokens[:3])
            minute = torch.from_numpy(sessions[0].minute_tokens[:2])
            np.testing.assert_allclose(restored.predict(daily, minute),
                                       result["online_agent"].predict(daily, minute))


if __name__ == "__main__":
    unittest.main()
