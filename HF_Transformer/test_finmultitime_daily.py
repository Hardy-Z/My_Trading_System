import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from finmultitime_daily import _filing_values, make_windows, news_features, NEWS_DIM


class CausalFeaturesTest(unittest.TestCase):
    def test_news_is_delayed_past_its_date(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "SPY.jsonl"
            path.write_text(json.dumps({"Date": "2024-01-02", "Article": "stocks rise"}) + "\n")
            dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
            with patch("finmultitime_daily._extract_members", return_value={"SPY": path}):
                values, counts = news_features(dates, Path(root), symbols=("SPY",))
            self.assertEqual(values.shape, (3, NEWS_DIM + 2))
            self.assertEqual(values[0, NEWS_DIM], 0)
            self.assertGreater(values[1, NEWS_DIM], 0)
            self.assertEqual(counts.tolist(), [1, 1])

    def test_filing_does_not_read_a_future_fact(self):
        filing = {"filing_date": "2024-02-01", "facts": {"us-gaap": {
            "Assets": {"units": {"USD": [
                {"end": "2023-12-31", "filed": "2024-01-31", "val": 100},
                {"end": "2023-12-31", "filed": "2024-02-15", "val": 1000},
            ]}},
            "Liabilities": {"units": {"USD": [{"end": "2023-12-31", "filed": "2024-01-31", "val": 40}]}},
        }}}
        values = _filing_values(filing)
        self.assertAlmostEqual(float(values[1]), .4)

    def test_window_ends_on_issue_date(self):
        dates = np.asarray(pd.date_range("2024-01-01", periods=6).strftime("%Y-%m-%d"), dtype="U10")
        sequence = np.arange(6, dtype=np.float32).reshape(6, 1)
        data = {"dates": dates, "labels": np.zeros((6, 3), np.float32),
                **{key: sequence for key in ("price", "news", "table", "image")}}
        windows, labels, issued = make_windows(data, window=3)
        self.assertEqual(issued.tolist(), [2, 3, 4])
        np.testing.assert_array_equal(windows["price"][0, :, 0], [0, 1, 2])
        self.assertEqual(len(labels), 3)


if __name__ == "__main__":
    unittest.main()
