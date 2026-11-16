import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'src'))
from backtest_signature_policy import (backtest_returns, percentile_allocation,
                                       performance, validate_predictions)


class SignaturePolicyTests(unittest.TestCase):
    def test_frozen_cdf_boundaries_order_and_ties(self):
        weights, ranks = percentile_allocation([-.03, -.02, 0, .02, .03], [-.02, 0, .02])
        np.testing.assert_allclose(weights, [.2, .3, .5, .7, .8])
        np.testing.assert_allclose(ranks, [0, 1/6, 1/2, 5/6, 1])
        self.assertAlmostEqual(percentile_allocation([0], [0, 0, 0])[0][0], .5)

    def test_future_forecasts_do_not_change_earlier_allocations(self):
        reference = [-.01, 0, .01]
        before = percentile_allocation([-.005, .005], reference)[0]
        after = percentile_allocation([-.005, .005, -100, 100], reference)[0]
        np.testing.assert_array_equal(before, after[:2])

    def test_self_financing_compounding_cash_and_sharpe(self):
        ledger = backtest_returns([.10, -.20], [.8, .2])
        np.testing.assert_allclose(ledger.wealth, [108, 103.68])
        np.testing.assert_allclose(ledger.cash_dollars, [20, 86.4])
        np.testing.assert_allclose(ledger.cash_dollars + ledger.invested_dollars, ledger.wealth_before)
        np.testing.assert_allclose(ledger.wealth_before + ledger.pnl_dollars, ledger.wealth)
        metrics = performance(ledger)
        self.assertAlmostEqual(metrics['total_return_pct'], 3.68)
        self.assertAlmostEqual(metrics['max_drawdown_pct'], 4)
        self.assertAlmostEqual(metrics['annualized_sharpe_rf0'], np.sqrt(14))

    def test_drawdown_includes_starting_capital(self):
        ledger = backtest_returns([-.1, .2], [.8, .2])
        np.testing.assert_allclose(ledger.wealth, [92, 95.68])
        self.assertAlmostEqual(performance(ledger)['max_drawdown_pct'], 8)

    def test_cash_only_and_log_to_simple_conversion(self):
        realized = np.expm1([np.log(1.10), np.log(.80)])
        np.testing.assert_allclose(backtest_returns(realized, [1, 1]).wealth, [110, 88])
        cash = backtest_returns(realized, [0, 0])
        np.testing.assert_allclose(cash.wealth, [100, 100])
        self.assertIsNone(performance(cash)['annualized_sharpe_rf0'])

    def test_reject_nonfinite_or_leveraged_inputs(self):
        for returns, weights in [([np.nan], [.5]), ([0], [1.1]), ([-1], [.5])]:
            with self.assertRaises(ValueError):
                backtest_returns(returns, weights)
        with self.assertRaises(ValueError):
            percentile_allocation([0], [])

    def test_prediction_dates_and_prices_must_match(self):
        prices = pd.DataFrame({'open': [100, 100, 110], 'close': [100, 110, 99]},
                              index=pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']))
        frame = pd.DataFrame({'issue_date': ['2024-01-02', '2024-01-03'],
                              'target_date': ['2024-01-03', '2024-01-04'],
                              'predicted_return': [.01, -.01],
                              'actual_return': np.log([1.1, .9])})
        self.assertEqual(len(validate_predictions(frame, prices)), 2)
        incorrect = frame.copy()
        incorrect.loc[0, 'target_date'] = '2024-01-04'
        with self.assertRaises(ValueError):
            validate_predictions(incorrect, prices)
        incorrect = frame.copy()
        incorrect.loc[0, 'actual_return'] = .1
        with self.assertRaises(AssertionError):
            validate_predictions(incorrect, prices)


if __name__ == '__main__':
    unittest.main()
