"""Mathematical, regression, and timing checks for the SPY-only method."""
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spy_adaptive_lasso import (AdaptiveTwoStepLasso, ForecastConfig, make_samples,
                                piecewise_linear_signature, replay,
                                predict_directions, reconstruct_price_comparison, score_predictions,
                                select_factors)


def synthetic_samples():
    rng = np.random.default_rng(17)
    n = 290
    factor = rng.normal(size=n)
    independent = rng.normal(size=n)
    spy_return = 0.004 * factor + 0.001 * np.roll(factor, 1) + rng.normal(0, .004, n)
    close = 100 * np.exp(np.cumsum(spy_return))
    market = {"dates": pd.bdate_range("2020-01-01", periods=n),
              "spy_close": close, "spy_return": spy_return,
              "factors": np.column_stack([factor, independent]).astype(np.float32),
              "factor_names": ["related", "independent"], "coverage": {}}
    config = ForecastConfig(window=10, signature_depth=2, regime_depth=2,
                            regime_gamma=0, lasso_alpha=.00001,
                            correlation_min=.05, max_train=200)
    return make_samples(market, config), config


class AdaptiveTwoStepLassoTests(unittest.TestCase):
    def test_separate_direction_thresholds_and_neutral_recall(self):
        forecast = np.array([.004, -.005, .001, -.002, 0., .0015])
        np.testing.assert_array_equal(predict_directions(forecast, .001, -.002),
                                      [1, -1, 0, -1, 0, 1])
        frame = pd.DataFrame({'actual_return': [.003, -.004, .002, -.003, -.001, -.001],
                              'predicted_return': forecast})
        metric = score_predictions(frame, .001, -.002).iloc[0]
        self.assertAlmostEqual(metric.direction_accuracy, 3 / 6)
        self.assertAlmostEqual(metric.up_recall, 1 / 2)
        self.assertAlmostEqual(metric.down_recall, 2 / 4)
        self.assertAlmostEqual(metric.balanced_accuracy, .5)
        self.assertAlmostEqual(metric.predicted_neutral_rate, 2 / 6)
        self.assertEqual(metric.actual_up_days, 2)
        self.assertEqual(metric.actual_down_days, 4)
        self.assertEqual(metric.predicted_up_days, 2)
        self.assertEqual(metric.predicted_down_days, 2)
        self.assertEqual(metric.predicted_neutral_days, 2)
        self.assertEqual(metric.correct_up_days, 1)
        self.assertEqual(metric.correct_down_days, 2)
        self.assertAlmostEqual(metric.up_precision, .5)
        self.assertAlmostEqual(metric.down_precision, 1.)
        self.assertAlmostEqual(metric.predicted_up_rate + metric.predicted_down_rate
                               + metric.predicted_neutral_rate, 1.)
        self.assertEqual(metric.threshold_log_return, .001)
        self.assertEqual(metric.down_threshold_log_return, -.002)

    def test_direction_threshold_defaults_preserve_binary_rule(self):
        forecast = np.array([-.003, 0., .001, .003])
        y = np.array([-.002, 0., .002, -.001])
        frame = pd.DataFrame({'actual_return': y, 'predicted_return': forecast})
        for threshold in [0., .001, -.001]:
            metric = score_predictions(frame, threshold=threshold).iloc[0]
            self.assertEqual(metric.direction_accuracy,
                             np.mean((forecast > threshold) == (y > 0)))
            self.assertEqual(metric.predicted_neutral_rate, 0.)
        pd.testing.assert_frame_equal(score_predictions(frame), score_predictions(frame, 0., 0.))

    def test_direction_precision_without_predictions_is_undefined(self):
        frame = pd.DataFrame({'actual_return': [.001, -.001], 'predicted_return': [0., 0.]})
        metric = score_predictions(frame, .001, -.001).iloc[0]
        self.assertEqual(metric.correct_up_days, 0)
        self.assertEqual(metric.correct_down_days, 0)
        self.assertTrue(np.isnan(metric.up_precision))
        self.assertTrue(np.isnan(metric.down_precision))

    def test_direction_thresholds_reject_nonfinite_and_overlap(self):
        frame = pd.DataFrame({'actual_return': [.001], 'predicted_return': [.001]})
        for up, down in [(np.nan, 0.), (0., np.inf), (-.001, .001)]:
            with self.subTest(up=up, down=down), self.assertRaises(ValueError):
                score_predictions(frame, up, down)

    def test_straight_line_and_order(self):
        line = np.array([[[0.], [.3], [1.]]])
        np.testing.assert_allclose(piecewise_linear_signature(line, 3)[0],
                                   [1, .5, 1 / 6], atol=1e-7)
        xy = np.array([[[0., 0.], [1., 0.], [1., 1.]]])
        yx = np.array([[[0., 0.], [0., 1.], [1., 1.]]])
        sx = piecewise_linear_signature(xy, 2)[0]
        sy = piecewise_linear_signature(yx, 2)[0]
        self.assertAlmostEqual(sx[3], 1)
        self.assertAlmostEqual(sy[3], 0)

    def test_screen_prunes_collinear_factors(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=250)
        x = np.column_stack([y + rng.normal(0, .02, 250),
                             y + rng.normal(0, .02, 250),
                             rng.normal(size=250)])
        chosen, correlations = select_factors(
            x, y, ForecastConfig(correlation_min=.2, collinearity_max=.9))
        self.assertEqual(len(chosen), 1)
        self.assertIn(int(chosen[0]), (0, 1))
        self.assertGreater(abs(correlations[chosen[0]]), .9)

    def test_second_step_is_weighted_ols(self):
        samples, config = synthetic_samples()
        issue_idx = 185
        matured = samples["target_dates"].searchsorted(samples["issue_dates"][issue_idx], side="right")
        train = np.arange(0, matured)
        model = AdaptiveTwoStepLasso(config).fit(samples, train, issue_idx)
        self.assertGreater(len(model.selected_features), 0)
        weights = model._regime_weights(samples, train, issue_idx, model.selected_factors)
        design = np.column_stack([samples["x"][train][:, model.selected_factors],
                                  samples["y_signature"][train]])
        z = np.clip((design - model.feature_mean) / model.feature_scale, -8, 8) - model.feature_offset
        residual = samples["targets"][train] - model.y_mean - z @ model.coefficients
        normal_equation = z[:, model.selected_features].T @ (weights * residual)
        np.testing.assert_allclose(normal_equation, 0, atol=1e-8)
        outside = np.setdiff1d(np.arange(z.shape[1]), model.selected_features)
        np.testing.assert_array_equal(model.coefficients[outside], 0)

    def test_future_label_cannot_change_prior_forecast(self):
        samples, config = synthetic_samples()
        first = 180
        start = str(samples["issue_dates"][first].date())
        end = str(samples["issue_dates"][205].date())
        original = replay(samples, config, start, end)
        altered = dict(samples)
        altered["targets"] = samples["targets"].copy()
        altered["targets"][first + 1:] += 4
        again = replay(altered, config, start, end)
        self.assertEqual(original.predicted_return.iloc[0], again.predicted_return.iloc[0])
        self.assertEqual(set(original.ticker) if "ticker" in original else {"SPY"}, {"SPY"})
        self.assertEqual(len(score_predictions(original)), 1)

    def test_each_predicted_price_uses_previous_actual_close(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        market = {"dates": dates, "spy_close": np.array([100., 110., 90., 99.])}
        frame = pd.DataFrame({
            "issue_date": [str(dates[1].date()), str(dates[2].date())],
            "target_date": [str(dates[2].date()), str(dates[3].date())],
            "actual_return": [np.log(90 / 110), np.log(99 / 90)],
            "predicted_return": [np.log(1.1), np.log(1.1)],
        })
        prices = reconstruct_price_comparison(frame, market)
        np.testing.assert_allclose(prices.previous_actual_close, [110., 90.])
        np.testing.assert_allclose(prices.predicted_close, [121., 99.])
        np.testing.assert_allclose(prices.actual_close, [90., 99.])


if __name__ == "__main__":
    unittest.main()
