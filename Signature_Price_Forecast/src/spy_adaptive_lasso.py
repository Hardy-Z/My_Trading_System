"""Causal SPY next-session log-return forecast via signature adaptive two-step Lasso.

The paper's "adaptive" part is a signature-kernel sample weight, not an
adaptive per-coefficient penalty. Step 1 is weighted Lasso; step 2 is weighted
OLS on its selected support. Every replay prediction refits using only labels
whose target close is already observed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SPY"
PRICE_FACTOR_INDICES = (7, 8, 10, 11, 12, 13, 14, 15)
PRICE_FACTOR_NAMES = (
    "SPY_high_low_range", "SPY_log_volume", "QQQ_high_low_range",
    "QQQ_log_volume", "SP500_mean_return", "SP500_return_dispersion",
    "SP500_fraction_up", "SP500_log_coverage",
)
TABLE_FACTOR_NAMES = (
    "filing_assets_scale", "filing_liabilities_assets", "filing_equity_assets",
    "filing_cash_assets", "filing_coverage",
)


@dataclass(frozen=True)
class ForecastConfig:
    window: int = 20
    signature_depth: int = 3
    regime_depth: int = 2
    horizon: int = 1
    lasso_alpha: float = 0.0005
    regime_gamma: float = 0.5
    correlation_min: float = 0.05
    collinearity_max: float = 0.95
    max_factors: int = 8
    regime_factors: int = 3
    max_train: int = 756
    lasso_passes: int = 80
    max_selected: int = 20

    def check(self) -> None:
        if not 5 <= self.window <= 252:
            raise ValueError("window must be 5..252")
        if not 1 <= self.signature_depth <= 4 or not 1 <= self.regime_depth <= 3:
            raise ValueError("signature_depth must be 1..4 and regime_depth 1..3")
        if not 1 <= self.horizon <= 20 or self.max_train < 100:
            raise ValueError("horizon must be 1..20 and max_train at least 100")
        if self.lasso_alpha < 0 or self.regime_gamma < 0:
            raise ValueError("lasso_alpha and regime_gamma must be nonnegative")
        if not 0 <= self.correlation_min < 1 or not 0 < self.collinearity_max <= 1:
            raise ValueError("correlation thresholds must lie in [0,1]")
        if min(self.max_factors, self.regime_factors, self.lasso_passes, self.max_selected) < 1:
            raise ValueError("factor and iteration counts must be positive")


def _outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", left, right, optimize=True).reshape(len(left), -1)


def piecewise_linear_signature(paths: np.ndarray, depth: int) -> np.ndarray:
    """Exact truncated signature for a batch of piecewise-linear paths."""
    paths = np.asarray(paths, np.float64)
    if paths.ndim != 3 or paths.shape[1] < 2 or not 1 <= depth <= 4:
        raise ValueError("paths must have shape (N, L>=2, channels); depth 1..4")
    n, _, channels = paths.shape
    levels = [np.ones((n, 1))] + [np.zeros((n, channels**k)) for k in range(1, depth + 1)]
    for increment in np.diff(paths, axis=1).transpose(1, 0, 2):
        exponential = [levels[0], increment]
        for level in range(2, depth + 1):
            exponential.append(_outer(exponential[level - 1], increment) / level)
        next_levels = [levels[0]]
        for level in range(1, depth + 1):
            value = levels[level] + exponential[level]
            for earlier in range(1, level):
                value = value + _outer(levels[earlier], exponential[level - earlier])
            next_levels.append(value)
        levels = next_levels
    return np.concatenate(levels[1:], axis=1).astype(np.float32)


def _local_etf_data(root: Path) -> pd.DataFrame:
    frames = []
    for ticker in ("SPY", "QQQ", "IWM"):
        frame = pd.read_csv(root / "SP500" / "attention_data" / f"{ticker}_daily.csv",
                            usecols=["date", "open", "close"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frames.append(frame.rename(columns=lambda c: f"{ticker}_{c}"))
    joined = pd.concat(frames, axis=1, join="inner").dropna()
    if joined.empty or not joined.index.is_unique or (joined <= 0).any().any():
        raise ValueError("ETF prices must be positive with unique common dates")
    return joined


def load_market_data(root: Path, *, use_finmultitime: bool = True,
                     start: str = "2018-01-01", end: str = "2025-03-28",
                     cache_dir: Path | None = None) -> dict:
    """Load SPY y and same-close quantitative X; exclude text and image inputs."""
    root = root.resolve()
    etf = _local_etf_data(root).loc[start:end]
    if len(etf) < 150:
        raise ValueError("Need at least 150 common ETF sessions")
    dates = pd.DatetimeIndex(etf.index)
    close = etf[[f"{ticker}_close" for ticker in ("SPY", "QQQ", "IWM")]].to_numpy(np.float64)
    opened = etf[[f"{ticker}_open" for ticker in ("SPY", "QQQ", "IWM")]].to_numpy(np.float64)
    daily_returns = np.vstack([np.zeros((1, 3)), np.diff(np.log(close), axis=0)])
    intraday = np.log(close / opened)
    volatility = pd.DataFrame(daily_returns, index=dates).rolling(20, min_periods=5).std().fillna(0).to_numpy()
    parts = [daily_returns[:, 1:], intraday[:, 1:], volatility[:, 1:]]
    names = ["QQQ_return", "IWM_return", "QQQ_intraday", "IWM_intraday",
             "QQQ_volatility20", "IWM_volatility20"]
    coverage = {"finmultitime_quantitative": False}
    if use_finmultitime:
        sys.path.insert(0, str(root / "HF_Transformer"))
        from finmultitime_daily import download_price_archive, price_features, table_features

        numeric_cache = cache_dir or Path(tempfile.gettempdir()) / "codex_finmultitime_source"
        target_dir = root / "SP500" / "attention_data"
        price_zip = download_price_archive(numeric_cache)
        base_cache = numeric_cache / "price_base.npz"
        base_manifest = numeric_cache / "price_base_manifest.json"
        source_stat = {"price_zip": [price_zip.stat().st_size, price_zip.stat().st_mtime_ns],
                       "targets": {symbol: [(target_dir / f"{symbol}_daily.csv").stat().st_size,
                                             (target_dir / f"{symbol}_daily.csv").stat().st_mtime_ns]
                                   for symbol in ("SPY", "QQQ", "IWM")}}
        if (base_cache.exists() and base_manifest.exists()
                and json.loads(base_manifest.read_text()) == source_stat):
            with np.load(base_cache) as base:
                source_dates = pd.DatetimeIndex(pd.to_datetime(base["dates"]))
                source_price = base["price"]
        else:
            source_dates, source_price, source_labels = price_features(price_zip, target_dir)
            np.savez_compressed(base_cache,
                                dates=np.asarray(source_dates.strftime("%Y-%m-%d"), dtype="U10"),
                                price=source_price, labels=source_labels)
            base_manifest.write_text(json.dumps(source_stat, indent=2))
        positions = source_dates.get_indexer(dates)
        if (positions < 0).any():
            raise ValueError("FinMultiTime does not cover every selected ETF session")
        price = source_price[positions]
        table, _ = table_features(dates, numeric_cache)
        parts += [price[:, PRICE_FACTOR_INDICES], table]
        names += list(PRICE_FACTOR_NAMES) + list(TABLE_FACTOR_NAMES)
        coverage = {"finmultitime_quantitative": True,
                    "filing_sessions": int(np.count_nonzero(table[:, -1] > 0)),
                    "factor_count": len(names),
                    "text_and_images_used": False}
    factors = np.column_stack(parts).astype(np.float32)
    if not np.isfinite(factors).all():
        raise ValueError("Quantitative factors contain non-finite values")
    return {"dates": dates, "spy_close": close[:, 0], "spy_return": daily_returns[:, 0],
            "factors": factors, "factor_names": names, "coverage": coverage}


def make_samples(market: dict, config: ForecastConfig) -> dict:
    """Issue-close features and next-horizon SPY log-return labels."""
    config.check()
    dates = market["dates"]
    log_spy = np.log(market["spy_close"])
    x = market["factors"]
    issue_indices = np.arange(config.window - 1, len(dates) - config.horizon)
    if len(issue_indices) < 120:
        raise ValueError("Need at least 120 labeled windows")
    n = len(issue_indices)
    y_paths = np.empty((n, config.window, 2), np.float32)
    y_paths[:, :, 0] = np.linspace(0, 1, config.window)
    x_paths = np.empty((n, config.window, x.shape[1]), np.float32)
    for row, issue in enumerate(issue_indices):
        y_slice = log_spy[issue - config.window + 1:issue + 1]
        y_paths[row, :, 1] = y_slice - y_slice[0]
        x_paths[row] = x[issue - config.window + 1:issue + 1]
    y_signature = piecewise_linear_signature(y_paths, config.signature_depth)
    target_indices = issue_indices + config.horizon
    targets = (log_spy[target_indices] - log_spy[issue_indices]).astype(np.float64)
    return {"issue_dates": dates[issue_indices], "target_dates": dates[target_indices],
            "x": x[issue_indices].astype(np.float64), "x_paths": x_paths,
            "y_paths": y_paths, "y_signature": y_signature.astype(np.float64),
            "y_issue": market["spy_return"][issue_indices].astype(np.float64),
            "targets": targets, "factor_names": market["factor_names"],
            "coverage": market["coverage"]}


def select_factors(x: np.ndarray, y_current: np.ndarray, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray]:
    """Training-only correlation screen followed by collinearity pruning."""
    config.check()
    centered = x - x.mean(axis=0)
    y_centered = y_current - y_current.mean()
    x_norm = np.sqrt(np.mean(centered**2, axis=0))
    y_norm = max(float(np.sqrt(np.mean(y_centered**2))), 1e-12)
    correlations = np.divide(np.mean(centered * y_centered[:, None], axis=0),
                             x_norm * y_norm, out=np.zeros(x.shape[1]), where=x_norm > 1e-10)
    correlations = np.clip(correlations, -1, 1)
    ranking = np.argsort(-np.abs(correlations), kind="stable")
    chosen = []
    for index in ranking:
        if abs(correlations[index]) < config.correlation_min and chosen:
            break
        if x_norm[index] <= 1e-10:
            continue
        if chosen:
            pairwise = np.mean(centered[:, chosen] * centered[:, index, None], axis=0)
            pairwise /= x_norm[chosen] * x_norm[index]
            if np.max(np.abs(pairwise)) >= config.collinearity_max:
                continue
        chosen.append(int(index))
        if len(chosen) >= config.max_factors:
            break
    if not chosen:
        raise ValueError("No nonconstant quantitative factors in the training data")
    return np.asarray(chosen, dtype=int), correlations


def _lasso_from_gram(gram: np.ndarray, xy: np.ndarray, alpha: float, passes: int) -> np.ndarray:
    """Coordinate descent for 0.5 * weighted MSE + alpha * L1."""
    coefficients = np.zeros_like(xy)
    for _ in range(passes):
        old = coefficients.copy()
        for j in range(len(xy)):
            if gram[j, j] <= 1e-12:
                continue
            partial = xy[j] - gram[j] @ coefficients + gram[j, j] * coefficients[j]
            coefficients[j] = np.sign(partial) * max(abs(partial) - alpha, 0) / gram[j, j]
        if np.max(np.abs(coefficients - old)) < 1e-8:
            break
    return coefficients


class AdaptiveTwoStepLasso:
    """Correlation screen -> signature weights -> weighted Lasso -> weighted OLS."""

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.selected_factors = None
        self.selected_features = None
        self.feature_mean = None
        self.feature_scale = None
        self.feature_offset = None
        self.y_mean = None
        self.coefficients = None
        self.effective_train_n = None

    def _regime_weights(self, samples: dict, train_idx: np.ndarray,
                        issue_idx: int, chosen: np.ndarray) -> np.ndarray:
        regime_columns = chosen[:self.config.regime_factors]
        x_scale = np.maximum(samples["x"][train_idx][:, regime_columns].std(axis=0), 1e-5)
        y_scale = max(samples["y_issue"][train_idx].std() * np.sqrt(self.config.window), 1e-4)
        indices = np.append(train_idx, issue_idx)
        y_paths = samples["y_paths"][indices].astype(np.float64)
        x_paths = samples["x_paths"][indices][:, :, regime_columns].astype(np.float64)
        x_paths = np.clip(x_paths / x_scale, -8, 8)
        x_paths -= x_paths[:, :1, :]
        joint = np.concatenate([y_paths[:, :, :1], y_paths[:, :, 1:2] / y_scale, x_paths], axis=2)
        signatures = piecewise_linear_signature(joint, self.config.regime_depth).astype(np.float64)
        history, current = signatures[:-1], signatures[-1]
        mean = history.mean(axis=0)
        scale = np.maximum(history.std(axis=0), 1e-5)
        history = np.clip((history - mean) / scale, -8, 8)
        current = np.clip((current - mean) / scale, -8, 8)
        # Finite normalized signature kernel: k(a,b)=<phi(a),phi(b)>/D.
        # Its induced distance is k(a,a)-2k(a,b)+k(b,b).
        distance = np.mean((history - current) ** 2, axis=1)
        logits = -self.config.regime_gamma * distance
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        self.effective_train_n = float(1 / np.sum(weights**2))
        return weights

    def fit(self, samples: dict, train_idx: np.ndarray, issue_idx: int):
        train_idx = np.asarray(train_idx, dtype=int)
        if len(train_idx) < 100:
            raise ValueError("Need at least 100 matured labels")
        if np.any(samples["target_dates"][train_idx] > samples["issue_dates"][issue_idx]):
            raise ValueError("Training includes a label not available at the issue close")
        selected, _ = select_factors(samples["x"][train_idx],
                                     samples["y_issue"][train_idx], self.config)
        self.selected_factors = selected
        weights = self._regime_weights(samples, train_idx, issue_idx, selected)
        x_train = np.column_stack([samples["x"][train_idx][:, selected],
                                   samples["y_signature"][train_idx]])
        labels = samples["targets"][train_idx]
        self.feature_mean = weights @ x_train
        self.feature_scale = np.maximum(np.sqrt(weights @ ((x_train - self.feature_mean) ** 2)), 1e-5)
        z = np.clip((x_train - self.feature_mean) / self.feature_scale, -8, 8)
        self.feature_offset = weights @ z
        z -= self.feature_offset
        self.y_mean = float(weights @ labels)
        centered_y = labels - self.y_mean
        gram = z.T @ (weights[:, None] * z)
        xy = z.T @ (weights * centered_y)
        first_step = _lasso_from_gram(gram, xy, self.config.lasso_alpha,
                                      self.config.lasso_passes)
        support = np.flatnonzero(np.abs(first_step) > 1e-10)
        if len(support) > self.config.max_selected:
            strongest = np.argsort(-np.abs(first_step[support]), kind="stable")
            support = np.sort(support[strongest[:self.config.max_selected]])
        self.selected_features = support
        self.coefficients = np.zeros(z.shape[1], np.float64)
        if len(support):
            # Paper Eq. 15: weighted OLS on the first-step Lasso support.
            sqrt_w = np.sqrt(weights)
            design = sqrt_w[:, None] * z[:, support]
            response = sqrt_w * centered_y
            self.coefficients[support] = np.linalg.lstsq(design, response, rcond=1e-8)[0]
        return self

    def predict_return(self, samples: dict, issue_idx: int) -> float:
        x = np.concatenate([samples["x"][issue_idx, self.selected_factors],
                            samples["y_signature"][issue_idx]])
        z = np.clip((x - self.feature_mean) / self.feature_scale, -8, 8) - self.feature_offset
        return float(self.y_mean + z @ self.coefficients)


def replay(samples: dict, config: ForecastConfig, start: str, end: str) -> pd.DataFrame:
    """Daily online refit: forecast using only labels matured by each issue close."""
    config.check()
    issue, target = samples["issue_dates"], samples["target_dates"]
    selected = np.flatnonzero((issue >= pd.Timestamp(start)) & (issue <= pd.Timestamp(end)))
    if len(selected) < 20:
        raise ValueError("Evaluation period needs at least 20 issue sessions")
    rows = []
    model = AdaptiveTwoStepLasso(config)
    for index in selected:
        matured = int(target.searchsorted(issue[index], side="right"))
        train_idx = np.arange(max(0, matured - config.max_train), matured)
        model.fit(samples, train_idx, int(index))
        rows.append({"issue_date": issue[index].date().isoformat(),
                     "target_date": target[index].date().isoformat(),
                     "actual_return": samples["targets"][index],
                     "predicted_return": model.predict_return(samples, int(index)),
                     "screened_factor_count": len(model.selected_factors),
                     "lasso_support_count": len(model.selected_features),
                     "selected_factors": ";".join(samples["factor_names"][j]
                                                  for j in model.selected_factors),
                     "effective_train_n": model.effective_train_n})
    return pd.DataFrame(rows)


def predict_directions(forecast, up_threshold: float = 0.0,
                       down_threshold: float | None = None) -> np.ndarray:
    """Return +1 (up), -1 (down), or 0 (neutral) from log-return cutoffs.

    Up requires forecast > up_threshold; down requires forecast <= down_threshold.
    The interval (down_threshold, up_threshold] is neutral. A missing down
    threshold uses the up threshold, preserving the original binary rule.
    """
    if down_threshold is None:
        down_threshold = up_threshold
    if not np.isfinite(up_threshold) or not np.isfinite(down_threshold):
        raise ValueError("Direction thresholds must be finite log returns")
    if down_threshold > up_threshold:
        raise ValueError("down_threshold must be <= up_threshold")
    forecast = np.asarray(forecast, dtype=np.float64)
    return np.where(forecast > up_threshold, 1,
                    np.where(forecast <= down_threshold, -1, 0))


def score_predictions(frame: pd.DataFrame, threshold: float = 0.0,
                      down_threshold: float | None = None) -> pd.DataFrame:
    """Return errors and direction metrics; neutral forecasts count as misses.

    ``threshold`` is the up cutoff, retained for compatibility with callers.
    Actual returns > 0 are up; all other actual returns are down, as before.
    """
    y = frame.actual_return.to_numpy(np.float64)
    forecast = frame.predicted_return.to_numpy(np.float64)
    actual_up = y > 0
    direction = predict_directions(forecast, threshold, down_threshold)
    predicted_up, predicted_down = direction == 1, direction == -1
    correct_up_days = int((predicted_up & actual_up).sum())
    correct_down_days = int((predicted_down & ~actual_up).sum())
    up_recall = float(predicted_up[actual_up].mean()) if actual_up.any() else np.nan
    down_recall = float(predicted_down[~actual_up].mean()) if (~actual_up).any() else np.nan
    return pd.DataFrame([{"ticker": TARGET, "n": len(frame),
                          "actual_up_days": int(actual_up.sum()),
                          "actual_down_days": int((~actual_up).sum()),
                          "predicted_up_days": int(predicted_up.sum()),
                          "predicted_down_days": int(predicted_down.sum()),
                          "predicted_neutral_days": int((direction == 0).sum()),
                          "correct_up_days": correct_up_days,
                          "correct_down_days": correct_down_days,
                          "up_precision": correct_up_days / int(predicted_up.sum()) if predicted_up.any() else np.nan,
                          "down_precision": correct_down_days / int(predicted_down.sum()) if predicted_down.any() else np.nan,
                          "threshold_log_return": float(threshold),
                          "down_threshold_log_return": float(threshold if down_threshold is None else down_threshold),
                          "direction_accuracy": float(np.mean(direction == np.where(actual_up, 1, -1))),
                          "up_recall": up_recall, "down_recall": down_recall,
                          "balanced_accuracy": (up_recall + down_recall) / 2,
                          "majority_baseline": float(max(actual_up.mean(), 1 - actual_up.mean())),
                          "up_rate": float(actual_up.mean()),
                          "predicted_up_rate": float(predicted_up.mean()),
                          "predicted_down_rate": float(predicted_down.mean()),
                          "predicted_neutral_rate": float((direction == 0).mean()),
                          "return_mae": float(np.mean(np.abs(forecast - y))),
                          "return_rmse": float(np.sqrt(np.mean((forecast - y) ** 2))),
                          "zero_return_mae": float(np.mean(np.abs(y)))}])


def print_direction_counts(metric: dict | pd.Series, label: str = "Evaluation") -> None:
    """Print day counts from the same direction rules used for accuracy."""
    print(f"{label} direction counts ({int(metric['n'])} days):")
    print(f"  Predicted UP: {int(metric['predicted_up_days'])} days; "
          f"DOWN: {int(metric['predicted_down_days'])} days; "
          f"NEUTRAL: {int(metric['predicted_neutral_days'])} days")
    print(f"  Actual UP: {int(metric['actual_up_days'])} days; "
          f"DOWN (including zero returns): {int(metric['actual_down_days'])} days")


def reconstruct_price_comparison(frame: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Evaluate return forecasts as prices using each issue day's actual close.

    The forecaster still produces only a log return. This evaluation helper
    does not feed a predicted price into the next day's forecast.
    """
    closes = pd.Series(np.asarray(market["spy_close"], np.float64),
                       index=pd.DatetimeIndex(market["dates"]))
    issue_dates = pd.DatetimeIndex(pd.to_datetime(frame.issue_date))
    target_dates = pd.DatetimeIndex(pd.to_datetime(frame.target_date))
    previous_close = closes.reindex(issue_dates).to_numpy()
    actual_close = closes.reindex(target_dates).to_numpy()
    if not np.isfinite(previous_close).all() or not np.isfinite(actual_close).all():
        raise ValueError("Issue or target close is missing from market data")
    implied_actual = previous_close * np.exp(frame.actual_return.to_numpy(np.float64))
    if not np.allclose(implied_actual, actual_close, rtol=1e-10, atol=1e-8):
        raise ValueError("Actual log returns do not match the market closes")
    predicted_close = previous_close * np.exp(frame.predicted_return.to_numpy(np.float64))
    return pd.DataFrame({"issue_date": frame.issue_date.to_numpy(),
                         "target_date": frame.target_date.to_numpy(),
                         "previous_actual_close": previous_close,
                         "actual_close": actual_close,
                         "predicted_close": predicted_close,
                         "actual_return": frame.actual_return.to_numpy(np.float64),
                         "predicted_return": frame.predicted_return.to_numpy(np.float64)})


def save_run(folder: Path, frame: pd.DataFrame, config: ForecastConfig,
             scores: pd.DataFrame, coverage: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "predictions.csv", index=False)
    scores.to_csv(folder / "metrics.csv", index=False)
    (folder / "run_info.json").write_text(json.dumps({
        "model": "SPY signature adaptive two-step Lasso",
        "target": "next-session SPY log return",
        "config": asdict(config), "coverage": coverage,
        "first_issue": frame.issue_date.min(), "last_issue": frame.issue_date.max(),
    }, indent=2))
