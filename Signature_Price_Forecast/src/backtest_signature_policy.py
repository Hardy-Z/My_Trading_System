"""Backtest saved signature forecasts using a frozen pre-test forecast percentile.

Run from the repository root:
    .venv/Scripts/python.exe Signature_Price_Forecast/src/backtest_signature_policy.py

The main close-to-close result is an idealized signal study: forecasts use the
issue close, so a fill at that exact close is not executable without latency.
The next-open-to-close comparison enters after the signal is available and
liquidates daily; it excludes overnight returns and is a different holding rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_allocation(predicted, reference, minimum=0.20, maximum=0.80):
    """Map forecasts to frozen empirical mid-CDF percentiles; ties get midrank."""
    forecasts = np.asarray(predicted, dtype=float)
    calibration = np.sort(np.asarray(reference, dtype=float))
    if forecasts.ndim != 1 or calibration.ndim != 1 or not len(calibration):
        raise ValueError("Forecasts and nonempty calibration must be one-dimensional")
    if not np.isfinite(forecasts).all() or not np.isfinite(calibration).all():
        raise ValueError("Forecasts and calibration must be finite")
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("Allocation bounds must satisfy 0 <= minimum <= maximum <= 1")
    left = calibration.searchsorted(forecasts, side="left")
    right = calibration.searchsorted(forecasts, side="right")
    percentiles = (left + right) / (2.0 * len(calibration))
    return minimum + (maximum - minimum) * percentiles, percentiles


def backtest_returns(asset_simple_returns, weights, initial_wealth=100.0):
    """Self-financing daily rebalancing with zero costs and zero cash return."""
    realized = np.asarray(asset_simple_returns, dtype=float)
    allocation = np.asarray(weights, dtype=float)
    if realized.ndim != 1 or allocation.shape != realized.shape or not len(realized):
        raise ValueError("Returns and weights must be nonempty aligned vectors")
    if not np.isfinite(realized).all() or not np.isfinite(allocation).all():
        raise ValueError("Returns and weights must be finite")
    if np.any(realized <= -1) or np.any((allocation < 0) | (allocation > 1)):
        raise ValueError("Asset returns must exceed -100%; allocation must be 0..1")
    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive and finite")
    portfolio_return = allocation * realized
    wealth = initial_wealth * np.cumprod(1 + portfolio_return)
    before = np.r_[initial_wealth, wealth[:-1]]
    peak = np.maximum.accumulate(np.r_[initial_wealth, wealth])[1:]
    return pd.DataFrame({
        "allocation": allocation,
        "asset_simple_return": realized,
        "portfolio_simple_return": portfolio_return,
        "wealth_before": before,
        "invested_dollars": before * allocation,
        "cash_dollars": before * (1 - allocation),
        "pnl_dollars": before * portfolio_return,
        "wealth": wealth,
        "drawdown": wealth / peak - 1,
    })


def performance(ledger, initial_wealth=100.0, sessions_per_year=252):
    """Cumulative return, daily-marked drawdown, sample-volatility Sharpe (rf=0)."""
    returns = ledger.portfolio_simple_return.to_numpy(float)
    standard_deviation = np.std(returns, ddof=1) if len(returns) > 1 else np.nan
    sharpe = (math.sqrt(sessions_per_year) * returns.mean() / standard_deviation
              if np.isfinite(standard_deviation) and standard_deviation > 1e-15 else None)
    return {
        "sessions": len(ledger),
        "initial_wealth": float(initial_wealth),
        "final_wealth": float(ledger.wealth.iloc[-1]),
        "total_return_pct": float(100 * (ledger.wealth.iloc[-1] / initial_wealth - 1)),
        "max_drawdown_pct": float(-100 * ledger.drawdown.min()),
        "annualized_sharpe_rf0": float(sharpe) if sharpe is not None else None,
        "mean_allocation_pct": float(100 * ledger.allocation.mean()),
        "min_allocation_pct": float(100 * ledger.allocation.min()),
        "max_allocation_pct": float(100 * ledger.allocation.max()),
    }


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_predictions(frame, prices):
    """Require consecutive one-session labels and verify every realized return."""
    frame = frame.copy()
    required = {"issue_date", "target_date", "predicted_return", "actual_return"}
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {required - set(frame)}")
    for column in ("issue_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame = frame.sort_values("issue_date").reset_index(drop=True)
    if frame.empty or not frame.issue_date.is_unique or not frame.target_date.is_unique:
        raise ValueError("Predictions must have nonempty, unique issue and target dates")
    if not np.isfinite(frame[["predicted_return", "actual_return"]].to_numpy(float)).all():
        raise ValueError("Prediction returns must be finite")
    indices = prices.index.get_indexer(frame.issue_date)
    target_indices = prices.index.get_indexer(frame.target_date)
    if np.any(indices < 0) or np.any(target_indices != indices + 1):
        raise ValueError("Each target must be the next available price session")
    if len(frame) > 1 and not np.array_equal(frame.issue_date.to_numpy()[1:], frame.target_date.to_numpy()[:-1]):
        raise ValueError("Missing or overlapping test holding periods")
    observed = np.log(prices.close.to_numpy()[target_indices] / prices.close.to_numpy()[indices])
    np.testing.assert_allclose(frame.actual_return, observed, atol=2e-12, rtol=0,
                               err_msg="Saved actual returns differ from source closes")
    return frame


def build_calibration(project, test, info, source_cache, output):
    from spy_adaptive_lasso import (AdaptiveTwoStepLasso, ForecastConfig,
                                    load_market_data, make_samples, replay)

    config = ForecastConfig(**info["config"])
    if config.horizon != 1:
        raise ValueError("This policy requires nonoverlapping one-session forecasts")
    market = load_market_data(project.parent, cache_dir=source_cache)
    samples = make_samples(market, config)
    first_issue = pd.Timestamp(test.issue_date.iloc[0])
    matured = int(samples["target_dates"].searchsorted(first_issue, side="right"))
    calibration_indices = np.arange(max(0, matured - config.max_train), matured)
    if len(calibration_indices) < 100:
        raise ValueError("Need at least 100 matured pre-test forecasts for calibration")
    if np.any(samples["issue_dates"][calibration_indices] >= first_issue):
        raise ValueError("Calibration includes a test-period forecast")

    # Check that calibration uses exactly the forecaster behind the saved results.
    anchors = []
    for row in sorted({0, len(test) // 2, len(test) - 1}):
        date = pd.Timestamp(test.issue_date.iloc[row])
        sample_index = int(samples["issue_dates"].get_indexer([date])[0])
        if sample_index < 0:
            raise ValueError("Saved test date is absent from the forecasting samples")
        eligible = int(samples["target_dates"].searchsorted(date, side="right"))
        train = np.arange(max(0, eligible - config.max_train), eligible)
        model = AdaptiveTwoStepLasso(config).fit(samples, train, sample_index)
        reproduced = model.predict_return(samples, sample_index)
        saved = float(test.predicted_return.iloc[row])
        if not np.isclose(reproduced, saved, rtol=1e-8, atol=1e-11):
            raise ValueError(f"Current model does not reproduce saved prediction on {date.date()}")
        anchors.append({"issue_date": str(date.date()), "saved": saved, "reproduced": reproduced})

    identity = hashlib.sha256()
    identity.update(json.dumps(asdict(config), sort_keys=True).encode())
    identity.update(str(first_issue.date()).encode())
    identity.update((project / "src" / "spy_adaptive_lasso.py").read_bytes())
    for name in ("spy_close", "factors"):
        identity.update(np.ascontiguousarray(market[name]).tobytes())
    identity = identity.hexdigest()
    saved_csv = output / "calibration_predictions.csv"
    saved_info = output / "calibration_info.json"
    if saved_csv.exists() and saved_info.exists():
        previous = json.loads(saved_info.read_text(encoding="utf8"))
        if previous.get("identity") == identity and previous.get("csv_sha256") == file_hash(saved_csv):
            print("Reusing verified pre-test calibration forecasts.", flush=True)
            return pd.read_csv(saved_csv), previous, anchors

    parts = []
    completed = 0
    for chunk in np.array_split(calibration_indices, math.ceil(len(calibration_indices) / 100)):
        parts.append(replay(samples, config,
                            str(samples["issue_dates"][chunk[0]].date()),
                            str(samples["issue_dates"][chunk[-1]].date())))
        completed += len(chunk)
        print(f"Pre-test calibration forecasts: {completed}/{len(calibration_indices)}", flush=True)
    calibration = pd.concat(parts, ignore_index=True)
    if len(calibration) != len(calibration_indices):
        raise ValueError("The calibration forecast count differs from the selected training window")
    if (pd.to_datetime(calibration.issue_date) >= first_issue).any() or (pd.to_datetime(calibration.target_date) > first_issue).any():
        raise ValueError("A calibration outcome was not known at the first test issue close")
    calibration.to_csv(saved_csv, index=False)
    metadata = {
        "identity": identity, "csv_sha256": file_hash(saved_csv),
        "method": "Frozen empirical mid-CDF of causal pre-test walk-forward forecasts",
        "forecast_count": len(calibration),
        "issue_first": calibration.issue_date.iloc[0],
        "issue_last": calibration.issue_date.iloc[-1],
        "latest_target": calibration.target_date.iloc[-1],
        "known_by_issue_close": str(first_issue.date()),
        "config": asdict(config),
        "selection": "Most recent max_train matured examples at the first saved test issue; model settings from saved validation selection",
    }
    saved_info.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return calibration, metadata, anchors


def save_plot(output, dates, ledgers, initial_wealth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ["#1764ab", "#808080", "#dc8500"]
    styles = ["-", "--", "-"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    for (name, ledger), color, style in zip(ledgers.items(), colors, styles):
        wealth = np.r_[initial_wealth, ledger.wealth.to_numpy()]
        drawdown = np.r_[0, ledger.drawdown.to_numpy()]
        axes[0].plot(dates, wealth, color=color, linestyle=style,
                     label=f"{name} | final ${wealth[-1]:.2f}", linewidth=1.5)
        axes[1].plot(dates, drawdown, color=color, linestyle=style, linewidth=1.2)
    axes[0].set(title="SPY held-out test | idealized close-to-close rebalancing",
                ylabel="Portfolio wealth (USD)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set(ylabel="Drawdown from peak", xlabel="Valuation date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    fig.savefig(output / "equity_and_drawdown.png", dpi=160)
    plt.close(fig)


def write_notebook(project):
    """A companion notebook displays results without rerunning the forecast notebook."""
    import nbformat
    path = project / "signature_trading_policy.ipynb"
    if path.exists():
        return  # Preserve any subsequent user edits.
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Signature forecast trading policy\n\n"
            "Initial wealth: **$100**. The SPY allocation is `0.20 + 0.60 × percentile`, "
            "using a frozen distribution of pre-test walk-forward forecasts. Fractional shares, "
            "zero costs, zero cash interest, and a 0% risk-free rate are assumed.\n\n"
            "The close-to-close result assumes an idealized same-close fill after observing "
            "that close. The next-open-to-close row trades after the signal is available "
            "and holds cash overnight. Maximum drawdown includes the initial $100 peak; "
            "Sharpe uses daily simple portfolio returns, sample standard deviation and 252 sessions/year."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n"
            "PROJECT = Path.cwd()\n"
            "if not (PROJECT / 'src' / 'backtest_signature_policy.py').exists():\n"
            "    PROJECT = PROJECT / 'Signature_Price_Forecast'\n"
            "OUTPUT = PROJECT / 'results' / 'trading_policy'\n"
            "summary = pd.read_csv(OUTPUT / 'summary.csv')\n"
            "display(summary[['strategy', 'execution', 'final_wealth', 'total_return_pct', "
            "'max_drawdown_pct', 'annualized_sharpe_rf0']].round(4))"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUTPUT / 'equity_and_drawdown.png')))"),
        nbformat.v4.new_code_cell("daily = pd.read_csv(OUTPUT / 'daily_close_to_close.csv')\n"
            "display(daily[['issue_date', 'target_date', 'predicted_return', 'training_percentile', "
            "'allocation', 'wealth_before', 'invested_dollars', 'cash_dollars', 'wealth', 'drawdown']].head(10))\n"
            "display(pd.DataFrame([json.loads((OUTPUT / 'calibration_info.json').read_text())]))"),
        nbformat.v4.new_code_cell("display(Markdown((OUTPUT / 'report.md').read_text(encoding='utf8')))"),
        nbformat.v4.new_markdown_cell("To recompute from the saved forecast CSV, run "
            "`python src/backtest_signature_policy.py` from this project folder. "
            "Calibration forecasts are reused only if their source identity and saved content match. "
            "The original forecasting notebook and its predictions are not overwritten."),
    ]
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def run(project, source_cache=None, initial_wealth=100.0):
    project = Path(project).absolute()
    output = project / "results" / "trading_policy"
    output.mkdir(parents=True, exist_ok=True)
    prediction_file = project / "results" / "daily_refit" / "predictions.csv"
    info_file = project / "results" / "daily_refit" / "run_info.json"
    input_hashes = {str(path.relative_to(project)): file_hash(path) for path in (prediction_file, info_file)}
    info = json.loads(info_file.read_text(encoding="utf8"))
    prices = pd.read_csv(project.parent / "data" / "etf_daily" / "SPY_daily.csv",
                         parse_dates=["date"]).set_index("date").sort_index()
    if not prices.index.is_unique or not np.isfinite(prices[["open", "close"]]).all().all() or (prices[["open", "close"]] <= 0).any().any():
        raise ValueError("Source open/close prices must be finite, positive and uniquely dated")
    test = validate_predictions(pd.read_csv(prediction_file), prices)
    if source_cache is None:
        candidates = [Path.home() / "AppData/Local/Temp/codex_finmultitime_source",
                      Path(tempfile.gettempdir()) / "codex_finmultitime_source"]
        source_cache = next((path for path in candidates if (path / "price_base.npz").exists()), None)
    calibration, calibration_info, anchors = build_calibration(project, test, info, source_cache, output)
    allocation, percentiles = percentile_allocation(test.predicted_return, calibration.predicted_return)
    realized_simple = np.expm1(test.actual_return.to_numpy(float))
    named_weights = {"Signature percentile 20-80%": allocation,
                     "Fixed 50% SPY / 50% cash": np.full(len(test), 0.5),
                     "Buy and hold SPY": np.ones(len(test))}
    ledgers = {name: backtest_returns(realized_simple, weights, initial_wealth)
               for name, weights in named_weights.items()}
    main_ledger = next(iter(ledgers.values()))
    details = pd.concat([test.reset_index(drop=True), main_ledger], axis=1)
    details["training_percentile"] = percentiles
    details["predicted_simple_return"] = np.expm1(test.predicted_return.to_numpy(float))
    details["fixed_50_wealth"] = ledgers["Fixed 50% SPY / 50% cash"].wealth
    details["buy_hold_wealth"] = ledgers["Buy and hold SPY"].wealth
    details.to_csv(output / "daily_close_to_close.csv", index=False)

    rows = [{"strategy": name, "execution": "Idealized close-to-close", **performance(ledger, initial_wealth)}
            for name, ledger in ledgers.items()]
    next_session = prices.reindex(pd.DatetimeIndex(test.target_date))
    intraday_returns = (next_session.close / next_session.open - 1).to_numpy(float)
    intraday_ledger = backtest_returns(intraday_returns, allocation, initial_wealth)
    intraday = pd.concat([test.reset_index(drop=True), intraday_ledger], axis=1)
    intraday["training_percentile"] = percentiles
    intraday["entry_open"] = next_session.open.to_numpy()
    intraday["exit_close"] = next_session.close.to_numpy()
    intraday.to_csv(output / "daily_next_open_to_close.csv", index=False)
    rows.append({"strategy": "Signature percentile 20-80%", "execution": "Next open to close; cash overnight",
                 **performance(intraday_ledger, initial_wealth)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    dates = pd.DatetimeIndex([test.issue_date.iloc[0]] + list(test.target_date))
    save_plot(output, dates, ledgers, initial_wealth)

    run_info = {
        "initial_wealth": initial_wealth, "asset": "SPY", "allocation_min": 0.2, "allocation_max": 0.8,
        "allocation_rule": "w = 0.20 + 0.60 * frozen_training_mid_CDF(predicted_log_return)",
        "cdf_ties": "(number strictly below + half the number equal) / calibration count",
        "test_issue_first": str(test.issue_date.iloc[0].date()),
        "test_issue_last": str(test.issue_date.iloc[-1].date()),
        "test_target_first": str(test.target_date.iloc[0].date()),
        "test_target_last": str(test.target_date.iloc[-1].date()),
        "calibration": calibration_info, "reproduced_prediction_anchors": anchors,
        "sessions_per_year": 252, "risk_free_rate_annual": 0.0,
        "cash_interest_rate": 0.0, "commission_and_slippage": 0.0,
        "fractional_shares": True, "short_selling": False, "borrowing": False,
        "main_execution": "Idealized issue-close entry/rebalance to next target close; same-close fill after close inputs is optimistic",
        "alternative_execution": "Enter/rebalance at target open, liquidate at target close; cash overnight, so excludes overnight price returns",
        "drawdown": "Daily valuation only, initial wealth included as first peak; excludes intraday troughs",
        "return_basis": "Use source price series as provided; no additional dividend cashflows, taxes or currency conversion",
        "sharpe": "sqrt(252) * mean(daily simple portfolio returns) / sample_std(ddof=1), zero risk-free rate",
        "input_sha256": input_hashes,
        "backtest_script_sha256": file_hash(Path(__file__)),
        "results": rows,
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2, allow_nan=False), encoding="utf8")
    table = ["| Policy | Holding period | Final wealth | Total return | Max drawdown | Annualized Sharpe |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        sr = "N/A" if row["annualized_sharpe_rf0"] is None else f"{row['annualized_sharpe_rf0']:.4f}"
        table.append(f"| {row['strategy']} | {row['execution']} | ${row['final_wealth']:.2f} | "
                     f"{row['total_return_pct']:.4f}% | {row['max_drawdown_pct']:.4f}% | {sr} |")
    report = ("# Signature forecast trading policy backtest\n\n"
        f"Initial wealth: **${initial_wealth:.2f}**. Test: **{len(test)} sessions**, "
        f"{run_info['test_target_first']} to {run_info['test_target_last']}.\n\n"
        + "\n".join(table) + "\n\n"
        "## Allocation\n\n"
        "Each day invest `20% + 60% × frozen training forecast percentile` of current wealth in SPY; "
        "hold the remainder as cash. A percentile is the number of calibration forecasts strictly below "
        "the current forecast plus half the ties, divided by the number of calibration forecasts. "
        "Forecasts outside the calibration range receive 20% or 80%. The rule is fixed; no test-return "
        "optimization or test-wide ranking is used.\n\n"
        f"Calibration: **{len(calibration)} causal walk-forward forecasts**, issued "
        f"{calibration_info['issue_first']} through {calibration_info['issue_last']}; "
        f"all outcomes were known by the first test issue close ({calibration_info['known_by_issue_close']}). "
        "The calibration window matches the first test model's matured training-example window. "
        "The saved validation-selected forecasting configuration is reused. The first, middle and last "
        "saved test predictions were reproduced to confirm the same forecasting pipeline.\n\n"
        f"Realized allocations: {100 * allocation.min():.2f}% to {100 * allocation.max():.2f}%; "
        f"mean {100 * allocation.mean():.2f}%.\n\n"
        "## Calculation and execution assumptions\n\n"
        "- Convert realized log returns to simple returns with `exp(log_return) - 1`.\n"
        "- Wealth evolves as `W_next = W_current × (1 + allocation × asset_simple_return)`.\n"
        "- Fractional shares are allowed; no borrowing or short selling.\n"
        "- Cash interest, risk-free rate, transaction costs and slippage are all zero.\n"
        "- Maximum drawdown is the largest daily decline from a running wealth peak, including the initial $100.\n"
        "- Annualized Sharpe is `sqrt(252) × mean(daily portfolio returns) / sample standard deviation`. "
        "The square-root annualization is the usual simplifying convention; see "
        "[Sharpe's discussion of differential returns and time dependence](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm).\n"
        "- The primary close-to-close backtest assumes an exact issue-close fill after observing that close. "
        "This is an idealized signal study and cannot establish an executable same-close result.\n"
        "- The next-open-to-close comparison uses only an already-issued forecast, trades from target-day "
        "open to close and holds cash overnight. Its holding period differs from the forecast target.\n"
        "- Results use the local SPY price series as provided, with no additional dividend cashflows, taxes "
        "or currency conversion. Drawdown is based on daily marks, not intraday prices.\n\n"
        "## Saved outputs\n\n"
        "`daily_close_to_close.csv` and `daily_next_open_to_close.csv` contain allocations, cash, "
        "investment amounts, daily P&L, wealth and drawdown. `calibration_predictions.csv` preserves the "
        "pre-test reference forecasts. `run_info.json` records settings, source hashes and reproduction checks.\n")
    (output / "report.md").write_text(report, encoding="utf8")
    write_notebook(project)
    for relative, before in input_hashes.items():
        if file_hash(project / relative) != before:
            raise RuntimeError(f"An original forecast input changed during the run: {relative}")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results: {output}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--initial-wealth", type=float, default=100.0)
    args = parser.parse_args()
    run(args.project, args.source_cache, args.initial_wealth)
