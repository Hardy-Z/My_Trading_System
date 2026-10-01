# SPY signature forecasting — reviewer guide

The [executed project report](signature_price_forecast.ipynb) examines next-session **SPY log-return** forecasting from signatures of the SPY close path and other numeric market factors. Model outputs are scalar log returns. Up/down accuracy uses the sign of those returns; plotted prices are reconstructed for interpretation from each preceding **actual** SPY close.

The model re-screens factors, calculates signature-kernel weights, selects with weighted Lasso, and refits with weighted OLS on every issue date. Results are saved in `results/daily_refit/`.

Both notebooks expose `DIRECTION_UP_RETURN_THRESHOLD` and `DIRECTION_DOWN_RETURN_THRESHOLD` in their settings cells. These are signed log-return cutoffs: for example, use `0.001` for up and `-0.002` for down. Forecasts above the up cutoff predict up; forecasts at or below the down cutoff predict down; the interval between them is neutral. Keep the down cutoff no greater than the up cutoff. Both default to zero, preserving the original binary rule. Neutral forecasts count as misses in direction accuracy and up/down recall, and `predicted_neutral_rate` reports their frequency. Return-error metrics continue to use all forecasts. Rerun the notebooks after changing settings to refresh their saved outputs.

The [fixed-settings train/test comparison](spy_train_test_comparison.ipynb) runs no validation search. It trains the pre-existing `ForecastConfig()` defaults and compares them with the configuration already chosen in the original report. Both refit daily using identical training windows, inputs and 310 test issue dates. Settings, training dates, full-period Corr and normalized MSE, accuracy, MAE, paired error uncertainty, and forecast plots are included. Its separate results are saved in `results/train_test_comparison/`. This comparison evaluates two configurations; skipping validation alone does not add training data or change fitting.

The [previous-day direction baseline](previous_day_direction_baseline.ipynb) predicts the next session's direction using the last completed session's close-to-close return sign. On the same 310 held-out sessions, it correctly predicts 158 directions (**50.97% accuracy**), compared with **57.74%** for always predicting up. It prints predicted and actual up/down counts and the confusion matrix. Its executed outputs and CSVs are saved under `results/previous_day_direction_baseline/`.

In the default matched test, fixed settings yield MAE **0.00627412**, direction accuracy **58.39%**, and Corr **0.04401**, versus **0.00627737**, **58.71%**, and **0.01515** for the original configuration. The paired MAE difference interval includes zero, so neither configuration has a clear overall advantage on this sample.

## Data and calendar

- Local daily bars: `../SP500/attention_data/{SPY,QQQ,IWM}_daily.csv`. SPY closes form response history and next-session labels; QQQ and IWM contribute six quantitative factors.
- Optional FinMultiTime numeric archive: eight price/breadth fields and five financial-table fields, via `../HF_Transformer/finmultitime_daily.py`. Filing fields are delayed until the session after filing. News text and chart images are not used. The default `USE_FINMULTITIME=True` gives 19 candidate quantitative factors. Initial cache construction needs internet access.
- Common bar dates: **2018-01-02 to 2025-03-28** (1,820 sessions). A 20-session lookback and one-session target give usable forecast issue dates **2018-01-30 to 2025-03-27**.
- Validation forecast issues: **2023-01-03 to 2023-12-29**. Model candidates are chosen by validation log-return mean absolute error (MAE).
- Held-out forecast issues: **2024-01-02 to 2025-03-27** (310 sessions). Their target close dates are **2024-01-03 to 2025-03-28**.
- Training is **rolling**, with at most 756 examples whose target closes have occurred by the forecast issue close. For the first held-out issue, the daily refit uses training issue dates **2020-12-29 to 2023-12-29**; its latest eligible target is **2024-01-02**. The notebook prints actual training boundaries at representative dates.

These are historical replays through March 2025, not live forecasts for 2026. The notebook states source coverage, date cutoffs, leakage controls, selection rules, baselines, and limitations next to the corresponding code and results.

## Default held-out result

| Method | Log-return MAE | Direction accuracy |
|---|---:|---:|
| Daily full two-step refit | 0.006277 | 58.71% |
| Zero-return / majority-direction baselines | 0.006338 | 57.74% |

The model's direction advantage over the majority-direction diagnostic is small, and it predicted up on 96.45% of test sessions. These numbers describe only the 310 held-out sessions above; no trading-cost or live-performance claim follows from them.

## Layout

```text
Signature_Price_Forecast/
  signature_price_forecast.ipynb   Executed report
  spy_train_test_comparison.ipynb Fixed-settings train/test comparison
  src/                            Forecast implementation
  tests/                          Timing and model checks
  tools/                          Notebook source builder
  results/daily_refit/            Forecasts, metrics and plots
  results/train_test_comparison/  Matched train/test comparison outputs
```

The paper motivating the adaptive signature and two-step Lasso workflow is [*Transportation Marketplace Rate Forecast Using Signature Transform*](../Papers/Transportation%20Marketplace%20Rate%20Forecast%20Using%20Signature%20Transform.pdf). The notebook does not establish a profitable trading rule.
