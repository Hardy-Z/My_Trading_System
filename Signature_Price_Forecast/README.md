# SPY Signature Forecasting

## Method

The [forecast notebook](signature_price_forecast.ipynb) predicts the next trading session's SPY log return using signatures of a 20-session price history and quantitative market factors. Each day, it screens factors by correlation, weights training samples by signature-kernel similarity, selects features with weighted Lasso, and refits with weighted OLS. Training uses up to 756 examples whose outcomes are already known. Model settings are selected on the validation period.

## Results

Held-out test: **310 forecasts**, issued January 2, 2024–March 27, 2025. Direction thresholds are zero; positive returns are up, and zero or negative returns are down.

| Method | Log-return MAE | Direction accuracy |
|---|---:|---:|
| Signature adaptive two-step Lasso | 0.006277 | 58.71% |
| Zero-return / majority-direction baselines | 0.006338 | 57.74% |

The baseline row reports zero-return MAE and always-up direction accuracy. Results are in [results/daily_refit](results/daily_refit/).

## Dataset

- **Prices:** local SPY, QQQ, and IWM daily open/close data in `../SP500/attention_data/`, with common coverage January 2, 2018–March 28, 2025 (1,820 sessions).
- **Factors:** 19 quantitative inputs: six QQQ/IWM return and volatility features, eight FinMultiTime price/breadth fields, and five numeric filing-table fields. Filing data become usable the session after filing; news and images are excluded.
- **Validation:** forecast issues in 2023. **Test:** forecast issues January 2, 2024–March 27, 2025, targeting January 3, 2024–March 28, 2025.
