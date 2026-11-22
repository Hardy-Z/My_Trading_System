# SPY Signature Forecasting

## Method

The [forecast notebook](signature_price_forecast.ipynb) predicts the next trading session's SPY log return using signatures of a 20-session price history and quantitative market factors. Each day, it screens factors by correlation, weights training samples by signature-kernel similarity, selects features with weighted Lasso, and refits with weighted OLS. Training uses up to 756 examples whose outcomes are already known. Model settings are selected on the validation period by lowest MSE.

## Results

Held-out test: **310 forecasts**, issued January 2, 2024–March 27, 2025. Every evaluation reports four metrics.

| Metric | Model | Reference |
|---|---:|---|
| Direction accuracy | 58.71% | 57.74% from always predicting up |
| Corr | 0.0151 | |
| MSE | 7.138e-05 | 7.167e-05 from always forecasting zero |
| ICIR | 0.155 | |

- **Direction accuracy:** share of sessions where the predicted direction matches the realized one. Direction thresholds are zero; positive returns are up, and zero or negative returns are down.
- **Corr:** Pearson correlation between predicted and realized log returns over all forecasts.
- **MSE:** mean squared error of the predicted log return.
- **ICIR:** mean of the monthly ICs divided by their sample standard deviation, not annualized. SPY is one series, so each month's IC is the Pearson correlation between that month's predicted and realized returns. The test has 15 months with a mean IC of 0.026; 9 months are positive.

Results are in [results/daily_refit](results/daily_refit/): the four metrics in `metrics.csv` and the monthly ICs in `monthly_ic.csv`.

## Dataset

- **Prices:** SPY, QQQ, and IWM daily open/close bars from the [quant-research-dataset](https://pypi.org/project/quant-research-dataset/) library (`qrd`, dataset `us/etf/daily_bars`), with common coverage January 2, 2018–March 28, 2025 (1,820 sessions).
- **Factors:** 19 quantitative inputs: six QQQ/IWM return and volatility features, eight FinMultiTime price/breadth fields, and five numeric filing-table fields. Filing data become usable the session after filing; news and images are excluded.
- **Validation:** forecast issues in 2023. **Test:** forecast issues January 2, 2024–March 27, 2025, targeting January 3, 2024–March 28, 2025.

## Setup

```text
pip install -r requirements.txt
```

The ETF bars are a local-only qrd dataset (Alpaca data is not on the Hugging Face Hub).
Point qrd at your local data folder with `QRD_DATA_DIR`, either as an environment variable
or as a line in the repository's `.env` file:

```text
QRD_DATA_DIR=C:/path/to/quant-research-dataset/prepared
```

`src/market_data.py` reads that setting and loads the bars with qrd.
