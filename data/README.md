# Data

## ETF daily prices (`etf_daily/`)

The forecast code reads these files:

- `etf_daily/SPY_daily.csv`
- `etf_daily/QQQ_daily.csv`
- `etf_daily/IWM_daily.csv`

Each file has one row per trading day, with at least these columns:

```text
date,open,close
```

The files come from Alpaca (daily bars, adjusted for splits and dividends).
They are **not uploaded to GitHub**, because Alpaca data is for personal use only.
To run the project, download the same data with your own Alpaca account and save it here.

## Other files

- `example_bars.csv`: a tiny sample of daily price bars, for testing.
- `sp500_candidate_profile.example.json`: a made-up company profile, for testing. All values are fake.

These two files were used by an older backtest tool. That tool is no longer in this repository.
