# Historical CSV data

CSV backtests accept completed daily bars with the following columns:

```text
timestamp,open,high,low,close,volume
```

Requirements:

- `timestamp` must be an ISO-8601 date or timestamp.
- `open`, `high`, `low`, and `close` must be finite positive numbers.
- `volume` is optional and defaults to zero.
- Timestamps must be unique.
- Rows may be unsorted; the loader sorts them chronologically.
- Include at least the strategy's minimum history before the selected start
  date. A 50-day SMA therefore needs at least 50 earlier daily rows.

The bundled `example_bars.csv` is a small functional sample. Run it with short
windows because it does not contain enough rows for the default 20/50 SMA:

```powershell
.\.venv\Scripts\python.exe backtest.py `
  --data-source csv `
  --csv-file data\example_bars.csv `
  --strategy sma `
  --short-window 2 `
  --long-window 3 `
  --symbol DEMO `
  --start 2024-01-04 `
  --end 2024-01-12 `
  --initial-cash 10000 `
  --slippage-bps 0
```

## Candidate fundamental profile

`sp500_candidate_profile.example.json` demonstrates the point-in-time inputs
required by the `sp500-candidate` strategy. Its company and values are fictional
and are provided only for software testing. Replace every field with information
that was publicly available on or before the backtest start date.

The market-cap guideline changes over time. Set `market_cap_threshold_usd` to
the S&P DJI threshold applicable on the profile's `as_of_date`, rather than
blindly applying the example's July 2026 value to older periods.

Field meanings:

| Field | Meaning |
|---|---|
| `company_name`, `symbol`, `as_of_date` | Candidate identity and the date on which every input was known |
| `is_us_company` | Whether the issuer meets the methodology's US-company test |
| `eligible_exchange` | Whether the primary listing is on an eligible US exchange |
| `eligible_security_type` | Whether the security type is eligible for the index |
| `already_in_sp500` | Must be false for a potential new entrant |
| `total_market_cap_usd` | Total company-level market capitalization in US dollars |
| `security_float_adjusted_market_cap_usd` | Float-adjusted capitalization of the candidate security |
| `investable_weight_factor` | Publicly investable float fraction, from 0 to 1 |
| `minimum_monthly_volume_last_six_months` | Lowest monthly share volume among the prior six months |
| `float_adjusted_liquidity_ratio` | Annual dollar value traded divided by float-adjusted market cap |
| `latest_quarter_gaap_net_income` | Most recent quarter's GAAP net income in US dollars |
| `trailing_four_quarters_gaap_net_income` | Sum of GAAP net income across the last four quarters |
| `months_since_ipo` | Completed months since the first exchange trading date |
| `market_cap_threshold_usd` | S&P 500 size guideline applicable on `as_of_date` |

Use numbers or quoted decimal strings. Monetary inputs may be negative only for
the two net-income fields. Booleans must use JSON `true` or `false` rather than
quoted text.
