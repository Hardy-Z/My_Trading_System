# Alpaca Paper Trading Quantitative Models

This repository contains a small, long-only quantitative trading model for
Alpaca Paper Trading. It is intentionally simple and heavily documented so the
order flow and risk controls are easy to review.

> This project is for education and paper-trading experiments only. It is not
> investment advice, and it does not guarantee future results.

## Strategies

### SMA regime strategy

The model trades one US equity or ETF (default: `SPY`) from completed daily
bars:

- Calculate a short simple moving average (default: 20 sessions).
- Calculate a long simple moving average (default: 50 sessions).
- Hold a long position when the short average is above the long average.
- Hold cash when the short average is below the long average.
- Do nothing when both averages are equal.

The runner is idempotent: it compares the desired strategy state with the
current paper-account position. Re-running it does not repeatedly buy while the
account is already long.

### Potential S&P 500 entrant strategy

The `sp500-candidate` strategy trades one user-selected candidate at a time:

1. Read a dated fundamental profile supplied as JSON.
2. Check the measurable S&P 500 outside-addition criteria: US domicile,
   eligible listing/security type, size, public float, liquidity, profitability,
   and minimum IPO history.
3. Buy only when every mechanical rule passes and the latest close is above a
   configurable trend SMA (default: 50 sessions).
4. After entry, hold until the completed daily close reaches a fixed percentage
   above the actual entry price (default: 20%). The sell fills at the next open.

This is a screening strategy, not an inclusion predictor. S&P DJI also considers
sector representation and uses committee judgment, so a company that passes all
implemented checks may never enter the index. The program does not discover
candidates or fetch fundamentals automatically; you must research a company and
provide its point-in-time profile. There is no stop loss or time-based exit in
this intentionally simple version.

## Safety controls

- `paper=True` is hard-coded when the Alpaca trading client is created.
- The model is long-only and never opens a short position.
- The default command is a dry run and cannot submit an order.
- `--execute` is required to send a paper order.
- New positions are limited to a configurable fraction of account equity.
- The runner skips execution when the US equity market is closed.
- The runner skips execution when an order for the symbol is already open.
- Alpaca credentials are loaded from `.env`, which is ignored by Git.

## Project layout

```text
.
|-- alpaca_quant/
|   |-- __init__.py
|   |-- alpaca_broker.py
|   |-- backtesting/
|   |   |-- __init__.py
|   |   |-- data.py
|   |   |-- engine.py
|   |   |-- models.py
|   |   `-- reporting.py
|   |-- config.py
|   |-- runner.py
|   `-- strategies/
|       |-- __init__.py
|       |-- base.py
|       |-- README.md
|       |-- registry.py
|       |-- sma.py
|       `-- sp500_candidate.py
|-- data/
|   |-- README.md
|   |-- example_bars.csv
|   `-- sp500_candidate_profile.example.json
|-- tests/
|   |-- test_backtest.py
|   |-- test_backtest_data.py
|   |-- test_config.py
|   |-- test_sp500_candidate.py
|   `-- test_strategy.py
|-- .env.example
|-- .gitignore
|-- backtest.py
|-- main.py
|-- requirements.txt
`-- readme.md
```

## Setup

The required packages are already installed in the `Quant_ENV` Conda
environment on this computer. Activate it before running a command:

```powershell
& C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1
conda activate Quant_ENV
```

If Conda reports that the shell is not initialized, run commands without
activation by prefixing them with:

```powershell
C:\ProgramData\Anaconda3\Scripts\conda.exe run -n Quant_ENV python
```

To create a separate environment on another computer:

1. Create and activate a virtual environment.

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies.

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy the environment template.

   ```powershell
   Copy-Item .env.example .env
   ```

4. Put your **paper-account** API key and secret in `.env`. Paper keys are
   available from the Alpaca dashboard after selecting the paper environment.

## Run the model

### SMA strategy

Preview the decision without submitting an order:

```powershell
python main.py
```

Submit the resulting order to Alpaca Paper Trading:

```powershell
python main.py --execute
```

The script runs once and exits. For daily automation, schedule it once during
regular US market hours after the daily data you want to use is available.

### Potential S&P 500 entrant strategy

First copy `data\sp500_candidate_profile.example.json`, replace the fictional
values with dated data for a real candidate, and set `TRADING_SYMBOL` in `.env`
to exactly the same ticker. Preview the decision without submitting an order:

```powershell
python main.py `
  --strategy sp500-candidate `
  --candidate-profile data\my_candidate.json `
  --trend-window 50 `
  --take-profit-pct 20
```

After inspecting the log, append `--execute` to permit an Alpaca paper order.
The target is based on Alpaca's reported average entry price.
`--take-profit-pct 20` means a target of `entry price x 1.20`.

## Run a historical backtest

The backtester accepts a registered strategy, data source, date range, starting
cash, allocation, commission, and slippage assumptions. It never submits an
order to Alpaca.

### Alpaca historical data

This command downloads adjusted daily IEX bars using the credentials in `.env`:

```powershell
.\.venv\Scripts\python.exe backtest.py `
  --strategy sma `
  --data-source alpaca `
  --symbol SPY `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --short-window 20 `
  --long-window 50 `
  --initial-cash 100000 `
  --allocation 1.0 `
  --commission-per-order 0 `
  --slippage-bps 5
```

### CSV historical data

The CSV schema is `timestamp,open,high,low,close,volume`. Include enough rows
before the selected start date to warm up the strategy indicators.

```powershell
.\.venv\Scripts\python.exe backtest.py `
  --strategy sma `
  --data-source csv `
  --csv-file data\example_bars.csv `
  --symbol DEMO `
  --start 2024-01-04 `
  --end 2024-01-12 `
  --short-window 2 `
  --long-window 3 `
  --initial-cash 10000 `
  --slippage-bps 0
```

Use `python backtest.py --help` to see every available input.

### Backtest the candidate strategy

The profile ticker must match `--symbol`, and `as_of_date` must be on or before
`--start`. This guard prevents the backtest from using fundamentals that were
published after the simulated decision date.

```powershell
python backtest.py `
  --strategy sp500-candidate `
  --candidate-profile data\sp500_candidate_profile.example.json `
  --data-source csv `
  --csv-file data\example_bars.csv `
  --symbol DEMO `
  --start 2024-01-04 `
  --end 2024-01-12 `
  --trend-window 3 `
  --take-profit-pct 2 `
  --initial-cash 10000 `
  --slippage-bps 0
```

The example uses a 2% target and fictional data only so a complete entry/exit
can be observed in a very small file. For a historical study, use the market-cap
threshold that applied on the profile date. The current code treats the single
profile as unchanged throughout one backtest; it does not model later fundamental
revisions.

### Backtest assumptions

- The strategy observes only completed daily closes.
- A signal generated at one close fills at the following trading day's open.
- Buy fills include upward slippage; sell fills include downward slippage.
- The engine supports fractional quantities and fixed per-order commissions.
- Open positions are marked to the final close and reported as unrealized P/L.
- Performance starts exactly at `--start`; earlier rows are indicator warm-up
  data and are excluded from the equity curve.
- Both strategies are long-only and never borrow cash or open a short.

### Result files

The default result location is:

```text
backtest_results/<strategy>_<symbol>_<start>_<end>/
```

Each run creates:

| File | Contents |
|---|---|
| `summary.json` | Return, profit, drawdown, volatility, Sharpe ratio, benchmark, exposure, and trade statistics |
| `equity_curve.csv` | Daily cash, position value, total equity, and drawdown |
| `trades.csv` | Every simulated buy/sell fill, fee, reason, and realized P/L |

The result directory is ignored by Git. Use `--output-dir` to choose a different
location.

Short backtests can produce misleadingly large annualized returns and unstable
Sharpe ratios. Always evaluate total return, maximum drawdown, trade count, and
the equity curve together.

## Configuration

The `.env` file accepts these settings:

| Variable | Default | Meaning |
|---|---:|---|
| `ALPACA_API_KEY` | required | Alpaca paper API key |
| `ALPACA_SECRET_KEY` | required | Alpaca paper secret key |
| `TRADING_SYMBOL` | `SPY` | US equity or ETF symbol |
| `SHORT_WINDOW` | `20` | Short SMA length in sessions |
| `LONG_WINDOW` | `50` | Long SMA length in sessions |
| `ALLOCATION_FRACTION` | `0.10` | Maximum fraction of equity used for a new position |

`ALLOCATION_FRACTION` must be greater than zero and no greater than `1.0`.
`SHORT_WINDOW` must be smaller than `LONG_WINDOW`.

## Adding another strategy

Each strategy lives in its own module under `alpaca_quant/strategies/`. Keep
signal calculations independent from Alpaca order execution so they remain easy
to test. To add a strategy, create a new module such as `rsi.py`, add offline
tests, and register its command-line name in `strategies/registry.py`. It will
then become selectable through `backtest.py --strategy`. See the strategy-folder
README for the expected separation of responsibilities.

## Tests

The strategy tests do not require credentials or a network connection:

```powershell
python -m unittest discover -s tests -v
```

## Official references

- [Alpaca-py getting started](https://alpaca.markets/sdks/python/getting_started.html)
- [Paper trading with `TradingClient`](https://alpaca.markets/sdks/python/trading.html)
- [Stock historical data](https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html)
- [Order requests](https://alpaca.markets/sdks/python/api_reference/trading/requests.html)
- [S&P U.S. Indices methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf)
