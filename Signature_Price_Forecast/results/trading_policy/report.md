# Signature forecast trading policy backtest

Initial wealth: **$100.00**. Test: **310 sessions**, 2024-01-03 to 2025-03-28.

| Policy | Holding period | Final wealth | Total return | Max drawdown | Annualized Sharpe |
|---|---|---:|---:|---:|---:|
| Signature percentile 20-80% | Idealized close-to-close | $110.76 | 10.7582% | 6.3473% | 1.1731 |
| Fixed 50% SPY / 50% cash | Idealized close-to-close | $109.59 | 9.5878% | 5.1171% | 1.1437 |
| Buy and hold SPY | Idealized close-to-close | $119.43 | 19.4297% | 10.0351% | 1.1437 |
| Signature percentile 20-80% | Next open to close; cash overnight | $97.01 | -2.9937% | 7.1187% | -0.3906 |

## Allocation

Each day invest `20% + 60% × frozen training forecast percentile` of current wealth in SPY; hold the remainder as cash. A percentile is the number of calibration forecasts strictly below the current forecast plus half the ties, divided by the number of calibration forecasts. Forecasts outside the calibration range receive 20% or 80%. The rule is fixed; no test-return optimization or test-wide ranking is used.

Calibration: **756 causal walk-forward forecasts**, issued 2020-12-29 through 2023-12-29; all outcomes were known by the first test issue close (2024-01-02). The calibration window matches the first test model's matured training-example window. The saved validation-selected forecasting configuration is reused. The first, middle and last saved test predictions were reproduced to confirm the same forecasting pipeline.

Realized allocations: 22.86% to 79.92%; mean 53.13%.

## Calculation and execution assumptions

- Convert realized log returns to simple returns with `exp(log_return) - 1`.
- Wealth evolves as `W_next = W_current × (1 + allocation × asset_simple_return)`.
- Fractional shares are allowed; no borrowing or short selling.
- Cash interest, risk-free rate, transaction costs and slippage are all zero.
- Maximum drawdown is the largest daily decline from a running wealth peak, including the initial $100.
- Annualized Sharpe is `sqrt(252) × mean(daily portfolio returns) / sample standard deviation`. The square-root annualization is the usual simplifying convention; see [Sharpe's discussion of differential returns and time dependence](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm).
- The primary close-to-close backtest assumes an exact issue-close fill after observing that close. This is an idealized signal study and cannot establish an executable same-close result.
- The next-open-to-close comparison uses only an already-issued forecast, trades from target-day open to close and holds cash overnight. Its holding period differs from the forecast target.
- Results use the local SPY price series as provided, with no additional dividend cashflows, taxes or currency conversion. Drawdown is based on daily marks, not intraday prices.

## Saved outputs

`daily_close_to_close.csv` and `daily_next_open_to_close.csv` contain allocations, cash, investment amounts, daily P&L, wealth and drawdown. `calibration_predictions.csv` preserves the pre-test reference forecasts. `run_info.json` records settings, source hashes and reproduction checks.
