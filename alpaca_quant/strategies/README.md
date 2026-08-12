# Strategy modules

Place every trading strategy in a separate Python module in this directory.
For example:

```text
strategies/
|-- base.py
|-- registry.py
|-- sma.py
|-- sp500_candidate.py
|-- rsi.py
`-- momentum.py
```

A strategy module should contain only deterministic signal logic:

- Accept market observations and the minimum required account state.
- Return a decision object or signal.
- Avoid API calls, credential access, order submission, logging configuration,
  and filesystem writes.
- Validate invalid or insufficient market data before producing a signal.
- Include offline unit tests for all signal and position-state combinations.

After adding a strategy module, register its command-line name and constructor
in `registry.py`. The backtest command can then select it with `--strategy`.

Broker communication and risk controls belong in `alpaca_broker.py` and
`runner.py`. Keeping this boundary makes new strategies safe to compare and
easy to test without an Alpaca account.
