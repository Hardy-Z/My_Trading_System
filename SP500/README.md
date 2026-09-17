# Online reinforcement learning for S&P 500 allocation

The agent allocates between signed S&P 500 exposure and signed cash, receives the **net daily portfolio return**, and updates its policy **once after each new completed session**. It starts with $100,000. The included example uses Alpaca SPY data as a tradable S&P 500 proxy. This is a historical/paper research framework; it does not submit orders.

Start with **`online_sp500_rl.ipynb`** for the theory, executed examples, results and latest action probabilities. The implementation is `online_rl.py`; its checkpoint contains the policy, value function, optimizer, random generator, portfolio, pending action and audit history.

## Stock and cash action space

Let equity be `E`, the stock target weight be `w`, and the execution price be `P`. After transaction costs, the target positions are:

`stock dollars = w × E`, `stock units x = w × E / P`, `cash dollars y = (1 − w) × E`.

Thus stock value plus cash equals equity. Stock and cash cannot both be chosen independently without specifying additional deposits, withdrawals or funding.

| Stock weight | Cash weight | Meaning |
|---:|---:|---|
| −100% | +200% | Short stock; retain short-sale proceeds as cash |
| −50% | +150% | Half-sized short |
| 0% | +100% | Cash |
| +50% | +50% | Half stock, half cash |
| +100% | 0% | Fully invested |
| +150% | −50% | Borrow cash to invest |
| +200% | −100% | Twice equity in stock |

The default policy assigns probabilities to these seven **target allocations**, not trade increments. The actual order is the difference between the target position and the position already held. This is a discrete grid of amounts; edit `Config.weights` or use `--weights=-1,-0.5,0,0.5,1` for another grid within the supported −100% to +200% bounds. Changing the grid requires a separate replay/checkpoint.

`latest_decision.json` includes each action's probability, stock/cash weights, signed dollar amounts, estimated signed stock units and the pending sampled action. Positive cash means a cash asset; negative cash means borrowing. Units and dollar amounts use the latest close and equity as estimates; actual target amounts depend on next-open equity, price and costs. A zero-stock target closes an existing stock position; it does not mean leaving that position unchanged. Probabilities describe policy choices, not the probability of profit.

## Daily learning sequence

1. After session **t** closes, construct features using prices through **t** and the current portfolio. Sample and save action `a[t]` from the current policy.
2. On **t+1**, the existing position receives the overnight return. Account balances accrue the configured financing costs over elapsed calendar days.
3. At the **t+1 open**, rebalance to the saved target allocation, deducting costs. The new allocation receives the open-to-close return.
4. After the **t+1 close**, calculate `reward[t+1] = equity[t+1] / equity[t] − 1`.
5. Make **one online actor/critic update** using the observed transition; only then sample the next allocation.

This prevents using today's close to obtain a fill at today's open. The reward includes the overnight movement of the previous position; that component is not controlled by the newly selected allocation, but is part of the account's actual daily return.

The implementation is **on-policy TD(0) actor-critic**, with a linear softmax actor, linear critic and Adam updates. The temporal-difference error is `delta = 100 × reward + gamma × V(next_state) − V(state)`, with `gamma=0.95`. The actor follows the sampled action's log-probability gradient times the TD advantage, with a small entropy term. The critic follows the stopped-gradient TD target. TD errors are clipped to ±10 for updates and gradients to norm 1. The 100 multiplier changes learning units; the reported reward remains the unmodified net arithmetic return. This is not batch PPO and does not refit the full history every day.

Inputs are the previous 20 close-to-close log returns, 5/20-session momentum and volatility, 20-session drawdown, current stock/cash weights and log equity relative to initial equity. Scaling uses fixed constants, not full-dataset statistics. Only past S&P 500 proxy prices and account state enter the policy. Both positive and negative cash/stock positions are represented.

## Run from the My_Trading_System folder

```powershell
python -m pip install -r SP500/requirements.txt

# Initial data download, using the existing project .env credentials.
python SP500/download_data.py

# One chronological online pass; choose a new folder for a fresh experiment.
python SP500/online_rl.py replay --output SP500/outputs_new

# Show the saved action without learning or resampling it.
python SP500/online_rl.py predict --checkpoint SP500/outputs/checkpoint.json

# Daily continuation: append completed bars, then learn only new sessions.
python SP500/download_data.py --append
python SP500/online_rl.py update --checkpoint SP500/outputs/checkpoint.json

# Tests
python -m unittest discover -s SP500 -p test_online_rl.py -v
```

The delivered `outputs` folder already contains a completed run. Initial download/replay commands refuse to overwrite existing data/checkpoints. `--append` retains the saved history and adds new sessions. The download uses SIP by default; `--feed iex` is available if your Alpaca data access requires it, but it is a different data source and should use a separate dataset/experiment.

`update` can catch up through multiple new sessions in order. Repeating it with the same history makes **zero** learning updates. No scheduler or background process is installed: run these commands when completed data is available. A real daily application can call `OnlinePortfolio.ingest()` with updated prices, then `save()`; it should use the same single-writer lock.

## Files and persistence

- `data/SPY_daily.csv` and `.metadata.json`: adjusted/raw prices, source, feed, retrieval date and checksum.
- `outputs/checkpoint.json`: authoritative state, including the pending action, Adam moments, RNG and full processed ledger. Written through an atomic file replacement under a writer lock.
- `outputs/latest_decision.json`: derived action probabilities and target amounts.
- `outputs/daily_learning.csv`: action date, fill session, policy version, equity, reward, costs and margins.
- `outputs/evaluation.json`, `baseline_*.csv`: comparisons from the original replay. They are **not refreshed by `update`**; their dates identify the original evaluation window.
- `outputs/equity_and_learning.png`: notebook-generated comparison from that replay.

If a process stops after saving the checkpoint but before exporting CSV/JSON, rerunning `update` recreates those exports without learning twice. A concurrent writer is rejected. After an abnormal process exit, inspect the `.lock` file and confirm the original process has stopped before removing a stale lock. No unsafe Python pickle is loaded.

Saved price history is hashed. Editing or deleting processed rows causes `update` to stop. Alpaca may rescale adjusted history after dividends or splits. Append mode rebases new adjusted data to the saved scale, preserving return ratios and all old observations; nonuniform historical revisions are rejected for review and a separate replay. Never replace adjusted prices with raw prices across corporate actions.

## Using your own historical S&P 500 dataset

Provide a CSV with `date,open,close,raw_close`. Dates must be increasing, unique session dates. `open` and `close` must share a consistent corporate-action/total-return adjustment basis. `raw_close` is optional and used only for indicative unit amounts; without it those units are in the supplied price series' units. An S&P 500 index series gives a theoretical index-exposure simulation; it is not itself a security that can be bought. A price-only index also omits dividend returns.

```powershell
python SP500/online_rl.py replay --csv SP500/data/my_sp500.csv --output SP500/my_experiment
```

The next-open convention requires open prices; a close-only CSV is rejected rather than inventing fills. The current New York calendar day is excluded to avoid partial daily bars. Input rows define the trading calendar; gaps longer than ten calendar days raise, but shorter missing sessions require independent data-quality checks.

## Accounting and assumptions

Trading cost is **5 bps of absolute traded stock notional**, covering a simple spread/slippage/commission allowance. Target positions are solved on **post-cost equity**, so stock plus cash plus costs exactly equals pre-trade equity. Defaults assume 0% interest on positive cash, 6% annual interest on negative cash and 3% annual stock borrow cost, charged on prior-close balances using ACT/365. These are configurable modeling assumptions, not quoted broker terms.

Adjusted returns approximate reinvested distributions for longs and their economic cost for shorts. Fractional exposure is allowed. There is no intraday price path, locate/borrow availability model, cash collateral restriction, tax model or live order/fill handling. Target stock exposure is bounded to [−1, 2], but market moves can make the portfolio drift outside those weights between rebalances. A close-based 25% equity-to-absolute-stock maintenance threshold forces liquidation with costs. Insolvent accounts stop at zero equity, treating capital as limited liability; real leveraged accounts can owe more. This is a daily margin approximation, not a brokerage margin engine.

## What the included run shows

The fixed configuration made **2,631** sequential daily updates on 2,691 Alpaca SPY sessions, after 60 warm-up sessions. The later reporting window is **2024-01-02 to 2026-09-16**. It is a prequential evaluation: each session is scored before learning from its reward, while the policy continues adapting. It is not a frozen-policy test. No hyperparameters were chosen using this reporting window.

| Strategy | Net return in reporting window | Maximum drawdown |
|---|---:|---:|
| Online actor-critic | 42.72% | −16.65% |
| 100% SPY | 63.44% | −18.76% |
| 150% SPY with borrowing | 87.41% | −27.55% |
| Cash at modeled 0% rate | 0.00% | 0.00% |

The agent **underperformed holding SPY**. This is a working learning framework, not evidence of a trading advantage. Results use one seed and one historical path; transaction and financing costs matter. Comparisons are normalized to each strategy's equity at the reporting-window start; raw dollar costs are not directly comparable because their account sizes differ. End-of-data portfolios are marked to market rather than forcibly liquidated so daily continuation remains possible.

References: [policy gradients and actor-critic](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), [SPY and the S&P 500](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy), [Alpaca historical bars](https://docs.alpaca.markets/reference/stockbars).
