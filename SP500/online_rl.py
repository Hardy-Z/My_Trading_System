"""Persistent online actor-critic for signed S&P 500 / cash allocation.

Historical/paper simulation only. Decisions after close execute at the next open.
The policy updates exactly once per newly completed session. No orders are sent.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCHEMA = 1


@dataclass(frozen=True)
class Config:
    weights: tuple = (-1., -.5, 0., .5, 1., 1.5, 2.)
    lookback: int = 20
    initial_equity: float = 100000.
    actor_lr: float = .0003
    critic_lr: float = .001
    gamma: float = .95
    entropy_coef: float = .001
    reward_scale: float = 100.
    advantage_clip: float = 10.
    transaction_bps: float = 5.
    cash_lending_rate: float = 0.
    cash_borrow_rate: float = .06
    stock_borrow_rate: float = .03
    maintenance_margin: float = .25
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, 'weights', tuple(float(w) for w in self.weights))
        if len(self.weights) < 2 or len(set(self.weights)) != len(self.weights):
            raise ValueError('Provide at least two distinct target stock weights.')
        if not all(np.isfinite(w) and -1 <= w <= 2 for w in self.weights):
            raise ValueError('This implementation supports stock weights from -1 to +2.')
        if 0. not in self.weights:
            raise ValueError('Include the zero-stock / cash action.')
        if not isinstance(self.lookback, int) or self.lookback < 20:
            raise ValueError('lookback must be an integer >= 20.')
        for key in ('initial_equity', 'actor_lr', 'critic_lr', 'reward_scale', 'advantage_clip'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite.')
        for key in ('entropy_coef', 'transaction_bps', 'cash_lending_rate', 'cash_borrow_rate', 'stock_borrow_rate'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative and finite.')
        if not 0 <= self.gamma < 1 or not 0 < self.maintenance_margin <= .5 or self.transaction_bps >= 100:
            raise ValueError('Invalid discount, maintenance margin or transaction cost.')


def load_prices(path: Path) -> pd.DataFrame:
    """CSV: date, open, close (same adjusted basis), raw_close (share estimates).

    Optional raw_open is accepted. No forward fills or invented prices. A close-
    only file is rejected because this execution convention needs next-day opens.
    """
    path = Path(path)
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    if not {'date', 'open', 'close'}.issubset(data):
        raise ValueError('CSV requires date, open and close; open/close must use the same adjustment basis.')
    data['date'] = pd.to_datetime(data['date'], errors='raise')
    data = data.set_index('date')
    if data.empty or data.index.hasnans or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError('Dates must be nonempty, unique and strictly increasing.')
    if data.index.tz is not None or not (data.index == data.index.normalize()).all():
        raise ValueError('Use timezone-free session dates, not intraday timestamps.')
    if 'raw_close' not in data:
        data['raw_close'] = data['close']
    columns = ['open', 'close', 'raw_close']
    data = data[columns].astype(float)
    if not np.isfinite(data.to_numpy()).all() or (data <= 0).any().any():
        raise ValueError('All prices must be finite and positive.')
    today = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date()
    data = data.loc[data.index.date < today]
    meta_path = path.with_suffix('.metadata.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Data checksum differs from its provenance file.')
        if meta.get('adjustment') != 'all':
            raise ValueError('Downloaded data must use all corporate-action adjustments.')
    return data


def price_row(date, row):
    return {'date': str(pd.Timestamp(date).date()), **{k: float(row[k]) for k in ('open', 'close', 'raw_close')}}


def extend_digest(digest, row):
    encoded = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256((digest + encoded).encode()).hexdigest()


def data_digest(data):
    digest = ''
    for date, row in data.iterrows():
        digest = extend_digest(digest, price_row(date, row))
    return digest


def features(history, stock_value, cash, config):
    """Fixed scales avoid fitting preprocessing on future observations."""
    closes = np.asarray([r['close'] for r in history], float)
    if len(closes) < config.lookback + 1:
        raise ValueError('Need lookback+1 past closes.')
    log_returns = np.diff(np.log(closes))[-config.lookback:]
    market = list(log_returns[::-1] / .02)
    for n in (5, 20):
        market.extend([log_returns[-n:].sum() / .05, log_returns[-n:].std() / .02])
    market.append((closes[-1] / closes[-20:].max() - 1) / .2)
    equity = stock_value + cash
    if equity <= 0:
        return np.zeros(config.lookback + 8)
    account = [stock_value / equity, cash / equity, np.log(equity / config.initial_equity)]
    state = np.clip(np.r_[market, account], -5., 5.)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite state features.')
    return state


class OnlineActorCritic:
    """Linear softmax actor and linear TD(0) critic, with persistent Adam state.

    One on-policy transition and one parameter update per completed session.
    No replay buffer, batch retraining, full-history scaling or hindsight labels.
    """
    def __init__(self, inputs, config):
        self.config = config
        self.actor = np.zeros((inputs + 1, len(config.weights)))
        self.critic = np.zeros(inputs + 1)
        self.m = {'actor': np.zeros_like(self.actor), 'critic': np.zeros_like(self.critic)}
        self.v = {'actor': np.zeros_like(self.actor), 'critic': np.zeros_like(self.critic)}
        self.updates = 0
        self.rng = np.random.default_rng(config.seed)

    def probabilities(self, state):
        logits = np.r_[state, 1.] @ self.actor
        exponentials = np.exp(logits - logits.max())
        return exponentials / exponentials.sum()

    def value(self, state):
        return float(np.r_[state, 1.] @ self.critic)

    def choose(self, state):
        probabilities = self.probabilities(state)
        return int(self.rng.choice(len(probabilities), p=probabilities)), probabilities

    @staticmethod
    def actor_gradient(state, probabilities, action, advantage, entropy_coef):
        onehot = np.eye(len(probabilities))[action]
        logp = np.log(np.maximum(probabilities, 1e-30))
        entropy = -float(probabilities @ logp)
        dlogits = advantage * (onehot - probabilities) - entropy_coef * probabilities * (logp + entropy)
        return np.outer(np.r_[state, 1.], dlogits)

    def learn(self, state, action, reward, next_state, terminal=False):
        cfg = self.config
        # Bootstrap is a stop-gradient target; the reported economic reward stays
        # the exact net arithmetic return. Scaling only changes numerical units.
        delta = cfg.reward_scale * reward + cfg.gamma * self.value(next_state) * (not terminal) - self.value(state)
        advantage = float(np.clip(delta, -cfg.advantage_clip, cfg.advantage_clip))
        probabilities = self.probabilities(state)
        gradients = {'actor': self.actor_gradient(state, probabilities, action, advantage, cfg.entropy_coef),
                     'critic': advantage * np.r_[state, 1.]}
        self.updates += 1
        for name, gradient in gradients.items():
            norm = np.linalg.norm(gradient)
            gradient = gradient * min(1., 1. / (norm + 1e-12))
            self.m[name] = .9 * self.m[name] + .1 * gradient
            self.v[name] = .999 * self.v[name] + .001 * gradient**2
            corrected_m = self.m[name] / (1 - .9**self.updates)
            corrected_v = self.v[name] / (1 - .999**self.updates)
            lr = cfg.actor_lr if name == 'actor' else cfg.critic_lr
            getattr(self, name)[:] += lr * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        return float(delta)

    def to_dict(self):
        return {'actor': self.actor.tolist(), 'critic': self.critic.tolist(),
                'm': {k: v.tolist() for k, v in self.m.items()}, 'v': {k: v.tolist() for k, v in self.v.items()},
                'updates': self.updates, 'rng_state': self.rng.bit_generator.state}

    def restore(self, saved):
        for name in ('actor', 'critic'):
            array = np.asarray(saved[name], float)
            if array.shape != getattr(self, name).shape or not np.isfinite(array).all():
                raise ValueError('Invalid checkpoint parameter shape or value.')
            setattr(self, name, array)
        self.m = {k: np.asarray(v, float) for k, v in saved['m'].items()}
        self.v = {k: np.asarray(v, float) for k, v in saved['v'].items()}
        self.updates = int(saved['updates'])
        self.rng.bit_generator.state = saved['rng_state']


def rebalance(equity, stock_value, weight, cost_rate):
    """Solve exact self-financing allocation AFTER proportional trading costs."""
    if equity <= 0 or not -1 <= weight <= 2 or not 0 <= cost_rate < .01:
        raise ValueError('Invalid rebalance arguments.')
    sign = np.sign(weight * equity - stock_value)
    post_cost_equity = (equity + cost_rate * sign * stock_value) / (1 + cost_rate * sign * weight)
    if post_cost_equity <= 0:
        raise ValueError('Trading costs exhaust equity.')
    target_stock = weight * post_cost_equity
    turnover = abs(target_stock - stock_value)
    costs = cost_rate * turnover
    cash = (1 - weight) * post_cost_equity
    return target_stock, cash, costs, turnover


class OnlinePortfolio:
    def __init__(self, config=Config(), fixed_weight=None):
        self.config, self.fixed_weight = config, fixed_weight
        if fixed_weight is not None and not -1 <= fixed_weight <= 2:
            raise ValueError('Baseline weight outside supported bounds.')
        self.agent = OnlineActorCritic(config.lookback + 8, config)
        self.stock_value, self.cash = 0., config.initial_equity
        self.history, self.ledger = [], []
        self.digest, self.consumed_count = '', 0
        self.pending, self.bankrupt = None, False

    def initialize(self, warmup):
        if self.history:
            raise ValueError('Already initialized.')
        if len(warmup) < self.config.lookback + 1:
            raise ValueError('Insufficient warm-up history.')
        for date, row in warmup.iterrows():
            self._remember(price_row(date, row))
        self._choose()

    def _remember(self, row):
        self.history.append(row)
        self.history = self.history[-(self.config.lookback + 1):]
        self.digest = extend_digest(self.digest, row)
        self.consumed_count += 1

    def _choose(self):
        state = features(self.history, self.stock_value, self.cash, self.config)
        action, probabilities = self.agent.choose(state)
        weight = self.config.weights[action] if self.fixed_weight is None else self.fixed_weight
        self.pending = {'decision_date': self.history[-1]['date'], 'state': state.tolist(),
                        'action': action, 'probabilities': probabilities.tolist(), 'stock_weight': weight,
                        'policy_version': self.agent.updates}

    def advance(self, date, row):
        if not self.history or self.pending is None:
            raise ValueError('Initialize before advancing; bankrupt accounts cannot continue.')
        incoming = price_row(date, row)
        last = self.history[-1]
        if incoming['date'] <= last['date']:
            raise ValueError('A session can be learned only once, in strictly increasing date order.')
        if not all(np.isfinite(incoming[k]) and incoming[k] > 0 for k in ('open', 'close', 'raw_close')):
            raise ValueError('New prices must be finite and positive.')
        cfg, pending = self.config, self.pending
        equity_before = self.stock_value + self.cash
        days = (pd.Timestamp(incoming['date']) - pd.Timestamp(last['date'])).days
        if days > 10:
            raise ValueError('More than 10 calendar days between sessions: check missing input data.')
        carry = days / 365.
        cash_income = max(self.cash, 0) * cfg.cash_lending_rate * carry
        cash_borrow_cost = max(-self.cash, 0) * cfg.cash_borrow_rate * carry
        stock_borrow_cost = max(-self.stock_value, 0) * cfg.stock_borrow_rate * carry
        cash_open = self.cash + cash_income - cash_borrow_cost - stock_borrow_cost
        stock_open = self.stock_value * incoming['open'] / last['close']
        equity_open = stock_open + cash_open
        cost_rate = cfg.transaction_bps / 10000.
        transaction_cost, turnover, forced_cost, margin_call = 0., 0., 0., False
        if equity_open <= cost_rate * abs(stock_open):
            # Even closing the old position would exhaust capital. Treat this
            # as account ruin rather than attempting an infeasible rebalance.
            forced_cost = cost_rate * abs(stock_open)
            self.stock_value, self.cash, self.bankrupt = 0., 0., True
        else:
            self.stock_value, self.cash, transaction_cost, turnover = rebalance(
                equity_open, stock_open, pending['stock_weight'], cost_rate)
            self.stock_value *= incoming['close'] / incoming['open']
            equity_close = self.stock_value + self.cash
            margin_call = abs(self.stock_value) > 0 and equity_close / abs(self.stock_value) < cfg.maintenance_margin
            if equity_close <= 0 or margin_call:
                forced_cost = cost_rate * abs(self.stock_value)
                self.cash, self.stock_value = max(0., equity_close - forced_cost), 0.
                self.bankrupt = self.cash <= 0
        reward = (self.stock_value + self.cash) / equity_before - 1
        self._remember(incoming)
        next_state = features(self.history, self.stock_value, self.cash, cfg)
        delta = 0.
        if self.fixed_weight is None:
            delta = self.agent.learn(np.asarray(pending['state']), pending['action'], reward, next_state, self.bankrupt)
        log = {'decision_date': pending['decision_date'], 'session': incoming['date'],
               'policy_version_used': pending['policy_version'], 'updates_after': self.agent.updates,
               'stock_target_weight': pending['stock_weight'], 'cash_target_weight': 1 - pending['stock_weight'],
               'equity_before': equity_before, 'equity_open_before_trade': equity_open,
               'equity': self.stock_value + self.cash, 'stock_value': self.stock_value, 'cash': self.cash,
               'reward_net_return': reward, 'transaction_cost': transaction_cost,
               'turnover_dollars': turnover, 'forced_liquidation_cost': forced_cost,
               'cash_income': cash_income, 'cash_borrow_cost': cash_borrow_cost,
               'stock_borrow_cost': stock_borrow_cost, 'calendar_days': days,
               'margin_call': bool(margin_call), 'bankrupt': bool(self.bankrupt), 'td_error_scaled': delta}
        log.update({f'p_stock_{w:+.2f}': float(p) for w, p in zip(cfg.weights, pending['probabilities'])})
        self.ledger.append(log)
        self.pending = None
        if not self.bankrupt:
            self._choose()
        return log

    def decision(self):
        if self.pending is None:
            return {'status': 'STOPPED_BANKRUPT', 'mode': 'RESEARCH_NO_ORDERS'}
        equity = self.stock_value + self.cash
        last = self.history[-1]
        choices = []
        for i, (w, p) in enumerate(zip(self.config.weights, self.pending['probabilities'])):
            choices.append({'action_index': i, 'probability': float(p), 'stock_weight': w, 'cash_weight': 1-w,
                'stock_side': 'long' if w > 0 else 'short' if w < 0 else 'flat',
                'cash_side': 'lend' if w < 1 else 'borrow' if w > 1 else 'zero',
                'estimated_stock_dollars': w * equity, 'estimated_stock_units': w * equity / last['raw_close'],
                'estimated_cash_dollars': (1-w) * equity})
        return {'mode': 'RESEARCH_NO_ORDERS', 'as_of_close': last['date'], 'policy_version': self.agent.updates,
                'execution': 'Target weights at the next session open, after costs; shown amounts use current equity and raw close.',
                'probability_meaning': 'Action-selection probabilities, not probabilities of making a profit.',
                'equity': equity, 'current_stock_value': self.stock_value, 'current_cash': self.cash,
                'actions': choices, 'selected_action': choices[self.pending['action']],
                'expected_stock_weight': float(np.dot(self.config.weights, self.pending['probabilities'])),
                'expected_cash_weight': float(1-np.dot(self.config.weights, self.pending['probabilities']))}

    def save(self, path):
        payload = {'schema': SCHEMA, 'config': asdict(self.config), 'agent': self.agent.to_dict(),
            'stock_value': self.stock_value, 'cash': self.cash, 'history': self.history, 'ledger': self.ledger,
            'digest': self.digest, 'consumed_count': self.consumed_count, 'pending': self.pending,
            'bankrupt': self.bankrupt, 'fixed_weight': self.fixed_weight}
        atomic_json(path, payload)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported checkpoint schema.')
        result = cls(Config(**payload['config']), payload['fixed_weight'])
        result.agent.restore(payload['agent'])
        for key in ('stock_value', 'cash', 'history', 'ledger', 'digest', 'consumed_count', 'pending', 'bankrupt'):
            setattr(result, key, payload[key])
        return result

    def ingest(self, prices):
        """Resume from a full CSV; verify the past and process only new sessions."""
        past = prices.loc[:self.history[-1]['date']]
        if len(past) != self.consumed_count or data_digest(past) != self.digest:
            raise ValueError('Previously consumed history changed. Do not continue on revised prices; replay into a new output folder.')
        fresh = prices.loc[prices.index > pd.Timestamp(self.history[-1]['date'])]
        processed = 0
        for date, row in fresh.iterrows():
            if self.bankrupt:
                break
            self.advance(date, row)
            processed += 1
        return processed


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


@contextmanager
def writer_lock(checkpoint):
    lock = Path(str(checkpoint) + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Another writer or a stale lock exists; inspect it before retrying.') from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def export(portfolio, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(portfolio.ledger).to_csv(output / 'daily_learning.csv', index=False)
    atomic_json(output / 'latest_decision.json', portfolio.decision())


def replay(data, config, warmup=60, fixed_weight=None):
    if warmup < config.lookback + 1 or len(data) <= warmup:
        raise ValueError('Need a sufficient warm-up and at least one subsequent session.')
    portfolio = OnlinePortfolio(config, fixed_weight)
    portfolio.initialize(data.iloc[:warmup])
    for date, row in data.iloc[warmup:].iterrows():
        portfolio.advance(date, row)
        if portfolio.bankrupt:
            break
    return portfolio


def metrics(ledger, since=None):
    frame = pd.DataFrame(ledger)
    if since:
        frame = frame.loc[frame.session >= since]
    if frame.empty:
        return {'sessions': 0}
    curve = np.r_[frame.equity_before.iloc[0], frame.equity.to_numpy()]
    returns = frame.reward_net_return.to_numpy()
    std = returns.std(ddof=1) if len(returns) > 1 else 0.
    return {'sessions': len(frame), 'first_session': frame.session.iloc[0], 'last_session': frame.session.iloc[-1],
            'return': float(curve[-1]/curve[0]-1), 'max_drawdown': float((curve/np.maximum.accumulate(curve)-1).min()),
            'sharpe_252_zero_hurdle': float(np.sqrt(252)*returns.mean()/std) if std > 1e-12 else None,
            'transaction_cost': float(frame.transaction_cost.sum()+frame.forced_liquidation_cost.sum()),
            'financing_cost': float(frame.cash_borrow_cost.sum()+frame.stock_borrow_cost.sum()),
            'margin_calls': int(frame.margin_call.sum()), 'bankrupt': bool(frame.bankrupt.any())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('replay', help='One chronological online learning pass and baseline comparison.')
    p.add_argument('--csv', type=Path, default=HERE / 'data' / 'SPY_daily.csv')
    p.add_argument('--output', type=Path, default=HERE / 'outputs')
    p.add_argument('--warmup', type=int, default=60)
    p.add_argument('--evaluation-start', default='2024-01-01')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weights', default='-1,-0.5,0,0.5,1,1.5,2')
    p = sub.add_parser('update', help='Consume newly completed sessions once and persist the updated policy.')
    p.add_argument('--csv', type=Path, default=HERE / 'data' / 'SPY_daily.csv')
    p.add_argument('--checkpoint', type=Path, default=HERE / 'outputs' / 'checkpoint.json')
    p = sub.add_parser('predict', help='Show the already sampled, pending action without changing policy or RNG.')
    p.add_argument('--checkpoint', type=Path, default=HERE / 'outputs' / 'checkpoint.json')
    args = parser.parse_args()
    if args.command == 'predict':
        print(json.dumps(OnlinePortfolio.load(args.checkpoint).decision(), indent=2, allow_nan=False))
        return
    data = load_prices(args.csv)
    if args.command == 'update':
        with writer_lock(args.checkpoint):
            portfolio = OnlinePortfolio.load(args.checkpoint)
            count = portfolio.ingest(data)
            if count:
                portfolio.save(args.checkpoint)
            export(portfolio, args.checkpoint.parent)
        print(f'Processed {count} new sessions. Policy version: {portfolio.agent.updates}. Last session: {portfolio.history[-1]["date"]}.')
        return
    cfg = Config(seed=args.seed, weights=tuple(float(x) for x in args.weights.split(',')))
    checkpoint = args.output / 'checkpoint.json'
    with writer_lock(checkpoint):
        if checkpoint.exists():
            raise ValueError('Checkpoint already exists; choose a new --output folder or use update.')
        portfolio = replay(data, cfg, args.warmup)
        portfolio.save(checkpoint)
        export(portfolio, args.output)
    results = {'online_actor_critic': portfolio}
    for name, weight in (('cash', 0.), ('long_100pct', 1.), ('long_150pct', 1.5)):
        baseline = replay(data, cfg, args.warmup, weight)
        pd.DataFrame(baseline.ledger).to_csv(args.output / f'baseline_{name}.csv', index=False)
        results[name] = baseline
    summary = {'evaluation_start': args.evaluation_start, 'config': asdict(cfg),
               'data_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               'protocol': 'Single chronological pass. After each day is scored, learn its reward before choosing the next action. No hyperparameters selected from this evaluation.',
               'all_sessions': {k: metrics(v.ledger) for k, v in results.items()},
               'evaluation_period': {k: metrics(v.ledger, args.evaluation_start) for k, v in results.items()}}
    atomic_json(args.output / 'evaluation.json', summary)
    print(json.dumps(summary['evaluation_period'], indent=2))
    print(f'Policy updated {portfolio.agent.updates} times. Files saved in {args.output}.')


if __name__ == '__main__':
    main()
