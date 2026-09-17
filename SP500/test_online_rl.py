from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from online_rl import (Config, OnlineActorCritic, OnlinePortfolio, rebalance,
    replay, features, price_row, load_prices, writer_lock)
from download_data import append_preserving_history


def prices(n=80):
    t = np.arange(n)
    close = 100*np.exp(.001*t + .015*np.sin(t/3))
    open_ = close * (1 + .003*np.cos(t))
    return pd.DataFrame({'open': open_, 'close': close, 'raw_close': close},
                        index=pd.bdate_range('2020-01-01', periods=n))


class OnlineRLTests(unittest.TestCase):
    def test_append_rebases_adjustments_without_revising_learned_rows(self):
        old = prices(30)
        refreshed = prices(35)
        refreshed[['open', 'close']] *= .99
        merged, factor = append_preserving_history(old, refreshed)
        pd.testing.assert_frame_equal(merged.iloc[:30], old)
        self.assertAlmostEqual(factor, 1/.99)
        self.assertAlmostEqual(merged.close.iloc[30]/merged.close.iloc[29],
                               refreshed.close.iloc[30]/refreshed.close.iloc[29])
        refreshed.iloc[10, 0] *= 1.01
        with self.assertRaisesRegex(ValueError, 'Historical prices were revised'):
            append_preserving_history(old, refreshed)

    def test_self_financing_with_costs_for_all_actions(self):
        for weight in Config().weights:
            for previous in (-50000., 0., 70000., 180000.):
                stock, cash, costs, turnover = rebalance(100000, previous, weight, .0005)
                self.assertAlmostEqual(stock + cash + costs, 100000)
                self.assertAlmostEqual(stock / (stock+cash), weight)
                self.assertAlmostEqual(costs, turnover * .0005)
                self.assertAlmostEqual(cash, 100000-previous-(stock-previous)-costs)

    def test_next_open_timing_no_overnight_profit_before_first_fill(self):
        data = prices(23)
        data.iloc[:21] = 100.
        data.iloc[21] = 120.
        data.iloc[22] = 132.
        cfg = Config(transaction_bps=0, cash_borrow_rate=0, stock_borrow_rate=0)
        agent = OnlinePortfolio(cfg, fixed_weight=1.)
        agent.initialize(data.iloc[:21])
        row = agent.advance(data.index[21], data.iloc[21])
        self.assertAlmostEqual(row['reward_net_return'], 0.)
        row = agent.advance(data.index[22], data.iloc[22])
        self.assertAlmostEqual(row['reward_net_return'], .1)
        self.assertAlmostEqual(row['equity'], 110000)

    def test_short_stock_positive_cash_and_borrowed_cash(self):
        stock, cash, _, _ = rebalance(100000, 0, -.5, 0)
        self.assertEqual((stock, cash), (-50000, 150000))
        stock, cash, _, _ = rebalance(100000, 0, 1.5, 0)
        self.assertEqual((stock, cash), (150000, -50000))

    def test_cash_and_stock_financing_calendar_days(self):
        data = prices(23)
        data.iloc[:] = 100.
        cfg = Config(transaction_bps=0, cash_borrow_rate=.365, stock_borrow_rate=.365)
        for weight, expected_basis in [(1.5, 50000), (-1., 100000)]:
            agent = OnlinePortfolio(cfg, fixed_weight=weight)
            agent.initialize(data.iloc[:21])
            agent.advance(data.index[21], data.iloc[21])
            next_date = data.index[21] + pd.Timedelta(days=3)
            row = agent.advance(next_date, data.iloc[22])
            self.assertAlmostEqual(row['cash_borrow_cost']+row['stock_borrow_cost'], expected_basis*.003)
            self.assertAlmostEqual(row['reward_net_return'], -expected_basis*.003/100000)

    def test_bankruptcy_and_margin_stop(self):
        data = prices(22)
        data.iloc[:21] = 100.
        data.iloc[21] = [100., 200., 200.]
        agent = OnlinePortfolio(Config(transaction_bps=0), fixed_weight=-1.)
        agent.initialize(data.iloc[:21])
        row = agent.advance(data.index[21], data.iloc[21])
        self.assertTrue(agent.bankrupt)
        self.assertEqual(row['reward_net_return'], -1.)
        self.assertIsNone(agent.pending)
        self.assertEqual(agent.decision()['status'], 'STOPPED_BANKRUPT')

    def test_gap_leaves_insufficient_equity_to_close_position(self):
        data = prices(23)
        data.iloc[:] = 100.
        agent = OnlinePortfolio(Config(transaction_bps=5, cash_borrow_rate=0), fixed_weight=2.)
        agent.initialize(data.iloc[:21])
        agent.advance(data.index[21], data.iloc[21])
        # Approximately a 50% overnight gap on a 2x-long account: small positive
        # residual equity is less than the cost of closing the old position.
        next_row = pd.Series({'open': 50.01, 'close': 50.01, 'raw_close': 50.01})
        result = agent.advance(data.index[22], next_row)
        self.assertTrue(agent.bankrupt)
        self.assertEqual(result['equity'], 0.)
        self.assertEqual(result['reward_net_return'], -1.)

    def test_online_updates_and_daily_reward_identity(self):
        data = prices(70)
        agent = replay(data, Config(), warmup=21)
        self.assertEqual(agent.agent.updates, 49)
        self.assertFalse(np.array_equal(agent.agent.actor, np.zeros_like(agent.agent.actor)))
        for i, row in enumerate(agent.ledger):
            self.assertEqual(row['updates_after'], i+1)
            self.assertEqual(row['policy_version_used'], i)
            self.assertAlmostEqual(row['reward_net_return'], row['equity']/row['equity_before']-1)
            self.assertLess(row['decision_date'], row['session'])
            self.assertAlmostEqual(row['stock_target_weight']+row['cash_target_weight'], 1)

    def test_resume_exactly_matches_uninterrupted_online_learning(self):
        data = prices()
        continuous = replay(data, Config(), warmup=21)
        interrupted = replay(data.iloc[:48], Config(), warmup=21)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'checkpoint.json'
            interrupted.save(checkpoint)
            resumed = OnlinePortfolio.load(checkpoint)
            self.assertEqual(resumed.ingest(data), len(data)-48)
            self.assertEqual(resumed.ledger, continuous.ledger)
            self.assertEqual(resumed.decision(), continuous.decision())
            np.testing.assert_array_equal(resumed.agent.actor, continuous.agent.actor)
            np.testing.assert_array_equal(resumed.agent.critic, continuous.agent.critic)
            before = json.dumps(resumed.agent.to_dict(), sort_keys=True)
            self.assertEqual(resumed.ingest(data), 0)
            self.assertEqual(json.dumps(resumed.agent.to_dict(), sort_keys=True), before)

    def test_revised_history_and_duplicates_rejected(self):
        data = prices()
        agent = replay(data.iloc[:50], Config(), warmup=21)
        modified = data.copy()
        modified.iloc[5, 0] *= 1.001
        with self.assertRaisesRegex(ValueError, 'Previously consumed history changed'):
            agent.ingest(modified)
        with self.assertRaisesRegex(ValueError, 'only once'):
            agent.advance(data.index[49], data.iloc[49])

    def test_prefix_invariance_when_future_data_changes(self):
        data = prices()
        original = replay(data, Config(), warmup=21)
        changed = data.copy()
        changed.iloc[60:] *= 1.1
        alternative = replay(changed, Config(), warmup=21)
        self.assertEqual(original.ledger[:39], alternative.ledger[:39])

    def test_actor_gradient_finite_differences(self):
        cfg = Config()
        model = OnlineActorCritic(3, cfg)
        rng = np.random.default_rng(5)
        model.actor[:] = rng.normal(size=model.actor.shape)*.1
        state = np.array([.2, -.4, 1.])
        action, advantage, entropy_coef = 2, -.7, .03
        probs = model.probabilities(state)
        gradient = model.actor_gradient(state, probs, action, advantage, entropy_coef)
        def objective():
            p = model.probabilities(state)
            return advantage*np.log(p[action])-entropy_coef*np.sum(p*np.log(p))
        for idx in np.ndindex(model.actor.shape):
            original, eps = model.actor[idx], 1e-6
            model.actor[idx] = original+eps
            plus = objective()
            model.actor[idx] = original-eps
            minus = objective()
            model.actor[idx] = original
            self.assertAlmostEqual(gradient[idx], (plus-minus)/(2*eps), places=6)

    def test_td_bootstrap_and_terminal_mask(self):
        cfg = Config(gamma=.5)
        model = OnlineActorCritic(2, cfg)
        model.critic[-1] = 2.
        delta = model.learn(np.zeros(2), 0, .01, np.zeros(2), terminal=False)
        self.assertEqual(delta, 0.)  # 1 + .5*2 - 2
        model.critic[-1] = 2.
        delta = model.learn(np.zeros(2), 0, .01, np.zeros(2), terminal=True)
        self.assertEqual(delta, -1.)  # 1 - 2

    def test_probability_and_dollar_allocation_output(self):
        agent = replay(prices(30), Config(), warmup=21)
        result = agent.decision()
        self.assertAlmostEqual(sum(a['probability'] for a in result['actions']), 1)
        for action in result['actions']:
            self.assertAlmostEqual(action['estimated_stock_dollars']+action['estimated_cash_dollars'], result['equity'])
        self.assertEqual(agent.decision(), result)  # Reading does not resample.
        json.dumps(result, allow_nan=False)

    def test_csv_and_lock_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.csv'
            path.write_text('date,close\n2020-01-01,100\n')
            with self.assertRaisesRegex(ValueError, 'open'):
                load_prices(path)
            path.write_text('date,open,close\n2020-01-01,-1,100\n')
            with self.assertRaises(ValueError):
                load_prices(path)
            checkpoint = Path(folder)/'checkpoint.json'
            with writer_lock(checkpoint):
                with self.assertRaises(RuntimeError):
                    with writer_lock(checkpoint):
                        pass
            self.assertFalse(Path(str(checkpoint)+'.lock').exists())


if __name__ == '__main__':
    unittest.main()
