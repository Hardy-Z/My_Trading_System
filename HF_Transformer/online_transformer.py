"""Causal next-minute ETF forecasts with a streaming Transformer.

At the close of minute t, the model uses completed daily bars before the
session and completed current-session minute bars through t. Its label is
whether close(t+1) exceeds close(t). An online update occurs only after the
minute-(t+1) close is observed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


SYMBOLS = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    daily_window: int = 32
    minute_window: int = 24
    width: int = 16
    heads: int = 4
    layers: int = 1
    learning_rate: float = 1e-4
    train_fraction: float = .70
    validation_fraction: float = .15

    def __post_init__(self) -> None:
        if min(self.daily_window, self.minute_window, self.width, self.heads, self.layers) < 1:
            raise ValueError("Transformer dimensions and windows must be positive.")
        if self.width % self.heads:
            raise ValueError("Transformer width must be divisible by the number of heads.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Leave nonempty chronological train, validation and test splits.")


@dataclass
class MinuteSession:
    date: pd.Timestamp
    daily_end: int
    bar_times: pd.DatetimeIndex
    minute_tokens: np.ndarray  # every completed aligned bar, shape [bars, 8]
    decision_indices: np.ndarray  # only bars followed by a consecutive 1-minute bar
    labels: np.ndarray  # next-minute up indicators, shape [decisions, 3]


def daily_tokens_from_prices(daily_prices: np.ndarray) -> np.ndarray:
    """Return fixed-scale close, overnight and intraday returns for each ETF."""
    prices = np.asarray(daily_prices, dtype=np.float64)
    if prices.ndim != 3 or prices.shape[1:] != (3, 2):
        raise ValueError("Expected [session, three ETFs, (adjusted open, adjusted close)].")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Daily prices must be positive and finite.")
    opens, closes = prices[:, :, 0], prices[:, :, 1]
    previous = np.concatenate((closes[:1], closes[:-1]))
    returns = np.stack((np.log(closes / previous),
                        np.log(opens / previous), np.log(closes / opens)), axis=-1)
    returns[0, :, :2] = 0
    return np.clip(returns / .02, -10, 10).reshape(len(prices), 9).astype("float32")


def minute_tokens_from_bars(bars: pd.DataFrame) -> np.ndarray:
    """Encode all completed bars of one session, including the latest bar."""
    times = bars.index
    local = times.tz_convert("America/New_York")
    if (bars.empty or not times.is_monotonic_increasing or times.has_duplicates
            or len(set(local.date)) != 1
            or not ((local.time >= time(9, 30)) & (local.time < time(16))).all()):
        raise ValueError("Expected one ordered regular-hours session of completed bars.")
    opens = bars.loc[:, ("open", list(SYMBOLS))].to_numpy(float)
    closes = bars.loc[:, ("close", list(SYMBOLS))].to_numpy(float)
    if not np.isfinite(opens).all() or not np.isfinite(closes).all() or (opens <= 0).any() or (closes <= 0).any():
        raise ValueError("Minute prices must be positive and finite.")
    previous = np.concatenate((opens[:1], closes[:-1]))
    consecutive = np.diff(times.asi8) == 60_000_000_000
    previous[1:] = np.where(consecutive[:, None], previous[1:], opens[1:])
    close_move = np.clip(np.log(closes / previous) / .002, -10, 10)
    intrabar = np.clip(np.log(closes / opens) / .002, -10, 10)
    minute_number = (local.hour * 60 + local.minute - 570).to_numpy()
    phase = 2 * math.pi * minute_number / 390
    return np.concatenate((close_move, intrabar,
                           np.column_stack((np.sin(phase), np.cos(phase)))), axis=1).astype("float32")


def make_sessions(daily_dates: pd.DatetimeIndex, daily_prices: np.ndarray,
                  minute_bars: pd.DataFrame) -> tuple[np.ndarray, list[MinuteSession]]:
    """Build historical replay sessions without crossing gaps or session ends."""
    daily_tokens = daily_tokens_from_prices(daily_prices)
    if len(daily_dates) != len(daily_tokens) or not daily_dates.is_monotonic_increasing:
        raise ValueError("Daily dates and prices are misaligned.")
    local_times = minute_bars.index.tz_convert("America/New_York")
    session_dates = pd.DatetimeIndex(local_times.date)
    sessions = []
    for session_date, positions_group in pd.Series(np.arange(len(minute_bars)), index=session_dates).groupby(level=0):
        daily_end = int(daily_dates.searchsorted(session_date, side="left"))
        if daily_end < 2:
            continue
        positions = positions_group.to_numpy()
        times = minute_bars.index[positions]
        bars = minute_bars.iloc[positions]
        closes = bars.loc[:, ("close", list(SYMBOLS))].to_numpy(float)
        consecutive = np.diff(times.asi8) == 60_000_000_000
        tokens = minute_tokens_from_bars(bars)
        eligible = np.flatnonzero(consecutive)
        if not len(eligible):
            continue
        labels = (closes[eligible + 1] > closes[eligible]).astype("float32")
        sessions.append(MinuteSession(pd.Timestamp(session_date), daily_end, times,
                                      tokens, eligible, labels))
    if len(sessions) < 3:
        raise ValueError("Need at least three sessions with consecutive, aligned ETF minute bars.")
    return daily_tokens, sessions


def current_context(daily_dates: pd.DatetimeIndex, daily_prices: np.ndarray,
                    completed_session_bars: pd.DataFrame,
                    cfg: Settings) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, pd.Timestamp]:
    """Prepare a live context using only completed bars through the present minute."""
    tokens = minute_tokens_from_bars(completed_session_bars)
    session_date = pd.Timestamp(completed_session_bars.index[-1].tz_convert("America/New_York").date())
    daily_end = int(daily_dates.searchsorted(session_date, side="left"))
    if daily_end < 2:
        raise ValueError("Need at least two completed daily sessions before the current session.")
    daily = daily_tokens_from_prices(daily_prices)
    if len(daily) != len(daily_dates):
        raise ValueError("Daily dates and prices are misaligned.")
    last_close = completed_session_bars.loc[:, ("close", list(SYMBOLS))].iloc[-1].to_numpy(float)
    return (torch.from_numpy(daily[max(0, daily_end-cfg.daily_window):daily_end]),
            torch.from_numpy(tokens[-cfg.minute_window:]), last_close,
            completed_session_bars.index[-1])


class ETFTransformer(nn.Module):
    """Separate Transformer encoders for prior daily and current-session minutes."""

    def __init__(self, cfg: Settings):
        super().__init__()
        self.cfg = cfg
        self.daily_input = nn.Linear(9, cfg.width)
        self.minute_input = nn.Linear(8, cfg.width)
        self.daily_age = nn.Embedding(cfg.daily_window, cfg.width)
        self.minute_age = nn.Embedding(cfg.minute_window, cfg.width)

        def encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.width, nhead=cfg.heads, dim_feedforward=2 * cfg.width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, num_layers=cfg.layers,
                                         enable_nested_tensor=False)

        self.daily_encoder = encoder()
        self.minute_encoder = encoder()
        self.head = nn.Sequential(nn.LayerNorm(2 * cfg.width),
                                  nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                  nn.Linear(cfg.width, 3))

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, daily: torch.Tensor, minute: torch.Tensor) -> torch.Tensor:
        if daily.ndim != 2 or daily.shape[1] != 9 or not 1 <= len(daily) <= self.cfg.daily_window:
            raise ValueError("Daily input must have shape [1..daily_window, 9].")
        if minute.ndim != 2 or minute.shape[1] != 8 or not 1 <= len(minute) <= self.cfg.minute_window:
            raise ValueError("Minute input must have shape [1..minute_window, 8].")
        d_age = torch.arange(len(daily) - 1, -1, -1, device=daily.device)
        m_age = torch.arange(len(minute) - 1, -1, -1, device=minute.device)
        d = self.daily_input(daily) + self.daily_age(d_age)
        m = self.minute_input(minute) + self.minute_age(m_age)
        d = self.daily_encoder(d.unsqueeze(0), mask=self._causal_mask(len(d), daily.device))[0, -1]
        m = self.minute_encoder(m.unsqueeze(0), mask=self._causal_mask(len(m), minute.device))[0, -1]
        return torch.sigmoid(self.head(torch.cat((d, m))))


class OnlineAgent:
    """Predict now; update weights only when that prediction's label is known."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.model = ETFTransformer(cfg)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.updates = 0

    def predict(self, daily: torch.Tensor, minute: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(daily, minute).numpy().copy()

    def observe(self, daily: torch.Tensor, minute: torch.Tensor, label: np.ndarray) -> float:
        """Call after the next minute closes; one gradient step per revealed label."""
        target = torch.as_tensor(label, dtype=torch.float32)
        self.model.train()
        probability = self.model(daily, minute)
        loss = (probability - target).square().mean()  # Brier loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"settings": asdict(self.cfg), "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates}, path)

    @classmethod
    def load(cls, path: str | Path) -> OnlineAgent:
        state = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(Settings(**state["settings"]))
        agent.model.load_state_dict(state["model"])
        agent.optimizer.load_state_dict(state["optimizer"])
        agent.updates = int(state["updates"])
        return agent


class StreamingTrainer:
    """Update the previous forecast before issuing one for a new completed bar."""

    def __init__(self, agent: OnlineAgent):
        self.agent = agent
        self.pending: tuple[torch.Tensor, torch.Tensor, np.ndarray, pd.Timestamp] | None = None

    def on_completed_bar(self, daily: torch.Tensor, minute: torch.Tensor,
                         close: np.ndarray, bar_start: pd.Timestamp) -> tuple[np.ndarray, bool]:
        close = np.asarray(close, dtype=float)
        stamp = pd.Timestamp(bar_start)
        if close.shape != (3,) or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("Current close must contain three positive ETF prices.")
        if stamp.tz is None:
            raise ValueError("Bar timestamp must include its timezone.")
        updated = False
        if self.pending is not None:
            prior_daily, prior_minute, prior_close, prior_stamp = self.pending
            if stamp <= prior_stamp:
                raise ValueError("Completed bars must arrive in chronological order.")
            if stamp - prior_stamp == pd.Timedelta(minutes=1):
                label = (close > prior_close).astype("float32")
                self.agent.observe(prior_daily, prior_minute, label)
                updated = True
        probability = self.agent.predict(daily, minute)
        self.pending = (daily.clone(), minute.clone(), close.copy(), stamp)
        return probability, updated


def run_replay(daily_tokens: np.ndarray, sessions: list[MinuteSession],
               cfg: Settings = Settings(), verbose: bool = False) -> dict:
    """Chronological prequential train, frozen validation, frozen/online test."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    n = len(sessions)
    train_end = int(n * cfg.train_fraction)
    val_end = int(n * (cfg.train_fraction + cfg.validation_fraction))
    if not 0 < train_end < val_end < n:
        raise ValueError("Need nonempty train, validation and test session splits.")
    agent = OnlineAgent(cfg)
    records = []

    def process(day: MinuteSession, split: str, version: str,
                policy: OnlineAgent, learn: bool) -> None:
        daily = torch.from_numpy(daily_tokens[max(0, day.daily_end-cfg.daily_window):day.daily_end])
        minute_all = torch.from_numpy(day.minute_tokens)
        for index, label in zip(day.decision_indices, day.labels):
            minute = minute_all[max(0, index-cfg.minute_window+1):index+1]
            probability = policy.predict(daily, minute)
            start = day.bar_times[index]
            record = {"input_bar_start_utc": start.isoformat(),
                      "decision_time_utc": (start + pd.Timedelta(minutes=1)).isoformat(),
                      "outcome_available_utc": (start + pd.Timedelta(minutes=2)).isoformat(),
                      "session_date": str(day.date.date()), "split": split, "version": version,
                      "updates_before_forecast": policy.updates}
            for k, symbol in enumerate(SYMBOLS):
                record[f"{symbol}_prob_up"] = float(probability[k])
                record[f"{symbol}_up"] = int(label[k])
            records.append(record)
            if learn:
                policy.observe(daily, minute, label)

    for j, session in enumerate(sessions[:train_end], 1):
        process(session, "train", "online", agent, True)
        if verbose and j % 10 == 0:
            print(f"Trained {j}/{train_end} sessions", flush=True)
    for session in sessions[train_end:val_end]:
        process(session, "validation", "frozen", agent, False)
    frozen = copy.deepcopy(agent)
    for session in sessions[val_end:]:
        process(session, "test", "frozen", frozen, False)
        process(session, "test", "online", agent, True)
    predictions = pd.DataFrame(records)
    return {"predictions": predictions, "online_agent": agent, "frozen_agent": frozen,
            "split_dates": {"train_end": str(sessions[train_end-1].date.date()),
                            "validation_end": str(sessions[val_end-1].date.date()),
                            "test_end": str(sessions[-1].date.date())},
            "split_sessions": {"train": train_end, "validation": val_end-train_end,
                               "test": n-val_end}}


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, version), frame in predictions.groupby(["split", "version"], sort=False):
        for symbol in SYMBOLS:
            p = frame[f"{symbol}_prob_up"].to_numpy(float)
            y = frame[f"{symbol}_up"].to_numpy(float)
            rows.append({"split": split, "version": version, "symbol": symbol,
                         "n": len(frame), "accuracy_at_half": float(((p >= .5) == y).mean()),
                         "brier": float(np.mean((p-y)**2)), "up_rate": float(y.mean())})
    return pd.DataFrame(rows)
