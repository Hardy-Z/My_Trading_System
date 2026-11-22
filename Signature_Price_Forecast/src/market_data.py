"""Market data for the forecast, loaded with quant-research-dataset (qrd).

The SPY, QQQ and IWM daily bars come from the qrd dataset ``us/etf/daily_bars``
(https://pypi.org/project/quant-research-dataset/). That dataset is local only
(Alpaca data is not on the Hugging Face Hub), so qrd must read from a local data
folder: set QRD_DATA_DIR as an environment variable or as a line in the
repository's .env file.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import quant_research_dataset as qrd

ROOT = Path(__file__).resolve().parents[2]  # repository root


def _env_file_value(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def configure() -> None:
    """Point qrd at QRD_DATA_DIR from the environment, else from the repository's .env file."""
    if os.environ.get("QRD_DATA_DIR"):
        return
    data_dir = _env_file_value(ROOT / ".env", "QRD_DATA_DIR")
    if data_dir:
        qrd.configure(data_dir=data_dir)


configure()


def etf_daily(symbol: str, columns: Sequence[str] = ("open", "close")) -> pd.DataFrame:
    """Daily bars of one ETF, indexed by date, adjusted for splits and dividends."""
    frame = qrd.get_etf_bars(symbol, freq="1d", columns=["date", *columns])
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame.set_index("date").sort_index()
