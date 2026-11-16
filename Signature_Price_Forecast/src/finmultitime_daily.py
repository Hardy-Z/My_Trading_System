"""Causal, four-modality FinMultiTime features and online next-day ETF model.

The public archive is read with HTTP ranges, so only selected news and filing
members are transferred. Price breadth uses every CSV in the price archive.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch import nn

BASE = "https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/"
API = "https://huggingface.co/api/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/tree/main/"
PRICE_ARCHIVE = "time_series/S%26P500_time_series.zip"
NEWS_ARCHIVE = "text/sp500_news.zip"
TABLE_ARCHIVE = "table/SP500_tabular.zip"
TARGETS = ("SPY", "QQQ", "IWM")
CONTEXT = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "WMT")
TABLE_CONTEXT = tuple(x for x in CONTEXT if x not in ("SPY", "QQQ"))
NEWS_DIM = 32
IMAGE_SIZE = 12


class RemoteRangeFile(io.RawIOBase):
    """Read a public zip through a bounded block cache, without fetching it all."""

    def __init__(self, url: str, block_size: int = 1 << 20, cache_blocks: int = 32):
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=45)
        response.raise_for_status()
        self.url = response.url
        self.length = int(response.headers["Content-Length"])
        self.position = 0
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else
                         self.position + offset if whence == 1 else self.length + offset)
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        result = bytearray()
        while size > 0 and self.position < self.length:
            start = self.position // self.block_size * self.block_size
            if start not in self.cache:
                end = min(start + self.block_size - 1, self.length - 1)
                for attempt in range(4):
                    try:
                        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
                        response.raise_for_status()
                        if response.status_code != 206 or len(response.content) != end - start + 1:
                            raise IOError("Server did not honor byte range")
                        self.cache[start] = response.content
                        break
                    except (requests.RequestException, IOError):
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            part = self.cache[start][self.position - start:self.position - start + size]
            result.extend(part)
            self.position += len(part)
            size -= len(part)
        return bytes(result)


def download_price_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "SP500_time_series.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    url = BASE + PRICE_ARCHIVE
    head = requests.head(url, allow_redirects=True, timeout=45)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    part = path.with_suffix(".zip.part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=90) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            raise IOError("Resume requested but server did not honor byte range")
        with part.open("ab" if offset else "wb") as out:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    out.write(chunk)
    if part.stat().st_size != expected or not zipfile.is_zipfile(part):
        raise IOError("Incomplete FinMultiTime price archive")
    part.replace(path)
    return path


def _date_index(values):
    return pd.to_datetime(values, utc=True).tz_localize(None).normalize()


def price_features(price_zip: Path, target_dir: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """ETF OHLC and all-symbol market breadth, aligned on ETF sessions."""
    frames = []
    for symbol in TARGETS:
        frame = pd.read_csv(target_dir / f"{symbol}_daily.csv")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frames.append(frame[["open", "close"]].rename(columns=lambda c: f"{symbol}_{c}"))
    etf = pd.concat(frames, axis=1, join="inner").dropna()
    with zipfile.ZipFile(price_zip) as archive:
        upper = []
        for symbol in TARGETS[:2]:
            with archive.open(f"S&P500_time_series/{symbol.lower()}.csv") as source:
                df = pd.read_csv(source, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
            df.index = pd.to_datetime(df["Date"].str[:10])
            df = df.reindex(etf.index)
            upper.extend([
                np.log(df["Close"] / df["Close"].shift(1)).to_numpy(),
                ((df["High"] - df["Low"]) / df["Close"]).to_numpy(),
                np.log1p(df["Volume"]).to_numpy(),
            ])
        dates = etf.index
        sums = np.zeros(len(dates), np.float64)
        squares = np.zeros(len(dates), np.float64)
        ups = np.zeros(len(dates), np.float64)
        counts = np.zeros(len(dates), np.float64)
        symbol_count = 0
        for member in archive.namelist():
            if not member.endswith(".csv"):
                continue
            with archive.open(member) as source:
                try:
                    df = pd.read_csv(source, usecols=["Date", "Close"])
                except (ValueError, pd.errors.ParserError):
                    continue
            d = pd.to_datetime(df["Date"].str[:10], errors="coerce")
            close_values = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
            close = pd.Series(np.where(close_values > 0, close_values, np.nan), index=d)
            close = close.loc[~close.index.duplicated(keep="last")].sort_index()
            returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).reindex(dates).to_numpy()
            good = np.isfinite(returns) & (np.abs(returns) < 0.7)
            values = np.where(good, returns, 0.0)
            sums += values
            squares += values ** 2
            ups += (values > 0) & good
            counts += good
            symbol_count += 1
        mean = sums / np.maximum(counts, 1)
        sd = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean ** 2, 0))
        breadth = [mean, sd, ups / np.maximum(counts, 1), np.log1p(counts)]
    etf_returns = [np.log(etf[f"{s}_close"] / etf[f"{s}_close"].shift(1)).to_numpy() for s in TARGETS]
    gap = [(etf[f"{s}_close"] / etf[f"{s}_open"] - 1).to_numpy() for s in TARGETS]
    features = np.column_stack(etf_returns + gap + upper + breadth).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = (etf[[f"{s}_close" for s in TARGETS]].shift(-1).to_numpy() >
              etf[[f"{s}_close" for s in TARGETS]].to_numpy()).astype(np.float32)
    print(f"Price breadth: {symbol_count} symbol CSVs, {len(dates)} ETF sessions")
    return dates, features, labels


def _extract_members(remote_archive: str, members: dict[str, str], cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {key: cache_dir / key / Path(name).name for key, name in members.items()}
    missing = {key: name for key, name in members.items() if not result[key].exists()}
    if missing:
        with zipfile.ZipFile(RemoteRangeFile(BASE + remote_archive)) as archive:
            available = set(archive.namelist())
            for key, name in missing.items():
                if name not in available:
                    print(f"Missing archive member: {name}")
                    continue
                path = result[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                with archive.open(name) as source, tmp.open("wb") as dest:
                    while chunk := source.read(1 << 20):
                        dest.write(chunk)
                tmp.replace(path)
                print("Cached", name)
    return {key: path for key, path in result.items() if path.exists()}


def _hash_text(text: str) -> np.ndarray:
    vector = np.zeros(NEWS_DIM, np.float32)
    for token in re.findall(r"[a-z]{3,}", text.lower()[:5000]):
        digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
        integer = int.from_bytes(digest, "little")
        vector[integer % NEWS_DIM] += 1 if integer & (1 << 31) else -1
    return vector / max(np.linalg.norm(vector), 1.0)


def news_features(dates: pd.DatetimeIndex, cache_dir: Path, symbols=CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    members = {symbol: f"sp500_news/{symbol}.jsonl" for symbol in symbols}
    files = _extract_members(NEWS_ARCHIVE, members, cache_dir / "news")
    result = np.zeros((len(dates), NEWS_DIM + 2), np.float32)
    raw_articles = 0
    for symbol, path in files.items():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    date = pd.Timestamp(item["Date"]).normalize()
                except (KeyError, ValueError, TypeError):
                    continue
                # Strictly later session: date-only news may have been published after close.
                idx = dates.searchsorted(date, side="right")
                if idx >= len(dates) or idx == 0:
                    continue
                text = str(item.get("Article", ""))
                if not text:
                    continue
                result[idx, :NEWS_DIM] += _hash_text(text)
                result[idx, NEWS_DIM] += 1
                result[idx, NEWS_DIM + 1] += min(len(text), 5000) / 5000
                raw_articles += 1
    count = np.maximum(result[:, NEWS_DIM:NEWS_DIM + 1], 1)
    result[:, :NEWS_DIM] /= count
    result[:, NEWS_DIM + 1:] /= count
    result[:, NEWS_DIM] = np.log1p(result[:, NEWS_DIM])
    return result, np.array([len(files), raw_articles])


TABLE_TAGS = ("Assets", "Liabilities", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue")


def _filing_values(filing: dict) -> np.ndarray | None:
    dated = pd.Timestamp(filing.get("filing_date", "1900-01-01"))
    facts = filing.get("facts", {}).get("us-gaap", {})
    values = []
    for tag in TABLE_TAGS:
        rows = facts.get(tag, {}).get("units", {}).get("USD", [])
        eligible = [x for x in rows if x.get("end", "9999") <= str(dated.date())
                    and x.get("filed", "9999") <= str(dated.date())
                    and isinstance(x.get("val"), (float, int))]
        if not eligible:
            values.append(np.nan)
        else:
            latest = max(eligible, key=lambda x: (x.get("end", ""), x.get("filed", "")))
            values.append(float(latest["val"]))
    if not np.isfinite(values[0]) or values[0] <= 0:
        return None
    assets = values[0]
    return np.array([np.log1p(assets) / 30,
                     values[1] / assets if np.isfinite(values[1]) else 0,
                     values[2] / assets if np.isfinite(values[2]) else 0,
                     values[3] / assets if np.isfinite(values[3]) else 0], np.float32)


def table_features(dates: pd.DatetimeIndex, cache_dir: Path, symbols=TABLE_CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    members = {symbol: f"financial_reports/{symbol.lower()}/condensed_consolidated_balance_sheets.json"
               for symbol in symbols}
    files = _extract_members(TABLE_ARCHIVE, members, cache_dir / "tables")
    per_symbol = []
    filing_count = 0
    for symbol, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        events = []
        for filing in data.get("filings", []):
            try:
                filed = pd.Timestamp(filing["filing_date"]).normalize()
                vector = _filing_values(filing)
            except (KeyError, ValueError, TypeError):
                continue
            if vector is None:
                continue
            # SEC filing date has no release time here; expose on the next session.
            idx = dates.searchsorted(filed, side="right")
            if idx < len(dates):
                events.append((idx, vector))
                filing_count += 1
        arr = np.zeros((len(dates), 4), np.float32)
        valid = np.zeros(len(dates), np.float32)
        last = None
        cursor = 0
        events.sort(key=lambda x: x[0])
        for i in range(len(dates)):
            while cursor < len(events) and events[cursor][0] <= i:
                last = events[cursor][1]
                cursor += 1
            if last is not None:
                arr[i] = last
                valid[i] = 1
        per_symbol.append((arr, valid))
    result = np.zeros((len(dates), 5), np.float32)
    for arr, valid in per_symbol:
        result[:, :4] += arr
        result[:, 4] += valid
    result[:, :4] /= np.maximum(result[:, 4:5], 1)
    result[:, 4] /= max(len(per_symbol), 1)
    return result, np.array([len(files), filing_count])


def _image_directory(symbol: str, session: requests.Session) -> str | None:
    group = f"S&P500_image_{symbol[0].lower()}"
    candidates = [f"image/image/{group}/{symbol.lower()}",
                  f"image/image/S&P500_image_s/{group}/{symbol.lower()}"]
    for candidate in candidates:
        response = session.get(API + quote(candidate, safe="/"), params={"limit": 100}, timeout=30)
        if response.ok and response.json():
            return candidate
    return None


def image_features(dates: pd.DatetimeIndex, cache_dir: Path, symbols=CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    event_vectors: list[tuple[int, str, np.ndarray]] = []
    symbols_found = 0
    for symbol in symbols:
        directory = _image_directory(symbol, session)
        if directory is None:
            print("No chart directory for", symbol)
            continue
        symbols_found += 1
        listing = session.get(API + quote(directory, safe="/"), params={"limit": 1000}, timeout=30)
        listing.raise_for_status()
        for record in listing.json():
            path = record["path"]
            match = re.search(r"_(\d{4})_H([12])_candlestick\.png$", path)
            if not match:
                continue
            year, half = map(int, match.groups())
            period_end = pd.Timestamp(year=year, month=6 if half == 1 else 12, day=30 if half == 1 else 31)
            # Only recent completed charts matter for the selected backtest.
            if period_end < dates[0] - pd.Timedelta(days=210):
                continue
            idx = dates.searchsorted(period_end, side="right")
            if idx >= len(dates):
                continue
            local = image_dir / f"{symbol}_{year}_H{half}.png"
            if not local.exists():
                response = session.get(BASE + quote(path, safe="/"), timeout=60)
                response.raise_for_status()
                local.write_bytes(response.content)
            try:
                with Image.open(local) as im:
                    rgb = im.convert("RGB")
                    width, height = rgb.size
                    rgb = rgb.crop((int(width * .10), int(height * .17),
                                    int(width * .90), int(height * .90)))
                    rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
                    vector = np.asarray(rgb, np.float32).reshape(-1) / 255
            except (OSError, ValueError):
                continue
            event_vectors.append((idx, symbol, vector))
    by_symbol: dict[str, list[tuple[int, np.ndarray]]] = {}
    for idx, symbol, vector in event_vectors:
        by_symbol.setdefault(symbol, []).append((idx, vector))
    result = np.zeros((len(dates), IMAGE_SIZE * IMAGE_SIZE * 3 + 1), np.float32)
    for symbol, events in by_symbol.items():
        events.sort(key=lambda x: x[0])
        cursor = 0
        last = None
        for i in range(len(dates)):
            while cursor < len(events) and events[cursor][0] <= i:
                last = events[cursor][1]
                cursor += 1
            if last is not None:
                result[i, :-1] += last
                result[i, -1] += 1
    result[:, :-1] /= np.maximum(result[:, -1:], 1)
    result[:, -1] /= max(symbols_found, 1)
    return result, np.array([symbols_found, len(event_vectors)])


def build_dataset(target_dir: Path, cache_dir: Path, start="2018-01-01", end="2025-03-28") -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = cache_dir / "prepared.npz"
    manifest = cache_dir / "prepared_manifest.json"
    settings = {"feature_version": 5, "start": start, "end": end, "context": list(CONTEXT), "news_dim": NEWS_DIM,
                "image_size": IMAGE_SIZE, "targets": list(TARGETS), "source": BASE,
                "target_files": {symbol: [(target_dir / f"{symbol}_daily.csv").stat().st_size,
                                          (target_dir / f"{symbol}_daily.csv").stat().st_mtime_ns]
                                 for symbol in TARGETS}}
    if prepared.exists() and manifest.exists() and json.loads(manifest.read_text()) == settings:
        with np.load(prepared) as data:
            return {key: data[key] for key in data.files}
    price_zip = download_price_archive(cache_dir)
    base_cache = cache_dir / "price_base.npz"
    base_manifest = cache_dir / "price_base_manifest.json"
    source_stat = {"price_zip": [price_zip.stat().st_size, price_zip.stat().st_mtime_ns],
                   "targets": {symbol: [(target_dir / f"{symbol}_daily.csv").stat().st_size,
                                        (target_dir / f"{symbol}_daily.csv").stat().st_mtime_ns]
                               for symbol in TARGETS}}
    if base_cache.exists() and base_manifest.exists() and json.loads(base_manifest.read_text()) == source_stat:
        with np.load(base_cache) as base:
            all_dates = pd.DatetimeIndex(pd.to_datetime(base["dates"]))
            price, labels = base["price"], base["labels"]
    else:
        all_dates, price, labels = price_features(price_zip, target_dir)
        np.savez_compressed(base_cache, dates=np.asarray(all_dates.strftime("%Y-%m-%d"), dtype="U10"),
                            price=price, labels=labels)
        base_manifest.write_text(json.dumps(source_stat, indent=2))
    select = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = all_dates[select]
    price, labels = price[select], labels[select]
    news, news_counts = news_features(dates, cache_dir)
    table, table_counts = table_features(dates, cache_dir)
    image, image_counts = image_features(dates, cache_dir)
    dataset = {"dates": np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"), "price": price,
               "news": news, "table": table, "image": image, "labels": labels,
               "news_counts": news_counts, "table_counts": table_counts,
               "image_counts": image_counts}
    np.savez_compressed(prepared, **dataset)
    manifest.write_text(json.dumps(settings, indent=2))
    return dataset


class MultiModalTransformer(nn.Module):
    def __init__(self, dims: dict[str, int], hidden=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.project = nn.ModuleDict({key: nn.Linear(value, hidden) for key, value in dims.items()})
        self.modality_scale = nn.Parameter(torch.ones(len(dims)))
        self.position = nn.Embedding(512, hidden)
        encoder = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = sum(scale * self.project[key](batch[key]) for scale, key in zip(self.modality_scale, self.project))
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.position(positions)[None, :, :]
        encoded = self.encoder(tokens)
        return torch.sigmoid(self.head(encoded[:, -1, :]))


def make_windows(dataset: dict, window=40) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    dates = pd.to_datetime(dataset["dates"])
    if not 2 <= window <= 512 or len(dates) < window + 2:
        raise ValueError("Window must be 2–512 sessions with at least two subsequent labels")
    keys = ("price", "news", "table", "image")
    # Issued after close i, label is close(i+1) > close(i).
    indices = np.arange(window - 1, len(dates) - 1)
    windows = {key: np.stack([dataset[key][i - window + 1:i + 1] for i in indices]) for key in keys}
    labels = dataset["labels"][indices]
    return windows, labels, indices


def standardize(windows: dict[str, np.ndarray], train_end: int) -> tuple[dict, dict]:
    """Fit normalization only on training windows; image RGB needs no scaling."""
    output, stats = {}, {}
    for key, value in windows.items():
        if key == "image":
            output[key] = value.astype(np.float32)
            continue
        flat = value[:train_end].reshape(-1, value.shape[-1])
        mean = flat.mean(0)
        std = np.maximum(flat.std(0), 1e-4)
        output[key] = np.clip((value - mean) / std, -8, 8).astype(np.float32)
        stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return output, stats


def _batch(windows, indices, device):
    return {key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
            for key, value in windows.items()}


def train_and_replay(dataset: dict, output_dir: Path, window=40, train_fraction=0.8,
                     epochs=3, batch_size=64, learning_rate=3e-4, hidden=64,
                     heads=4, layers=2, online_updates=1, device="cpu") -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows, labels, issued = make_windows(dataset, window)
    if len(labels) < 120:
        raise ValueError("Need at least 120 labeled next-day windows")
    train_end = int(len(labels) * train_fraction)
    if train_end < 60 or len(labels) - train_end < 30:
        raise ValueError("Need at least 60 training and 30 test windows")
    windows, stats = standardize(windows, train_end)
    dims = {key: val.shape[-1] for key, val in windows.items()}
    torch.manual_seed(17)
    model = MultiModalTransformer(dims, hidden, heads, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    loss_fn = nn.BCELoss()
    # Chronological mini-batches keep the optimization order aligned with time.
    for epoch in range(epochs):
        model.train()
        losses = []
        for left in range(0, train_end, batch_size):
            ids = np.arange(left, min(left + batch_size, train_end))
            pred = model(_batch(windows, ids, device))
            loss = loss_fn(pred, target[ids])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch {epoch+1}/{epochs}: train BCE {np.mean(losses):.4f}")
    model.eval()
    train_pred = []
    with torch.no_grad():
        for left in range(0, train_end, batch_size):
            ids = np.arange(left, min(left + batch_size, train_end))
            train_pred.append(model(_batch(windows, ids, device)).cpu().numpy())
    train_pred = np.vstack(train_pred)
    test_pred = []
    # First test prediction is after train cutoff; each later step receives the
    # preceding sample's realized next-day label before it predicts.
    for i in range(train_end, len(labels)):
        if i > train_end and online_updates:
            model.train()
            old = np.array([i - 1])
            for _ in range(online_updates):
                pred = model(_batch(windows, old, device))
                loss = loss_fn(pred, target[old])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            test_pred.append(model(_batch(windows, np.array([i]), device)).cpu().numpy()[0])
    probability = np.vstack([train_pred, np.asarray(test_pred)])
    issue_dates = pd.to_datetime(dataset["dates"])[issued]
    label_dates = pd.to_datetime(dataset["dates"])[issued + 1]
    rows = []
    for i in range(len(labels)):
        for j, ticker in enumerate(TARGETS):
            rows.append({"issue_date": issue_dates[i].date().isoformat(),
                         "label_date": label_dates[i].date().isoformat(),
                         "split": "train_in_sample" if i < train_end else "test_prequential",
                         "ticker": ticker, "prob_up": float(probability[i, j]),
                         "actual_up": int(labels[i, j]),
                         "correct": int((probability[i, j] >= 0.5) == labels[i, j])})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    summary = {}
    for (split, ticker), group in frame.groupby(["split", "ticker"]):
        p, y = group.prob_up.to_numpy(), group.actual_up.to_numpy()
        accuracy = float(((p >= .5) == y).mean())
        up_rate = float(y.mean())
        summary[f"{split}_{ticker}"] = {"accuracy": accuracy,
                                        "brier": float(np.mean((p - y) ** 2)),
                                        "up_rate": up_rate,
                                        "majority_baseline": max(up_rate, 1 - up_rate),
                                        "n": len(group)}
    info = {"summary": summary, "config": {"window": window, "train_fraction": train_fraction,
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "hidden": hidden, "heads": heads, "layers": layers,
            "online_updates": online_updates, "device": device},
            "train_last_label_date": label_dates[train_end - 1].date().isoformat(),
            "test_first_label_date": label_dates[train_end].date().isoformat(),
            "normalization": stats, "feature_dims": dims,
            "modalities": {"news": dataset["news_counts"].tolist(),
                           "table": dataset["table_counts"].tolist(),
                           "image": dataset["image_counts"].tolist()}}
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    torch.save({"state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "feature_dims": dims, "normalization": stats,
                "config": info["config"], "last_label_date": label_dates[-1].date().isoformat()},
               output_dir / "online_checkpoint.pt")
    return frame, info


def advance_online(checkpoint_path: Path, current_window: dict[str, np.ndarray],
                   prior_window: dict[str, np.ndarray] | None = None,
                   prior_label: np.ndarray | None = None) -> np.ndarray:
    """Apply one newly revealed label, predict current window, persist new weights.

    Windows have shape (time, feature). The caller must supply the same four
    modalities and the previous issued window if its next-day label just arrived.
    """
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = MultiModalTransformer(saved["feature_dims"], config["hidden"],
                                  config["heads"], config["layers"])
    model.load_state_dict(saved["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
    optimizer.load_state_dict(saved["optimizer_state"])

    def normalize(sample):
        result = {}
        for key, dim in saved["feature_dims"].items():
            value = np.asarray(sample[key], np.float32)
            if value.ndim != 2 or value.shape != (config["window"], dim):
                raise ValueError(f"{key} must have shape {(config['window'], dim)}")
            if key in saved["normalization"]:
                stat = saved["normalization"][key]
                value = np.clip((value - np.asarray(stat["mean"])) / np.asarray(stat["std"]), -8, 8)
            result[key] = torch.as_tensor(value[None], dtype=torch.float32)
        return result

    if (prior_window is None) != (prior_label is None):
        raise ValueError("Supply both prior_window and its realized prior_label")
    if prior_window is not None:
        label = torch.as_tensor(np.asarray(prior_label, np.float32).reshape(1, 3))
        model.train()
        for _ in range(config["online_updates"]):
            loss = nn.functional.binary_cross_entropy(model(normalize(prior_window)), label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        result = model(normalize(current_window)).numpy()[0]
    saved["state_dict"] = model.state_dict()
    saved["optimizer_state"] = optimizer.state_dict()
    torch.save(saved, checkpoint_path)
    return result


def plot_accuracy(frame: pd.DataFrame, output: Path, rolling=60):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ticker in zip(axes, TARGETS):
        one = frame[frame.ticker == ticker]
        for split, color, title in [("train_in_sample", "#4477aa", "train (in sample)"),
                                    ("test_prequential", "#ee7733", "test (online)")]:
            part = one[one.split == split]
            ax.plot(pd.to_datetime(part.label_date), part.correct.rolling(rolling, min_periods=10).mean(),
                    label=title, color=color)
        ax.axhline(.5, color="gray", ls="--", lw=1)
        ax.set(title=ticker, ylim=(0, 1), xlabel="Label date")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel(f"Rolling {rolling}-day accuracy")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
