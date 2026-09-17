"""Download completed SPY daily bars from Alpaca; no order endpoints are used."""
import argparse
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent


def append_preserving_history(existing, fresh):
    """Rebase refreshed adjusted prices to the saved basis; keep past rows exact.

    A new dividend/split can rescale all previously adjusted prices. Common scale
    changes are harmless; other revisions are rejected instead of rewriting the
    observations used by the online policy.
    """
    if not existing.index.isin(fresh.index).all():
        raise ValueError('Refresh is missing previously saved sessions.')
    anchor = existing.index[-1]
    factor = existing.loc[anchor, 'close'] / fresh.loc[anchor, 'close']
    candidate = fresh.copy()
    candidate[['open', 'close']] *= factor
    if not np.allclose(existing.to_numpy(), candidate.loc[existing.index, existing.columns].to_numpy(), rtol=1e-7, atol=1e-8):
        raise ValueError('Historical prices were revised beyond a common adjustment scale. Review the data and replay separately.')
    return pd.concat([existing, candidate.loc[candidate.index > anchor]]), float(factor)


def download(output=HERE / 'data' / 'SPY_daily.csv', start='2016-01-04', feed='sip', append=False):
    from dotenv import load_dotenv
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from online_rl import load_prices
    load_dotenv(HERE.parent / '.env', override=False)
    key, secret = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY')
    if not key or not secret:
        raise ValueError('Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the project .env.')
    ny = ZoneInfo('America/New_York')
    end = datetime.now(timezone.utc).astimezone(ny).date() - timedelta(days=1)
    client = StockHistoricalDataClient(key, secret)
    parts = []
    for adjustment, prefix in ((Adjustment.ALL, ''), (Adjustment.RAW, 'raw_')):
        request = StockBarsRequest(symbol_or_symbols='SPY', timeframe=TimeFrame.Day,
            start=datetime.combine(pd.Timestamp(start).date(), time.min, tzinfo=ny),
            end=datetime.combine(end + timedelta(days=1), time.min, tzinfo=ny),
            adjustment=adjustment, feed=DataFeed(feed))
        bars = client.get_stock_bars(request).data.get('SPY', [])
        rows = [{'date': b.timestamp.astimezone(ny).date(), prefix+'open': b.open,
                 prefix+'close': b.close} for b in bars
                if pd.Timestamp(start).date() <= b.timestamp.astimezone(ny).date() <= end]
        if not rows:
            raise ValueError('Alpaca returned no SPY bars.')
        parts.append(pd.DataFrame(rows).set_index('date'))
    frame = pd.concat(parts, axis=1).sort_index()
    if frame.isna().any().any():
        raise ValueError('Raw and adjusted session coverage differs.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_sha, factor = None, 1.
    if output.exists():
        if not append:
            raise ValueError('Data file exists. Use --append to preserve online history, or choose another --output.')
        load_prices(output)  # Validate the existing checksum before retaining it.
        previous_sha = hashlib.sha256(output.read_bytes()).hexdigest()
        existing = pd.read_csv(output, index_col='date', parse_dates=True)
        frame.index = pd.to_datetime(frame.index)
        frame, factor = append_preserving_history(existing, frame)
    frame.to_csv(output, float_format='%.12g')
    meta = {'symbol': 'SPY', 'source': 'Alpaca historical stock bars', 'feed': feed,
            'adjustment': 'all', 'raw_columns_adjustment': 'raw', 'timeframe': '1Day',
            'first_date': str(frame.index[0]), 'last_date': str(frame.index[-1]),
            'rows': len(frame), 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'append_mode': append, 'new_data_adjustment_basis_factor': factor, 'previous_sha256': previous_sha,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    load_prices(output)
    print(f"Saved {len(frame)} SPY sessions: {meta['first_date']} through {meta['last_date']}.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'data' / 'SPY_daily.csv')
    p.add_argument('--start', default='2016-01-04')
    p.add_argument('--feed', choices=['sip', 'iex'], default='sip')
    p.add_argument('--append', action='store_true', help='Preserve saved history and append completed new sessions.')
    args = p.parse_args()
    download(args.output, args.start, args.feed, args.append)
