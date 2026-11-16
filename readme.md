# My Trading System

This project tries to predict tomorrow's SPY return.
It uses the **signature transform**, a math tool that turns a price path into numbers a model can learn from.

## Folders

| Folder | What is inside |
|---|---|
| [Signature_Price_Forecast](Signature_Price_Forecast/) | The main project: code, notebooks, tests, and results |
| [Papers](Papers/) | Research papers the project is based on |
| [data](data/) | Data files and notes on how to get the price data |

## Result in one line

On 310 test days (Jan 2024 – Mar 2025), the model guessed the up/down direction right **58.7%** of the time.
Always guessing "up" scores 57.7%.
See [Signature_Price_Forecast](Signature_Price_Forecast/) for details.

## Quick start

1. Install Python 3.9 or newer, then run `pip install -r Signature_Price_Forecast/requirements.txt`.
2. Put SPY, QQQ, and IWM daily price files in `data/etf_daily/` (see [data/README.md](data/README.md)).
3. Open `Signature_Price_Forecast/signature_price_forecast.ipynb` and run it.
