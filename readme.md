# My Trading System

This project tries to predict tomorrow's SPY return.
It uses the **signature transform**, a math tool that turns a price path into numbers a model can learn from.

## Folders

| Folder | What is inside |
|---|---|
| [Signature_Price_Forecast](Signature_Price_Forecast/) | The main project: code, notebooks, tests, and results |
| [Papers](Papers/) | Research papers the project is based on |

## Result in one line

On 310 test days (Jan 2024 – Mar 2025), the model guessed the up/down direction right **58.7%** of the time.
Always guessing "up" scores 57.7%.
See [Signature_Price_Forecast](Signature_Price_Forecast/) for details.

## Quick start

1. Install Python 3.10 or newer, then run `pip install -r Signature_Price_Forecast/requirements.txt`.
2. Price data comes from the [quant-research-dataset](https://pypi.org/project/quant-research-dataset/) library. Set `QRD_DATA_DIR` in `.env` to your local qrd data folder (see [Signature_Price_Forecast](Signature_Price_Forecast/README.md#setup)).
3. Open `Signature_Price_Forecast/signature_price_forecast.ipynb` and run it.
