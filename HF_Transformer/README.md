# HF Transformer

## FinMultiTime next-day model

Open [finmultitime_tuning_guide.ipynb](finmultitime_tuning_guide.ipynb) in `Quant_ENV` for the four-modality FinMultiTime SPY/QQQ/IWM next-day direction model. It explains the source data, as-of rules, chronological training and test, online updates, and fine-tuning. It compares three configurations on 2023 validation data, uses the frozen model as a benchmark, chooses the best online configuration, and lets you enter your own thresholds before evaluating on 2024–2025 data. Changing a threshold only requires rerunning the threshold, report, and plot cells. Outputs are saved under `finmultitime_tuning_outputs/`.

Implementation is in [finmultitime_daily.py](finmultitime_daily.py). The first run downloads the public 483 MB price archive to the OS temporary folder; it reads only selected members of the large news and table archives by HTTP range. Local Alpaca-adjusted ETF daily CSVs in `../SP500/attention_data` provide all three labels. The public FinMultiTime price snapshot ends 2025-03-28, so this notebook is a historical backtest.

## Existing next-minute model

Open [hf_transformer_online.ipynb](hf_transformer_online.ipynb) in `Quant_ENV` for the model, causal timing, online replay, plots, tuning settings, and checkpoint usage.

The notebook reads the existing adjusted SPY/QQQ/IWM daily and Alpaca SIP minute snapshots in `../SP500/attention_data` and `../SP500/minute_data`. To populate missing data, run `python SP500/download_etf_attention_data.py --output-dir SP500/attention_data` and `python SP500/download_etf_minute_data.py --history` from the project root with Alpaca credentials in `.env`.

Run `python -m unittest test_finmultitime_daily test_online_transformer` from this folder for timing and checkpoint checks. The next-day notebook saves probabilities, metrics, an accuracy plot, and a checkpoint under `finmultitime_outputs/`; the earlier minute notebook uses `outputs/`.
