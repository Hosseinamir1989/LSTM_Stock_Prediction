# LSTM Stock Price Prediction Pipeline

An end-to-end pipeline for forecasting next-day closing stock prices using LSTM neural networks, developed as part of a Bachelor's thesis. The pipeline covers the full ML workflow: raw data ingestion → technical indicator enrichment → brute-force feature combination search → walk-forward validation → final model retraining → PnL backtesting → thesis-ready visualizations.

## Research Question

> Do technical indicators add predictive value to an LSTM model for next-day stock price forecasting compared to a naive persistence baseline?

All non-empty subsets of seven technical indicator groups (RSI, MACD, SMA, EMA, Bollinger Bands, Momentum, Rolling Volatility) are evaluated across multiple large-cap US stocks over the period 2020–2025.

## Features

- **Brute-force feature combination search** — evaluates every subset of indicator groups
- **Leakage-free preprocessing** — scaler fitted on training data only; applied to val/test
- **Chronological 70/10/20 split** — train/validation/test; no shuffling
- **Final retraining on 80%** — selected models retrained on TRAIN+VAL, evaluated once on TEST
- **Walk-forward validation** — optional expanding-window cross-validation
- **Three model selection criteria** — best validation RMSE, directional accuracy (DA), PnL ratio
- **PnL backtesting** — simulated buy/sell trading with configurable threshold and fees
- **Naive Persistence & Buy-and-Hold baselines** — included automatically
- **Multiple target modes** — next-day price (`price`), log/simple return (`return`), binary direction (`direction`)
- **Stationarity reporting** — ADF + KPSS tests on training data only; optional stationary-only feature filtering

## Getting Started

### 1. Install Requirements

```sh
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` to set the ticker, split ratios, LSTM hyperparameters, and trading threshold.

### 3. Run

Open and run `LSTM_experiment.ipynb` cell by cell. Each cell is self-contained and annotated.

## Project Structure

```
LSTM_BA/
├── config/
│   └── config.yaml                  # Central configuration (ticker, splits, LSTM params)
├── stock_data_histogram/            # Raw daily OHLCV CSV files per ticker (2020–2025)
├── lstm_models/                     # Saved model files (gitignored)
├── runs/                            # Experiment outputs per ticker (gitignored)
├── data_loader.py                   # Loads stock CSVs, parses dates
├── indicators.py                    # Technical indicators (RSI, MACD, SMA, EMA, BB, MOM, VOL)
├── preprocessing.py                 # Windowing, normalization, stationarity transforms
├── model.py                         # LSTM model architecture (build_lstm_model)
├── evaluation.py                    # Metrics (RMSE, MAE, MAPE, R², DA), PnL simulation
├── LSTM_experiment.ipynb            # Main notebook — full pipeline
├── requirements.txt                 # Python dependencies
└── README.md
```

- `requirements.txt`: Python package dependencies.
- `README.md`: This README file.






