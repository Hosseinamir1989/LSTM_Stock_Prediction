# LSTM Stock Price Prediction Pipeline

A research-grade, end-to-end pipeline for forecasting **next-day closing stock prices** using LSTM (Long Short-Term Memory) neural networks, developed as part of a Bachelor's thesis in Finance / Data Science.

The pipeline covers the full machine-learning workflow: raw data ingestion → technical indicator enrichment → feature combination search → walk-forward validation → final model retraining → PnL backtesting → SHAP-based explainability → thesis-ready visualizations.

---

## Research Question

> Do technical indicators add predictive value to an LSTM model for next-day stock price forecasting compared to a naive persistence baseline?

The pipeline systematically evaluates **all non-empty subsets** of seven technical indicator groups (RSI, MACD, SMA, EMA, Bollinger Bands, Momentum, Rolling Volatility) across multiple large-cap US stocks over the period 2020–2025.

---

## Features

- **Brute-force feature combination search** — evaluates every subset of indicator groups; no manual feature selection
- **Walk-forward expanding-window validation** — time-series-safe model selection with no look-ahead bias
- **Three model selection criteria** — best validation RMSE, best directional accuracy (DA), best PnL ratio
- **Final retraining stage** — selected models are retrained on TRAIN+VAL and evaluated exactly once on the held-out TEST set
- **PnL backtesting** — simulated buy/hold/sell trading with configurable initial cash and signal threshold
- **SHAP explainability** — feature-level importance for the final LSTM models using gradient-based SHAP
- **Naive Persistence & Buy-and-Hold baselines** — hard-to-beat reference models included automatically
- **Multiple target modes** — next-day price (`price`), log/simple return (`return`), or binary direction (`direction`)
- **Stationarity reporting** — ADF + KPSS tests on every feature column; optional stationary-only filtering
- **Structured artifact saving** — every run saves models, scalers, histories, PnL logs, plots, and JSON summaries to a versioned run folder

---

## Project Structure

```
LSTM_BA/
│
├── config/
│   └── config.yaml                    # Central configuration file
│
├── stock_data_histogram/              # Raw daily OHLCV CSV files per ticker (2020–2025)
│   ├── AAPL_stock_data_2020-01-01_2025-01-01.csv
│   ├── AMZN_stock_data_2020-01-01_2025-01-01.csv
│   ├── GOOGL_stock_data_2020-01-01_2025-01-01.csv
│   └── ...                            # 35 tickers total (US + German stocks)
│
├── runs/                              # All experiment outputs, one folder per ticker
│   └── {TICKER}_price_experiment/
│       ├── config.yaml                # Config snapshot for reproducibility
│       ├── tables/                    # Result CSVs: all combos, selected models, final metrics, baseline
│       ├── plots/                     # Prediction plots, signal charts, heatmaps
│       ├── models/                    # Saved .keras model files (search + final stage)
│       ├── scalers/                   # Fitted MinMaxScaler objects (joblib .pkl)
│       ├── histories/                 # Training loss curves per model (CSV)
│       ├── logs/                      # Per-model JSON summaries and global run_logs.json
│       ├── pnl_logs/                  # Trade-by-trade PnL logs (CSV, TRAIN/VAL/TEST)
│       ├── analysis/                  # Indicator contribution and generalization gap charts
│       ├── shap/                      # SHAP feature importance outputs
│       ├── presentation/              # Thesis-ready charts and summary tables
│       └── final/                     # Final retrained model predictions and PnL
│
├── output/                            # Stationarity reports and standalone combo result CSVs
│
├── data_loader.py                     # Loads stock CSVs, parses dates, selects OHLCV columns
├── indicators.py                      # Computes all technical indicators (MACD, RSI, SMA, EMA, BB, MOM, VOL)
├── preprocessing.py                   # Normalization, sliding window generation, stationarity transforms
├── model.py                           # LSTM model architecture (build_lstm_model)
├── evaluation.py                      # Metrics (RMSE, MAE, MAPE, R², DA), PnL simulation, plot helpers
├── evaluate_lstm_combos.py            # Standalone brute-force indicator combo evaluator
├── feature_analysis.py                # Feature drop importance analysis
├── run_saver.py                       # RunSaver class — saves all artifacts per experiment run
├── standalone_delta_analyse.py        # Scatter plot: actual vs predicted price deltas
├── evaluation_tables.py               # Table generation and formatting utilities
├── main.py                            # CLI entry point for single-ticker training and evaluation
│
└── LSTM_BA________________________.ipynb   # Main interactive notebook (full pipeline)
```

---

## LSTM Architecture

Defined in `model.py`:

```
Input shape: (window_size=10, n_features)
    └─► LSTM(100 units, activation='tanh', recurrent_activation='sigmoid')
    └─► Dropout(0.2)
    └─► Dense(1, activation='linear')

Optimizer : Adam (learning_rate = 0.0001)
Loss      : Mean Squared Error (MSE)
```

**Design rationale:**
- A single LSTM layer is used to keep the model parsimonious and avoid overfitting on the relatively short daily time series (~1,250 trading days).
- 100 hidden units are sufficient to capture the temporal patterns across a 10-day lookback window.
- Dropout (rate = 0.2) is applied after the LSTM layer for regularization.
- A low learning rate (0.0001) ensures stable convergence.

---

## Technical Indicators

All indicators are computed in `indicators.py` using the settings in `config.yaml`.

| Group | Computed Features | Default Window |
|---|---|---|
| **MACD** | `MACD`, `MACD_signal`, `MACD_hist` | fast=12, slow=26, signal=9 |
| **RSI** | `RSI` | 14 days |
| **SMA** | `SMA` | 20 days |
| **EMA** | `EMA` | 20 days |
| **BB** | `BB_upper`, `BB_middle`, `BB_lower` | 20 days, 2σ |
| **MOM** | `Momentum` (close − 10-day rolling mean) | 10 days |
| **VOL** | `RollingVolatility` (log-return std) | 20 days |

Base features always included: `Close`, `Open` (and optionally `High`, `Low`, `Volume` if stationary filter permits).

With 7 groups there are **127 non-empty subsets** evaluated per ticker, plus one base-features-only run.

---

## Pipeline Stages (Notebook)

The main notebook (`LSTM_BA________________________.ipynb`) runs the following stages in order:

| Cell | Stage | Description |
|---|---|---|
| 1 | Setup | Imports, config loading, global constants, reproducibility seed |
| 5 | Stationarity | ADF + KPSS tests; `ACTIVE_FEATURE_SET` filter |
| 6 | Feature groups | `INDICATOR_GROUPS` dict, `BASE_FEATURES`, `groups_to_feature_cols()` |
| 7 | Walk-forward engine | `eval_split()`, `run_one_fold()`, `run_one_combo_walk_forward()` |
| 10–11 | Search | Brute-force loop over all indicator subsets |
| 12 | Baseline | Naive Persistence + Buy-and-Hold evaluation |
| 14 | Presentation | Thesis-ready charts, co-occurrence heatmap, top-combo tables |
| 15 | Final retraining | `retrain_selected_model()` — train on TRAIN+VAL, evaluate on TEST |
| 16 | Artifact saving | `save_official_artifacts()` — saves everything to `runs/` |
| 17 | SHAP | Gradient SHAP for the final LSTM models |
| 26 | Analysis | Full result breakdown: metrics table, signal comparison, confusion matrix, PnL chart |

---

## Data Split Strategy

The dataset is split **chronologically** — no random shuffling.

```
|─────── TRAIN (70%) ──────|── VAL (10%) ──|── TEST (20%) ──|
  used in walk-forward folds  model selection   never touched
                                                 until final eval
```

**Walk-forward validation** (expanding window):
- Minimum training size: 50% of data
- Validation window: 10% of data per fold
- Step size: 5% per fold
- Models are selected by averaged validation metrics across all folds

**Final retraining:**
- Selected models are retrained on TRAIN+VAL combined (80% of data)
- Evaluated exactly once on the TEST set (20%)

---

## Configuration (`config/config.yaml`)

```yaml
tickers:
  - GOOGL          # Active ticker — comment/uncomment to switch

target:
  type: price      # Options: price | return | direction

split:
  train_end: 0.70  # End of training split
  val_end: 0.80    # End of validation split (test = remaining 20%)

walk_forward:
  enable: true
  expanding: true
  min_train_size: 0.50
  val_size: 0.10
  step_size: 0.05

lstm:
  window_size: 10   # Lookback window (days)
  epochs: 60
  batch_size: 8
  learning_rate: 0.0001

trading:
  threshold_pct: 0.002   # 0.2% minimum predicted move to trigger Buy/Sell
  initial_cash: 1000

technical_indicators:
  rsi_window: 14
  sma_window: 20
  ema_window: 20
  bb_window: 20
  bb_std: 2
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  momentum_window: 10
  vol_window: 20

features:
  stationary: false   # If true, filters out non-stationary features before training
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average absolute deviation |
| **MAPE** | Mean Absolute Percentage Error — scale-independent accuracy |
| **R²** | Coefficient of determination — proportion of variance explained |
| **DA** | Directional Accuracy — fraction of days the predicted direction matches actual |
| **SA** | Signal Accuracy — fraction of trading signals (Buy/Sell) that were correct |
| **PnL Ratio** | Simulated portfolio return relative to initial cash |

---

## Evaluated Tickers

Data covers **2020-01-01 to 2025-01-01** (daily, adjusted close prices).

Full experiments were run for:

| Ticker | Company |
|---|---|
| AAPL | Apple Inc. |
| AMZN | Amazon.com Inc. |
| GOOGL | Alphabet Inc. |
| NVDA | NVIDIA Corporation |
| BRK-B | Berkshire Hathaway |

The `stock_data_histogram/` folder also contains data for 30 additional tickers including MSFT, META, TSLA, JPM, BAC, and several German stocks (SAP, SIE, BAS, DTE, VOW3).

---

## Key Results Summary

### Naive Persistence Baseline (Test Set)

| Metric | AAPL | AMZN | GOOGL |
|---|---|---|---|
| RMSE | 2.89 | 3.28 | 2.92 |
| MAE | 2.12 | 2.44 | 2.10 |
| MAPE | 1.03% | 1.31% | 1.28% |
| R² | 0.987 | 0.962 | 0.963 |
| Directional Accuracy | **0%** | **0%** | **0%** |

*The naive baseline always predicts zero change, so it can never correctly predict direction.*

### Best LSTM Model — BEST_VAL_RMSE (Test Set)

| Metric | AAPL | AMZN | GOOGL |
|---|---|---|---|
| Feature combo | MACD+SMA+EMA+BB+MOM | MACD+EMA+BB+VOL | MACD+SMA+EMA+BB+MOM |
| RMSE | 5.85 | 4.96 | 5.86 |
| MAE | 4.78 | 3.68 | 4.86 |
| MAPE | 2.26% | 1.99% | 2.93% |
| R² | 0.948 | 0.913 | 0.851 |
| Directional Accuracy | 40.1% | 51.4% | 41.7% |

### Interpretation

The LSTM models do **not** outperform naive persistence on price-level accuracy metrics (RMSE, MAE, MAPE, R²). This is expected: stock prices are strongly autocorrelated, so predicting "tomorrow = today" is already a very strong baseline for absolute price error.

However, the LSTM achieves **40–51% directional accuracy** while the naive baseline scores 0% DA. This demonstrates that the model captures genuine directional signal, which is more relevant for trading applications than absolute price error. MACD appears in 6 out of 9 best-performing feature combinations across all three tickers, making it the most consistently useful indicator group.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd LSTM_BA
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Full reproducibility:** `requirements_freeze.txt` contains the exact version of every package (including transitive dependencies) as captured from the development environment. Use it for a bit-for-bit reproduction:
> ```bash
> pip install -r requirements_freeze.txt
> ```

### Python version

This project was developed and tested with **Python 3.11**.

```
C:\Users\Hossein\AppData\Local\Microsoft\WindowsApps\python3.11.exe
```

### Main dependencies

| Package | Version (tested) | Purpose |
|---|---|---|
| `tensorflow` | 2.19.0 | LSTM model training and inference |
| `keras` | 3.9.2 | High-level neural network API |
| `scikit-learn` | 1.6.1 | MinMaxScaler, evaluation metrics |
| `numpy` | 2.1.0 | Numerical computation |
| `pandas` | 2.2.3 | Data loading and manipulation |
| `scipy` | 1.15.2 | Statistical utilities |
| `statsmodels` | 0.14.4 | ADF and KPSS stationarity tests |
| `shap` | 0.51.0 | Feature importance (gradient SHAP) |
| `matplotlib` | 3.10.1 | All plots and charts |
| `seaborn` | 0.13.2 | Heatmaps and statistical plots |
| `PyYAML` | 6.0.2 | Config file parsing |
| `joblib` | 1.4.2 | Scaler serialization |
| `ipykernel` | 6.29.5 | Jupyter notebook kernel |

---

## Usage

### Option 1 — Interactive Notebook (recommended)

Open `LSTM_BA________________________.ipynb` in Jupyter Lab or VS Code.

1. Set the active ticker in `config/config.yaml`
2. Set `target.type` to `price`, `return`, or `direction`
3. Run all cells in order

### Option 2 — CLI script

```bash
python main.py
```

Runs a single train/val/test cycle for the ticker specified in `config.yaml`. Outputs metrics and plots.

### Option 3 — Standalone brute-force combo search

```bash
python evaluate_lstm_combos.py
```

Evaluates all indicator subsets independently without the full notebook pipeline.

---

## Output Artifacts

After a full notebook run, the following are saved under `runs/{TICKER}_price_experiment/`:

| File / Folder | Contents |
|---|---|
| `tables/results_price.csv` | All 128 feature combinations with walk-forward validation metrics |
| `tables/selected_models_price.csv` | The three selected models (RMSE / DA / PnL) and their validation metrics |
| `tables/final_retrained_models_price.csv` | Final test-set metrics for the 3 retrained models |
| `tables/baseline_results_price.csv` | Naive Persistence and Buy-and-Hold baseline metrics |
| `tables/indicator_contribution_price.csv` | Per-indicator average metric contribution |
| `models/*.keras` | Saved Keras models — one per selected model, search and final stage |
| `scalers/*_scaler_X.pkl` | Fitted feature scaler (MinMaxScaler) |
| `scalers/*_scaler_y.pkl` | Fitted target scaler (MinMaxScaler) |
| `histories/*_history.csv` | Epoch-by-epoch training loss for each model |
| `logs/*_summary.json` | Full metric snapshot per model (all metrics, all splits) |
| `logs/run_logs.json` | Global run summary (config, selected models, baseline) |
| `pnl_logs/*_pnl_log.csv` | Trade-by-trade buy/sell log with PnL per trade |
| `plots/*_predictions.png` | Actual vs predicted price plots |
| `analysis/` | Indicator contribution bar charts, generalization gap analysis |
| `shap/` | SHAP feature importance values and plots |
| `presentation/` | Thesis-ready versions of all key charts and tables |






