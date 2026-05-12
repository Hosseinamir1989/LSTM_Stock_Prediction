import os
import random
import yaml
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Project modules ---
from data_loader import load_stock_data
from indicators import add_indicators
from preprocessing import (
    impute_numeric_median,
    normalize_features,
    df_to_windowed_df,
    windowed_df_to_date_X_y,
    ensure_datetime_index,
    stationarity_report,
    add_stationary_transforms,
)
from model import build_lstm_model
from evaluation import (
    plot_predictions, compute_metrics, compute_directional_accuracy, simulate_pnl
)
from feature_analysis import feature_drop_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------
# Config & reproducibility
# ------------------------------------------------------------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---- split percents in config ----
sp = config.get("split", {})
TRAIN_END = float(sp.get("train_end", 0.90))
VAL_END   = float(sp.get("val_end",   0.96))

# ---- target type: "price" (next-day Close) OR "log_return" (next-day log return)
tgt_cfg = config.get("target", {})
TARGET_TYPE = str(tgt_cfg.get("type", "price")).lower()  # "price" | "log_return"

# ---- stationarity transforms options
st_cfg = config.get("stationarity", {})
ST_ENABLE = bool(st_cfg.get("enable", True))
ZWIN      = int(st_cfg.get("zscore_window", 30))
ADD_PCT   = bool(st_cfg.get("add_pct", True))
ADD_LOG   = bool(st_cfg.get("add_log", True))

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
ticker = config["tickers"][0]
dfs = load_stock_data(config["paths"]["csv_folder"], [ticker])
df = ensure_datetime_index(dfs[0])  # date index enforced

# ------------------------------------------------------------------
# Indicators & stationary transforms
# ------------------------------------------------------------------
df = impute_numeric_median(df)
df = add_indicators(df)  # MACD, RSI, BB, etc. (no dropna)  :contentReference[oaicite:3]{index=3}
df = impute_numeric_median(df)

# add stationary transforms for price-like columns
if ST_ENABLE:
    price_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = add_stationary_transforms(df, price_cols=price_cols, zwin=ZWIN, add_pct=ADD_PCT, add_log=ADD_LOG)
    df = impute_numeric_median(df)

# stationarity report (optional / safe if statsmodels missing)
try:
    cols_to_check = [c for c in ["Open","High","Low","Close","Volume","RSI","MACD","BB_upper","BB_lower","RollingVolatility","SMA","EMA","Log_Return"] if c in df.columns]
    rep = stationarity_report(df, cols_to_check, save_path=f"output/{ticker}_stationarity_report.csv")
    logging.info("Stationarity report saved to output/")
except Exception as e:
    logging.warning(f"Stationarity report skipped: {e}")

# ------------------------------------------------------------------
# Target
# ------------------------------------------------------------------
if TARGET_TYPE == "log_return":
    # predict next-day log-return r_{t+1} = log(C_{t+1}) - log(C_t)
    df["Target"] = np.log(df["Close"]).diff().shift(-1)
else:
    # predict next-day Close (your original)
    df["Target"] = df["Close"].shift(-1)

df = df.dropna(subset=["Target"]).copy()

# ------------------------------------------------------------------
# Normalize & window
# ------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ["Close", "Target"]]
df_scaled, _, scaler_y = normalize_features(df.drop(columns=["Close"]), target_col="Target")  # :contentReference[oaicite:4]{index=4}

window_size = config["lstm"]["window_size"]
windowed_df = df_to_windowed_df(df_scaled, window_size, target_col="Target")  # :contentReference[oaicite:5]{index=5}

# robust safety pass
windowed_df = windowed_df.replace([np.inf, -np.inf], np.nan)
num_cols_win = windowed_df.select_dtypes(include=[np.number]).columns
if len(num_cols_win) > 0:
    windowed_df[num_cols_win] = windowed_df[num_cols_win].fillna(windowed_df[num_cols_win].median(skipna=True))

dates, X, y = windowed_df_to_date_X_y(windowed_df, window_size)  # :contentReference[oaicite:6]{index=6}
dates = pd.to_datetime(dates)  # ensure real dates for plots

# ------------------------------------------------------------------
# Train/Val/Test split by percent (TRAIN_END, VAL_END)
# ------------------------------------------------------------------
n = len(dates)
i_tr_end  = max(1, min(int(n * TRAIN_END), n - 2))
i_val_end = max(i_tr_end + 1, min(int(n * VAL_END),  n - 1))

X_train, X_val, X_test = X[:i_tr_end], X[i_tr_end:i_val_end], X[i_val_end:]
y_train, y_val, y_test = y[:i_tr_end], y[i_tr_end:i_val_end], y[i_val_end:]
dates_train, dates_val, dates_test = dates[:i_tr_end], dates[i_tr_end:i_val_end], dates[i_val_end:]

# indices for drop-feature helper
q_train_idx, q_val_idx = i_tr_end, i_val_end

# ------------------------------------------------------------------
# Build & train LSTM
# ------------------------------------------------------------------
params = config["lstm"]
model = build_lstm_model(window_size, X.shape[2], params["learning_rate"])  # :contentReference[oaicite:7]{index=7}

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    verbose=1
)

# ------------------------------------------------------------------
# Predictions (inverse-scaled)
# ------------------------------------------------------------------
def inv_pred(m, Xsplit):
    return scaler_y.inverse_transform(m.predict(Xsplit)).flatten()

y_train_hat = inv_pred(model, X_train)
y_val_hat   = inv_pred(model, X_val)
y_test_hat  = inv_pred(model, X_test)

y_train_true = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
y_val_true   = scaler_y.inverse_transform(y_val.reshape(-1,1)).flatten()
y_test_true  = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

# ------------------------------------------------------------------
# If target = log_return, reconstruct next-day prices for plotting/metrics
# ------------------------------------------------------------------
if TARGET_TYPE == "log_return":
    close_series = df["Close"]
    def reconstruct_prices(date_idx, yhat_ret):
        base = close_series.reindex(pd.to_datetime(date_idx)).values  # price at t
        pred_next = base * np.exp(yhat_ret)  # predicted C_{t+1}
        true_next = close_series.shift(-1).reindex(pd.to_datetime(date_idx)).values  # actual C_{t+1}
        return true_next, pred_next

    y_train_orig, y_train_pred = reconstruct_prices(dates_train, y_train_hat)
    y_val_orig,   y_val_pred   = reconstruct_prices(dates_val,   y_val_hat)
    y_test_orig,  y_test_pred  = reconstruct_prices(dates_test,  y_test_hat)
else:
    # already prices
    y_train_orig, y_train_pred = y_train_true, y_train_hat
    y_val_orig,   y_val_pred   = y_val_true,   y_val_hat
    y_test_orig,  y_test_pred  = y_test_true,  y_test_hat

# ------------------------------------------------------------------
# Save deltas
# ------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
pd.DataFrame({
    "Date": dates_test, "Actual": y_test_orig, "Predicted": y_test_pred,
    "Delta": y_test_pred - y_test_orig
}).to_csv(f"output/test_deltas_{ticker}.csv", index=False)

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
plot_predictions(
    dates_train, y_train_orig, y_train_pred,
    dates_val,   y_val_orig,   y_val_pred,
    dates_test,  y_test_orig,  y_test_pred
)

def plot_test_zoomed(dates_z, y_true_z, y_pred_z, zoom=120):
    z = min(zoom, len(dates_z))
    plt.figure(figsize=(14,4))
    plt.plot(dates_z[-z:], y_true_z[-z:], label='Test Actual')
    plt.plot(dates_z[-z:], y_pred_z[-z:], label='Test Predicted')
    plt.title(f"Test Zoom (last {z} points)")
    plt.xlabel("Date"); plt.ylabel("Close Price")
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
    plt.show()

plot_test_zoomed(dates_test, y_test_orig, y_test_pred, zoom=120)

# ------------------------------------------------------------------
# Metrics & PnL
# ------------------------------------------------------------------
rmse, mae = compute_metrics(y_test_orig, y_test_pred)
dir_acc   = compute_directional_accuracy(y_test_orig, y_test_pred)
with open(f"output/directional_accuracy_{ticker}.txt", "w") as f:
    f.write(f"Directional Accuracy: {dir_acc:.4f}\n")
print(f"\n✅ Test Evaluation for {ticker}: RMSE={rmse:.4f}, MAE={mae:.4f}, DA={dir_acc:.4f}")

pnl_result, trade_log_df = simulate_pnl(
    y_test_orig, y_test_pred, initial_cash=80000, ticker=ticker, output_folder="output"
)
logging.info("PnL: %s", pnl_result)

# ------------------------------------------------------------------
# Feature drop analysis
# ------------------------------------------------------------------
try:
    results_df = feature_drop_analysis(
        df_scaled, feature_cols, window_size,
        X, y, scaler_y,
        dates_test, y_test_orig,
        rmse, mae,
        q_train_idx, q_val_idx,
        params["learning_rate"], params["epochs"], params["batch_size"]
    )
    print("\n📊 Feature Drop Analysis (top 10):")
    print(results_df.head(10))
except Exception as e:
    logging.warning(f"feature_drop_analysis skipped: {e}")
