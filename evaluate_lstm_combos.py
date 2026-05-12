# -*- coding: utf-8 -*-
"""
evaluate_lstm_combos.py — full factorial combos with VAL-vs-TEST selection, DA & PnL champions
-----------------------------------------------------------------------------------------------
What it does
1) Evaluates ALL non-empty combinations of available technical indicators for an LSTM.
2) Saves a master CSV with BOTH VALIDATION and TEST metrics: RMSE, MAE, DA, PnL Ratio.
3) Tracks three champions:
     - Best RMSE (TEST, lowest)
     - Best DA  (winner selected by SELECT_ON, reports TEST metrics)
     - Best PnL (winner selected by SELECT_ON, reports TEST metrics)
4) Heatmaps: indicator↔metrics, metric↔metric, effect sizes, pairwise synergy, top-N, all-combos z-scores.
5) Supports time-series screening (last-N / fraction / stride) to speed up scans.

Project deps:
- data_loader.load_stock_data
- indicators.add_indicators
- preprocessing.normalize_features, df_to_windowed_df, windowed_df_to_date_X_y
- model.build_lstm_model  (signature expected: build_lstm_model(window_size, n_features, learning_rate))
- evaluation.compute_metrics, evaluation.compute_directional_accuracy, simulate_pnl

Reads config at: config/config.yaml
"""

import os
import itertools
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_loader import load_stock_data
from indicators import add_indicators
from preprocessing import normalize_features, df_to_windowed_df, windowed_df_to_date_X_y
from model import build_lstm_model
from evaluation import compute_metrics, compute_directional_accuracy, simulate_pnl

# ---------------------------------------------------------------------------------
# Selection policy (match CNN script)
#   - "val": avoid test leakage — choose champions by VALIDATION metrics,
#            but report the TEST metrics for the chosen combos.
#   - "test": choose champions directly by TEST metrics (legacy behavior).
# ---------------------------------------------------------------------------------
SELECT_ON = "val"   # "val" (recommended) or "test"

# ---------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def ensure_output_dir(path="output_combos"):
    os.makedirs(path, exist_ok=True)
    return path


def all_indicator_combinations(indicator_list):
    return [c for r in range(1, len(indicator_list) + 1) for c in itertools.combinations(indicator_list, r)]


def zscore_robust(series: pd.Series) -> pd.Series:
    vals = series.values.astype(float)
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros_like(vals, dtype=float), index=series.index)
    return pd.Series((vals - mu) / sd, index=series.index)


def time_sample_df(df: pd.DataFrame,
                   method: str = "last_n",
                   fraction: float = 0.35,
                   last_n: int = 2000,
                   stride: int = 1) -> pd.DataFrame:
    """
    Make screening faster by reducing training size while preserving temporal order.

    method:
      - "last_n": keep last N rows
      - "fraction": keep last fraction of rows
      - "stride": keep every k-th row (downsample uniformly)
    """
    if df.empty:
        return df

    if method == "last_n":
        return df.iloc[-min(last_n, len(df)):]
    elif method == "fraction":
        n = max(32, int(len(df) * float(fraction)))
        return df.iloc[-n:]
    elif method == "stride":
        return df.iloc[::max(1, int(stride))]
    else:
        return df


def evaluate_combo(df, features, window_size, epochs, batch_size, learning_rate, ticker, outdir):
    """
    Train LSTM on given feature set and return BOTH validation and test metrics,
    plus test arrays for optional saving.

    Returns dict with keys:
      rmse_val, mae_val, da_val, pnl_val
      rmse_test, mae_test, da_test, pnl_test
      y_true_test, y_pred_test
      model (fitted Keras model, for optional saving on winners)
    """
    df_subset = df[["Target"] + features].copy()

    # Scale + window
    df_scaled, _, scaler_y = normalize_features(df_subset, target_col="Target")
    windowed_df = df_to_windowed_df(df_scaled, window_size, target_col="Target")
    dates, X, y = windowed_df_to_date_X_y(windowed_df, window_size)

    # Guard
    n = len(dates)
    if n < 20:
        raise ValueError(f"Too few samples after windowing: n={n}")

    # Chronological split 90/6/4 (train/val/test)
    i_tr = int(n * 0.90)
    i_va = int(n * 0.96)
    X_train, X_val, X_test = X[:i_tr], X[i_tr:i_va], X[i_va:]
    y_train, y_val, y_test = y[:i_tr], y[i_tr:i_va], y[i_va:]

    # Model
    model = build_lstm_model(window_size, X.shape[2], learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=min(5, max(2, epochs // 6)), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=min(3, max(1, epochs // 10)), min_lr=1e-6, verbose=0),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    # Predict (inverse-transform) — VALIDATION
    y_val_pred_scaled = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_scaled = y_val.reshape(-1, 1)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val_scaled).flatten()

    # Predict (inverse-transform) — TEST
    y_test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()

    # Clean invalids (VAL)
    m_val = np.isfinite(y_val_orig) & np.isfinite(y_val_pred)
    y_val_orig = y_val_orig[m_val]
    y_val_pred = y_val_pred[m_val]

    # Clean invalids (TEST)
    m_test = np.isfinite(y_test_orig) & np.isfinite(y_test_pred)
    y_test_orig = y_test_orig[m_test]
    y_test_pred = y_test_pred[m_test]

    if len(y_val_orig) == 0 or len(y_test_orig) == 0:
        raise ValueError("No valid samples after cleaning NaN/Inf (val/test).")

    # Metrics — VALIDATION
    rmse_val, mae_val = compute_metrics(y_val_orig, y_val_pred)
    da_val = compute_directional_accuracy(y_val_orig, y_val_pred)
    pnl_val, _ = simulate_pnl(y_val_orig, y_val_pred, initial_cash=80000, ticker=ticker, output_folder=outdir)
    pnl_val_ratio = float(pnl_val.get("P&L Ratio", 0.0))

    # Metrics — TEST
    rmse_test, mae_test = compute_metrics(y_test_orig, y_test_pred)
    da_test = compute_directional_accuracy(y_test_orig, y_test_pred)
    pnl_test, _ = simulate_pnl(y_test_orig, y_test_pred, initial_cash=80000, ticker=ticker, output_folder=outdir)
    pnl_test_ratio = float(pnl_test.get("P&L Ratio", 0.0))

    return {
        # VAL
        "rmse_val": float(rmse_val),
        "mae_val": float(mae_val),
        "da_val": float(da_val),
        "pnl_val": float(pnl_val_ratio),
        # TEST
        "rmse_test": float(rmse_test),
        "mae_test": float(mae_test),
        "da_test": float(da_test),
        "pnl_test": float(pnl_test_ratio),
        # Arrays & model
        "y_true_test": y_test_orig,
        "y_pred_test": y_test_pred,
        "model": model
    }


def build_design_matrix(combo_strings, indicators):
    X = np.zeros((len(combo_strings), len(indicators)), dtype=int)
    for i, s in enumerate(combo_strings):
        feats = [f.strip() for f in s.split(",")] if isinstance(s, str) else []
        for j, ind in enumerate(indicators):
            X[i, j] = 1 if ind in feats else 0
    return pd.DataFrame(X, columns=indicators)


def plot_heatmap(df, title, out_path):
    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    else:
        im = plt.imshow(df.values, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
        plt.yticks(range(df.shape[0]), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", out_path)


def plot_heatmap_values(df, title, out_path):
    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.heatmap(df, annot=True, cmap="coolwarm", square=True, fmt=".2f")
    else:
        data = np.where(np.isfinite(df.values), df.values, 0.0)
        im = plt.imshow(data, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
        plt.yticks(range(df.shape[0]), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i, j]
                if np.isfinite(v):
                    plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", out_path)


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    # Repro
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    tf.random.set_seed(42)

    # Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ticker = config["tickers"][0]
    csv_folder = config["paths"]["csv_folder"]
    window_size = int(config["lstm"]["window_size"])
    lr = float(config["lstm"]["learning_rate"])

    # Screening (speed up big scans)
    screening_cfg = config.get("screening", {})
    screening_enabled = bool(screening_cfg.get("enabled", True))
    screen_method = str(screening_cfg.get("method", "last_n"))        # "last_n" | "fraction" | "stride"
    screen_fraction = float(screening_cfg.get("fraction", 0.35))
    screen_last_n = int(screening_cfg.get("last_n", 2000))
    screen_stride = int(screening_cfg.get("stride", 1))
    screen_epochs = int(screening_cfg.get("epochs", max(8, int(config["lstm"]["epochs"] * 0.3))))
    screen_batch = int(screening_cfg.get("batch_size", config["lstm"]["batch_size"]))

    # Full (if screening disabled)
    full_epochs = int(config["lstm"]["epochs"])
    full_batch = int(config["lstm"]["batch_size"])

    # OUTPUT FOLDER
    outdir = ensure_output_dir("output_combos")

    # Data
    df = load_stock_data(csv_folder, [ticker])[0]
    df = add_indicators(df)

    # Fix index if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            logging.warning("Index restored to DateTimeIndex from 'Date' column.")
        else:
            logging.warning("DataFrame lacks a DateTimeIndex; date formatting in windowing may fail.")

    # Candidate indicators (only keep those present)
    candidate_indicators = ["RSI", "BB_upper", "BB_lower", "Momentum", "MACD", "SMA", "EMA", "RollingVolatility"]
    indicators = [c for c in candidate_indicators if c in df.columns]
    if not indicators:
        raise RuntimeError("No candidate indicators found in dataframe columns.")

    # Target T+1
    df["Target"] = df["Close"].shift(-1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Apply screening sample once globally so all combos are compared fairly
    df_for_eval = df.copy()
    if screening_enabled:
        df_for_eval = time_sample_df(
            df_for_eval, method=screen_method, fraction=screen_fraction,
            last_n=screen_last_n, stride=screen_stride
        )
        logging.info("Screening enabled: method=%s, len(df)=%d", screen_method, len(df_for_eval))
        epochs = screen_epochs
        batch_size = screen_batch
    else:
        epochs = full_epochs
        batch_size = full_batch

    # All non-empty combos
    combos = all_indicator_combinations(indicators)
    logging.info("Total combinations: %d", len(combos))

    # Evaluate
    rows = []

    # Champions
    best_rmse_test = np.inf
    best_rmse_combo = None
    best_rmse_model = None

    # Winners tracked for both selection bases
    best_da_val = -np.inf;  best_da_val_combo = None;  best_da_val_report = {};  best_da_val_model = None
    best_da_test = -np.inf; best_da_test_combo = None; best_da_test_report = {}; best_da_test_model = None

    best_pnl_val = -np.inf;  best_pnl_val_combo = None;  best_pnl_val_report = {};  best_pnl_val_model = None
    best_pnl_test = -np.inf; best_pnl_test_combo = None; best_pnl_test_report = {}; best_pnl_test_model = None

    for idx, combo in enumerate(combos, start=1):
        try:
            res = evaluate_combo(
                df=df_for_eval,
                features=list(combo),
                window_size=window_size,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                ticker=ticker,
                outdir=outdir
            )

            # Row for CSV (include BOTH VAL and TEST metrics)
            rows.append({
                "Index": idx,
                "Features": ", ".join(combo),
                # VALIDATION
                "RMSE_val": round(res["rmse_val"], 6),
                "MAE_val": round(res["mae_val"], 6),
                "Directional Accuracy_val": round(res["da_val"], 6),
                "PnL Ratio_val": round(res["pnl_val"], 6),
                # TEST
                "RMSE": round(res["rmse_test"], 6),
                "MAE": round(res["mae_test"], 6),
                "Directional Accuracy": round(res["da_test"], 6),
                "PnL Ratio": round(res["pnl_test"], 6),
            })

            # --- Update winners (RMSE by TEST — legacy, for reference)
            if res["rmse_test"] < best_rmse_test:
                best_rmse_test = res["rmse_test"]
                best_rmse_combo = combo
                best_rmse_model = res["model"]

            # --- DA winners (VAL selection)
            if res["da_val"] > best_da_val:
                best_da_val = res["da_val"]
                best_da_val_combo = combo
                best_da_val_model = res["model"]
                best_da_val_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- DA winners (TEST selection)
            if res["da_test"] > best_da_test:
                best_da_test = res["da_test"]
                best_da_test_combo = combo
                best_da_test_model = res["model"]
                best_da_test_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- PnL winners (VAL selection)
            if res["pnl_val"] > best_pnl_val:
                best_pnl_val = res["pnl_val"]
                best_pnl_val_combo = combo
                best_pnl_val_model = res["model"]
                best_pnl_val_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- PnL winners (TEST selection)
            if res["pnl_test"] > best_pnl_test:
                best_pnl_test = res["pnl_test"]
                best_pnl_test_combo = combo
                best_pnl_test_model = res["model"]
                best_pnl_test_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            logging.info(
                "✅ %d/%d %s | VAL: DA %.3f PnL %.3f | TEST: RMSE %.4f MAE %.4f DA %.3f PnL %.3f",
                idx, len(combos), combo,
                res["da_val"], res["pnl_val"],
                res["rmse_test"], res["mae_test"], res["da_test"], res["pnl_test"]
            )

        except Exception as e:
            logging.warning("❌ Failed on combo %s: %s", combo, str(e))

    if not rows:
        raise RuntimeError("No successful combinations evaluated. Check data and pipeline.")

    # =========================
    # Save results (master + sorted views)
    # =========================
    results_df = pd.DataFrame(rows)

    csv_master = os.path.join(outdir, f"lstm_combo_results_{ticker}.csv")
    results_df.to_csv(csv_master, index=False)

    # Sorted convenience views (VAL & TEST)
    csv_by_rmse      = os.path.join(outdir, f"lstm_combo_results_by_rmseTEST_{ticker}.csv")
    csv_by_da_val    = os.path.join(outdir, f"lstm_combo_results_by_daVAL_{ticker}.csv")
    csv_by_da_test   = os.path.join(outdir, f"lstm_combo_results_by_daTEST_{ticker}.csv")
    csv_by_pnl_val   = os.path.join(outdir, f"lstm_combo_results_by_pnlVAL_{ticker}.csv")
    csv_by_pnl_test  = os.path.join(outdir, f"lstm_combo_results_by_pnlTEST_{ticker}.csv")

    results_df.sort_values(by="RMSE", ascending=True).to_csv(csv_by_rmse, index=False)
    results_df.sort_values(by="Directional Accuracy_val", ascending=False).to_csv(csv_by_da_val, index=False)
    results_df.sort_values(by="Directional Accuracy", ascending=False).to_csv(csv_by_da_test, index=False)
    results_df.sort_values(by="PnL Ratio_val", ascending=False).to_csv(csv_by_pnl_val, index=False)
    results_df.sort_values(by="PnL Ratio", ascending=False).to_csv(csv_by_pnl_test, index=False)

    print(f"✅ Saved master: {csv_master}")
    print(f"   Sorted by RMSE (TEST): {csv_by_rmse}")
    print(f"   Sorted by DA (VAL):    {csv_by_da_val}")
    print(f"   Sorted by DA (TEST):   {csv_by_da_test}")
    print(f"   Sorted by PnL (VAL):   {csv_by_pnl_val}")
    print(f"   Sorted by PnL (TEST):  {csv_by_pnl_test}")

    # =========================
    # Heatmaps & analyses (use TEST metrics for continuity)
    # =========================
    D = build_design_matrix(results_df["Features"].tolist(), indicators)

    # TEST metrics for visuals
    M = results_df[["RMSE", "MAE", "Directional Accuracy", "PnL Ratio"]].reset_index(drop=True)

    # Flip errors so "higher is better"
    M_adj = M.copy()
    M_adj["-RMSE"] = -M_adj["RMSE"]
    M_adj["-MAE"]  = -M_adj["MAE"]
    M_adj = M_adj[["-RMSE", "-MAE", "Directional Accuracy", "PnL Ratio"]]

    # Heatmap 1: indicator presence vs metrics
    corr_ind_vs_metrics = pd.DataFrame(index=indicators, columns=M_adj.columns, dtype=float)
    for ind in indicators:
        for met in M_adj.columns:
            corr_ind_vs_metrics.loc[ind, met] = np.corrcoef(D[ind].values, M_adj[met].values)[0, 1]

    heatmap1_path = os.path.join(outdir, f"lstm_heatmap_indicator_vs_metrics_{ticker}.png")
    plot_heatmap(corr_ind_vs_metrics, "LSTM: Indicator Presence vs Metrics (corr)", heatmap1_path)

    # Heatmap 2: metric-to-metric correlations
    corr_metrics = M.corr()
    heatmap2_path = os.path.join(outdir, f"lstm_heatmap_metric_to_metric_{ticker}.png")
    plot_heatmap(corr_metrics, "LSTM: Metric-to-Metric Correlations (TEST)", heatmap2_path)

    # =========================
    # EXTRA: Effect sizes & Pairwise synergy
    # =========================
    effects = pd.DataFrame(index=D.columns, columns=M_adj.columns, dtype=float)
    for ind in D.columns:
        present_mask = D[ind] == 1
        absent_mask  = D[ind] == 0
        for met in M_adj.columns:
            effects.loc[ind, met] = M_adj.loc[present_mask, met].mean() - M_adj.loc[absent_mask, met].mean()

    effects_csv = os.path.join(outdir, f"lstm_indicator_effect_sizes_{ticker}.csv")
    effects.to_csv(effects_csv)
    effects_png = os.path.join(outdir, f"lstm_indicator_effect_sizes_{ticker}.png")
    plot_heatmap(effects, "LSTM: Indicator Effect Sizes (present − absent)\n(higher = better)", effects_png)

    def pairwise_synergy(metric_name):
        cols = list(D.columns)
        mat = np.full((len(cols), len(cols)), np.nan, dtype=float)
        for i, a in enumerate(cols):
            A = (D[a] == 1)
            if A.sum() < 2:
                continue
            for j, b in enumerate(cols):
                if i == j:
                    continue
                B = (D[b] == 1)
                AB = A & B
                if AB.sum() < 2 or B.sum() < 2:
                    continue
                mAB = M_adj.loc[AB, metric_name].mean()
                mA  = M_adj.loc[A,  metric_name].mean()
                mB  = M_adj.loc[B,  metric_name].mean()
                mat[i, j] = mAB - 0.5 * (mA + mB)
        return pd.DataFrame(mat, index=cols, columns=cols)

    for metric in ["-RMSE", "Directional Accuracy", "PnL Ratio"]:
        S = pairwise_synergy(metric)
        synergy_png = os.path.join(outdir, f"lstm_pairwise_synergy_{metric.replace(' ', '_')}_{ticker}.png")
        plot_heatmap_values(S, f"LSTM: Pairwise Synergy — {metric}", synergy_png)

    # =========================
    # EXTRA: Top-N and ALL-combos z-score heatmaps
    # =========================
    Z = M_adj.apply(zscore_robust)
    composite = Z.sum(axis=1)
    topN = min(20, len(results_df))
    top_idx = np.argsort(-composite.values)[:topN]
    H = results_df.iloc[top_idx][["RMSE", "MAE", "Directional Accuracy", "PnL Ratio"]].copy()
    H.index = results_df.iloc[top_idx]["Features"].tolist()

    top_png = os.path.join(outdir, f"lstm_top_combos_heatmap_{ticker}.png")
    plt.figure(figsize=(9.5, 8))
    if _HAS_SNS:
        sns.heatmap(H, annot=True, fmt=".3g", cmap="coolwarm")
    else:
        im = plt.imshow(H.values, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(H.shape[1]), H.columns, rotation=45, ha="right")
        plt.yticks(range(H.shape[0]), H.index)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                plt.text(j, i, f"{H.values[i, j]:.3g}", ha="center", va="center")
    plt.title("LSTM: Top Combinations — Performance Heatmap (TEST)")
    plt.tight_layout(); plt.savefig(top_png, dpi=300); plt.close()
    logging.info("Saved heatmap → %s", top_png)

    # ALL-combos z-score heatmap
    E = M_adj.copy()
    E.index = results_df["Features"]
    E = E[["-RMSE", "-MAE", "Directional Accuracy", "PnL Ratio"]]
    Ez = E.apply(zscore_robust).replace([np.inf, -np.inf], np.nan)

    combo_scores = pd.DataFrame({
        "Features": results_df["Features"],
        "RMSE": results_df["RMSE"],
        "MAE": results_df["MAE"],
        "Directional Accuracy": results_df["Directional Accuracy"],
        "PnL Ratio": results_df["PnL Ratio"],
        "CompositeScore": Ez.sum(axis=1).values
    }).sort_values("CompositeScore", ascending=False)
    combo_scores_csv = os.path.join(outdir, f"lstm_combo_scores_{ticker}.csv")
    combo_scores.to_csv(combo_scores_csv, index=False)

    Ez_sorted = Ez.loc[combo_scores["Features"].values]
    height = max(8, 0.25 * len(Ez_sorted))
    combos_heatmap_png = os.path.join(outdir, f"lstm_combos_effectiveness_heatmap_{ticker}.png")

    plt.figure(figsize=(10, height))
    if _HAS_SNS:
        sns.heatmap(Ez_sorted, cmap="coolwarm", center=0, cbar=True)
    else:
        data = np.where(np.isfinite(Ez_sorted.values), Ez_sorted.values, 0.0)
        im = plt.imshow(data, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(Ez_sorted.shape[1]), Ez_sorted.columns, rotation=45, ha="right")
        plt.yticks(range(Ez_sorted.shape[0]), Ez_sorted.index)
    plt.title("LSTM: Combination Effectiveness Heatmap (TEST z-scores; higher = better)")
    plt.xlabel("Metrics"); plt.ylabel("Indicator Combination")
    plt.tight_layout(); plt.savefig(combos_heatmap_png, dpi=300); plt.close()
    logging.info("Saved heatmap → %s", combos_heatmap_png)

    if _HAS_SNS:
        Ez_clust = Ez.copy()
        var_series = Ez_clust.var(axis=0, ddof=0)
        drop_cols = var_series[~np.isfinite(var_series) | (var_series == 0)].index.tolist()
        if drop_cols:
            Ez_clust = Ez_clust.drop(columns=drop_cols)
        Ez_clust = Ez_clust.dropna(how="all", axis=0).dropna(how="all", axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if Ez_clust.shape[0] >= 2 and Ez_clust.shape[1] >= 1:
            clus_height = max(8, 0.25 * Ez_clust.shape[0])
            cg = sns.clustermap(Ez_clust, cmap="coolwarm", center=0, figsize=(10, clus_height))
            cg.fig.suptitle("LSTM: Combination Effectiveness Clustered Heatmap (TEST z-scores)", y=1.02)
            combos_clustermap_png = os.path.join(outdir, f"lstm_combos_effectiveness_clustermap_{ticker}.png")
            cg.fig.savefig(combos_clustermap_png, dpi=300, bbox_inches="tight")
            plt.close(cg.fig)
            logging.info("Saved heatmap → %s", combos_clustermap_png)

    # =========================
    # Decide final DA & PnL winners per SELECT_ON policy
    # =========================
    if SELECT_ON.lower() == "val":
        da_winner_combo = best_da_val_combo
        da_winner_sel_value = best_da_val
        da_winner_report = best_da_val_report
        da_winner_model = best_da_val_model

        pnl_winner_combo = best_pnl_val_combo
        pnl_winner_sel_value = best_pnl_val
        pnl_winner_report = best_pnl_val_report
        pnl_winner_model = best_pnl_val_model
        sel_note = "selected by VALIDATION; reporting TEST metrics below"
    else:
        da_winner_combo = best_da_test_combo
        da_winner_sel_value = best_da_test
        da_winner_report = best_da_test_report
        da_winner_model = best_da_test_model

        pnl_winner_combo = best_pnl_test_combo
        pnl_winner_sel_value = best_pnl_test
        pnl_winner_report = best_pnl_test_report
        pnl_winner_model = best_pnl_test_model
        sel_note = "selected by TEST metrics"

    # =========================
    # Save text summary of winners
    # =========================
    best_txt = os.path.join(outdir, f"best_lstm_combos_{ticker}.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("=== Best LSTM Indicator Combinations ===\n")
        f.write(f"Selection policy: {SELECT_ON.upper()} ({sel_note})\n\n")
        f.write(f"By RMSE (TEST, lowest): {best_rmse_combo} | RMSE_test={best_rmse_test:.6f}\n\n")
        f.write(f"By Directional Accuracy (winner {SELECT_ON.upper()}): {da_winner_combo} | "
                f"DA_{SELECT_ON.lower()}={da_winner_sel_value:.6f} | "
                f"TEST -> RMSE={da_winner_report.get('rmse_test', np.nan):.6f}, "
                f"MAE={da_winner_report.get('mae_test', np.nan):.6f}, "
                f"DA={da_winner_report.get('da_test', np.nan):.6f}, "
                f"PnL={da_winner_report.get('pnl_test', np.nan):.6f}\n\n")
        f.write(f"By PnL Ratio (winner {SELECT_ON.upper()}): {pnl_winner_combo} | "
                f"PnL_{SELECT_ON.lower()}={pnl_winner_sel_value:.6f} | "
                f"TEST -> RMSE={pnl_winner_report.get('rmse_test', np.nan):.6f}, "
                f"MAE={pnl_winner_report.get('mae_test', np.nan):.6f}, "
                f"DA={pnl_winner_report.get('da_test', np.nan):.6f}, "
                f"PnL={pnl_winner_report.get('pnl_test', np.nan):.6f}\n")

    # Optionally save winner models (weights) for reuse
    # RMSE winner (TEST)
    if best_rmse_model is not None:
        rmse_model_path = os.path.join(outdir, "best_lstm_model_by_RMSE_TEST.h5")
        best_rmse_model.save(rmse_model_path)
        with open(os.path.join(outdir, "best_lstm_features_by_RMSE_TEST.txt"), "w", encoding="utf-8") as f:
            f.write(",".join(best_rmse_combo))

    # DA winner (per SELECT_ON)
    if da_winner_model is not None and da_winner_combo is not None:
        da_model_path = os.path.join(outdir, f"best_lstm_model_by_DA_{SELECT_ON.upper()}.h5")
        da_winner_model.save(da_model_path)
        with open(os.path.join(outdir, f"best_lstm_features_by_DA_{SELECT_ON.upper()}.txt"), "w", encoding="utf-8") as f:
            f.write(",".join(da_winner_combo))

    # PnL winner (per SELECT_ON)
    if pnl_winner_model is not None and pnl_winner_combo is not None:
        pnl_model_path = os.path.join(outdir, f"best_lstm_model_by_PnL_{SELECT_ON.upper()}.h5")
        pnl_winner_model.save(pnl_model_path)
        with open(os.path.join(outdir, f"best_lstm_features_by_PnL_{SELECT_ON.upper()}.txt"), "w", encoding="utf-8") as f:
            f.write(",".join(pnl_winner_combo))

    # =========================
    # Console summary
    # =========================
    print(f"🏆 Selection policy: {SELECT_ON.upper()} ({sel_note})")
    print(f"🏆 Best LSTM by RMSE (TEST): {best_rmse_combo} (RMSE={best_rmse_test:.6f})")
    print(f"🏆 Best LSTM by DA ({SELECT_ON.upper()}): {da_winner_combo} "
          f"(DA_{SELECT_ON.lower()}={da_winner_sel_value:.6f}) | "
          f"TEST: RMSE={da_winner_report.get('rmse_test', np.nan):.6f}, "
          f"MAE={da_winner_report.get('mae_test', np.nan):.6f}, "
          f"DA={da_winner_report.get('da_test', np.nan):.6f}, "
          f"PnL={da_winner_report.get('pnl_test', np.nan):.6f}")
    print(f"🏆 Best LSTM by PnL ({SELECT_ON.upper()}): {pnl_winner_combo} "
          f"(PnL_{SELECT_ON.lower()}={pnl_winner_sel_value:.6f}) | "
          f"TEST: RMSE={pnl_winner_report.get('rmse_test', np.nan):.6f}, "
          f"MAE={pnl_winner_report.get('mae_test', np.nan):.6f}, "
          f"DA={pnl_winner_report.get('da_test', np.nan):.6f}, "
          f"PnL={pnl_winner_report.get('pnl_test', np.nan):.6f}")
    print("🖼  Heatmaps saved:")
    print(" -", heatmap1_path)
    print(" -", heatmap2_path)
    print(" -", effects_png)
    print(" -", os.path.join(outdir, f"lstm_pairwise_synergy_-RMSE_{ticker}.png"))
    print(" -", os.path.join(outdir, f"lstm_pairwise_synergy_Directional_Accuracy_{ticker}.png"))
    print(" -", os.path.join(outdir, f"lstm_pairwise_synergy_PnL_Ratio_{ticker}.png"))
    print(" -", top_png)
    print(" -", combos_heatmap_png)
    if _HAS_SNS:
        print(" -", os.path.join(outdir, f"lstm_combos_effectiveness_clustermap_{ticker}.png"))
    print("📄 Tables saved:")
    print(" -", csv_master)
    print(" -", csv_by_rmse)
    print(" -", csv_by_da_val)
    print(" -", csv_by_da_test)
    print(" -", csv_by_pnl_val)
    print(" -", csv_by_pnl_test)
    print(" -", combo_scores_csv)


if __name__ == "__main__":
    main()
