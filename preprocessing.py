import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# =========================
# Missing-data & numeric-safety
# =========================
def impute_numeric_median(df):
    """
    Replace +/-inf with NaN, then fill numeric columns with column-wise medians.
    Non-numeric columns are left untouched.
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(skipna=True))
    return df

def ensure_datetime_index(df):
    """Force DateTimeIndex if a 'Date' column exists. No-op if already time-indexed."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
    return df

# =========================
# Stationarity helpers
# =========================
def _adf_p(x):
    try:
        from statsmodels.tsa.stattools import adfuller
        return adfuller(pd.Series(x).dropna(), autolag="AIC")[1]
    except Exception:
        return np.nan

def _kpss_p(x):
    try:
        from statsmodels.tsa.stattools import kpss
        return kpss(pd.Series(x).dropna(), regression="c", nlags="auto")[1]
    except Exception:
        return np.nan

def stationarity_report(df, cols, save_path=None):
    """
    ADF (H0: unit root / non-stationary) and KPSS (H0: stationary).
    'likely stationary' if ADF_p < .05 and KPSS_p > .05.
    """
    rows = []
    for c in cols:
        s = df[c].values
        p_adf  = _adf_p(s)
        p_kpss = _kpss_p(s)
        if (p_adf < 0.05) and (p_kpss > 0.05):
            verdict = "likely stationary"
        elif (p_adf >= 0.05) and (p_kpss <= 0.05):
            verdict = "likely non-stationary"
        else:
            verdict = "inconclusive"
        rows.append({"feature": c, "ADF_p": p_adf, "KPSS_p": p_kpss, "verdict": verdict})
    rep = pd.DataFrame(rows)
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        rep.to_csv(save_path, index=False)
    return rep

def add_stationary_transforms(
    df,
    price_cols=("Open","High","Low","Close","Volume"),
    zwin=30,
    add_pct=True,
    add_log=True
):
    """
    Add percent change, log-returns, and rolling z-scores for price-like columns.
    """
    df = df.copy()
    for c in price_cols:
        if c in df.columns:
            if add_pct:
                df[f"{c}_ret"] = df[c].pct_change()
            if add_log:
                df[f"{c}_logret"] = np.log(df[c] / df[c].shift(1))
            m = df[c].rolling(zwin).mean()
            s = df[c].rolling(zwin).std()
            df[f"{c}_z{zwin}"] = (df[c] - m) / s
    return df

# =========================
# Scaling & windowing
# =========================
def normalize_features(df, target_col="Close"):
    """
    MinMax scale features and target separately.
    Cleans numerics before and after scaling.
    """
    df = impute_numeric_median(df)

    feature_cols = [c for c in df.columns if c != target_col]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[[target_col]])

    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    df_scaled[target_col] = y_scaled

    df_scaled = impute_numeric_median(df_scaled)
    return df_scaled, scaler_X, scaler_y

def df_to_windowed_df(df, window_size=5, target_col="Close"):
    """
    Convert a DataFrame to a windowed format for time series forecasting.
    """
    feature_cols = [col for col in df.columns if col != target_col]

    windowed_data, target_dates = [], []
    for i in range(window_size, len(df)):
        row = []
        for col in feature_cols:
            row.extend(df[col].iloc[i - window_size:i].values)
        row.append(df[target_col].iloc[i])
        target_dates.append(df.index[i].strftime('%Y-%m-%d'))
        windowed_data.append(row)

    columns = ["Target Date"] + [f"{col}_t-{j}" for col in feature_cols for j in range(window_size, 0, -1)] + ["Target"]
    windowed_df = pd.DataFrame(windowed_data, columns=columns[1:])
    windowed_df.insert(0, "Target Date", target_dates)
    logging.info(f"Windowed DataFrame shape: {windowed_df.shape}")

    # robust to NaNs/Infs
    windowed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = windowed_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        windowed_df[num_cols] = windowed_df[num_cols].fillna(windowed_df[num_cols].median(skipna=True))
    return windowed_df

def windowed_df_to_date_X_y(windowed_df, window_size):
    """
    Convert windowed DataFrame back to arrays.
    """
    dates = windowed_df["Target Date"].values
    X = windowed_df.drop(columns=["Target Date", "Target"]).values
    y = windowed_df["Target"].values
    n_features = int(X.shape[1] / window_size)
    X = X.reshape((len(dates), window_size, n_features))
    return dates, X.astype(np.float32), y.astype(np.float32)



def normalize_features_train_only(df_train, df_other, target_col):
    feature_cols = [c for c in df_train.columns if c != target_col]

    scaler_X = MinMaxScaler().fit(df_train[feature_cols])
    scaler_y = MinMaxScaler().fit(df_train[[target_col]])

    def _transform(df):
        Xs = scaler_X.transform(df[feature_cols])
        ys = scaler_y.transform(df[[target_col]])
        out = pd.DataFrame(Xs, columns=feature_cols, index=df.index)
        out[target_col] = ys
        return impute_numeric_median(out)

    return _transform(df_train), _transform(df_other), scaler_X, scaler_y