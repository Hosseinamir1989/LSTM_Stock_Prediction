import pandas as pd
import numpy as np
import logging

def add_indicators(df, cfg=None):
    """
    Add core technical indicators.
    - Preserves DatetimeIndex (NO dropna/reset_index here)
    - Accepts optional `cfg` dict (e.g., config["technical_indicators"])
    """
    logging.info("Adding technical indicators")
    df = df.copy()
    cfg = (cfg or {})

    # windows
    rsi_w      = int(cfg.get("rsi_window", 14))
    sma_w      = int(cfg.get("sma_window", 20))
    ema_w      = int(cfg.get("ema_window", 20))
    bb_w       = int(cfg.get("bb_window", 20))
    bb_std     = float(cfg.get("bb_std", 2))
    vol_w      = int(cfg.get("vol_window", 20))
    mom_w      = int(cfg.get("momentum_window", 10))
    macd_fast  = int(cfg.get("macd_fast", 12))
    macd_slow  = int(cfg.get("macd_slow", 26))
    macd_sig   = int(cfg.get("macd_signal", 9))
    add_log_return = bool(cfg.get("log_return", False))
    add_directional = bool(cfg.get("directional_features", False))

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    temp_log_return = np.log(df["Close"] / df["Close"].shift(1))

    if add_log_return:
        df["Log_Return"] = temp_log_return
        df["Log_Return_lag1"] = temp_log_return.shift(1)
        df["Log_Return_lag2"] = temp_log_return.shift(2)
        df["Log_Return_lag3"] = temp_log_return.shift(3)

    # MACD
    fast_ema = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = fast_ema - slow_ema
    df["MACD_signal"] = df["MACD"].ewm(span=macd_sig, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(rsi_w).mean()
    roll_down = down.rolling(rsi_w).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving averages
    df["SMA"] = df["Close"].rolling(window=sma_w).mean()
    df["EMA"] = df["Close"].ewm(span=ema_w, adjust=False).mean()

    # Bollinger Bands
    ma_bb = df["Close"].rolling(window=bb_w).mean()
    std_bb = df["Close"].rolling(window=bb_w).std()
    df["BB_middle"] = ma_bb
    df["BB_upper"] = ma_bb + (bb_std * std_bb)
    df["BB_lower"] = ma_bb - (bb_std * std_bb)

    # Volatility
    df["RollingVolatility"] = temp_log_return.rolling(window=vol_w).std()

    # Momentum
    df["Momentum"] = df["Close"] - df["Close"].rolling(window=mom_w).mean()

    # Directional / regime features
    if add_directional:
        df["Above_SMA"] = (df["Close"] > df["SMA"]).astype(int)
        df["Above_EMA"] = (df["Close"] > df["EMA"]).astype(int)
        df["MACD_Positive"] = (df["MACD"] > 0).astype(int)
        df["RSI_Above_50"] = (df["RSI"] > 50).astype(int)
        df["BB_Breakout_Up"] = (df["Close"] > df["BB_upper"]).astype(int)
        df["BB_Breakout_Down"] = (df["Close"] < df["BB_lower"]).astype(int)

    logging.info(f"Indicators added. DataFrame shape: {df.shape}")
    return df