import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)


# =========================
# Plotting helpers
# =========================
def plot_predictions(dates_train, y_train, train_pred,
                     dates_val, y_val, val_pred,
                     dates_test, y_test, test_pred,
                     title="LSTM Predictions"):
    """Plot train/val/test actual vs predicted."""
    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, y_train, label="Train Actual")
    plt.plot(dates_train, train_pred, label="Train Predicted")
    plt.plot(dates_val, y_val, label="Validation Actual")
    plt.plot(dates_val, val_pred, label="Validation Predicted")
    plt.plot(dates_test, y_test, label="Test Actual")
    plt.plot(dates_test, test_pred, label="Test Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zoomed_test(dates_test, y_test, test_pred, zoom=100, title=None):
    """Zoom into the last `zoom` points of the test set."""
    zoom = int(zoom)
    title = title or f"Zoomed-in Test Set (last {zoom} samples)"
    plt.figure(figsize=(12, 5))
    plt.plot(dates_test[-zoom:], y_test[-zoom:], label="Actual", linewidth=2)
    plt.plot(dates_test[-zoom:], test_pred[-zoom:], label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_pred):
    """RMSE, MAE, MAPE, R2 for regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
     
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan


    logging.info("RMSE: %.4f, MAE: %.4f, MAPE: %.4f, R2: %.4f", rmse, mae, mape, r2)
    return rmse, mae, r2, mape


def compute_directional_accuracy_next_day(y_true, y_pred, ignore_flat=True):
    """
    Directional accuracy for 1-step-ahead forecast.

    Assumption: y_pred[t] predicts y_true[t+1].
    True direction at t: sign(y_true[t+1] - y_true[t])
    Pred direction at t: sign(y_pred[t]     - y_true[t])

    ignore_flat=True ignores cases where the true move is exactly 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_pred), len(y_true) - 1)
    if n <= 0:
        return np.nan

    today = y_true[:n]
    true_next = y_true[1:n + 1]
    pred_next = y_pred[:n]

    true_dir = np.sign(true_next - today)
    pred_dir = np.sign(pred_next - today)

    if ignore_flat:
        mask = (true_dir != 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(true_dir[mask] == pred_dir[mask]))

    return float(np.mean(true_dir == pred_dir))


def compute_signal_accuracy_next_day(y_true, y_pred, threshold=0.0, ignore_flat=True):
    """
    Signal Accuracy (Buy/Sell/Hold) for 1-step-ahead forecast.

    Assumption: y_pred[t] predicts y_true[t+1].
    True signal at t: sign(y_true[t+1] - y_true[t]) mapped to {-1,0,1}
    Pred signal at t: sign(y_pred[t]   - y_true[t]) with a deadzone threshold

    threshold: deadzone in *price units* (if you pass prices) or return units
               (if you pass returns). Example: 0.0 for price, or 0.001 for 0.1% return.
    ignore_flat=True ignores true 0-move days in accuracy calculation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_pred), len(y_true) - 1)
    if n <= 0:
        return np.nan

    today = y_true[:n]
    true_next = y_true[1:n + 1]
    pred_next = y_pred[:n]

    true_move = true_next - today
    pred_move = pred_next - today

    true_sig = np.where(true_move > 0, 1, np.where(true_move < 0, -1, 0))
    pred_sig = np.where(pred_move > threshold, 1, np.where(pred_move < -threshold, -1, 0))

    if ignore_flat:
        mask = (true_sig != 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(true_sig[mask] == pred_sig[mask]))

    return float(np.mean(true_sig == pred_sig))


# Optional: keep your old "generate_signals" + compute_signal_accuracy (fixed)
def generate_signals(prices_today, prices_next):
    """1=Buy, -1=Sell, 0=Hold based on next vs today."""
    prices_today = np.asarray(prices_today, dtype=float)
    prices_next = np.asarray(prices_next, dtype=float)

    signals = np.zeros(len(prices_today), dtype=int)
    signals[prices_next > prices_today] = 1
    signals[prices_next < prices_today] = -1
    return signals


def compute_signal_accuracy(y_true, y_pred):
    """
    Legacy signal accuracy based on predicted-series transitions
    (not recommended for trading rule evaluation, but kept for compatibility).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    if n < 2:
        return np.nan

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    today_true = y_true[:-1]
    next_true = y_true[1:]

    today_pred = y_pred[:-1]   # ✅ fixed
    next_pred = y_pred[1:]

    true_signals = generate_signals(today_true, next_true)
    pred_signals = generate_signals(today_pred, next_pred)

    return float(accuracy_score(true_signals, pred_signals))


# =========================
# Horizon evaluation
# =========================
def evaluate_multi_horizon_accuracy(y_true, y_pred, horizon_days=(1, 2, 3)):
    """
    Evaluate RMSE/MAE/R2 for multiple horizons, assuming y_pred is T+1 style series.
    For horizon h:
      compare y_true[t+h] vs y_pred[t]  (shift y_true forward by h, y_pred backward by h)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = []
    for h in horizon_days:
        h = int(h)
        if len(y_true) <= h or len(y_pred) <= 0:
            continue

        n = min(len(y_pred), len(y_true) - h)
        if n <= 0:
            continue

        shifted_true = y_true[h:h + n]
        valid_pred = y_pred[:n]

        rmse = float(np.sqrt(mean_squared_error(shifted_true, valid_pred)))
        mae = float(mean_absolute_error(shifted_true, valid_pred))
        r2 = float(r2_score(shifted_true, valid_pred))

        results.append({"Horizon": f"T+{h}", "RMSE": rmse, "MAE": mae, "R2": r2})

    return results


def evaluate_directional_accuracy_by_horizon(y_true, y_pred, horizon_days=(1, 2, 3), ignore_flat=True):
    """
    Directional accuracy for multiple horizons, assuming y_pred is T+1 style series.
    For horizon h:
      true_dir at t = sign(y_true[t+h] - y_true[t])
      pred_dir at t = sign(y_pred[t]   - y_true[t])
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = []
    for h in horizon_days:
        h = int(h)
        if len(y_true) <= h or len(y_pred) <= 0:
            continue

        n = min(len(y_pred), len(y_true) - h)
        if n <= 0:
            continue

        today = y_true[:n]
        true_h = y_true[h:h + n]
        pred_next = y_pred[:n]

        true_dir = np.sign(true_h - today)
        pred_dir = np.sign(pred_next - today)

        if ignore_flat:
            mask = (true_dir != 0)
            if mask.sum() == 0:
                dir_acc = np.nan
            else:
                dir_acc = float(np.mean(true_dir[mask] == pred_dir[mask]))
        else:
            dir_acc = float(np.mean(true_dir == pred_dir))

        results.append({"Horizon": f"T+{h}", "Directional Accuracy": dir_acc})

    return results


# =========================
# PnL Simulation
# =========================
def simulate_pnl(
    y_true,
    y_pred,
    initial_cash=80000,
    ticker="Unknown",
    output_folder="output",
    force_first_buy=False,           # recommended: False for fair evaluation
    exclude_unpaired_last_buy=True,
    fee_rate=0.0005,                 # 5 bps per trade
    slippage_bps=2,                  # 2 bps
    stop_loss_pct=None,
    take_profit_pct=None,
    max_hold_days=None,
    threshold_pct=0.0                # deadzone on predicted move in pct (0.002=0.2%)
):
    """
    Trading simulator aligned with 1-step-ahead prediction indexing.

    Assumption:
        y_pred[t] predicts y_true[t+1].

    Decision at time t, executed at t+1 close:
        pred_move_pct = (y_pred[t] - y_true[t]) / y_true[t]
        Buy  if pred_move_pct >  threshold_pct and no position is open.
        Sell if pred_move_pct < -threshold_pct and a position is open.

    Important metric definition:
        P&L Ratio = Final Portfolio Value / Initial Capital

    This matches the thesis definition. The old implementation used an average
    gain/loss ratio, which is more similar to a profit factor and can become
    unrealistically large when total/average losses are close to zero.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    n = min(len(y_pred), len(y_true) - 1)
    if n <= 0:
        result = {
            "Ticker": ticker,
            "Final Cash": round(float(initial_cash), 2),
            "Final Portfolio Value": round(float(initial_cash), 2),
            "Total Gain": 0.0,
            "Total Loss": 0.0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "P&L Ratio": 1.0,
            "Profit Factor": 0.0,
            "Win Ratio": 0.0,
        }
        return result, pd.DataFrame()

    cash = float(initial_cash)
    shares = 0
    entry_price = 0.0
    entry_day = None

    trade_log = []
    total_gain = 0.0
    total_loss = 0.0
    winning_trades = 0
    losing_trades = 0

    def exec_buy(day_idx, raw_price):
        nonlocal cash, shares, entry_price, entry_day, trade_log

        px = float(raw_price) * (1.0 + slippage_bps / 10000.0)
        max_shares = int(cash // px)
        if max_shares <= 0:
            return False

        # Ensure that the notional value plus transaction fee fits into cash.
        while max_shares > 0:
            notional = max_shares * px
            fee = notional * fee_rate
            if notional + fee <= cash:
                break
            max_shares -= 1
        if max_shares <= 0:
            return False

        notional = max_shares * px
        fee = notional * fee_rate

        cash -= (notional + fee)
        shares = max_shares
        entry_price = px
        entry_day = day_idx

        trade_log.append({
            "Day": day_idx,
            "Action": "Buy",
            "Price": px,
            "Shares": shares,
            "Fee": fee,
            "Cash_After": cash,
        })
        return True

    def exec_sell(day_idx, raw_price, reason="Sell"):
        nonlocal cash, shares, entry_price, entry_day
        nonlocal total_gain, total_loss, winning_trades, losing_trades, trade_log

        if shares <= 0:
            return False

        px = float(raw_price) * (1.0 - slippage_bps / 10000.0)
        notional = shares * px
        fee = notional * fee_rate

        pnl = (px - entry_price) * shares - fee
        if pnl > 0:
            total_gain += pnl
            winning_trades += 1
        else:
            total_loss += abs(pnl)
            losing_trades += 1

        cash += (notional - fee)

        trade_log.append({
            "Day": day_idx,
            "Action": reason,
            "Price": px,
            "Shares": shares,
            "PnL": pnl,
            "Fee": fee,
            "Cash_After": cash,
        })

        shares = 0
        entry_price = 0.0
        entry_day = None
        return True

    # Main loop: t=0..n-1 corresponds to execution day t+1.
    for t in range(n):
        today_price = float(y_true[t])
        tomorrow_price = float(y_true[t + 1])
        predicted_price = float(y_pred[t])
        day_exec = t + 1

        if force_first_buy and t == 0 and shares == 0:
            exec_buy(day_exec, tomorrow_price)
            continue

        # Risk controls are checked against the execution day's true close price.
        if shares > 0:
            if stop_loss_pct is not None and tomorrow_price <= entry_price * (1.0 - float(stop_loss_pct)):
                exec_sell(day_exec, tomorrow_price, reason="Sell (StopLoss)")
                continue
            if take_profit_pct is not None and tomorrow_price >= entry_price * (1.0 + float(take_profit_pct)):
                exec_sell(day_exec, tomorrow_price, reason="Sell (TakeProfit)")
                continue
            if max_hold_days is not None and entry_day is not None and (day_exec - entry_day) >= int(max_hold_days):
                exec_sell(day_exec, tomorrow_price, reason="Sell (MaxHold)")
                continue

        pred_move_pct = (predicted_price - today_price) / today_price if today_price != 0 else 0.0

        if pred_move_pct > float(threshold_pct) and shares == 0:
            exec_buy(day_exec, tomorrow_price)
        elif pred_move_pct < -float(threshold_pct) and shares > 0:
            exec_sell(day_exec, tomorrow_price, reason="Sell")

    # Optional: remove a final open buy if it has no matching sell in the test period.
    # This keeps the analysis based only on completed trades.
    if exclude_unpaired_last_buy and shares > 0:
        last_buy_idx = None
        for j in range(len(trade_log) - 1, -1, -1):
            if trade_log[j].get("Action") == "Buy":
                last_buy_idx = j
                break

        if last_buy_idx is not None:
            buy_entry = trade_log[last_buy_idx]
            px = float(buy_entry["Price"])
            sh = int(buy_entry["Shares"])
            fee = float(buy_entry.get("Fee", 0.0))

            # Undo the buy because it was not paired with a sell inside the evaluation period.
            cash += (sh * px + fee)
            trade_log = trade_log[:last_buy_idx] + trade_log[last_buy_idx + 1:]

        shares = 0
        entry_price = 0.0
        entry_day = None

    # Final portfolio value includes any open position if exclude_unpaired_last_buy=False.
    final_price = float(y_true[n])
    final_position_value = shares * final_price
    final_portfolio_value = cash + final_position_value

    pnl_ratio = final_portfolio_value / float(initial_cash) if initial_cash != 0 else np.nan
    profit_factor = total_gain / total_loss if total_loss > 0 else (np.inf if total_gain > 0 else 0.0)
    total_trades = winning_trades + losing_trades
    win_ratio = winning_trades / total_trades if total_trades > 0 else 0.0

    result = {
        "Ticker": ticker,
        "Final Cash": round(cash, 2),
        "Final Portfolio Value": round(float(final_portfolio_value), 2),
        "Open Position Value": round(float(final_position_value), 2),
        "Total Gain": round(float(total_gain), 2),
        "Total Loss": round(float(total_loss), 2),
        "Winning Trades": int(winning_trades),
        "Losing Trades": int(losing_trades),
        "P&L Ratio": round(float(pnl_ratio), 4),
        "Profit Factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else np.inf,
        "Win Ratio": round(float(win_ratio), 4),
    }

    trade_log_df = pd.DataFrame(trade_log)
    os.makedirs(output_folder, exist_ok=True)
    trade_log_df.to_csv(os.path.join(output_folder, f"pnl_research_style_{ticker}.csv"), index=False)

    return result, trade_log_df


def sanity_check_simulate_pnl(output_folder="output/pnl_sanity_checks"):
    """
    Simple sanity checks for simulate_pnl().

    These checks are intentionally small and interpretable:
      1. No-trade case: P&L Ratio should remain 1.0.
      2. Buy-and-hold-like case: open position is valued at the final price.
      3. One completed losing trade: P&L Ratio should be below 1.0.

    The tests use fee_rate=0 and slippage_bps=0 so that the expected behavior is easy to inspect.
    """
    checks = []

    # 1) No trade: predictions equal today's price, so no Buy/Sell signal is triggered.
    y_true = np.array([100, 101, 102, 103], dtype=float)
    y_pred = np.array([100, 101, 102], dtype=float)
    res, trades = simulate_pnl(
        y_true, y_pred,
        initial_cash=1000,
        ticker="SANITY_NO_TRADE",
        output_folder=output_folder,
        threshold_pct=0.0,
        fee_rate=0.0,
        slippage_bps=0,
    )
    checks.append({
        "Case": "No trade",
        "Expected": "P&L Ratio = 1.0 and zero trades",
        "P&L Ratio": res["P&L Ratio"],
        "Final Portfolio Value": res["Final Portfolio Value"],
        "Trades": len(trades),
        "Pass": np.isclose(res["P&L Ratio"], 1.0) and len(trades) == 0,
    })

    # 2) Buy-and-hold-like: model predicts upward movement every day.
    # Keep the final open position and value it at the final price.
    y_true = np.array([100, 110, 120, 130], dtype=float)
    y_pred = np.array([111, 121, 131], dtype=float)
    res, trades = simulate_pnl(
        y_true, y_pred,
        initial_cash=1000,
        ticker="SANITY_BUY_HOLD",
        output_folder=output_folder,
        threshold_pct=0.0,
        fee_rate=0.0,
        slippage_bps=0,
        exclude_unpaired_last_buy=False,
    )
    checks.append({
        "Case": "Buy-and-hold-like",
        "Expected": "P&L Ratio > 1.0 in rising market",
        "P&L Ratio": res["P&L Ratio"],
        "Final Portfolio Value": res["Final Portfolio Value"],
        "Trades": len(trades),
        "Pass": res["P&L Ratio"] > 1.0 and len(trades) >= 1,
    })

    # 3) One completed losing trade: buy after first signal, sell after later negative signal.
    y_true = np.array([100, 100, 90, 90], dtype=float)
    y_pred = np.array([110, 80, 80], dtype=float)
    res, trades = simulate_pnl(
        y_true, y_pred,
        initial_cash=1000,
        ticker="SANITY_ONE_LOSS",
        output_folder=output_folder,
        threshold_pct=0.0,
        fee_rate=0.0,
        slippage_bps=0,
    )
    checks.append({
        "Case": "One completed losing trade",
        "Expected": "P&L Ratio < 1.0 and at least one losing trade",
        "P&L Ratio": res["P&L Ratio"],
        "Final Portfolio Value": res["Final Portfolio Value"],
        "Trades": len(trades),
        "Pass": res["P&L Ratio"] < 1.0 and res["Losing Trades"] >= 1,
    })

    return pd.DataFrame(checks)



def compute_directional_accuracy_from_reference(current_price, next_true, next_pred, ignore_flat=True):
    """
    Directional accuracy for next-day price prediction.

    current_price[i] = today's actual price for sample i
    next_true[i]     = actual next-day price for sample i
    next_pred[i]     = predicted next-day price for sample i
    """
    current_price = np.asarray(current_price, dtype=float)
    next_true = np.asarray(next_true, dtype=float)
    next_pred = np.asarray(next_pred, dtype=float)

    n = min(len(current_price), len(next_true), len(next_pred))
    if n == 0:
        return np.nan

    current_price = current_price[:n]
    next_true = next_true[:n]
    next_pred = next_pred[:n]

    true_dir = np.sign(next_true - current_price)
    pred_dir = np.sign(next_pred - current_price)

    if ignore_flat:
        mask = (true_dir != 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(true_dir[mask] == pred_dir[mask]))

    return float(np.mean(true_dir == pred_dir))


def compute_signal_accuracy_from_reference(current_price, next_true, next_pred,
                                           threshold_pct=0.0, ignore_flat=True):
    """
    Signal accuracy for next-day price prediction.

    Signals:
      1 = buy
      0 = hold
     -1 = sell
    """
    current_price = np.asarray(current_price, dtype=float)
    next_true = np.asarray(next_true, dtype=float)
    next_pred = np.asarray(next_pred, dtype=float)

    n = min(len(current_price), len(next_true), len(next_pred))
    if n == 0:
        return np.nan

    current_price = current_price[:n]
    next_true = next_true[:n]
    next_pred = next_pred[:n]

    true_move_pct = (next_true - current_price) / current_price
    pred_move_pct = (next_pred - current_price) / current_price

    true_sig = np.where(true_move_pct > 0, 1, np.where(true_move_pct < 0, -1, 0))
    pred_sig = np.where(pred_move_pct > threshold_pct, 1,
                        np.where(pred_move_pct < -threshold_pct, -1, 0))

    if ignore_flat:
        mask = (true_sig != 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(true_sig[mask] == pred_sig[mask]))

    return float(np.mean(true_sig == pred_sig))
