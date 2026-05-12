import os
import pandas as pd
import logging
import glob

def load_stock_data(folder, tickers):
    df_list = []
    for t in tickers:
        logging.info(f"Loading CSV for {t}")
        filepath = os.path.join(folder, f"{t}.csv")
        if not os.path.exists(filepath):
            # Use glob to handle pattern matching if needed
            pattern = os.path.join(folder, f"{t}*.csv")
            files = glob.glob(pattern)
            if files:
                filepath = files[0]
            else:
                logging.warning(f"No CSV found for {t}")
                continue
        df = pd.read_csv(filepath)
        df = df[["Date", "Close", "Open"]]
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df.set_index("Date", inplace=True)
        logging.info(f"✔ Loaded {t}: {df.shape}")
        df_list.append(df)
    if not df_list:
        raise ValueError("No valid CSVs loaded.")
    return df_list
