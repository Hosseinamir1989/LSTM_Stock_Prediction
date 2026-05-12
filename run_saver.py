import os, json, joblib, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path
import yaml

class RunSaver:
    """
    Per-run recorder that saves only what you ask for:
    - config snapshot (optional)
    - dataframes (CSV), json blobs, npz artifacts
    - models (.h5 by default)
    - training history (CSV/JSON)
    - freeform notes

    No environment/package metadata is saved.
    """
    def __init__(self, model_name: str, ticker: str, root: str = "runs"):
        self.run_id  = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.base    = Path(root) / model_name.upper() / ticker / self.run_id
        self.plots   = self.base / "plots"
        self.models  = self.base / "models"
        self.tables  = self.base / "tables"
        self.art     = self.base / "artifacts"
        self.logs    = self.base / "logs"
        for p in [self.base, self.plots, self.models, self.tables, self.art, self.logs]:
            p.mkdir(parents=True, exist_ok=True)

    # ---- generic helpers ----
    def save_config(self, cfg):
        """Write a YAML snapshot of your current config dict."""
        with open(self.base / "config_snapshot.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def save_df(self, df: pd.DataFrame, name: str):
        p = self.tables / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_json(self, obj, name: str):
        p = self.tables / f"{name}.json"
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=float)
        return p

    def save_npz(self, name: str, **arrays):
        p = self.art / f"{name}.npz"
        np.savez_compressed(p, **arrays)
        return p

    def save_model(self, model, name: str):
        p = self.models / f"{name}.h5"
        model.save(p)
        return p

    def save_scalers(self, scaler_X=None, scaler_y=None, name="scalers"):
        if scaler_X is not None:
            joblib.dump(scaler_X, self.art / f"{name}_X.pkl")
        if scaler_y is not None:
            joblib.dump(scaler_y, self.art / f"{name}_y.pkl")

    def save_history(self, history, name="history"):
        if history is None:
            return
        try:
            pd.DataFrame(history.history).to_csv(self.tables / f"{name}.csv", index=False)
        except Exception:
            with open(self.tables / f"{name}.json", "w") as f:
                json.dump(getattr(history, "history", {}), f, indent=2)

    def note(self, text: str, name="notes"):
        with open(self.logs / f"{name}.txt", "a") as f:
            f.write(text.rstrip() + "\n")
