import os
import pandas as pd
import matplotlib.pyplot as plt

# 📁 Folder path
folder_path = "./output"
delta_files = [f for f in os.listdir(folder_path) if f.startswith("test_deltas_") and f.endswith(".csv")]

if not delta_files:
    raise FileNotFoundError("No test_deltas_*.csv found.")

# 🔁 Loop through each model's result
for file in delta_files:
    path = os.path.join(folder_path, file)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    name = file.replace("test_deltas_", "").replace(".csv", "")

    # 📊 Create scatter plot
    plt.figure(figsize=(14, 6))
    plt.scatter(df["Date"], df["Actual"], color='blue', label="Actual Price", s=25)
    plt.scatter(df["Date"], df["Predicted"], color='red', label="Predicted Price", s=25)

    # Draw delta as vertical lines between actual and predicted
    for i in range(len(df)):
        plt.plot([df["Date"][i], df["Date"][i]], [df["Actual"][i], df["Predicted"][i]], color='gray', alpha=0.4, linewidth=1)

    plt.title(f"🔍 {name}: Predicted vs Actual Prices with Deltas")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # 💾 Save
    out_path = os.path.join(folder_path, f"scatter_actual_vs_pred_{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Saved scatter plot for {name} → {out_path}")
