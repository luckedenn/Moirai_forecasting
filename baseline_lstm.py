# baseline_lstm.py (FIXED: multi-step + fair context_len)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================================
# Reproducibility (biar ga berubah-ubah)
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(num / den)) * 100.0

def choose_test_len(total_len: int, pdt: int, frac: float = 0.15) -> int:
    raw = max(pdt, int(total_len * frac))
    test = (raw // pdt) * pdt
    return test if test >= pdt else pdt

def load_series(csv_path: str, freq: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    if freq in ("M", "MS", "ME"):
        df.index = df.index.to_period("M").to_timestamp("M")
        use_freq = "M"
    else:
        use_freq = freq

    df = df.asfreq(use_freq)
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"])
    return df, use_freq

# =========================================================
# Dataset MULTI-STEP (seq -> pred_len)
# =========================================================
class SeqDatasetMulti(Dataset):
    """
    Input  : (context_len,)
    Target : (pred_len,)
    """
    def __init__(self, series: np.ndarray, context_len: int, pred_len: int):
        self.series = series.astype(np.float32)
        self.context_len = context_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.series) - self.context_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.context_len]  # (context_len,)
        y = self.series[idx + self.context_len : idx + self.context_len + self.pred_len]  # (pred_len,)
        return (
            torch.from_numpy(x).unsqueeze(-1).float(),  # (context_len, 1)
            torch.from_numpy(y).float()                 # (pred_len,)
        )

# =========================================================
# LSTM model MULTI-OUTPUT
# =========================================================
class LSTMForecastMulti(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_len=7, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.rnn = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, T, H)
        h_last = out[:, -1, :]        # (B, H)
        y = self.head(h_last)         # (B, pred_len)
        return y

# =========================================================
# Training loop MULTI-STEP
# =========================================================
def train_lstm_multi(
    train_arr: np.ndarray,
    context_len: int,
    pred_len: int,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    device: str = "cpu"
):
    ds = SeqDatasetMulti(train_arr, context_len, pred_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMForecastMulti(1, hidden_size, num_layers, pred_len).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        losses = []
        for xb, yb in dl:
            xb = xb.to(device)                 # (B, context_len, 1)
            yb = yb.to(device)                 # (B, pred_len)

            optim.zero_grad()
            pred = model(xb)                   # (B, pred_len)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"[LSTM-Multi] epoch {ep+1}/{epochs} - loss: {np.mean(losses):.6f}")

    model.eval()
    return model

# =========================================================
# Predict DIRECT multi-step (tanpa autoregressive)
# =========================================================
@torch.no_grad()
def direct_forecast(model: nn.Module, context: np.ndarray, device="cpu"):
    """
    context: (context_len,) scaled
    return : (pred_len,) scaled
    """
    x = torch.from_numpy(context.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
    pred_scaled = model(x).squeeze(0).cpu().numpy()
    return pred_scaled

# =========================================================
# Runner per dataset
# =========================================================
def run_one_dataset(
    name: str,
    csv: str,
    freq: str,
    pred_len: int,
    context_len: int,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    results_dir: str = "results_baseline_lstm",
    device: str = None
):
    print("\n" + "=" * 60)
    print(f"🚀 LSTM BASELINE (FIXED MULTI-STEP) — {name.upper()}")
    print("=" * 60)

    outdir = ensure_dir(os.path.join(results_dir, name))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    df, use_freq = load_series(csv, freq)
    series = df["value"].values.astype(np.float32)

    total_len = len(series)
    TEST = choose_test_len(total_len, pred_len, frac=0.15)
    windows = TEST // pred_len

    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG[name].get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pred_len

    print(f"📂 Range: {df.index.min().date()} → {df.index.max().date()} | freq={use_freq}")
    print(f"🧮 Rows: {total_len} | TEST={TEST} | windows={windows}")
    print(f"⚙️ context_len={context_len} | pred_len={pred_len}")

    # Train/test split
    train_vals = series[: total_len - TEST]
    test_vals  = series[total_len - TEST :]

    # scaling pakai train saja
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1)
    test_scaled  = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1)

    if len(train_scaled) <= context_len + pred_len:
        raise ValueError(
            f"Train length ({len(train_scaled)}) harus > context_len+pred_len ({context_len+pred_len})."
        )

    # Train multi-step model
    model = train_lstm_multi(
        train_arr=train_scaled,
        context_len=context_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lr=lr,
        device=device
    )

    rows = []
    maes, rmses, smapes = [], [], []

    full_scaled = np.concatenate([train_scaled, test_scaled])

    for w in range(windows):
        start_idx = len(train_scaled) + w * pred_len
        end_idx = start_idx + pred_len

        # context sebelum window
        ctx_start = start_idx - context_len
        if ctx_start < 0:
            raise ValueError("context_len terlalu besar untuk window pertama.")
        context = full_scaled[ctx_start:start_idx]  # (context_len,)

        # DIRECT multi-step forecast (scaled)
        pred_scaled = direct_forecast(model, context, device=device)  # (pred_len,)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

        # ground truth
        y_true = series[start_idx:end_idx].astype(np.float32)

        # metrics
        m_mae = mae(y_true, pred)
        m_rmse = rmse(y_true, pred)
        m_smape = smape(y_true, pred)
        maes.append(m_mae); rmses.append(m_rmse); smapes.append(m_smape)

        ts_idx = pd.date_range(start=df.index[start_idx], periods=pred_len, freq=use_freq)

        for t, yt, yp in zip(ts_idx, y_true, pred):
            rows.append({"window": w, "timestamp": pd.Timestamp(t), "y_true": float(yt), "y_pred": float(yp)})

        print(f"[LSTM-Multi] window {w+1}/{windows} → MAE={m_mae:.4f}, sMAPE={m_smape:.2f}%")

    # save outputs
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, f"{name}_lstm_forecasts.csv"), index=False)

    metrics = {
        "Model": "LSTM",
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "sMAPE_mean": float(np.mean(smapes)),
        "sMAPE_std": float(np.std(smapes)),
        "windows": int(windows),
        "pred_len": int(pred_len),
        "context_len": int(context_len),
        "epochs": int(epochs),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "freq": use_freq
    }

    with open(os.path.join(outdir, f"{name}_lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_lstm_results(df_out, name, outdir, windows)

    print("✅ LSTM metrics:", metrics)
    return metrics

# =========================================================
# Plotting last window only (fair)
# =========================================================
def plot_lstm_results(df_predictions: pd.DataFrame, name: str, outdir: str, total_windows: int):
    last_window_idx = total_windows - 1
    window_data = df_predictions[df_predictions["window"] == last_window_idx]

    if len(window_data) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(window_data["timestamp"], window_data["y_true"], label="Ground Truth", linewidth=2)
        plt.plot(window_data["timestamp"], window_data["y_pred"], label="LSTM Prediction", linewidth=2, alpha=0.85)

        plt.title(f"{name} — LSTM Multi-step Baseline (last window)")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}_lstm_last_window.png"), dpi=150)
        plt.close()

        print(f"📊 Plot saved: {outdir}/{name}_lstm_last_window.png")
    else:
        print(f"⚠️ Warning: No data for last window {last_window_idx+1}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    HP = LIGHT_TRAINING_CONFIG["lstm"]

    DATASETS = [
        ("weather_melbourne",
         STANDARD_CONFIG["weather_melbourne"]["csv"],
         STANDARD_CONFIG["weather_melbourne"]["freq"],
         STANDARD_CONFIG["weather_melbourne"]["pred_len"],
         STANDARD_CONFIG["weather_melbourne"]["context_len"]),

        ("finance_aapl",
         STANDARD_CONFIG["finance_aapl"]["csv"],
         STANDARD_CONFIG["finance_aapl"]["freq"],
         STANDARD_CONFIG["finance_aapl"]["pred_len"],
         STANDARD_CONFIG["finance_aapl"]["context_len"]),

        ("co2_maunaloa_monthly",
         STANDARD_CONFIG["co2_maunaloa_monthly"]["csv"],
         STANDARD_CONFIG["co2_maunaloa_monthly"]["freq"],
         STANDARD_CONFIG["co2_maunaloa_monthly"]["pred_len"],
         STANDARD_CONFIG["co2_maunaloa_monthly"]["context_len"]),

        ("etth1",
         STANDARD_CONFIG["etth1"]["csv"],
         STANDARD_CONFIG["etth1"]["freq"],
         STANDARD_CONFIG["etth1"]["pred_len"],
         STANDARD_CONFIG["etth1"]["context_len"]),
    ]

    ensure_dir("results_baseline_lstm")

    all_metrics = []
    for (name, csv, freq, pred_len, context_len) in DATASETS:
        try:
            m = run_one_dataset(
                name=name, csv=csv, freq=freq,
                pred_len=pred_len, context_len=context_len,
                epochs=HP["epochs"],
                batch_size=HP["batch_size"],
                hidden_size=HP["hidden_size"],
                num_layers=HP["num_layers"],
                lr=HP["lr"],
                results_dir="results_baseline_lstm"
            )
            m["dataset"] = name
            all_metrics.append(m)
        except Exception as e:
            print(f"❌ Error on {name}: {e}")

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv("results_baseline_lstm/summary_lstm.csv", index=False)
        print("\n📊 Ringkasan tersimpan di results_baseline_lstm/summary_lstm.csv")
