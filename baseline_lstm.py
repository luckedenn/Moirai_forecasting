# baseline_lstm.py
import os
import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================
# Utils
# =========================
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
    """
    Membaca CSV [timestamp,value], set index & frekuensi, ffill missing.
    Mengembalikan (pd.DataFrame berindex waktu, DatetimeIndex frekuensi fix).
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    # Normalisasi frekuensi:
    # - Bulanan: pakai akhir-bulan agar stabil ('M')
    # - AAPL: pakai 'B' (business day)
    if freq in ("M", "MS", "ME"):
        df.index = df.index.to_period("M").to_timestamp("M")
        use_freq = "M"
    else:
        use_freq = freq

    df = df.asfreq(use_freq)
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"])
    return df, use_freq

# =========================
# Dataset untuk LSTM (seq->next)
# =========================
class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, lookback: int):
        self.series = series.astype(np.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.series) - self.lookback

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.lookback]           # (lookback,)
        y = self.series[idx + self.lookback]                 # scalar (next step)
        return torch.from_numpy(x).unsqueeze(-1).float(), torch.tensor([y], dtype=torch.float32)

# =========================
# LSTM model
# =========================
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.rnn(x)
        # ambil hidden terakhir
        h_last = out[:, -1, :]           # (B, H)
        y = self.head(h_last)            # (B, 1)
        return y

# =========================
# Training loop
# =========================
def train_lstm(train_arr: np.ndarray, lookback: int, epochs: int = 10, batch_size: int = 64,
               hidden_size: int = 64, num_layers: int = 2, lr: float = 1e-3, device: str = "cpu"):
    ds = SeqDataset(train_arr, lookback)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMForecast(1, hidden_size, num_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        losses = []
        for xb, yb in dl:
            xb = xb.float().to(device)   # (B, T, 1) - ensure float32
            yb = yb.float().to(device)   # (B, 1) - ensure float32
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"[LSTM] epoch {ep+1}/{epochs} - loss: {np.mean(losses):.6f}")

    model.eval()
    return model

# =========================
# Autoregressive rollout
# =========================
@torch.no_grad()
def rollout_forecast(model: nn.Module, last_context: np.ndarray, horizon: int, device: str = "cpu") -> np.ndarray:
    """
    last_context: array shape (lookback,), sudah dalam skala yang SAMA dengan data latihan (scaled).
    Return: prediksi (horizon,) dalam skala YANG SAMA dengan input (scaled).
    """
    ctx = last_context.astype(np.float32).copy()
    preds = []
    for _ in range(horizon):
        x = torch.from_numpy(ctx[-len(last_context):]).unsqueeze(0).unsqueeze(-1).float().to(device)  # (1, T, 1)
        yhat = model(x).squeeze().item()  # scalar
        preds.append(yhat)
        ctx = np.append(ctx, yhat)
    return np.array(preds, dtype=np.float32)

# =========================
# Runner per dataset
# =========================
def run_one_dataset(
    name: str,
    csv: str,
    freq: str,
    pdt: int,
    lookback: int,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    results_dir: str = "results_baseline_lstm",
    device: str = None,
):
    print("\n" + "=" * 60)
    print(f"🚀 LSTM BASELINE — {name.upper()}")
    print("=" * 60)

    outdir = ensure_dir(os.path.join(results_dir, name))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # load & split
    df, use_freq = load_series(csv, freq)
    series = df["value"].values.astype(np.float32)
    total_len = len(series)
    TEST = choose_test_len(total_len, pdt, frac=0.15)
    windows = TEST // pdt
    
    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pdt  # Sesuaikan TEST dengan windows yang dibatasi

    print(f"📂 Range: {df.index.min().date()} → {df.index.max().date()}  | freq={use_freq}")
    print(f"🧮 Rows: {total_len} | TEST={TEST} | windows={windows} | lookback={lookback} | horizon={pdt}")

    # train/test split
    train_vals = series[: total_len - TEST]
    test_vals  = series[total_len - TEST :]

    # scaling pakai train saja (StandardScaler untuk performa lebih baik)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1)
    test_scaled  = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1)

    # pastikan length cukup
    if len(train_scaled) <= lookback:
        raise ValueError(f"Train length ({len(train_scaled)}) harus > lookback ({lookback}). Kurangi lookback / tambah data.")

    # train model (sekali)
    model = train_lstm(
        train_arr=train_scaled,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lr=lr,
        device=device,
    )

    rows = []
    maes, rmses, smapes = [], [], []

    # rolling evaluation di test split
    # test window w: gunakan konteks dari ujung train + hasil prediksi windows sebelumnya
    # namun agar fair & sederhana, untuk setiap window kita ambil konteks asli dari data gabungan (train+test),
    # bukan hasil prediksi window sebelumnya (jadi tiap window independen).
    full_scaled = np.concatenate([train_scaled, test_scaled])  # skala konsisten

    for w in range(windows):
        # posisi awal window pada full_series (scaled)
        start_idx = len(train_scaled) + w * pdt
        end_idx = start_idx + pdt

        # ambil konteks lookback tepat sebelum start_idx
        ctx_start = start_idx - lookback
        if ctx_start < 0:
            raise ValueError("Lookback terlalu besar untuk posisi window pertama.")
        context = full_scaled[ctx_start:start_idx]  # shape: (lookback,)

        # rollout autoregressive selama horizon pdt
        pred_scaled = rollout_forecast(model, context, horizon=pdt, device=device)
        # inverse scale
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

        # ground truth asli window ini (di domain asli)
        y_true = series[start_idx:end_idx].astype(np.float32)

        # metrik
        m_mae = mae(y_true, pred)
        m_rmse = rmse(y_true, pred)
        m_smape = smape(y_true, pred)
        maes.append(m_mae); rmses.append(m_rmse); smapes.append(m_smape)

        # timestamp index
        ts_idx = pd.date_range(start=df.index[start_idx], periods=pdt, freq=use_freq)

        for t, yt, yp in zip(ts_idx, y_true, pred):
            rows.append({"window": w, "timestamp": pd.Timestamp(t), "y_true": float(yt), "y_pred": float(yp)})

        print(f"[LSTM] window {w+1}/{windows} → MAE={m_mae:.4f}, sMAPE={m_smape:.2f}%")

    # simpan hasil
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
        "PDT": int(pdt),
        "lookback": int(lookback),
        "epochs": int(epochs),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "freq": use_freq,
    }
    with open(os.path.join(outdir, f"{name}_lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot hasil prediksi vs ground truth untuk beberapa windows
    plot_lstm_results(df_out, name, outdir, windows)

    print("LSTM metrics:", metrics)
    return metrics


# =========================
# Plotting function
# =========================
def plot_lstm_results(df_predictions: pd.DataFrame, name: str, outdir: str, total_windows: int):
    """
    Plot hasil prediksi LSTM vs ground truth
    """
    # Plot hingga maksimal 4 windows untuk clarity
    n_plots = min(4, total_windows)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for i in range(n_plots):
        window_data = df_predictions[df_predictions['window'] == i]
        
        if len(window_data) > 0:
            axes[i].plot(window_data['timestamp'], window_data['y_true'], 
                        label='Ground Truth', marker='o', linewidth=2, markersize=4)
            axes[i].plot(window_data['timestamp'], window_data['y_pred'], 
                        label='LSTM Prediction', marker='s', linewidth=2, markersize=4, alpha=0.8)
            
            # Hitung MAE untuk window ini
            mae_window = np.mean(np.abs(window_data['y_true'] - window_data['y_pred']))
            
            axes[i].set_title(f'LSTM Forecast - Window {i+1} (MAE: {mae_window:.4f})', 
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Timestamp')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Rotate x-axis labels jika banyak
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'LSTM Baseline Results - {name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_lstm_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {outdir}/{name}_lstm_forecast.png")


# =========================
# Main – jalankan untuk 3 dataset
# =========================
if __name__ == "__main__":
    # Import konfigurasi ringan dari light_config
    from light_config import LIGHT_TRAINING_CONFIG
    
    # Hyperparameter ringan untuk perbandingan fair
    HP = LIGHT_TRAINING_CONFIG["lstm"]

    # Import konfigurasi ringan
    import sys
    sys.path.append('..')
    from light_config import STANDARD_CONFIG
    
    DATASETS = [
        # (name, csv_path, freq, PDT(horizon), lookback(context))
        ("weather_melbourne", STANDARD_CONFIG["weather_melbourne"]["csv"], 
         STANDARD_CONFIG["weather_melbourne"]["freq"], STANDARD_CONFIG["weather_melbourne"]["pred_len"], 
         STANDARD_CONFIG["weather_melbourne"]["lookback"]),
        ("finance_aapl", STANDARD_CONFIG["finance_aapl"]["csv"], 
         STANDARD_CONFIG["finance_aapl"]["freq"], STANDARD_CONFIG["finance_aapl"]["pred_len"], 
         STANDARD_CONFIG["finance_aapl"]["lookback"]),
        ("co2_maunaloa_monthly", STANDARD_CONFIG["co2_maunaloa_monthly"]["csv"], 
         STANDARD_CONFIG["co2_maunaloa_monthly"]["freq"], STANDARD_CONFIG["co2_maunaloa_monthly"]["pred_len"], 
         STANDARD_CONFIG["co2_maunaloa_monthly"]["lookback"]),
    ]

    ensure_dir("results_baseline_lstm")

    all_metrics = []
    for (name, csv, freq, pdt, lookback) in DATASETS:
        try:
            m = run_one_dataset(
                name=name, csv=csv, freq=freq, pdt=pdt, lookback=lookback,
                epochs=HP["epochs"], batch_size=HP["batch_size"], hidden_size=HP["hidden_size"],
                num_layers=HP["num_layers"], lr=HP["lr"], results_dir="results_baseline_lstm"
            )
            m["dataset"] = name
            all_metrics.append(m)
        except Exception as e:
            print(f"❌ Error on {name}: {e}")

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv("results_baseline_lstm/summary_lstm.csv", index=False)
        print("\n📊 Ringkasan tersimpan di results_baseline_lstm/summary_lstm.csv")
