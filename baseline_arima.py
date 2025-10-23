# baseline_arima.py
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from pmdarima import auto_arima
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

warnings.filterwarnings("ignore")


# ============== Utils & Metrics ==============
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def mae(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def smape(y, yhat, eps: float = 1e-8) -> float:
    num = np.abs(yhat - y)
    den = (np.abs(y) + np.abs(yhat) + eps) / 2.0
    return float(np.mean(num / den) * 100.0)

def choose_test_len(total_len: int, pdt: int, frac: float = 0.15, max_windows: int = 10) -> int:
    """
    Pilih test length dengan batasan maksimal windows untuk mempercepat
    """
    raw = max(pdt, int(total_len * frac))
    test = (raw // pdt) * pdt
    
    # Batasi maksimal windows untuk mempercepat
    max_test = max_windows * pdt
    test = min(test, max_test)
    
    return test if test >= pdt else pdt


# ============== Data Loading ==============
def load_series(csv_path: str, freq: str) -> Tuple[pd.Series, str]:
    """
    Baca CSV [timestamp,value], set index waktu & frekuensi stabil, ffill.
    Kembalikan (series, use_freq)
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    # Normalisasi frekuensi: bulanan ke akhir bulan untuk stabilitas
    if freq in ("M", "MS", "ME"):
        df.index = df.index.to_period("M").to_timestamp("M")
        use_freq = "M"
    elif freq == "B":
        # Business day deprecated, gunakan daily
        use_freq = "D"
        print(f"   ⚠️ Warning: Business day frequency 'B' deprecated, using 'D'")
    else:
        use_freq = freq

    df = df.asfreq(use_freq)
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"])
    return df["value"], use_freq


# ============== Musiman Otomatis ==============
def seasonal_setup(use_freq: str) -> Tuple[bool, int]:
    """
    Tentukan apakah seasonal dan periode musiman (m) berdasarkan frekuensi.
    - M (bulanan): m=12
    - D (harian): m=7 (mingguan) - tapi bisa disable untuk kecepatan
    - B (hari kerja): m=5 (pekan kerja)
    """
    if use_freq == "M":
        return True, 12
    if use_freq == "D":
        return False, 1  # disable seasonal untuk daily data (lebih cepat)
    if use_freq == "B":
        return False, 1  # disable seasonal untuk business day (lebih cepat)
    # default: non-seasonal
    return False, 1


# ============== Rolling Forecast dengan auto_arima (Fast Version) ==============
def rolling_forecast_arima(series: pd.Series, pdt: int, frac_test: float = 0.15, fast_mode: bool = True, max_windows: int = None):
    n = len(series)
    
    # Gunakan max_windows dari parameter atau default 8 jika fast_mode
    if max_windows is None:
        max_windows = 8 if fast_mode else None
    TEST = choose_test_len(n, pdt, frac_test, max_windows) if max_windows else choose_test_len(n, pdt, frac_test)
    windows = TEST // pdt

    train = series.iloc[: n - TEST]
    test = series.iloc[n - TEST :]

    is_seasonal, m = seasonal_setup(series.index.freqstr or "D")
    
    print(f"🏃‍♂️ Fast mode: {fast_mode} | Windows: {windows} | Test points: {TEST}")

    rows, maes, rmses, smapes = [], [], [], []

    for w in range(windows):
        print(f"[ARIMA] Processing window {w+1}/{windows}...", end=" ")
        
        # Gunakan semua data sampai awal window sebagai histori
        end = len(train) + w * pdt
        hist = series.iloc[:end]

        # Fit ARIMA pada histori (optimasi untuk kecepatan)
        # Import parameter ringan
        from light_config import LIGHT_TRAINING_CONFIG
        arima_config = LIGHT_TRAINING_CONFIG["arima"]
        
        model = auto_arima(
            hist,
            start_p=0, start_q=0,      # mulai dari yang sederhana
            max_p=arima_config["max_p"], max_q=arima_config["max_q"],  # lebih ringan
            start_P=0, start_Q=0,
            max_P=arima_config["max_P"], max_Q=arima_config["max_Q"],  # lebih ringan
            seasonal=is_seasonal,
            m=m,
            d=None, D=None,            # auto
            test="adf",                # uji stasioneritas
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,             # gunakan stepwise (lebih cepat)
            maxiter=arima_config["maxiter"],  # lebih ringan
            information_criterion="aic",
            n_jobs=-1,                 # parallel processing
            with_intercept=True,
        )

        # Forecast horizon pdt
        fc = model.predict(n_periods=pdt)
        y_true = test.iloc[w * pdt : (w + 1) * pdt].values
        y_pred = np.asarray(fc, dtype=float)

        # Metrik
        m_mae = mae(y_true, y_pred)
        m_rmse = rmse(y_true, y_pred)
        m_smape = smape(y_true, y_pred)
        maes.append(m_mae); rmses.append(m_rmse); smapes.append(m_smape)

        # Timestamp untuk window ini
        idx = test.index[w * pdt : (w + 1) * pdt]
        for t, yt, yp in zip(idx, y_true, y_pred):
            rows.append(
                {"window": int(w), "timestamp": pd.Timestamp(t), "y_true": float(yt), "y_pred": float(yp)}
            )

        print(f"MAE={m_mae:.3f}  sMAPE={m_smape:.2f}%")

    metrics = {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "sMAPE_mean": float(np.mean(smapes)),
        "sMAPE_std": float(np.std(smapes)),
        "windows": int(windows),
        "PDT": int(pdt),
        "seasonal": bool(is_seasonal),
        "m": int(m),
        "fast_mode": bool(fast_mode),
    }
    return rows, metrics


def plot_arima_results(df_predictions: pd.DataFrame, name: str, outdir: str):
    """
    Plot hasil prediksi ARIMA untuk beberapa windows terbaik
    """
    # Hitung metrik untuk setiap window
    windows = df_predictions['window'].unique()
    window_metrics = []
    
    for w in windows:
        window_data = df_predictions[df_predictions['window'] == w]
        w_mae = mae(window_data['y_true'].values, window_data['y_pred'].values)
        window_metrics.append((w, w_mae))
    
    # Sort by MAE dan ambil 4 windows terbaik
    window_metrics.sort(key=lambda x: x[1])
    best_windows = [w for w, _ in window_metrics[:4]]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, window_idx in enumerate(best_windows):
        if i >= 4:
            break
            
        window_data = df_predictions[df_predictions['window'] == window_idx].copy()
        window_data = window_data.sort_values('timestamp')
        
        if len(window_data) > 0:
            axes[i].plot(window_data['timestamp'], window_data['y_true'], 
                        'o-', label='Ground Truth', linewidth=2, markersize=6)
            axes[i].plot(window_data['timestamp'], window_data['y_pred'], 
                        's--', label='ARIMA Pred', linewidth=2, markersize=6, alpha=0.8)
            
            # Hitung metrik untuk window ini
            w_mae = mae(window_data['y_true'].values, window_data['y_pred'].values)
            w_smape = smape(window_data['y_true'].values, window_data['y_pred'].values)
            
            axes[i].set_title(f'Window {window_idx+1} (MAE: {w_mae:.3f}, sMAPE: {w_smape:.1f}%)', 
                            fontweight='bold')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(best_windows), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f'ARIMA Baseline Results - {name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_arima_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {outdir}/{name}_arima_forecast.png")


# ============== Runner per Dataset ==============
def run_arima(name: str, csv: str, freq: str, pdt: int, outdir_root: str = "results_baseline_arima", fast_mode: bool = True):
    outdir = ensure_dir(os.path.join(outdir_root, name))
    series, use_freq = load_series(csv, freq)

    print("\n" + "=" * 60)
    print(f"🚀 ARIMA BASELINE — {name.upper()}")
    print("=" * 60)
    print(f"📂 Range: {series.index.min().date()} → {series.index.max().date()} | freq={use_freq}")
    print(f"🔧 Horizon (PDT): {pdt} | Fast mode: {fast_mode}")

    # Gunakan max_windows dari konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", 8)
    rows, metrics = rolling_forecast_arima(series, pdt=pdt, frac_test=0.15, fast_mode=fast_mode, max_windows=max_windows_config)

    # Simpan hasil
    df_predictions = pd.DataFrame(rows)
    df_predictions.to_csv(os.path.join(outdir, f"{name}_arima_forecasts.csv"), index=False)
    with open(os.path.join(outdir, f"{name}_arima_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot hasil
    plot_arima_results(df_predictions, name, outdir)

    print("📈 ARIMA metrics:", metrics)
    return {"dataset": name, **metrics}


# ============== Main ==============
if __name__ == "__main__":
    # Import konfigurasi ringan
    from light_config import STANDARD_CONFIG
    
    # (name, csv_path, freq, PDT)
    DATASETS = [
        ("weather_melbourne", STANDARD_CONFIG["weather_melbourne"]["csv"], 
         STANDARD_CONFIG["weather_melbourne"]["freq"], STANDARD_CONFIG["weather_melbourne"]["pred_len"]),
        ("finance_aapl", STANDARD_CONFIG["finance_aapl"]["csv"], 
         STANDARD_CONFIG["finance_aapl"]["freq"], STANDARD_CONFIG["finance_aapl"]["pred_len"]),
        ("co2_maunaloa_monthly", STANDARD_CONFIG["co2_maunaloa_monthly"]["csv"], 
         STANDARD_CONFIG["co2_maunaloa_monthly"]["freq"], STANDARD_CONFIG["co2_maunaloa_monthly"]["pred_len"]),
    ]

    ensure_dir("results_baseline_arima")
    all_results = []
    for name, csv, freq, pdt in DATASETS:
        try:
            res = run_arima(name, csv, freq, pdt)
            all_results.append(res)
        except Exception as e:
            print(f"❌ Error on {name}: {e}")

    if all_results:
        pd.DataFrame(all_results).to_csv("results_baseline_arima/summary_arima.csv", index=False)
        print("\n📊 Ringkasan → results_baseline_arima/summary_arima.csv")
