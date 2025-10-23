import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== dependencies ======
try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Pastikan paket berikut terpasang:")
    print("  pip install gluonts")
    print("  pip install uni2ts   (atau sudah pip install -e '.[notebook]' di repo uni2ts)")
    raise

# ====== utils & metrics ======
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

def load_series(csv_path: str, freq: str):
    """
    Membaca CSV [timestamp,value] → kembalikan PandasDataset & DataFrame reguler.
    Menormalkan index ke freq yang diminta & forward-fill celah.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    # Normalisasi frekuensi
    if freq in ("M", "MS", "ME"):
        # GluonTS paling stabil dengan 'M'. Jika index bulanan lain, konversi ke akhir bulan.
        df.index = df.index.to_period("M").to_timestamp("M")
        use_freq = "M"
    elif freq == "B":
        # Business day deprecated, gunakan 'D' dengan filter weekdays
        print(f"   ⚠️ Warning: Business day frequency 'B' deprecated, using 'D'")
        use_freq = "D"
    else:
        use_freq = freq

    df = df.asfreq(use_freq)
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"])

    ds = PandasDataset({"value": df["value"]}, freq=use_freq)
    return ds, df.reset_index(), use_freq

def choose_test_len(total_len: int, pdt: int, frac: float = 0.15) -> int:
    raw = max(pdt, int(total_len * frac))
    test = (raw // pdt) * pdt
    return test if test >= pdt else pdt

def safe_to_float_array(values_dict_or_arr):
    """
    Ekstraksi nilai numerik robust dari label/prediksi GluonTS:
    dict(label) dengan key 'target' / array-like → np.ndarray float
    """
    if isinstance(values_dict_or_arr, dict) and "target" in values_dict_or_arr:
        arr = values_dict_or_arr["target"]
        return np.asarray(arr, dtype=np.float64).flatten()
    if isinstance(values_dict_or_arr, dict):
        vals = []
        for _, v in sorted(values_dict_or_arr.items(), key=lambda kv: kv[0]):
            if isinstance(v, (list, np.ndarray)):
                if len(v) > 0:
                    vals.append(float(np.ravel(v)[0]))
            else:
                vals.append(float(v))
        return np.array(vals, dtype=float).flatten()
    # array-like biasa
    return np.asarray(values_dict_or_arr, dtype=np.float64).flatten()

# ====== core few-shot MoE ======
def run_fewshot_moe(
    name: str,
    csv_path: str,
    pred_len: int,
    context_len: int,
    freq: str,
    n_shots: int = 3,
    batch_size: int = 16,
    results_dir: str = "results_fewshot_moe",
):
    print("\n" + "=" * 60)
    print(f"🚀 DATASET (MoE Few-shot): {name.upper()}")
    print("=" * 60)
    outdir = ensure_dir(os.path.join(results_dir, name))

    # Load data & dataset
    ds, df, use_freq = load_series(csv_path, freq=freq)
    total_len = len(df)
    test_len = pred_len * n_shots  # few-shot: ambil n_shots window di ekor
    if total_len < context_len + test_len:
        raise ValueError(
            f"Data terlalu pendek. Butuh ≥ {context_len + test_len}, ada {total_len} baris."
        )

    print(f"📂 Range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print("📊 Konfigurasi:")
    print(f"   • freq: {use_freq}")
    print(f"   • pred_len: {pred_len}")
    print(f"   • context_len: {context_len}")
    print(f"   • n_shots: {n_shots}  (test_len={test_len})")

    # Split & buat rolling windows few-shot
    train, test_template = split(ds, offset=-test_len)
    test_data = test_template.generate_instances(
        prediction_length=pred_len,
        windows=n_shots,
        distance=pred_len,  # non-overlap
    )

    # Siapkan model Moirai-MoE (pralatih)
    print("\n🤖 Memuat Moirai-MoE (Salesforce/moirai-moe-1.0-R-small)...")
    module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small")
    model = MoiraiMoEForecast(
        module=module,
        prediction_length=pred_len,
        context_length=context_len,
        patch_size=16,
        num_samples=LIGHT_TRAINING_CONFIG["moirai"]["num_samples"],  # ringan untuk uncertainty
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size)

    print("\n🔮 Menjalankan prediksi...")
    forecasts = list(predictor.predict(test_data.input))
    input_it = iter(test_data.input)
    label_it = iter(test_data.label)

    rows = []
    maes, rmses, smapes = [], [], []
    last_window_payload = None  # untuk plot

    for i, fc in enumerate(forecasts):
        inp = next(input_it)
        label = next(label_it)

        # ground truth
        y_true = safe_to_float_array(label)[:pred_len]

        # point forecast & uncertainty
        if hasattr(fc, "samples") and fc.samples is not None:
            samples = np.asarray(fc.samples, dtype=np.float64)  # [num_samples, pred_len]
            q10 = np.quantile(samples, 0.10, axis=0)
            q50 = np.quantile(samples, 0.50, axis=0)
            q90 = np.quantile(samples, 0.90, axis=0)
        else:
            # fallback ke quantile API
            q10 = safe_to_float_array(fc.quantile(0.1))[:pred_len]
            q50 = safe_to_float_array(fc.quantile(0.5))[:pred_len]
            q90 = safe_to_float_array(fc.quantile(0.9))[:pred_len]

        # samakan panjang
        min_len = min(len(y_true), len(q50), pred_len)
        y_true, q10, q50, q90 = y_true[:min_len], q10[:min_len], q50[:min_len], q90[:min_len]

        # timestamp window ini
        try:
            start = fc.start_date
            if hasattr(start, "to_timestamp"):
                start_ts = pd.Timestamp(start.to_timestamp())
            else:
                start_ts = pd.Timestamp(start)
        except Exception:
            # Fallback jika ada masalah dengan start_date
            start_ts = pd.Timestamp.now()
        
        try:
            ts_idx = pd.date_range(start=start_ts, periods=min_len, freq=use_freq)
        except Exception:
            # Fallback ke daily frequency
            ts_idx = pd.date_range(start=start_ts, periods=min_len, freq='D')

        # metrik (pakai median q50 sbg point-forecast)
        maes.append(mae(y_true, q50))
        rmses.append(rmse(y_true, q50))
        smapes.append(smape(y_true, q50))

        for t, yt, p10, p50, p90 in zip(ts_idx, y_true, q10, q50, q90):
            rows.append(
                {
                    "window": int(i + 1),
                    "timestamp": pd.Timestamp(t),
                    "y_true": float(yt),
                    "y_pred_p10": float(p10),
                    "y_pred_p50": float(p50),
                    "y_pred_p90": float(p90),
                }
            )

        print(f"   Window {i+1}/{n_shots} → MAE: {maes[-1]:.4f}, sMAPE: {smapes[-1]:.2f}%")

        last_window_payload = (ts_idx, y_true, q10, q50, q90)

    # Simpan CSV & metrics
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, f"{name}_moe_predictions.csv"), index=False)

    metrics = {
        "dataset": name,
        "freq": use_freq,
        "prediction_length": int(pred_len),
        "context_length": int(context_len),
        "n_shots": int(n_shots),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "sMAPE_mean": float(np.mean(smapes)),
        "sMAPE_std": float(np.std(smapes)),
    }
    with open(os.path.join(outdir, f"{name}_moe_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n📈 Summary Metrics (MoE few-shot):")
    print(
        f"   • MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}\n"
        f"   • RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}\n"
        f"   • sMAPE: {metrics['sMAPE_mean']:.2f}% ± {metrics['sMAPE_std']:.2f}%"
    )

    # Plot window terakhir dengan fan chart
    if last_window_payload is not None:
        ts_idx, y_true, q10, q50, q90 = last_window_payload
        plt.figure(figsize=(10, 4))
        plt.plot(ts_idx, y_true, label="Ground Truth", linewidth=2)
        plt.plot(ts_idx, q50, label="Median (q50)", linewidth=2)
        plt.fill_between(ts_idx, q10, q90, alpha=0.2, label="Uncertainty (q10–q90)")
        plt.title(f"{name} — Moirai-MoE Few-shot (last window)")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}_moe_last_window.png"), dpi=150)
        plt.close()

    print(f"\n💾 Disimpan ke: {outdir}")
    return metrics


# ====== konfigurasi dataset ======
DATASETS = [
    {
        "name": "weather_melbourne",
        "csv": STANDARD_CONFIG["weather_melbourne"]["csv"],
        "pred_len": STANDARD_CONFIG["weather_melbourne"]["pred_len"],
        "context_len": STANDARD_CONFIG["weather_melbourne"]["context_len"],
        "freq": STANDARD_CONFIG["weather_melbourne"]["freq"],
        "n_shots": STANDARD_CONFIG["weather_melbourne"]["n_shots"],
    },
    {
        "name": "finance_aapl",
        "csv": STANDARD_CONFIG["finance_aapl"]["csv"],
        "pred_len": STANDARD_CONFIG["finance_aapl"]["pred_len"],
        "context_len": STANDARD_CONFIG["finance_aapl"]["context_len"],
        "freq": STANDARD_CONFIG["finance_aapl"]["freq"],
        "n_shots": STANDARD_CONFIG["finance_aapl"]["n_shots"],
    },
    {
        "name": "co2_maunaloa_monthly",
        "csv": STANDARD_CONFIG["co2_maunaloa_monthly"]["csv"],
        "pred_len": STANDARD_CONFIG["co2_maunaloa_monthly"]["pred_len"],
        "context_len": STANDARD_CONFIG["co2_maunaloa_monthly"]["context_len"],
        "freq": STANDARD_CONFIG["co2_maunaloa_monthly"]["freq"],
        "n_shots": STANDARD_CONFIG["co2_maunaloa_monthly"]["n_shots"],
    },
]

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 FEW-SHOT TIME SERIES FORECASTING — Moirai-MoE")
    print("=" * 60)

    ensure_dir("results_fewshot_moe")
    all_results = []
    for cfg in DATASETS:
        try:
            res = run_fewshot_moe(
                name=cfg["name"],
                csv_path=cfg["csv"],
                pred_len=cfg["pred_len"],
                context_len=cfg["context_len"],
                freq=cfg["freq"],
                n_shots=cfg["n_shots"],
                batch_size=LIGHT_TRAINING_CONFIG["moirai"]["batch_size"],
                results_dir="results_fewshot_moe",
            )
            all_results.append(res)
        except Exception as e:
            print(f"❌ Error on {cfg['name']}: {e}")

    # gabungkan ringkasan
    if all_results:
        df_sum = pd.DataFrame(all_results)
        df_sum.to_csv("results_fewshot_moe/summary_all_moe.csv", index=False)
        print("\n📊 Ringkasan seluruh dataset tersimpan di results_fewshot_moe/summary_all_moe.csv")
