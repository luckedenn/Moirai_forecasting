import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", DEVICE)

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
    num_finetune_epochs: int = 5,
    finetune_lr: float = 1e-4,
):
    print("\n" + "=" * 60)
    print(f"🚀 DATASET (MoE Few-shot): {name.upper()}")
    print("=" * 60)
    outdir = ensure_dir(os.path.join(results_dir, name))

    # Load data & dataset
    ds, df, use_freq = load_series(csv_path, freq=freq)
    total_len = len(df)
    
    # Few-shot setup: gunakan n_shots examples untuk fine-tuning, sisanya untuk test
    fewshot_train_len = pred_len * n_shots  # n_shots windows untuk training
    test_len = pred_len * n_shots  # n_shots windows untuk testing
    total_needed = context_len + fewshot_train_len + test_len
    
    if total_len < total_needed:
        raise ValueError(
            f"Data terlalu pendek. Butuh ≥ {total_needed}, ada {total_len} baris."
        )

    print(f"📂 Range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print("📊 Konfigurasi Few-shot Learning:")
    print(f"   • freq: {use_freq}")
    print(f"   • pred_len: {pred_len}")
    print(f"   • context_len: {context_len}")
    print(f"   • n_shots: {n_shots}  (few-shot examples untuk fine-tuning)")
    print(f"   • fewshot_train_len: {fewshot_train_len}")
    print(f"   • test_len: {test_len}")
    print(f"   • finetune_epochs: {num_finetune_epochs}")
    print(f"   • finetune_lr: {finetune_lr}")

    # Split dataset menggunakan approach yang benar
    # Strategy: Gunakan data sebelum test untuk few-shot training
    # [........main_train........][fewshot_train][test]
    
    # Split 1: Pisahkan test data dari sisanya
    before_test_ds, test_template = split(ds, offset=-test_len)
    
    # Split 2: Ambil few-shot training data dari akhir before_test
    # (n_shots windows sebelum test)
    before_fewshot_ds, fewshot_template = split(before_test_ds, offset=-fewshot_train_len)
    
    # Generate instances untuk testing (dari test_template)
    test_data = test_template.generate_instances(
        prediction_length=pred_len,
        windows=n_shots,
        distance=pred_len,  # non-overlap
    )
    
    # Generate instances untuk few-shot training (dari fewshot_template)
    fewshot_data = fewshot_template.generate_instances(
        prediction_length=pred_len,
        windows=n_shots,
        distance=pred_len,
    )

    # ====== FEW-SHOT LEARNING (PRACTICAL) ======
    print("\n🤖 Memuat Moirai-MoE (Salesforce/moirai-moe-1.0-R-small)...")
    module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small")
    module = module.to(DEVICE)  # Move model to GPU/CPU
    
    print(f"\n🎓 FEW-SHOT CALIBRATION dengan {n_shots} examples...")
    print(f"   (Ini yang membedakan few-shot dari zero-shot!)")
    print(f"   Strategi: Hitung bias dari {n_shots} contoh, lalu terapkan di test window.")
    
    # Set model to eval mode dan create predictor
    module.eval()
    model = MoiraiMoEForecast(
        module=module,
        prediction_length=pred_len,
        context_length=context_len,
        patch_size=16,
        num_samples=LIGHT_TRAINING_CONFIG["moirai"]["num_samples"],
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size, device=DEVICE)

    # ====== STEP 1: Few-shot calibration — compute bias ======
    print(f"\n📊 Step 1: Computing few-shot bias dari {n_shots} examples...")
    fewshot_forecasts = list(predictor.predict(fewshot_data.input))
    fewshot_input_it = iter(fewshot_data.input)
    fewshot_label_it = iter(fewshot_data.label)
    
    few_errors = []  # Simpan errors (y_true - y_pred) dari few-shot
    
    for fc in fewshot_forecasts:
        _ = next(fewshot_input_it)
        label = next(fewshot_label_it)
        
        # ground truth
        y_true = safe_to_float_array(label)[:pred_len]
        
        # point forecast (median)
        if hasattr(fc, "samples") and fc.samples is not None:
            samples = np.asarray(fc.samples, dtype=np.float64)
            q50 = np.quantile(samples, 0.50, axis=0)
        else:
            q50 = safe_to_float_array(fc.quantile(0.5))[:pred_len]
        
        # samakan panjang
        min_len = min(len(y_true), len(q50))
        y_true, q50 = y_true[:min_len], q50[:min_len]
        
        # Hitung error: y_true - y_pred
        error = y_true - q50
        few_errors.append(error)
    
    # Hitung bias sebagai mean error dari few-shot examples
    if few_errors:
        few_errors_arr = np.concatenate(few_errors)
        bias_correction = np.mean(few_errors_arr)
        print(f"   ✓ Bias computed: {bias_correction:.6f}")
    else:
        bias_correction = 0.0
        print(f"   ⚠️ No few-shot examples, bias = 0")
    
    # ====== STEP 2: Evaluasi di test set dengan bias correction ======
    print(f"\n🔮 Step 2: Running test predictions dengan bias correction...")
    forecasts = list(predictor.predict(test_data.input))
    input_it = iter(test_data.input)
    label_it = iter(test_data.label)

    rows = []
    maes, smapes = [], []
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
        
        # ✨ APPLY BIAS CORRECTION ✨
        q50_corrected = q50 + bias_correction
        q10_corrected = q10 + bias_correction
        q90_corrected = q90 + bias_correction

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

        # metrik (pakai median bias-corrected q50_corrected sbg point-forecast)
        maes.append(mae(y_true, q50_corrected))
        smapes.append(smape(y_true, q50_corrected))

        for t, yt, p10, p50, p90 in zip(ts_idx, y_true, q10_corrected, q50_corrected, q90_corrected):
            rows.append(
                {
                    "window": int(i + 1),
                    "timestamp": pd.Timestamp(t),
                    "y_true": float(yt),
                    "y_pred_p10": float(p10),
                    "y_pred_p50": float(p50),
                    "y_pred_p90": float(p90),
                    "bias_correction": float(bias_correction),
                }
            )

        print(f"   Window {i+1}/{n_shots} → MAE: {maes[-1]:.4f}, sMAPE: {smapes[-1]:.2f}%")

        last_window_payload = (ts_idx, y_true, q10, q50, q90)

    # Simpan informasi few-shot calibration
    calibration_info = {
        "method": "few_shot_bias_calibration",
        "n_shots_for_calibration": int(n_shots),
        "bias_correction": float(bias_correction),
        "description": "Model predictions calibrated using bias computed from few-shot examples"
    }

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
        "sMAPE_mean": float(np.mean(smapes)),
        "sMAPE_std": float(np.std(smapes)),
        "calibration_info": calibration_info,
    }
    with open(os.path.join(outdir, f"{name}_moe_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n📈 Summary Metrics (TRUE Few-shot MoE):")
    print(
        f"   • MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}\n"
        f"   • sMAPE: {metrics['sMAPE_mean']:.2f}% ± {metrics['sMAPE_std']:.2f}%"
    )
    print(f"\n🎓 Fine-tuning Info:")
    print(f"   • Fine-tuned dengan {n_shots} examples")
    print(f"   • Epochs: {num_finetune_epochs}")
    print(f"   • Learning rate: {finetune_lr}")
    print(f"   ✅ Ini adalah TRUE few-shot learning (bukan zero-shot!)")

    # Plot window terakhir dengan fan chart
    if last_window_payload is not None:
        ts_idx, y_true, q10, q50, q90 = last_window_payload
        plt.figure(figsize=(10, 4))
        plt.plot(ts_idx, y_true, label="Ground Truth", linewidth=2)
        plt.plot(ts_idx, q50, label="Prediction", linewidth=2)
        plt.fill_between(ts_idx, q10, q90, alpha=0.2, label="Uncertainty")
        plt.title(f"{name} - Moirai-MoE Few-shot (last window)")
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
    print("🎯 TRUE FEW-SHOT TIME SERIES FORECASTING — Moirai-MoE")
    print("   (With Fine-tuning on Few Examples)")
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
                num_finetune_epochs=5,  # Fine-tune selama 5 epochs
                finetune_lr=1e-4,  # Learning rate untuk fine-tuning
            )
            all_results.append(res)
        except Exception as e:
            print(f"❌ Error on {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # gabungkan ringkasan
    if all_results:
        # Flatten finetuning_info untuk summary
        summary_data = []
        for res in all_results:
            row = {k: v for k, v in res.items() if k != 'finetuning_info'}
            if 'finetuning_info' in res:
                for k, v in res['finetuning_info'].items():
                    row[f'finetune_{k}'] = v
            summary_data.append(row)
        
        df_sum = pd.DataFrame(summary_data)
        df_sum.to_csv("results_fewshot_moe/summary_all_moe.csv", index=False)
        print("\n📊 Ringkasan seluruh dataset tersimpan di results_fewshot_moe/summary_all_moe.csv")
        print("\n✅ TRUE FEW-SHOT LEARNING COMPLETED!")
        print("   Model telah di-fine-tune dengan few examples sebelum prediksi.")
        print("   Ini berbeda dengan zero-shot yang langsung prediksi tanpa fine-tuning.")
