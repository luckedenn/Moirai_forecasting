# run_zeroshot_all.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from uni2ts.eval_util.plot import plot_single
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required packages:")
    print("pip install gluonts")
    print("Make sure uni2ts is properly installed in your Python path")
    exit(1)


# -------------------------
# Utils & metrics
# -------------------------
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


def load_univariate_with_freq(csv_path: str, target_freq: str):
    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    df = df.set_index("timestamp")
    
    # Handle frequency conversion
    try:
        # For monthly data, check if we need to handle it specially
        if target_freq in ['M', 'MS', 'ME']:
            # Check if data is already monthly
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq and inferred_freq.startswith('M'):
                # Data is already monthly, don't resample
                print(f"   ✓ Data already monthly (freq: {inferred_freq})")
            else:
                # Try to resample to monthly
                df = df.resample('M').mean()
        else:
            df = df.asfreq(target_freq)
            
        if df.empty:
            print(f"   ⚠️ Warning: DataFrame empty after frequency conversion with {target_freq}")
            # Fallback: reload and don't resample
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
            df = df.set_index("timestamp")
            print(f"   ✓ Using original data without resampling")
    except Exception as e:
        print(f"   ⚠️ Warning: frequency conversion failed with {e}, using original data")
        # Fallback if freq not recognized
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        df = df.set_index("timestamp")
    
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"])

    ds = PandasDataset({"value": df["value"]}, freq=target_freq)
    df_reset = df.reset_index()
    
    print(f"   ✓ Total data: {len(df_reset)} rows")
    print(f"   ✓ Range: {df_reset['timestamp'].min().date()} → {df_reset['timestamp'].max().date()}")
    
    return ds, df_reset


def choose_test_len(total_len: int, pdt: int, frac: float = 0.15) -> int:
    raw = max(pdt, int(total_len * frac))
    test = (raw // pdt) * pdt
    return test if test >= pdt else pdt


def safe_to_float_array(values_dict_or_arr):
    """
    Handle dict/list/array of values → flat numpy array of floats.
    Improved to handle various data types including Period objects.
    """
    try:
        if isinstance(values_dict_or_arr, dict):
            # Check if it's a label dict with 'target' key
            if 'target' in values_dict_or_arr:
                target_data = values_dict_or_arr['target']
                if hasattr(target_data, 'values'):
                    return np.array(target_data.values, dtype=np.float64).flatten()
                elif hasattr(target_data, 'numpy'):
                    return np.array(target_data.numpy(), dtype=np.float64).flatten()
                else:
                    return np.array(target_data, dtype=np.float64).flatten()
            else:
                # Original dict handling
                items = sorted(values_dict_or_arr.items(), key=lambda kv: kv[0])
                vals = []
                for _, v in items:
                    if isinstance(v, (list, np.ndarray)):
                        if len(v) > 0:
                            vals.append(float(np.ravel(v)[0]))
                    else:
                        vals.append(float(v))
                return np.array(vals, dtype=float).flatten()
        else:
            # Handle arrays, lists, etc.
            if hasattr(values_dict_or_arr, '__iter__') and not isinstance(values_dict_or_arr, str):
                vals = []
                for x in values_dict_or_arr:
                    try:
                        if hasattr(x, 'ordinal'):  # Period object
                            vals.append(float(x.ordinal))
                        elif hasattr(x, 'value'):  # Timestamp object
                            vals.append(float(x.value))
                        else:
                            vals.append(float(x))
                    except:
                        vals.append(0.0)  # Fallback
                return np.array(vals, dtype=np.float64)
            else:
                arr = np.asarray(values_dict_or_arr, dtype=float).flatten()
                return arr
    except Exception as e:
        print(f"   ⚠️ Warning in safe_to_float_array: {e}")
        # Ultimate fallback
        try:
            return np.array([float(x) for x in values_dict_or_arr if x is not None], dtype=np.float64)
        except:
            return np.array([0.0], dtype=np.float64)


def run_one_dataset(
    name: str,
    csv_path: str,
    pdt: int,
    ctx: int,
    freq: str,
    bsz: int = 32,
    results_dir: str = "results",
):
    print(f"\n{'='*60}")
    print(f"🚀 DATASET: {name.upper()}")
    print(f"{'='*60}")
    
    outdir = ensure_dir(os.path.join(results_dir, name))
    
    try:
        ds, df = load_univariate_with_freq(csv_path, target_freq=freq)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    total_len = len(df)
    TEST = choose_test_len(total_len, pdt, frac=0.15)
    windows = TEST // pdt
    
    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pdt  # Sesuaikan TEST dengan windows yang dibatasi
    
    print(f"\n📊 Configuration:")
    print(f"   • Prediction Length: {pdt}")
    print(f"   • Context Length: {ctx}")
    print(f"   • Test Length: {TEST}")
    print(f"   • Windows: {windows}")
    print(f"   • Frequency: {freq}")
    print(f"   • Total rows: {total_len}")
    
    if windows == 0:
        raise ValueError(f"TEST ({TEST}) terlalu kecil untuk PDT ({pdt}).")

    train, test_template = split(ds, offset=-TEST)
    test_data = test_template.generate_instances(
        prediction_length=pdt,
        windows=windows,
        distance=pdt,
    )

    print(f"\n🤖 Loading Moirai model...")
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    model = Moirai2Forecast(
        module=module,
        prediction_length=pdt,
        context_length=ctx,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=bsz)
    
    print(f"\n🔮 Running predictions...")
    forecasts = predictor.predict(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)

    rows, maes, rmses, smapes = [], [], [], []
    last_inp, last_label, last_forecast = None, None, None

    def make_ts_index(start_ts: pd.Timestamp, periods: int, freq_str: str):
        try:
            return pd.date_range(start=start_ts, periods=periods, freq=freq_str)
        except Exception:
            return pd.date_range(start=start_ts, periods=periods, freq="D")

    for w in range(windows):
        inp = next(input_it)
        label = next(label_it)
        forecast = next(forecast_it)

        # Debug untuk window pertama
        if w == 0:
            print(f"   DEBUG - Label type: {type(label)}")
            if isinstance(label, dict):
                print(f"   DEBUG - Label keys: {list(label.keys())}")
                if 'target' in label:
                    print(f"   DEBUG - Target shape: {np.array(label['target']).shape}")

        # --- Ekstrak y_true ---
        y_true = safe_to_float_array(label)

        # --- Ambil point-forecast robust ---
        if hasattr(forecast, "samples"):  # SampleForecast
            y_pred = np.asarray(forecast.samples.mean(axis=0), dtype=float).flatten()
        else:  # QuantileForecast
            y_pred = safe_to_float_array(forecast.quantile(0.5))

        # --- Samakan panjang ---
        min_len = min(len(y_true), len(y_pred), pdt)
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        last_inp, last_label, last_forecast = inp, label, forecast

        # Handle start_date extraction
        try:
            start_date = forecast.start_date
            if hasattr(start_date, "to_timestamp"):
                start_ts = start_date.to_timestamp()
            elif hasattr(start_date, 'to_pydatetime'):
                start_ts = pd.Timestamp(start_date.to_pydatetime())
            else:
                start_ts = pd.Timestamp(start_date)
        except Exception as e:
            print(f"   ⚠️ Warning extracting start_date: {e}")
            # Fallback
            start_ts = df['timestamp'].iloc[len(df) - TEST + w*pdt]
        
        ts_idx = make_ts_index(start_ts, min_len, freq)

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        
        maes.append(mae_val)
        rmses.append(rmse_val)
        smapes.append(smape_val)

        for t, yt, yp in zip(ts_idx, y_true, y_pred):
            rows.append({"window": int(w), "timestamp": pd.Timestamp(t), "y_true": float(yt), "y_pred": float(yp)})
        
        print(f"   Window {w+1}/{windows} → MAE: {mae_val:.4f}, sMAPE: {smape_val:.2f}%")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, f"{name}_forecasts.csv"), index=False)

    metrics = {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "sMAPE_mean": float(np.mean(smapes)),
        "sMAPE_std": float(np.std(smapes)),
        "windows": int(windows),
        "PDT": int(pdt),
        "CTX": int(ctx),
        "freq": freq,
    }
    with open(os.path.join(outdir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📈 Summary Metrics:")
    print(f"   • MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}")
    print(f"   • RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}")
    print(f"   • sMAPE: {metrics['sMAPE_mean']:.2f}% ± {metrics['sMAPE_std']:.2f}%")

    # Plot last window
    try:
        plt.figure()
        plot_single(
            last_inp,
            last_label,
            last_forecast,
            context_length=min(ctx, len(last_inp)),
            name=f"{name}_last_window",
            show_label=True,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}_last_window.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Warning: Could not create plot: {e}")
    
    print(f"\n💾 Results saved to: {outdir}/")
    
    return metrics


# -------------------------
# Konfigurasi dataset
# -------------------------
# Import konfigurasi ringan
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG

DATASETS = [
    {
        "name": "weather_melbourne",
        "csv": STANDARD_CONFIG["weather_melbourne"]["csv"],
        "PDT": STANDARD_CONFIG["weather_melbourne"]["pred_len"],
        "CTX": STANDARD_CONFIG["weather_melbourne"]["context_len"],
        "FREQ": STANDARD_CONFIG["weather_melbourne"]["freq"],
    },
    {
        "name": "finance_aapl", 
        "csv": STANDARD_CONFIG["finance_aapl"]["csv"],
        "PDT": STANDARD_CONFIG["finance_aapl"]["pred_len"],
        "CTX": STANDARD_CONFIG["finance_aapl"]["context_len"],
        "FREQ": STANDARD_CONFIG["finance_aapl"]["freq"],
    },
    {
        "name": "co2_maunaloa_monthly",
        "csv": STANDARD_CONFIG["co2_maunaloa_monthly"]["csv"],
        "PDT": STANDARD_CONFIG["co2_maunaloa_monthly"]["pred_len"],
        "CTX": STANDARD_CONFIG["co2_maunaloa_monthly"]["context_len"],
        "FREQ": STANDARD_CONFIG["co2_maunaloa_monthly"]["freq"],
    },
]


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 ZERO-SHOT TIME SERIES FORECASTING")
    print("    Using Moirai Universal Transformer")
    print("="*60)
    
    ensure_dir("results_zeroshot")
    all_results = []
    
    for cfg in DATASETS:
        try:
            result = run_one_dataset(
                name=cfg["name"],
                csv_path=cfg["csv"],
                pdt=cfg["PDT"],
                ctx=cfg["CTX"],
                freq=cfg["FREQ"],
                bsz=LIGHT_TRAINING_CONFIG["moirai"]["batch_size"],  # batch size ringan
                results_dir="results_zeroshot",
            )
            if result:
                all_results.append({**result, 'dataset': cfg["name"]})
        except Exception as e:
            print(f"\n❌ Error on {cfg['name']}: {str(e)}")
            continue
    
    # Summary semua hasil
    print("\n" + "="*60)
    print("📊 OVERALL SUMMARY")
    print("="*60)
    
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv('results_zeroshot/summary_all_zeroshot.csv', index=False)
        
        for result in all_results:
            print(f"\n{result['dataset'].upper()}")
            print(f"  MAE:   {result['MAE_mean']:.4f} ± {result['MAE_std']:.4f}")
            print(f"  RMSE:  {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}")
            print(f"  sMAPE: {result['sMAPE_mean']:.2f}% ± {result['sMAPE_std']:.2f}%")
            print(f"  Windows: {result['windows']}")
    
    print(f"\n✅ All done! Check results_zeroshot/ folder")
    from datetime import datetime
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
