import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required packages:")
    print("pip install gluonts")
    print("Make sure uni2ts is properly installed in your Python path")
    exit(1)

# ===========================================
# FUNGSI UTILITAS
# ===========================================

def ensure_dir(path):
    """Buat folder jika belum ada"""
    os.makedirs(path, exist_ok=True)
    return path


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, eps=1e-10):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def smape(y_true, y_pred, eps=1e-10):
    """Symmetric Mean Absolute Percentage Error"""
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return np.mean(num / den) * 100


# ===========================================
# LOAD DATA
# ===========================================

def load_data(csv_path, freq):
    """
    Load CSV dengan kolom [timestamp, value]
    Return: PandasDataset dan DataFrame
    """
    print(f"📂 Loading: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'value']).sort_values('timestamp')
    df = df.set_index('timestamp')
    
    print(f"   ✓ Data loaded: {len(df)} rows before frequency adjustment")
    
    # Fix untuk frequency - gunakan string sederhana
    # MS (Month Start) kadang bermasalah, gunakan 'M' atau convert ke 'ME'
    if freq == 'ME':
        freq_use = 'M'  # GluonTS menggunakan 'M' untuk monthly
    elif freq == 'MS':
        freq_use = 'M'  # Convert MS ke M
    else:
        freq_use = freq
    
    # Resample ke frekuensi tertentu & forward fill
    try:
        # Untuk monthly data, jangan gunakan asfreq jika sudah monthly
        if freq_use == 'M' and df.index.freq is None:
            # Check if data is already roughly monthly
            time_diffs = df.index.to_series().diff().dropna()
            avg_diff = time_diffs.mean()
            if pd.Timedelta(days=25) <= avg_diff <= pd.Timedelta(days=35):
                print(f"   ✓ Data already appears monthly, skipping resample")
                # Don't resample, data is already monthly
                pass
            else:
                df = df.asfreq(freq_use)
        elif freq_use != 'M':
            df = df.asfreq(freq_use)
        
        if df.empty:
            print(f"   ⚠️ Warning: DataFrame empty after asfreq with {freq_use}")
            # Fallback: don't resample, just use original frequency
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'value']).sort_values('timestamp')
            df = df.set_index('timestamp')
    except Exception as e:
        print(f"   ⚠️ Warning: asfreq failed with {e}, using original data")
        # Fallback jika freq tidak dikenali
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'value']).sort_values('timestamp')
        df = df.set_index('timestamp')
        freq_use = 'D'
    
    df['value'] = df['value'].ffill()
    df = df.dropna(subset=['value'])
    
    # Buat GluonTS dataset - gunakan freq asli untuk compatibility
    ds = PandasDataset({'value': df['value']}, freq=freq)
    df_reset = df.reset_index()
    
    print(f"   ✓ Total data: {len(df_reset)} rows")
    print(f"   ✓ Range: {df_reset['timestamp'].min().date()} → {df_reset['timestamp'].max().date()}")
    
    return ds, df_reset


# ===========================================
# FEW-SHOT FORECASTING
# ===========================================

def run_fewshot(name, csv_path, pred_len, context_len, freq, n_shots=3):
    """
    Few-shot forecasting:
    - n_shots: jumlah window prediksi yang dilakukan
    - Hanya gunakan data training minimal
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 DATASET: {name.upper()}")
    print(f"{'='*60}")
    
    # Setup output
    outdir = ensure_dir(os.path.join('results_fewshot', name))
    
    # Load data
    ds, df = load_data(csv_path, freq)
    total_len = len(df)
    
    # Hitung test length berdasarkan n_shots
    test_len = pred_len * n_shots
    
    if total_len < context_len + test_len:
        raise ValueError(f"Data terlalu pendek! Need {context_len + test_len}, got {total_len}")
    
    print(f"\n📊 Configuration:")
    print(f"   • Prediction Length: {pred_len}")
    print(f"   • Context Length: {context_len}")
    print(f"   • Few-shot Windows: {n_shots}")
    print(f"   • Test Length: {test_len}")
    print(f"   • Frequency: {freq}")
    
    # Split data
    train, test_template = split(ds, offset=-test_len)
    test_data = test_template.generate_instances(
        prediction_length=pred_len,
        windows=n_shots,
        distance=pred_len
    )
    
    # Load model Moirai (small = ringan untuk laptop)
    print(f"\n🤖 Loading Moirai model...")
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    
    model = Moirai2Forecast(
        module=module,
        prediction_length=pred_len,
        context_length=context_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0
    )
    
    predictor = model.create_predictor(batch_size=8)
    
    # Prediksi
    print(f"\n🔮 Running predictions...")
    forecasts = list(predictor.predict(test_data.input))
    
    # Evaluasi
    all_metrics = []
    all_predictions = []
    
    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    
    for i, forecast in enumerate(forecasts):
        inp = next(input_it)
        label = next(label_it)
        
        # DEBUG: Print object types for first window
        if i == 0:
            print(f"   DEBUG - Label type: {type(label)}")
            if isinstance(label, dict):
                print(f"   DEBUG - Label keys: {list(label.keys())}")
                if 'target' in label:
                    print(f"   DEBUG - Target shape: {np.array(label['target']).shape}")
                    print(f"   DEBUG - Target preview: {label['target'][:5] if len(label['target']) > 5 else label['target']}")
            print(f"   DEBUG - Forecast type: {type(forecast)}")
        
        
        # Ekstrak ground truth dengan konversi ROBUST
        try:
            if isinstance(label, dict):
                # Label adalah dict dengan 'target' key
                if 'target' in label:
                    target_data = label['target']
                    if hasattr(target_data, 'values'):
                        y_true = np.array(target_data.values, dtype=np.float64).flatten()
                    elif hasattr(target_data, 'numpy'):
                        y_true = np.array(target_data.numpy(), dtype=np.float64).flatten()
                    else:
                        y_true = np.array(target_data, dtype=np.float64).flatten()
                else:
                    # Jika tidak ada 'target', ambil first value
                    first_val = list(label.values())[0]
                    if hasattr(first_val, 'values'):
                        y_true = np.array(first_val.values, dtype=np.float64).flatten()
                    else:
                        y_true = np.array(first_val, dtype=np.float64).flatten()
            elif hasattr(label, 'values'):
                # Pandas Series/DataFrame
                y_true = np.array(label.values, dtype=np.float64).flatten()
            elif hasattr(label, 'numpy'):
                # PyTorch tensor
                y_true = np.array(label.numpy(), dtype=np.float64).flatten()
            else:
                # Array biasa - handle Period objects
                if hasattr(label, '__iter__'):
                    y_true = []
                    for x in label:
                        if hasattr(x, 'ordinal'):  # Period object
                            y_true.append(float(x.ordinal))
                        elif hasattr(x, 'value'):  # Timestamp object
                            y_true.append(float(x.value))
                        else:
                            y_true.append(float(x))
                    y_true = np.array(y_true, dtype=np.float64)
                else:
                    y_true = np.array([label], dtype=np.float64).flatten()
        except Exception as e:
            print(f"   ⚠️ Warning extracting y_true: {e}")
            # Ultimate fallback - try to extract numeric values more carefully
            try:
                if isinstance(label, dict) and 'target' in label:
                    # Try to get target array directly
                    target = label['target']
                    if hasattr(target, '__iter__'):
                        y_true = []
                        for x in target:
                            try:
                                y_true.append(float(x))
                            except:
                                y_true.append(0.0)
                        y_true = np.array(y_true, dtype=np.float64)
                    else:
                        y_true = np.array([float(target)], dtype=np.float64)
                else:
                    y_true = np.zeros(pred_len, dtype=np.float64)
            except:
                y_true = np.zeros(pred_len, dtype=np.float64)
        
        # Ekstrak prediksi dengan konversi ROBUST
        try:
            if hasattr(forecast, 'samples'):
                # SampleForecast
                y_pred = np.array(forecast.samples.mean(axis=0), dtype=np.float64).flatten()
            else:
                # QuantileForecast
                q = forecast.quantile(0.5)
                if isinstance(q, dict):
                    first_val = list(q.values())[0]
                    if hasattr(first_val, 'values'):
                        y_pred = np.array(first_val.values, dtype=np.float64).flatten()
                    else:
                        y_pred = np.array(first_val, dtype=np.float64).flatten()
                elif hasattr(q, 'values'):
                    y_pred = np.array(q.values, dtype=np.float64).flatten()
                elif hasattr(q, 'numpy'):
                    y_pred = np.array(q.numpy(), dtype=np.float64).flatten()
                else:
                    # Handle Period objects or other complex types
                    if hasattr(q, '__iter__'):
                        y_pred = []
                        for x in q:
                            if hasattr(x, 'ordinal'):  # Period object
                                y_pred.append(float(x.ordinal))
                            elif hasattr(x, 'value'):  # Timestamp object
                                y_pred.append(float(x.value))
                            else:
                                y_pred.append(float(x))
                        y_pred = np.array(y_pred, dtype=np.float64)
                    else:
                        y_pred = np.array([q], dtype=np.float64).flatten()
        except Exception as e:
            print(f"   ⚠️ Warning extracting y_pred: {e}")
            # Fallback to zeros with same length as y_true
            y_pred = np.zeros_like(y_true) if 'y_true' in locals() else np.zeros(pred_len, dtype=np.float64)
        
        # Pastikan panjang sama
        min_len = min(len(y_true), len(y_pred), pred_len)
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # DEBUG: Print info untuk window pertama
        if i == 0:
            print(f"   DEBUG - Label type: {type(label)}")
            print(f"   DEBUG - y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
            print(f"   DEBUG - y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            print(f"   DEBUG - min_len: {min_len}")
        
        # Hitung metrik
        metrics = {
            'window': i + 1,
            'MAE': float(mae(y_true, y_pred)),
            'RMSE': float(rmse(y_true, y_pred)),
            'MAPE': float(mape(y_true, y_pred)),
            'sMAPE': float(smape(y_true, y_pred))
        }
        all_metrics.append(metrics)
        
        # Simpan prediksi dengan timestamp yang benar
        try:
            # Handle berbagai tipe start_date
            start = forecast.start_date
            if hasattr(start, 'to_timestamp'):
                start_ts = pd.Timestamp(start.to_timestamp())
            elif hasattr(start, 'to_pydatetime'):
                start_ts = pd.Timestamp(start.to_pydatetime())
            elif hasattr(start, 'value'):
                start_ts = pd.Timestamp(start.value)
            elif isinstance(start, str):
                start_ts = pd.Timestamp(start)
            else:
                start_ts = pd.Timestamp(start)
        except Exception as e:
            print(f"   ⚠️ Warning extracting start_date: {e}")
            # Fallback ke timestamp dari test data
            try:
                start_ts = df['timestamp'].iloc[len(df) - test_len + i*pred_len]
            except:
                # Ultimate fallback
                start_ts = pd.Timestamp.now()
        
        # Generate timestamps berdasarkan frequency
        try:
            if freq in ['M', 'ME', 'MS']:  # Monthly
                timestamps = pd.date_range(start=start_ts, periods=min_len, freq='M')
            elif freq == 'D':  # Daily
                timestamps = pd.date_range(start=start_ts, periods=min_len, freq='D')
            elif freq == 'H':  # Hourly
                timestamps = pd.date_range(start=start_ts, periods=min_len, freq='H')
            else:
                timestamps = pd.date_range(start=start_ts, periods=min_len, freq='D')
        except Exception as e:
            print(f"   ⚠️ Warning generating timestamps: {e}")
            # Fallback: gunakan index dari df
            try:
                idx_start = len(df) - test_len + i*pred_len
                idx_end = min(idx_start + min_len, len(df))
                if idx_end > idx_start:
                    timestamps = df['timestamp'].iloc[idx_start:idx_end].values
                else:
                    # Ultimate fallback: generate simple timestamps
                    timestamps = pd.date_range(start=start_ts, periods=min_len, freq='D')
            except Exception as e2:
                print(f"   ⚠️ Final fallback for timestamps: {e2}")
                timestamps = pd.date_range(start=start_ts, periods=min_len, freq='D')
        
        for j in range(min_len):
            all_predictions.append({
                'window': i + 1,
                'timestamp': pd.Timestamp(timestamps[j]),
                'y_true': float(y_true[j]),
                'y_pred': float(y_pred[j]),
                'error': float(abs(y_true[j] - y_pred[j]))
            })
        
        print(f"   Window {i+1}/{n_shots} → MAE: {metrics['MAE']:.4f}, sMAPE: {metrics['sMAPE']:.2f}%")
    
    # Agregat metrik
    df_metrics = pd.DataFrame(all_metrics)
    summary = {
        'dataset': name,
        'prediction_length': pred_len,
        'context_length': context_len,
        'frequency': freq,
        'n_shots': n_shots,
        'MAE_mean': float(df_metrics['MAE'].mean()),
        'MAE_std': float(df_metrics['MAE'].std()),
        'RMSE_mean': float(df_metrics['RMSE'].mean()),
        'RMSE_std': float(df_metrics['RMSE'].std()),
        'MAPE_mean': float(df_metrics['MAPE'].mean()),
        'sMAPE_mean': float(df_metrics['sMAPE'].mean()),
        'sMAPE_std': float(df_metrics['sMAPE'].std())
    }
    
    # Save results
    df_pred = pd.DataFrame(all_predictions)
    df_pred.to_csv(os.path.join(outdir, f'{name}_predictions.csv'), index=False)
    
    with open(os.path.join(outdir, f'{name}_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot
    plot_results(df_pred, name, outdir, n_shots)
    
    print(f"\n📈 Summary Metrics:")
    print(f"   • MAE: {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}")
    print(f"   • RMSE: {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f}")
    print(f"   • sMAPE: {summary['sMAPE_mean']:.2f}% ± {summary['sMAPE_std']:.2f}%")
    print(f"\n💾 Results saved to: {outdir}/")
    
    return summary


# ===========================================
# PLOTTING
# ===========================================

def plot_results(df_pred, name, outdir, n_shots):
    """Plot hasil prediksi vs ground truth"""
    
    fig, axes = plt.subplots(n_shots, 1, figsize=(12, 3*n_shots))
    if n_shots == 1:
        axes = [axes]
    
    for i in range(n_shots):
        window_data = df_pred[df_pred['window'] == i+1]
        
        axes[i].plot(window_data['timestamp'], window_data['y_true'], 
                     label='Ground Truth', marker='o', linewidth=2)
        axes[i].plot(window_data['timestamp'], window_data['y_pred'], 
                     label='Prediction', marker='s', linewidth=2, alpha=0.7)
        
        axes[i].set_title(f'Window {i+1}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Timestamp')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_forecast.png'), dpi=150)
    plt.close()


# ===========================================
# MAIN
# ===========================================

if __name__ == '__main__':
    
    # Konfigurasi dataset
    DATASETS = [    
        {
            'name': 'weather_melbourne',
            'csv': 'data/weather_melbourne/weather_melbourne_full.csv',
            'pred_len': 30,        # Prediksi 7 hari
            'context_len': 365,    # Gunakan 30 hari sebelumnya
            'freq': 'D',          # Daily
            'n_shots': 3          # 3 window prediksi
        },
        {
            'name': 'finance_aapl',
            'csv': 'data/finance_aapl/finance_aapl_full.csv',
            'pred_len': 20,        # Prediksi 5 hari (1 minggu trading)
            'context_len': 250,    # 30 hari konteks
            'freq': 'B',
            'n_shots': 3
        },
        {
            'name': 'co2_maunaloa',
            'csv': 'data/co2_maunaloa_monthly/co2_maunaloa_monthly_full.csv',
            'pred_len': 12,        # Prediksi 6 bulan
            'context_len': 180,    # 3 tahun konteks
            'freq': 'M',          # Monthly (GluonTS standard)
            'n_shots': 3
        }
    ]
    
    # Jalankan untuk semua dataset
    all_results = []
    
    print("\n" + "="*60)
    print("🎯 FEW-SHOT TIME SERIES FORECASTING")
    print("    Using Moirai Universal Transformer")
    print("="*60)
    
    for config in DATASETS:
        try:
            result = run_fewshot(
                name=config['name'],
                csv_path=config['csv'],
                pred_len=config['pred_len'],
                context_len=config['context_len'],
                freq=config['freq'],
                n_shots=config['n_shots']
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error on {config['name']}: {str(e)}")
            continue
    
    # Summary semua hasil
    print("\n" + "="*60)
    print("📊 OVERALL SUMMARY")
    print("="*60)
    
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv('results_fewshot/summary_all.csv', index=False)
    
    for _, row in df_summary.iterrows():
        print(f"\n{row['dataset'].upper()}")
        print(f"  MAE:   {row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}")
        print(f"  RMSE:  {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}")
        print(f"  sMAPE: {row['sMAPE_mean']:.2f}% ± {row['sMAPE_std']:.2f}%")
    
    print(f"\n✅ All done! Check results_fewshot/ folder")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")