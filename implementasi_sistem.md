# 5. IMPLEMENTASI SISTEM FORECASTING

## 5.1 Implementasi Preprocessing Data

Alur preprocessing data time series dimulai dengan membaca dataset dari direktori data lokal dan memuat ke dalam DataFrame pandas. Langkah awal adalah standardisasi format tanggal dan frekuensi data untuk konsistensi, kemudian data dibersihkan dari nilai missing, outlier, dan inconsistency menggunakan berbagai fungsi validasi. Setelah itu, normalisasi dilakukan untuk menyesuaikan skala data agar kompatibel dengan model neural network. Selanjutnya, data dibagi menjadi train, validation, dan test set dengan proporsi yang telah ditentukan. Proses berlanjut dengan windowing untuk membuat sequence data yang sesuai dengan kebutuhan model time series. Semua hasil preprocessing disimpan dalam format yang siap untuk training dan evaluasi. Proses ini memastikan data bersih, konsisten, dan siap untuk diproses oleh berbagai model forecasting.

### Tabel 5.1 Implementasi Preprocessing Data

```python
# utilities/data_loader.py - Lines 1-45
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def load_series(csv_path: str, freq: str) -> Tuple[pd.DataFrame, str]:
    """
    Load dan preprocessing data time series
    
    Args:
        csv_path: Path ke file CSV
        freq: Target frequency (D, M, H, etc.)
    
    Returns:
        Tuple berisi DataFrame dan frequency yang digunakan
    """
    # Membaca dataset dari CSV
    df = pd.read_csv(csv_path)
    
    # Validasi kolom yang diperlukan
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'timestamp' dan 'value'")
    
    # Konversi timestamp ke datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sorting berdasarkan timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Set timestamp sebagai index
    df.set_index('timestamp', inplace=True)
    
    # Handling missing values dengan interpolasi
    df['value'] = df['value'].interpolate(method='linear')
    
    # Remove outliers menggunakan IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['value'] < (Q1 - 1.5 * IQR)) | 
              (df['value'] > (Q3 + 1.5 * IQR)))]
    
    # Resampling ke frequency yang diinginkan
    if freq != 'infer':
        df = df.resample(freq).mean().interpolate()
    
    # Validasi final
    assert not df['value'].isna().any(), "Masih ada missing values"
    
    return df, freq
```

## 5.2 Implementasi Konfigurasi Standardisasi

Sistem menggunakan konfigurasi terpusat untuk memastikan fair comparison antar model. Konfigurasi mencakup parameter dataset, training, dan evaluasi yang seragam.

### Tabel 5.2 Konfigurasi Standardisasi

```python
# light_config.py - Lines 1-60
STANDARD_CONFIG = {
    # Konfigurasi dataset yang konsisten untuk fair comparison
    "weather_melbourne": {
        "csv": "data/weather_melbourne/weather_melbourne_full.csv",
        "pred_len": 7,          # 1 week forecast (standardized)
        "context_len": 30,      # 1 month context (standardized)
        "freq": "D",            # Daily frequency
        "lookback": 30,         # untuk LSTM baseline
        "n_shots": 6,           # untuk few-shot learning (standardized)
        "max_windows": 6        # evaluasi windows (standardized)
    },
    
    "finance_aapl": {
        "csv": "data/finance_aapl/finance_aapl_full.csv",
        "pred_len": 5,          # 1 week trading forecast
        "context_len": 30,      # 1 month context
        "freq": "D",            # Daily frequency (standardized)
        "lookback": 30,         # untuk LSTM baseline
        "n_shots": 6,           # untuk few-shot learning (standardized)
        "max_windows": 6        # evaluasi windows (standardized)
    },
    
    "co2_maunaloa_monthly": {
        "csv": "data/co2_maunaloa_monthly/co2_maunaloa_monthly_full.csv", 
        "pred_len": 6,          # 6 months forecast
        "context_len": 24,      # 2 years context
        "freq": "M",            # Monthly frequency
        "lookback": 24,         # untuk LSTM baseline
        "n_shots": 6,           # untuk few-shot learning (standardized)
        "max_windows": 6        # evaluasi windows (standardized)
    }
}

# Training parameters yang lightweight dan konsisten
LIGHT_TRAINING_CONFIG = {
    "lstm": {
        "epochs": 5,            # reduced untuk efisiensi
        "batch_size": 64,       # optimal untuk memory
        "hidden_size": 32,      # lightweight architecture
        "num_layers": 1,        # simplified model
        "lr": 1e-3             # learning rate
    },
    
    "arima": {
        "fast_mode": True,      # accelerated fitting
        "max_p": 2,             # reduced complexity
        "max_q": 2,             # reduced complexity
        "max_P": 1,             # seasonal parameter
        "max_Q": 1,             # seasonal parameter
        "maxiter": 10           # reduced iterations
    },
    
    "moirai": {
        "batch_size": 8,        # small batch untuk efficiency
        "num_samples": 50       # reduced sampling untuk speed
    }
}
```

## 5.3 Implementasi Model Baseline

### 5.3.1 ARIMA Statistical Baseline

```python
# baseline/baseline_arima.py - Lines 90-120
def rolling_forecast_arima(series: pd.Series, pdt: int, 
                          frac_test: float = 0.15, 
                          fast_mode: bool = True, 
                          max_windows: int = None):
    """
    Rolling window forecasting dengan auto ARIMA
    
    Args:
        series: Time series data
        pdt: Prediction horizon
        frac_test: Fraction untuk test set
        fast_mode: Enable fast mode untuk efficiency
        max_windows: Maximum evaluation windows
    """
    n = len(series)
    
    # Gunakan max_windows dari konfigurasi atau default
    if max_windows is None:
        max_windows = 8 if fast_mode else None
    
    # Hitung test length dengan batasan windows
    TEST = choose_test_len(n, pdt, frac_test, max_windows) if max_windows else choose_test_len(n, pdt, frac_test)
    windows = TEST // pdt

    train = series.iloc[: n - TEST]
    test = series.iloc[n - TEST :]

    # Setup seasonal parameters
    is_seasonal, m = seasonal_setup(series.index.freqstr or "D")
    
    print(f"🏃‍♂️ Fast mode: {fast_mode} | Windows: {windows} | Test points: {TEST}")

    rows, maes, rmses, smapes = [], [], [], []

    # Rolling window prediction
    for w in range(windows):
        print(f"[ARIMA] Processing window {w+1}/{windows}...", end=" ")
        
        # Update training data dengan hasil prediksi sebelumnya
        if w == 0:
            train_extended = train.copy()
        else:
            # Tambahkan actual values dari windows sebelumnya ke training
            prev_actual = test.iloc[w*pdt:(w+1)*pdt]
            train_extended = pd.concat([train_extended, prev_actual])
        
        # Fit ARIMA model
        try:
            model = auto_arima(
                train_extended.values,
                seasonal=is_seasonal,
                m=m if is_seasonal else 1,
                max_p=2, max_q=2, max_P=1, max_Q=1,
                suppress_warnings=True,
                error_action='ignore',
                stepwise=True if fast_mode else False,
                maxiter=10 if fast_mode else 50
            )
            
            # Prediksi untuk window ini
            forecast = model.predict(n_periods=pdt)
            actual = test.iloc[w*pdt:(w+1)*pdt].values
            
            # Hitung metrics
            mae_val = np.mean(np.abs(actual - forecast))
            rmse_val = np.sqrt(np.mean((actual - forecast) ** 2))
            smape_val = 100 * np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast)))
            
            # Simpan hasil
            for i, (f, a) in enumerate(zip(forecast, actual)):
                rows.append({
                    'window': w + 1,
                    'step': i + 1,
                    'actual': a,
                    'forecast': f,
                    'abs_error': abs(a - f)
                })
            
            maes.append(mae_val)
            rmses.append(rmse_val)
            smapes.append(smape_val)
            
            print(f"MAE={mae_val:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
            # Fallback ke naive forecast
            naive_forecast = train_extended.iloc[-1:].values[0]
            forecast = np.full(pdt, naive_forecast)
            # Continue dengan metrics calculation...
    
    # Aggregate results
    metrics = {
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses), 
        "sMAPE": np.mean(smapes),
        "MAE_std": np.std(maes),
        "RMSE_std": np.std(rmses),
        "sMAPE_std": np.std(smapes),
        "windows": windows
    }
    
    return rows, metrics
```

### 5.3.2 LSTM Neural Network Baseline

```python
# baseline/baseline_lstm.py - Lines 150-200
def run_lstm_forecast(name: str, csv: str, freq: str, pdt: int, 
                     lookback: int = 30, epochs: int = 5):
    """
    LSTM forecasting dengan rolling window evaluation
    
    Args:
        name: Dataset name
        csv: Path ke CSV file
        freq: Data frequency
        pdt: Prediction horizon
        lookback: Sequence length untuk LSTM
        epochs: Training epochs
    """
    print(f"🚀 LSTM BASELINE — {name.upper()}")
    
    # Load dan preprocess data
    df, use_freq = load_series(csv, freq)
    series = df["value"].values.astype(np.float32)
    total_len = len(series)
    TEST = choose_test_len(total_len, pdt, frac=0.15)
    windows = TEST // pdt
    
    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pdt  # Sesuaikan TEST dengan windows yang dibatasi

    print(f"🧮 Rows: {total_len} | TEST={TEST} | windows={windows} | lookback={lookback} | horizon={pdt}")

    # Train/test split
    train_vals = series[: total_len - TEST]
    test_vals  = series[total_len - TEST :]

    # Scaling menggunakan StandardScaler untuk stabilitas training
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1)
    test_scaled = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1)
    
    # Prepare sequences untuk LSTM
    X_train, y_train = create_sequences(train_scaled, lookback, pdt)
    
    # Build dan train LSTM model
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, output_size=pdt)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_train), 64):  # batch training
            batch_X = X_train[i:i+64]
            batch_y = y_train[i:i+64]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train):.6f}")
    
    # Rolling window evaluation
    model.eval()
    predictions = []
    actuals = []
    
    for w in range(windows):
        # Siapkan context untuk prediksi window ini
        start_idx = w * pdt
        if w == 0:
            context = train_scaled[-lookback:]
        else:
            # Gunakan kombinasi train + actual dari windows sebelumnya
            prev_actual = test_scaled[:start_idx]
            context = np.concatenate([train_scaled, prev_actual])[-lookback:]
        
        # Prediksi menggunakan LSTM
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0).unsqueeze(-1)
            pred_scaled = model(context_tensor).squeeze().numpy()
        
        # Inverse scaling
        pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        actual_original = series[total_len - TEST + start_idx:total_len - TEST + start_idx + pdt]
        
        predictions.extend(pred_original)
        actuals.extend(actual_original)
        
        # Hitung metrics untuk window ini
        mae_w = np.mean(np.abs(actual_original - pred_original))
        smape_w = 100 * np.mean(2 * np.abs(actual_original - pred_original) / 
                               (np.abs(actual_original) + np.abs(pred_original)))
        
        print(f"[LSTM] window {w+1}/{windows} → MAE={mae_w:.4f}, sMAPE={smape_w:.2f}%")
    
    # Calculate overall metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    metrics = {
        "MAE": np.mean(np.abs(actuals - predictions)),
        "RMSE": np.sqrt(np.mean((actuals - predictions) ** 2)),
        "sMAPE": 100 * np.mean(2 * np.abs(actuals - predictions) / 
                              (np.abs(actuals) + np.abs(predictions))),
        "windows": int(windows),
        "total_predictions": len(predictions)
    }
    
    return metrics
```

## 5.4 Implementasi Model Moirai

### 5.4.1 Zero-shot Moirai Universal Transformer

```python
# moirai/run_zeroshot_all.py - Lines 160-220
def run_zeroshot_forecast(name, csv_path, pred_len, context_len, freq):
    """
    Zero-shot forecasting dengan Moirai Universal Transformer
    
    Args:
        name: Dataset identifier
        csv_path: Path ke data CSV
        pred_len: Prediction horizon
        context_len: Context window length
        freq: Data frequency
    """
    print(f"🚀 ZERO-SHOT MOIRAI — {name.upper()}")
    
    # Load data dengan handling frequency
    try:
        ds, df = load_univariate_with_freq(csv_path, target_freq=freq)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    total_len = len(df)
    TEST = choose_test_len(total_len, pred_len, frac=0.15)
    windows = TEST // pred_len
    
    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pred_len
    
    print(f"\n📊 Configuration:")
    print(f"   • Prediction Length: {pred_len}")
    print(f"   • Context Length: {context_len}")
    print(f"   • Test Length: {TEST}")
    print(f"   • Windows: {windows}")
    print(f"   • Frequency: {freq}")
    
    if windows == 0:
        print("❌ No test windows available")
        return
    
    # Initialize Moirai model
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-1.0-R-large"),
        prediction_length=pred_len,
        context_length=context_len,
        patch_size="auto",
        num_samples=50,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    
    # Split dataset untuk evaluation
    train_ds, test_template = split(ds, offset=-TEST)
    
    # Rolling window evaluation
    all_metrics = []
    all_forecasts = []
    
    for w in range(windows):
        print(f"\n🔮 Window {w+1}/{windows}")
        
        # Update dataset untuk window ini
        current_offset = -TEST + (w * pred_len)
        if w == 0:
            current_train_ds = train_ds
        else:
            # Extend training dengan actual values dari windows sebelumnya
            extended_train_ds, _ = split(ds, offset=current_offset)
            current_train_ds = extended_train_ds
        
        # Generate predictions
        forecasts = model.forecast(current_train_ds, num_samples=50)
        forecast_list = list(forecasts)
        
        if len(forecast_list) == 0:
            print(f"⚠️ No forecasts generated for window {w+1}")
            continue
        
        # Extract prediction dan actual values
        forecast_item = forecast_list[0]  # Ambil forecast pertama (single series)
        pred_mean = forecast_item.mean
        pred_samples = forecast_item.samples
        
        # Get actual values untuk window ini
        actual_start = total_len - TEST + (w * pred_len)
        actual_end = actual_start + pred_len
        actual_values = df.iloc[actual_start:actual_end]['value'].values
        
        # Ensure same length
        min_len = min(len(pred_mean), len(actual_values))
        pred_mean = pred_mean[:min_len]
        actual_values = actual_values[:min_len]
        
        # Calculate metrics untuk window ini
        mae_val = np.mean(np.abs(actual_values - pred_mean))
        rmse_val = np.sqrt(np.mean((actual_values - pred_mean) ** 2))
        smape_val = 100 * np.mean(2 * np.abs(actual_values - pred_mean) / 
                                 (np.abs(actual_values) + np.abs(pred_mean)))
        
        all_metrics.append({
            'window': w + 1,
            'MAE': mae_val,
            'RMSE': rmse_val,
            'sMAPE': smape_val
        })
        
        # Store forecasts untuk analisis
        for i in range(min_len):
            all_forecasts.append({
                'window': w + 1,
                'step': i + 1,
                'actual': actual_values[i],
                'forecast': pred_mean[i],
                'abs_error': abs(actual_values[i] - pred_mean[i])
            })
        
        print(f"   Window {w+1}/{windows} → MAE: {mae_val:.4f}, sMAPE: {smape_val:.2f}%")
    
    # Aggregate metrics across all windows
    if all_metrics:
        mae_values = [m['MAE'] for m in all_metrics]
        rmse_values = [m['RMSE'] for m in all_metrics]
        smape_values = [m['sMAPE'] for m in all_metrics]
        
        final_metrics = {
            'dataset': name,
            'model': 'Zero-shot',
            'MAE': np.mean(mae_values),
            'RMSE': np.mean(rmse_values),
            'sMAPE': np.mean(smape_values),
            'MAE_std': np.std(mae_values),
            'RMSE_std': np.std(rmse_values),
            'sMAPE_std': np.std(smape_values),
            'windows': windows,
            'total_predictions': len(all_forecasts)
        }
        
        return final_metrics
    else:
        print("❌ No successful predictions")
        return None
```

### 5.4.2 Few-shot MoE (Mixture of Experts)

```python
# moirai/run_fewshot_moe.py - Lines 100-180
def run_fewshot_moe(
    name: str,
    csv_path: str, 
    pred_len: int,
    context_len: int,
    freq: str,
    n_shots: int = 6,
) -> dict:
    """
    Few-shot forecasting dengan Moirai Mixture of Experts
    
    Args:
        name: Dataset identifier
        csv_path: Path ke data CSV
        pred_len: Prediction horizon
        context_len: Context length
        freq: Data frequency  
        n_shots: Number of few-shot examples
    """
    print(f"🚀 FEW-SHOT MOE — {name.upper()}")
    
    # Load dan preprocess data
    ds, df = load_univariate_with_freq(csv_path, target_freq=freq)
    
    # Few-shot setup: gunakan n_shots window di akhir untuk test
    test_len = pred_len * n_shots  # few-shot: ambil n_shots window di ekor
    total_len = len(df)
    
    print(f"\n📊 Configuration:")
    print(f"   • pred_len: {pred_len}")
    print(f"   • context_len: {context_len}")
    print(f"   • freq: {freq}")
    print(f"   • n_shots: {n_shots}  (test_len={test_len})")
    print(f"   • total_len: {total_len}")
    
    # Initialize Moirai-MoE model
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
        prediction_length=pred_len,
        context_length=context_len,
        patch_size="auto",
        num_samples=50,  # Reduced untuk efficiency
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        windows=n_shots,
    )
    
    print(f"🤖 Memuat Moirai-MoE (Salesforce/moirai-moe-1.0-R-small)...")
    
    # Split data: train sampai -test_len, test = test_len terakhir
    train_ds, test_template = split(ds, offset=-test_len)
    
    print(f"\n🔮 Menjalankan prediksi...")
    
    # Few-shot evaluation dengan rolling window
    maes, rmses, smapes = [], [], []
    all_predictions = []
    
    for i in range(n_shots):
        print(f"   Window {i+1}/{n_shots}", end="")
        
        # Update training data untuk include actual values dari windows sebelumnya
        current_offset = -test_len + (i * pred_len)
        if i == 0:
            current_train_ds = train_ds
        else:
            # Extend training dengan actual dari windows sebelumnya
            extended_train_ds, _ = split(ds, offset=current_offset)
            current_train_ds = extended_train_ds
        
        # Generate forecast dengan MoE
        forecasts = model.forecast(current_train_ds, num_samples=50)
        forecast_list = list(forecasts)
        
        if len(forecast_list) > 0:
            # Extract MoE prediction (ensemble dari multiple experts)
            forecast_item = forecast_list[0]
            pred_mean = forecast_item.mean  # Averaged prediction dari semua experts
            
            # Get actual values
            actual_start = total_len - test_len + (i * pred_len)
            actual_end = actual_start + pred_len
            actual_values = df.iloc[actual_start:actual_end]['value'].values
            
            # Ensure same length
            min_len = min(len(pred_mean), len(actual_values))
            pred_mean = pred_mean[:min_len]
            actual_values = actual_values[:min_len]
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_values - pred_mean))
            rmse = np.sqrt(np.mean((actual_values - pred_mean) ** 2))
            smape = 100 * np.mean(2 * np.abs(actual_values - pred_mean) / 
                                 (np.abs(actual_values) + np.abs(pred_mean)))
            
            maes.append(mae)
            rmses.append(rmse)
            smapes.append(smape)
            
            # Store detailed predictions
            for j in range(min_len):
                all_predictions.append({
                    'window': i + 1,
                    'step': j + 1,
                    'actual': actual_values[j],
                    'forecast': pred_mean[j],
                    'abs_error': abs(actual_values[j] - pred_mean[j])
                })
            
            print(f" → MAE: {mae:.4f}, sMAPE: {smape:.2f}%")
        else:
            print(" → No forecast generated")
    
    # Calculate summary metrics
    if maes:
        summary_metrics = {
            'dataset': name,
            'model': 'Few-shot MoE',
            'MAE': np.mean(maes),
            'RMSE': np.mean(rmses),
            'sMAPE': np.mean(smapes),
            'MAE_std': np.std(maes),
            'RMSE_std': np.std(rmses),
            'sMAPE_std': np.std(smapes),
            'n_shots': n_shots,
            'windows': n_shots,
            'total_predictions': len(all_predictions)
        }
        
        print(f"\n📈 Summary Metrics (MoE few-shot):")
        print(f"   • MAE: {summary_metrics['MAE']:.4f} ± {summary_metrics['MAE_std']:.4f}")
        print(f"   • RMSE: {summary_metrics['RMSE']:.4f} ± {summary_metrics['RMSE_std']:.4f}")
        print(f"   • sMAPE: {summary_metrics['sMAPE']:.2f}% ± {summary_metrics['sMAPE_std']:.2f}%")
        
        return summary_metrics
    else:
        print("❌ No successful predictions")
        return None
```

## 5.5 Implementasi Sistem Evaluasi Unified

```python
# run_all_standardized.py - Lines 80-140
def main():
    """
    Main function untuk menjalankan semua model dengan konfigurasi standardisasi
    """
    print("STANDARDIZED MODEL COMPARISON")
    print("Konfigurasi seragam dan ringan untuk fair comparison")
    print("="*60)
    
    # Tampilkan summary konfigurasi
    show_config_summary()
    
    # Konfirmasi sebelum eksekusi
    print(f"\n Jalankan semua model dengan konfigurasi ini? [y/N]: ", end="")
    confirm = input().lower().strip()
    
    if confirm not in ['y', 'yes']:
        print(" Dibatalkan oleh user")
        return
    
    # Daftar model untuk evaluasi standardisasi
    models = [
        {
            "name": "ARIMA Baseline",
            "command": "python baseline_arima.py",
            "description": "Running ARIMA statistical baseline model"
        },
        {
            "name": "LSTM Baseline", 
            "command": "python baseline_lstm.py",
            "description": "Running LSTM neural network baseline model"
        },
        {
            "name": "Zero-shot Moirai",
            "command": "python run_zeroshot_all.py",
            "description": "Running Zero-shot Moirai universal transformer"
        },
        {
            "name": "Few-shot MoE",
            "command": "python run_fewshot_moe.py", 
            "description": "Running Few-shot Moirai Mixture of Experts"
        }
    ]
    
    # Track eksekusi dan results
    results = []
    total_start_time = time.time()
    
    # Eksekusi sequential untuk setiap model
    for i, model in enumerate(models, 1):
        print(f"\n  RUNNING MODEL {i}/{len(models)}: {model['name']}")
        
        # Jalankan model dengan environment setup
        success = run_command(model['command'], model['description'])
        
        results.append({
            'model': model['name'],
            'success': success,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if success:
            print(f"[SUCCESS] {model['name']} completed successfully")
        else:
            print(f"[FAILED] {model['name']} failed")
            print("Continuing with next model...")
    
    # Generate summary report
    total_duration = time.time() - total_start_time
    successful = len([r for r in results if r['success']])
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    print(f"Successful: {successful}/{len(models)} models")
    print(f"Failed: {len(models) - successful}/{len(models)} models")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "[SUCCESS]" if result['success'] else "[FAILED]"
        print(f"  {result['model']:<20} | {status} | {result['timestamp']}")
    
    # Suggest next steps
    if successful == len(models):
        print(f"\n[SUCCESS] ALL MODELS COMPLETED SUCCESSFULLY!")
        print(f"Ready for standardized comparison analysis")
        print(f"\nNEXT STEPS:")
        print(f"  1. Run: python analysis_standardized_results.py")
        print(f"  2. Check individual model result directories")
        print(f"  3. Generate comparison tables and plots")
    
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
```

Proses implementasi ini memastikan preprocessing data yang konsisten, konfigurasi standardisasi yang fair, dan evaluasi yang seragam untuk semua model, menghasilkan perbandingan yang valid antara ARIMA, LSTM, Zero-shot Moirai, dan Few-shot MoE.