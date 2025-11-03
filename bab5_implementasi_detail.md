# BAB 5: IMPLEMENTASI SISTEM FORECASTING

## 5.1 Implementasi Data Preparation

Alur preprocessing data time series dimulai dengan membaca dataset dari direktori data lokal menggunakan `prepare_dataset.py` dan memuat ke dalam DataFrame pandas. Langkah awal adalah standardisasi format tanggal dan frekuensi data untuk konsistensi, kemudian data dibersihkan dari nilai missing, outlier, dan inconsistency menggunakan berbagai fungsi validasi. Setelah itu, data dibagi menjadi train, validation, dan test set dengan proporsi yang telah ditentukan berdasarkan urutan waktu. Proses ini memastikan data bersih, konsisten, dan siap untuk diproses oleh berbagai model forecasting.

### Tabel 5.1 Implementasi Data Preparation

```python
# prepare_dataset.py - Data Preparation
import os
import io
import pandas as pd
import requests
from datetime import datetime

def time_split(df, ts_col, frac_train=0.70, frac_val=0.15):
    """Split by time order: 70% train, 15% val, 15% test."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

def save_splits(df, domain_name):
    outdir = os.path.join(DATA_DIR, domain_name)
    os.makedirs(outdir, exist_ok=True)
    train, val, test = time_split(df, "timestamp")
    df.to_csv(os.path.join(outdir, f"{domain_name}_full.csv"), index=False)
    train.to_csv(os.path.join(outdir, f"{domain_name}_train.csv"), index=False)
    val.to_csv(os.path.join(outdir, f"{domain_name}_val.csv"), index=False)
    test.to_csv(os.path.join(outdir, f"{domain_name}_test.csv"), index=False)
```

**Penjelasan dari kode program implementasi data preparation pada Tabel 5.1:**

1. **Baris 1-6** berfungsi untuk mengimpor library yang diperlukan:

   - `os` untuk operasi sistem file dan direktori
   - `io` untuk operasi input/output
   - `pandas` untuk manipulasi dan analisis data
   - `requests` untuk mengambil data dari web API
   - `datetime` untuk handling format tanggal dan waktu

2. **Baris 8** berfungsi untuk mendefinisikan fungsi `time_split` yang melakukan pembagian dataset berdasarkan urutan waktu dengan proporsi 70% training, 15% validation, dan 15% testing.

3. **Baris 10** berfungsi untuk mengurutkan DataFrame berdasarkan kolom timestamp (`ts_col`) dan reset index untuk memastikan data terurut secara kronologis.

4. **Baris 11-13** berfungsi untuk menghitung jumlah data untuk setiap split:

   - `n`: total jumlah data
   - `n_train`: jumlah data training (70% dari total)
   - `n_val`: jumlah data validation (15% dari total)

5. **Baris 14-16** berfungsi untuk membagi dataset menjadi tiga bagian:

   - `train`: data dari awal hingga n_train
   - `val`: data dari n_train hingga n_train+n_val
   - `test`: data sisanya untuk testing

6. **Baris 17** berfungsi untuk mengembalikan ketiga split dataset sebagai tuple.

7. **Baris 19** berfungsi untuk mendefinisikan fungsi `save_splits` yang menyimpan hasil pembagian dataset ke dalam file CSV terpisah.

8. **Baris 20-22** berfungsi untuk membuat direktori output jika belum ada dan memanggil fungsi `time_split` untuk membagi data berdasarkan timestamp.

9. **Baris 23-26** berfungsi untuk menyimpan setiap split dataset ke file CSV:
   - `_full.csv`: dataset lengkap
   - `_train.csv`: data training
   - `_val.csv`: data validation
   - `_test.csv`: data testing

## 5.2 Implementasi ARIMA Baseline

ARIMA (AutoRegressive Integrated Moving Average) diimplementasikan sebagai baseline statistical dengan automatic parameter selection dan rolling window evaluation untuk memastikan evaluasi yang fair.

### Tabel 5.2 Implementasi ARIMA Forecasting

```python
# baseline_arima.py - ARIMA Model Implementation
def rolling_forecast_arima(series: pd.Series, pdt: int,
                          frac_test: float = 0.15,
                          fast_mode: bool = True,
                          max_windows: int = None):
    n = len(series)

    # Gunakan max_windows dari konfigurasi atau default
    if max_windows is None:
        max_windows = 8 if fast_mode else None

    # Hitung test length dengan batasan windows
    TEST = choose_test_len(n, pdt, frac_test, max_windows)
    windows = TEST // pdt

    train = series.iloc[: n - TEST]
    test = series.iloc[n - TEST :]

    # Setup seasonal parameters
    is_seasonal, m = seasonal_setup(series.index.freqstr or "D")

    rows, maes, rmses, smapes = [], [], [], []

    # Rolling window prediction
    for w in range(windows):
        if w == 0:
            train_extended = train.copy()
        else:
            prev_actual = test.iloc[w*pdt:(w+1)*pdt]
            train_extended = pd.concat([train_extended, prev_actual])

        # Fit ARIMA model
        model = auto_arima(
            train_extended.values,
            seasonal=is_seasonal,
            m=m if is_seasonal else 1,
            max_p=2, max_q=2, max_P=1, max_Q=1,
            suppress_warnings=True,
            stepwise=True if fast_mode else False,
            maxiter=10 if fast_mode else 50
        )

        # Prediksi untuk window ini
        forecast = model.predict(n_periods=pdt)
        actual = test.iloc[w*pdt:(w+1)*pdt].values

        # Hitung metrics
        mae_val = np.mean(np.abs(actual - forecast))
        rmse_val = np.sqrt(np.mean((actual - forecast) ** 2))
        smape_val = 100 * np.mean(2 * np.abs(actual - forecast) /
                                 (np.abs(actual) + np.abs(forecast)))

        maes.append(mae_val)
        rmses.append(rmse_val)
        smapes.append(smape_val)

    # Aggregate results
    metrics = {
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "sMAPE": np.mean(smapes),
        "windows": windows
    }

    return rows, metrics
```

**Penjelasan dari kode program implementasi ARIMA pada Tabel 5.2:**

1. **Baris 1-5** berfungsi untuk mendefinisikan fungsi `rolling_forecast_arima` dengan parameter series data, prediction horizon (pdt), fraction test, mode cepat, dan maksimal windows untuk evaluasi.

2. **Baris 6** berfungsi untuk mendapatkan panjang total series data yang akan digunakan untuk perhitungan pembagian train/test.

3. **Baris 8-10** berfungsi untuk menentukan batasan maksimal windows evaluasi, dengan default 8 windows jika fast_mode aktif untuk meningkatkan efisiensi komputasi.

4. **Baris 12-13** berfungsi untuk menghitung panjang data test berdasarkan fraksi yang ditentukan dan batasan windows, kemudian menentukan jumlah windows evaluasi.

5. **Baris 15-16** berfungsi untuk membagi data menjadi train dan test set berdasarkan panjang TEST yang telah dihitung sebelumnya.

6. **Baris 18-19** berfungsi untuk setup parameter seasonal dengan melakukan deteksi otomatis berdasarkan frekuensi data dan menentukan periode seasonal (m).

7. **Baris 21** berfungsi untuk menginisialisasi list untuk menyimpan hasil prediksi dan metrics evaluasi dari setiap window.

8. **Baris 23-30** berfungsi untuk melakukan rolling window evaluation di mana setiap iterasi mengupdate training data dengan menambahkan actual values dari windows sebelumnya untuk simulasi real-time forecasting.

9. **Baris 32-40** berfungsi untuk fitting model ARIMA menggunakan `auto_arima` dengan parameter:

   - `seasonal`: menggunakan seasonal ARIMA jika pola seasonal terdeteksi
   - `max_p, max_q`: maksimal parameter AR dan MA (dibatasi 2 untuk efisiensi)
   - `stepwise`: algoritma stepwise untuk optimisasi cepat
   - `maxiter`: maksimal iterasi optimisasi (10 untuk fast mode)

10. **Baris 42-44** berfungsi untuk generate prediksi untuk horizon prediction yang ditentukan dan mengambil actual values untuk window evaluasi saat ini.

11. **Baris 46-50** berfungsi untuk menghitung evaluation metrics:

    - `mae_val`: Mean Absolute Error untuk mengukur akurasi prediksi
    - `rmse_val`: Root Mean Square Error untuk penalti error besar
    - `smape_val`: Symmetric Mean Absolute Percentage Error untuk persentase error

12. **Baris 52-54** berfungsi untuk menyimpan metrics setiap window ke dalam list untuk proses agregasi akhir.

13. **Baris 56-62** berfungsi untuk mengagregasi metrics dari semua windows dengan menghitung rata-rata dan menyimpan jumlah windows yang dievaluasi.

14. **Baris 64** berfungsi untuk mengembalikan hasil prediksi detail dan metrics agregat sebagai output fungsi.

## 5.3 Implementasi LSTM Neural Network

LSTM (Long Short-Term Memory) diimplementasikan sebagai baseline neural network dengan autoregressive forecasting dan standardized scaling untuk stabilitas training.

### Tabel 5.3 Implementasi LSTM Model

```python
# baseline_lstm.py - LSTM Neural Network Implementation
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, lookback, pred_len):
    X, y = [], []
    for i in range(len(data) - lookback - pred_len + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+pred_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(y)

def run_lstm_forecast(name: str, csv: str, freq: str, pdt: int,
                     lookback: int = 30, epochs: int = 5):
    # Load dan preprocess data
    df, use_freq = load_series(csv, freq)
    series = df["value"].values.astype(np.float32)
    total_len = len(series)
    TEST = choose_test_len(total_len, pdt, frac=0.15)
    windows = TEST // pdt

    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pdt

    # Train/test split
    train_vals = series[: total_len - TEST]
    test_vals  = series[total_len - TEST :]

    # Scaling menggunakan StandardScaler
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
        for i in range(0, len(X_train), 64):
            batch_X = X_train[i:i+64]
            batch_y = y_train[i:i+64]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
```

**Penjelasan dari kode program implementasi LSTM pada Tabel 5.3:**

1. **Baris 1-4** berfungsi untuk mendefinisikan class `LSTMModel` yang mewarisi dari `nn.Module` dengan parameter input_size, hidden_size, num_layers, dan output_size untuk arsitektur model.

2. **Baris 5-7** berfungsi untuk menginisialisasi parameter model:

   - `hidden_size`: ukuran hidden state LSTM (32 units)
   - `num_layers`: jumlah layer LSTM (1 layer untuk model lightweight)
   - `lstm`: layer LSTM dengan batch_first=True untuk compatibility

3. **Baris 8** berfungsi untuk mendefinisikan fully connected layer yang menghubungkan LSTM output ke prediction output dengan ukuran sesuai prediction horizon.

4. **Baris 10-15** berfungsi untuk mendefinisikan forward pass:

   - Inisialisasi hidden state (h0) dan cell state (c0) dengan zeros
   - Forward pass melalui LSTM layer
   - Linear transformation dari hidden state terakhir ke output prediction

5. **Baris 17-22** berfungsi untuk mendefinisikan fungsi `create_sequences` yang membuat sliding window sequences untuk training LSTM dengan parameter lookback window dan prediction length.

6. **Baris 25-27** berfungsi untuk mendefinisikan fungsi utama `run_lstm_forecast` dengan parameter dataset name, CSV path, frequency, prediction horizon, lookback window, dan epochs.

7. **Baris 28-32** berfungsi untuk loading dan preprocessing data:

   - Load series menggunakan fungsi load_series
   - Convert ke float32 untuk kompatibilitas PyTorch
   - Hitung total length dan test length

8. **Baris 34-37** berfungsi untuk implementasi window standardization:

   - Batasi windows sesuai konfigurasi standar (6 windows)
   - Adjust TEST length sesuai windows yang dibatasi

9. **Baris 39-41** berfungsi untuk membagi data menjadi training dan testing set berdasarkan calculated TEST length.

10. **Baris 43-46** berfungsi untuk scaling data menggunakan StandardScaler:

    - Fit scaler pada training data saja
    - Transform kedua training dan testing data
    - Reshape untuk kompatibilitas dengan scaler

11. **Baris 48-49** berfungsi untuk membuat sequences data untuk training LSTM menggunakan fungsi create_sequences dengan lookback window dan prediction horizon.

12. **Baris 51-54** berfungsi untuk inisialisasi model dan training components:

    - LSTMModel dengan 32 hidden units dan 1 layer
    - Adam optimizer dengan learning rate 1e-3
    - MSELoss sebagai loss function

13. **Baris 56-67** berfungsi untuk training loop:
    - Set model ke training mode
    - Iterasi melalui epochs yang ditentukan
    - Batch training dengan ukuran batch 64
    - Forward pass, loss calculation, backpropagation, dan parameter update

## 5.4 Implementasi Zero-shot Moirai

Zero-shot Moirai diimplementasikan sebagai universal transformer yang dapat melakukan forecasting tanpa fine-tuning pada dataset spesifik, menggunakan model pre-trained Salesforce/moirai-1.0-R-large.

### Tabel 5.4 Implementasi Zero-shot Moirai

```python
# run_zeroshot_all.py - Zero-shot Moirai Implementation
def run_zeroshot_forecast(name, csv_path, pred_len, context_len, freq):
    # Load data dengan handling frequency
    ds, df = load_univariate_with_freq(csv_path, target_freq=freq)

    total_len = len(df)
    TEST = choose_test_len(total_len, pred_len, frac=0.15)
    windows = TEST // pred_len

    # Batasi windows sesuai konfigurasi standar
    max_windows_config = STANDARD_CONFIG.get(name, {}).get("max_windows", windows)
    windows = min(windows, max_windows_config)
    TEST = windows * pred_len

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

    for w in range(windows):
        # Update dataset untuk window ini
        current_offset = -TEST + (w * pred_len)
        if w == 0:
            current_train_ds = train_ds
        else:
            extended_train_ds, _ = split(ds, offset=current_offset)
            current_train_ds = extended_train_ds

        # Generate predictions
        forecasts = model.forecast(current_train_ds, num_samples=50)
        forecast_list = list(forecasts)

        # Extract prediction dan actual values
        forecast_item = forecast_list[0]
        pred_mean = forecast_item.mean

        # Get actual values untuk window ini
        actual_start = total_len - TEST + (w * pred_len)
        actual_end = actual_start + pred_len
        actual_values = df.iloc[actual_start:actual_end]['value'].values

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
```

**Penjelasan dari kode program implementasi Zero-shot Moirai pada Tabel 5.4:**

1. **Baris 1-2** berfungsi untuk mendefinisikan fungsi `run_zeroshot_forecast` dengan parameter dataset name, CSV path, prediction length, context length, dan frequency untuk konfigurasi model.

2. **Baris 3-4** berfungsi untuk loading data menggunakan `load_univariate_with_freq` yang secara otomatis menangani konversi frequency dan format data untuk compatibility dengan GluonTS.

3. **Baris 6-8** berfungsi untuk menghitung test length dan jumlah windows evaluation berdasarkan total data length dan prediction horizon.

4. **Baris 10-12** berfungsi untuk implementasi standardisasi windows:

   - Ambil max_windows dari konfigurasi standar
   - Batasi windows untuk fair comparison (6 windows)
   - Adjust TEST length sesuai windows yang dibatasi

5. **Baris 14-23** berfungsi untuk inisialisasi model Moirai dengan konfigurasi:

   - `Salesforce/moirai-1.0-R-large`: pre-trained model berukuran large
   - `prediction_length`: horizon forecasting sesuai dataset
   - `context_length`: panjang context window untuk input
   - `num_samples`: jumlah samples untuk uncertainty quantification
   - `target_dim=1`: univariate time series

6. **Baris 25-26** berfungsi untuk split dataset menjadi training dan testing menggunakan GluonTS split function dengan offset negatif untuk mengambil data terakhir sebagai test.

7. **Baris 28-29** berfungsi untuk inisialisasi list untuk menyimpan metrics dari setiap window evaluation.

8. **Baris 31-37** berfungsi untuk rolling window setup:

   - Update dataset untuk setiap window evaluation
   - Untuk window pertama gunakan original training set
   - Untuk window selanjutnya extend training dengan actual values sebelumnya

9. **Baris 39-41** berfungsi untuk generate forecasts menggunakan model Moirai:

   - Call model.forecast dengan current training dataset
   - Specify num_samples untuk probabilistic forecasting
   - Convert generator output ke list untuk processing

10. **Baris 43-45** berfungsi untuk extract prediction results:

    - Ambil forecast item pertama (single time series)
    - Extract mean prediction dari distribution output

11. **Baris 47-50** berfungsi untuk retrieve actual values untuk window saat ini:

    - Calculate start dan end index berdasarkan window position
    - Extract actual values dari original DataFrame

12. **Baris 52-56** berfungsi untuk menghitung evaluation metrics:

    - MAE: Mean Absolute Error untuk akurasi prediksi
    - RMSE: Root Mean Square Error untuk penalti error besar
    - sMAPE: Symmetric Mean Absolute Percentage Error

13. **Baris 58-63** berfungsi untuk menyimpan metrics setiap window ke dalam dictionary dengan informasi window number dan ketiga metrics untuk analisis per-window performance.

## 5.5 Implementasi Few-shot MoE (Fokus Penelitian)

Few-shot MoE (Mixture of Experts) Moirai diimplementasikan sebagai model utama penelitian yang menggunakan arsitektur expert routing untuk adaptive forecasting dengan minimal training examples.

### Tabel 5.5 Implementasi Few-shot MoE

```python
# run_fewshot_moe.py - Few-shot MoE Implementation
def run_fewshot_moe(
    name: str,
    csv_path: str,
    pred_len: int,
    context_len: int,
    freq: str,
    n_shots: int = 6,
) -> dict:
    # Load dan preprocess data
    ds, df, use_freq = load_series(csv_path, freq=freq)

    # Few-shot setup: gunakan n_shots window di akhir untuk test
    test_len = pred_len * n_shots
    total_len = len(df)

    # Initialize Moirai-MoE model
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
        prediction_length=pred_len,
        context_length=context_len,
        patch_size="auto",
        num_samples=50,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        windows=n_shots,
    )

    # Split data: train sampai -test_len, test = test_len terakhir
    train_ds, test_template = split(ds, offset=-test_len)

    # Few-shot evaluation dengan rolling window
    maes, rmses, smapes = [], [], []

    for i in range(n_shots):
        # Update training data untuk include actual values dari windows sebelumnya
        current_offset = -test_len + (i * pred_len)
        if i == 0:
            current_train_ds = train_ds
        else:
            extended_train_ds, _ = split(ds, offset=current_offset)
            current_train_ds = extended_train_ds

        # Generate forecast dengan MoE
        forecasts = model.forecast(current_train_ds, num_samples=50)
        forecast_list = list(forecasts)

        if len(forecast_list) > 0:
            # Extract MoE prediction (ensemble dari multiple experts)
            forecast_item = forecast_list[0]
            pred_mean = forecast_item.mean

            # Get actual values
            actual_start = total_len - test_len + (i * pred_len)
            actual_end = actual_start + pred_len
            actual_values = df.iloc[actual_start:actual_end]['value'].values

            # Calculate metrics
            mae = np.mean(np.abs(actual_values - pred_mean))
            rmse = np.sqrt(np.mean((actual_values - pred_mean) ** 2))
            smape = 100 * np.mean(2 * np.abs(actual_values - pred_mean) /
                                 (np.abs(actual_values) + np.abs(pred_mean)))

            maes.append(mae)
            rmses.append(rmse)
            smapes.append(smape)

    # Calculate summary metrics
    summary_metrics = {
        'dataset': name,
        'model': 'Few-shot MoE',
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'sMAPE': np.mean(smapes),
        'n_shots': n_shots,
        'windows': n_shots,
    }

    return summary_metrics
```

**Penjelasan dari kode program implementasi Few-shot MoE pada Tabel 5.5:**

1. **Baris 1-8** berfungsi untuk mendefinisikan fungsi `run_fewshot_moe` dengan parameter lengkap termasuk n_shots yang menentukan jumlah examples untuk few-shot learning (default 6 sesuai standardisasi).

2. **Baris 9-10** berfungsi untuk loading dan preprocessing data menggunakan load_series yang mengembalikan GluonTS dataset, DataFrame, dan frequency yang digunakan.

3. **Baris 12-14** berfungsi untuk setup few-shot configuration:

   - `test_len`: total length untuk evaluation (n_shots × pred_len)
   - `total_len`: panjang total dataset untuk indexing

4. **Baris 16-26** berfungsi untuk inisialisasi model Moirai-MoE dengan konfigurasi khusus:

   - `Salesforce/moirai-moe-1.0-R-small`: pre-trained MoE model
   - `windows=n_shots`: specify jumlah few-shot examples
   - Parameter lain sama dengan zero-shot untuk consistency

5. **Baris 28-29** berfungsi untuk split dataset dengan offset negatif untuk mengambil n_shots windows terakhir sebagai test set.

6. **Baris 31-32** berfungsi untuk inisialisasi lists untuk menyimpan metrics dari setiap few-shot evaluation window.

7. **Baris 34-41** berfungsi untuk few-shot rolling window evaluation:

   - Iterasi sebanyak n_shots windows
   - Update training dataset untuk setiap window
   - Include actual values dari windows sebelumnya untuk adaptive learning

8. **Baris 43-45** berfungsi untuk generate forecast menggunakan MoE architecture:

   - Model secara otomatis routing ke expert yang sesuai
   - Multiple experts menghasilkan predictions yang di-ensemble
   - num_samples untuk uncertainty quantification

9. **Baris 47-50** berfungsi untuk extract ensemble prediction:

   - MoE menghasilkan weighted average dari multiple experts
   - `pred_mean`: final prediction dari expert ensemble
   - Automatic expert selection berdasarkan input context

10. **Baris 52-55** berfungsi untuk retrieve actual values untuk window evaluation saat ini berdasarkan calculated indices.

11. **Baris 57-61** berfungsi untuk menghitung evaluation metrics:

    - Same metrics dengan model lain untuk fair comparison
    - MAE, RMSE, sMAPE untuk comprehensive evaluation

12. **Baris 63-65** berfungsi untuk menyimpan metrics setiap few-shot window untuk final aggregation.

13. **Baris 67-75** berfungsi untuk calculate summary metrics:

    - Aggregate metrics across all few-shot windows
    - Include metadata seperti n_shots dan model name
    - Return format konsisten untuk comparative analysis

14. **Baris 77** berfungsi untuk mengembalikan summary metrics sebagai dictionary untuk integration dengan evaluation framework.

Implementasi ini memastikan preprocessing data yang konsisten, konfigurasi standardisasi yang fair, dan evaluasi yang seragam untuk semua model, menghasilkan perbandingan yang valid antara ARIMA, LSTM, Zero-shot Moirai, dan Few-shot MoE dengan fokus pada arsitektur Mixture of Experts sebagai kontribusi utama penelitian.
