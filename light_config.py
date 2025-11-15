STANDARD_CONFIG = {
    # Dataset configurations yang ringan dan konsisten dengan WINDOWS SERAGAM
    "weather_melbourne": {
        "csv": "data/weather_melbourne/weather_melbourne_full.csv",
        "pred_len": 7,          # 1 week forecast (natural unit untuk weather)
        "context_len": 120,     # 4 bulan context (120 hari) - SERAGAM
        "freq": "D",
        "lookback": 60,         # 2 bulan lookback - SERAGAM: 60 timesteps
        "n_shots": 6,           # SERAGAM: 6 windows untuk semua model
        "max_windows": 6        # SERAGAM: maksimal 6 windows untuk evaluasi
    },
    
    "finance_aapl": {
        "csv": "data/finance_aapl/finance_aapl_full.csv",
        "pred_len": 5,          # 1 week trading forecast (5 hari kerja)
        "context_len": 120,     # 120 hari trading (~6 bulan) - SERAGAM
        "freq": "D",            # gunakan daily untuk konsistensi
        "lookback": 60,         # 60 hari trading (~3 bulan) - SERAGAM: 60 timesteps
        "n_shots": 6,           # SERAGAM: 6 windows untuk semua model
        "max_windows": 6        # SERAGAM: maksimal 6 windows
    },
    
    "co2_maunaloa_monthly": {
        "csv": "data/co2_maunaloa_monthly/co2_maunaloa_monthly_full.csv", 
        "pred_len": 6,          # 6 months forecast (half year)
        "context_len": 120,     # 10 tahun context (120 bulan) - SERAGAM
        "freq": "M",
        "lookback": 60,         # 5 tahun lookback (60 bulan) - SERAGAM: 60 timesteps
        "n_shots": 6,           # SERAGAM: 6 windows untuk semua model  
        "max_windows": 6        # SERAGAM: maksimal 6 windows (cukup untuk monthly data)
    }
}

# Training parameters yang ringan
LIGHT_TRAINING_CONFIG = {
    # LSTM parameters (ringan)
    "lstm": {
        "epochs": 5,            # reduced dari 10
        "batch_size": 64,       # reduced dari 128
        "hidden_size": 32,      # reduced dari 64
        "num_layers": 1,        # reduced dari 2
        "lr": 1e-3
    },
    
    # ARIMA parameters (fast mode)
    "arima": {
        "fast_mode": True,
        "max_windows": 6,       # batasi windows
        "max_p": 2,             # reduced dari 3
        "max_q": 2,             # reduced dari 3
        "max_P": 1,
        "max_Q": 1,
        "maxiter": 10           # reduced dari 20
    },
    
    # Moirai parameters
    "moirai": {
        "batch_size": 8,        # small batch untuk ringan
        "num_samples": 50       # reduced dari 100 untuk MoE
    }
}

