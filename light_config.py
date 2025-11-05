STANDARD_CONFIG = {
    # Dataset configurations yang ringan dan konsisten dengan WINDOWS SERAGAM
    "weather_melbourne": {
        "csv": "data/weather_melbourne/weather_melbourne_full.csv",
        "pred_len": 7,          # 1 week forecast (ringan)
        "context_len": 30,      # 1 month context (ringan)
        "freq": "D",
        "lookback": 30,         # untuk LSTM
        "n_shots": 6,           # SERAGAM: 6 windows untuk semua model
        "max_windows": 6        # SERAGAM: maksimal 6 windows untuk evaluasi
    },
    
    "finance_aapl": {
        "csv": "data/finance_aapl/finance_aapl_full.csv",
        "pred_len": 5,          # 1 week trading forecast
        "context_len": 30,      # 1 month context
        "freq": "D",            # gunakan daily untuk konsistensi
        "lookback": 30,         # untuk LSTM
        "n_shots": 6,           # SERAGAM: 6 windows untuk semua model
        "max_windows": 6        # SERAGAM: maksimal 6 windows
    },
    
    "co2_maunaloa_monthly": {
        "csv": "data/co2_maunaloa_monthly/co2_maunaloa_monthly_full.csv", 
        "pred_len": 6,          # 6 months forecast
        "context_len": 24,      # 2 years context
        "freq": "M",
        "lookback": 24,         # untuk LSTM
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

# Test configuration
EVAL_CONFIG = {
    "test_fraction": 0.2,       # 20% untuk test (lebih banyak dari 15%)
    "min_test_windows": 3,      # minimal 3 windows
    "max_test_windows": 6       # maksimal 6 windows untuk efficiency
}