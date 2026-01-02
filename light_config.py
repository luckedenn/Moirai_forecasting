STANDARD_CONFIG = {
    "weather_melbourne": {
        "csv": "data/weather_melbourne/weather_melbourne_full.csv",
        "pred_len": 7,          
        "context_len": 120,     
        "freq": "D",
        "lookback": 120,         
        "n_shots": 6,           
        "max_windows": 6        
    },
    
    "finance_aapl": {
        "csv": "data/finance_aapl/finance_aapl_full.csv",
        "pred_len": 5,          
        "context_len": 120,     
        "freq": "D",            
        "lookback": 120,         
        "n_shots": 6,           
        "max_windows": 6        
    },
    
    "co2_maunaloa_monthly": {
        "csv": "data/co2_maunaloa_monthly/co2_maunaloa_monthly_full.csv", 
        "pred_len": 6,          
        "context_len": 120,     
        "freq": "M",
        "lookback": 120,         
        "n_shots": 6,           
        "max_windows": 6        
    },
}

# Training parameters yang ringan
LIGHT_TRAINING_CONFIG = {
    # LSTM parameters (ringan)
    "lstm": {
        "epochs": 10,           
        "batch_size": 64,      
        "hidden_size": 32,     
        "num_layers": 1,       
        "lr": 1e-3
    },
    
    # ARIMA parameters (fast mode)
    "arima": {
        "fast_mode": True,
        "max_windows": 6,       
        "max_p": 2,            
        "max_q": 2,            
        "max_P": 1,
        "max_Q": 1,
        "maxiter": 10          
    },
    
    # Moirai parameters
    "moirai": {
        "batch_size": 6,        
        "num_samples": 100       
    }
}

