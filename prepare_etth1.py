#!/usr/bin/env python3
"""
Script untuk memproses ETTh1 dataset
Mengambil kolom 'date' dan 'OT' saja untuk forecasting
"""

import os
import pandas as pd

def prepare_etth1():
    """
    Load ETTh1.csv dan prepare untuk forecasting
    Input: date, OT (oil temperature)
    Output: timestamp, value
    """
    print("📊 Preparing ETTh1 dataset...")
    
    # Create output directory
    output_dir = "data/etth1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv("data/ETTh1.csv")
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Columns: {df.columns.tolist()}")
    
    # Rename columns untuk konsistensi dengan dataset lain
    df_prepared = pd.DataFrame({
        'timestamp': pd.to_datetime(df['date']),
        'value': df['OT']
    })
    
    # Sort by timestamp
    df_prepared = df_prepared.sort_values('timestamp').reset_index(drop=True)
    
    # Info
    print(f"\n📈 ETTh1 Dataset Info:")
    print(f"   • Total rows: {len(df_prepared)}")
    print(f"   • Date range: {df_prepared['timestamp'].min()} to {df_prepared['timestamp'].max()}")
    print(f"   • Frequency: Hourly (H)")
    print(f"   • Value range: {df_prepared['value'].min():.2f} to {df_prepared['value'].max():.2f}")
    print(f"   • Missing values: {df_prepared.isnull().sum().sum()}")
    
    # Calculate split sizes (70% train, 15% val, 15% test)
    n = len(df_prepared)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train_df = df_prepared.iloc[:n_train]
    val_df = df_prepared.iloc[n_train:n_train + n_val]
    test_df = df_prepared.iloc[n_train + n_val:]
    
    # Save splits
    df_prepared.to_csv(f"{output_dir}/etth1_full.csv", index=False)
    train_df.to_csv(f"{output_dir}/etth1_train.csv", index=False)
    val_df.to_csv(f"{output_dir}/etth1_val.csv", index=False)
    test_df.to_csv(f"{output_dir}/etth1_test.csv", index=False)
    
    print(f"\n💾 Saved to {output_dir}/")
    print(f"   • Full: {len(df_prepared)} rows")
    print(f"   • Train: {len(train_df)} rows ({n_train/n*100:.1f}%)")
    print(f"   • Val: {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
    print(f"   • Test: {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")
    
    print("\n✅ ETTh1 dataset prepared successfully!")
    print(f"\n📝 Dataset characteristics:")
    print(f"   • Domain: Electricity Transformer Temperature")
    print(f"   • Metric: Oil Temperature (OT)")
    print(f"   • Frequency: Hourly")
    print(f"   • Use case: Industrial sensor forecasting")
    
    return df_prepared

if __name__ == "__main__":
    prepare_etth1()
