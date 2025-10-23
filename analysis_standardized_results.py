#!/usr/bin/env python
"""
Analisis hasil dari semua model dengan konfigurasi standar yang seragam
Menghasilkan perbandingan fair antar model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from light_config import STANDARD_CONFIG

# Set style
sns.set_style("whitegrid")
plt.style.use('default')

def load_standardized_results():
    """Load hasil dari semua model dengan konfigurasi standar"""
    
    print("📂 Loading standardized results...")
    
    results = []
    
    # ARIMA results
    try:
        arima_path = "results_baseline_arima/summary_arima.csv"
        if os.path.exists(arima_path):
            arima_df = pd.read_csv(arima_path)
            for _, row in arima_df.iterrows():
                results.append({
                    'Model': 'ARIMA',
                    'Dataset': row['dataset'],
                    'MAE': row['MAE_mean'],
                    'MAE_std': row['MAE_std'],
                    'RMSE': row['RMSE_mean'],
                    'RMSE_std': row['RMSE_std'],
                    'sMAPE': row['sMAPE_mean'],
                    'sMAPE_std': row['sMAPE_std'],
                    'Windows': row['windows']
                })
            print(f"✅ ARIMA: {len(arima_df)} datasets")
    except Exception as e:
        print(f"⚠️ ARIMA loading failed: {e}")
    
    # LSTM results
    try:
        lstm_path = "results_baseline_lstm/summary_lstm.csv"
        if os.path.exists(lstm_path):
            lstm_df = pd.read_csv(lstm_path)
            for _, row in lstm_df.iterrows():
                results.append({
                    'Model': 'LSTM',
                    'Dataset': row['dataset'],
                    'MAE': row['MAE_mean'],
                    'MAE_std': row['MAE_std'],
                    'RMSE': row['RMSE_mean'],
                    'RMSE_std': row['RMSE_std'],
                    'sMAPE': row['sMAPE_mean'],
                    'sMAPE_std': row['sMAPE_std'],
                    'Windows': row['windows']
                })
            print(f"✅ LSTM: {len(lstm_df)} datasets")
    except Exception as e:
        print(f"⚠️ LSTM loading failed: {e}")
    
    # Zero-shot results
    try:
        zeroshot_path = "results_zeroshot/summary_all_zeroshot.csv"
        if os.path.exists(zeroshot_path):
            zeroshot_df = pd.read_csv(zeroshot_path)
            for _, row in zeroshot_df.iterrows():
                results.append({
                    'Model': 'Zero-shot',
                    'Dataset': row['dataset'],
                    'MAE': row['MAE_mean'],
                    'MAE_std': row['MAE_std'],
                    'RMSE': row['RMSE_mean'],
                    'RMSE_std': row['RMSE_std'],
                    'sMAPE': row['sMAPE_mean'],
                    'sMAPE_std': row['sMAPE_std'],
                    'Windows': row['windows']
                })
            print(f"✅ Zero-shot: {len(zeroshot_df)} datasets")
    except Exception as e:
        print(f"⚠️ Zero-shot loading failed: {e}")
    
    # Few-shot MoE results
    try:
        moe_path = "results_fewshot_moe/summary_all_moe.csv"
        if os.path.exists(moe_path):
            moe_df = pd.read_csv(moe_path)
            for _, row in moe_df.iterrows():
                results.append({
                    'Model': 'Few-shot MoE',
                    'Dataset': row['dataset'],
                    'MAE': row['MAE_mean'],
                    'MAE_std': row['MAE_std'],
                    'RMSE': row['RMSE_mean'],
                    'RMSE_std': row['RMSE_std'],
                    'sMAPE': row['sMAPE_mean'],
                    'sMAPE_std': row['sMAPE_std'],
                    'Windows': row['n_shots']  # Use actual n_shots from data
                })
            print(f"✅ Few-shot MoE: {len(moe_df)} datasets")
    except Exception as e:
        print(f"⚠️ Few-shot MoE loading failed: {e}")
    
    if not results:
        raise ValueError("No results found! Please run the models first.")
    
    df = pd.DataFrame(results)
    
    # Standardize dataset names
    name_mapping = {
        'weather_melbourne': 'Weather Melbourne',
        'finance_aapl': 'Finance AAPL',
        'co2_maunaloa_monthly': 'CO2 Mauna Loa',
        'co2_maunaloa': 'CO2 Mauna Loa'
    }
    df['Dataset'] = df['Dataset'].map(name_mapping).fillna(df['Dataset'])
    
    print(f"\n📊 Total results loaded: {len(df)} records")
    print(f"📋 Models: {df['Model'].unique().tolist()}")
    print(f"📋 Datasets: {df['Dataset'].unique().tolist()}")
    
    return df

def create_standardized_summary_table(df):
    """Buat tabel summary dengan konfigurasi yang sama"""
    
    print("\n" + "="*80)
    print("📊 STANDARDIZED MODEL COMPARISON")
    print("   All models with same prediction length, context, and evaluation")
    print("="*80)
    
    # Show configuration used
    print(f"\n⚙️ STANDARDIZED CONFIGURATION:")
    for dataset_name, config in STANDARD_CONFIG.items():
        display_name = dataset_name.replace('_', ' ').title()
        print(f"  📊 {display_name}:")
        print(f"     Pred: {config['pred_len']}, Context: {config['context_len']}, " +
              f"Freq: {config['freq']}, Max Windows: {config['max_windows']}")
    
    # Create summary table
    print(f"\n📋 PERFORMANCE RESULTS:")
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        print(f"\n🎯 {dataset.upper()}")
        print("-" * 70)
        
        dataset_df = df[df['Dataset'] == dataset].sort_values('MAE')
        
        print(f"{'Rank':<6}{'Model':<15}{'MAE':<12}{'RMSE':<12}{'sMAPE (%)':<12}{'Windows':<8}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(dataset_df.iterrows(), 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            mae_str = f"{row['MAE']:.3f}±{row['MAE_std']:.3f}"
            rmse_str = f"{row['RMSE']:.3f}±{row['RMSE_std']:.3f}"
            smape_str = f"{row['sMAPE']:.2f}±{row['sMAPE_std']:.2f}"
            
            print(f"{rank_emoji:<6}{row['Model']:<15}{mae_str:<12}{rmse_str:<12}{smape_str:<12}{row['Windows']:<8}")
    
    return df

def create_standardized_visualizations(df):
    """Buat visualisasi untuk hasil yang sudah distandardisasi"""
    
    os.makedirs("standardized_comparison", exist_ok=True)
    
    # 1. Performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['MAE', 'RMSE', 'sMAPE']
    titles = ['Mean Absolute Error', 'Root Mean Square Error', 'Symmetric MAPE (%)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.barplot(data=df, x='Dataset', y=metric, hue='Model', ax=axes[i])
        axes[i].set_title(f'{title}\n(Standardized Configuration)', fontweight='bold')
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('standardized_comparison/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Ranking heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create ranking matrix
    models = df['Model'].unique()
    datasets = df['Dataset'].unique()
    ranking_matrix = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset].sort_values('MAE')
        ranks = {model: i+1 for i, model in enumerate(dataset_df['Model'])}
        ranking_matrix.append([ranks.get(model, len(models)+1) for model in models])
    
    ranking_df = pd.DataFrame(ranking_matrix, columns=models, index=datasets)
    
    sns.heatmap(ranking_df, annot=True, cmap='RdYlGn_r', center=3, ax=ax,
                cbar_kws={'label': 'Rank (1=Best)'}, fmt='d')
    ax.set_title('Model Ranking by Dataset\n(Standardized Configuration)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('standardized_comparison/ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to standardized_comparison/")

def overall_analysis(df):
    """Analisis overall performa model"""
    
    print("\n" + "="*80)
    print("🏆 OVERALL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Overall ranking berdasarkan rata-rata MAE
    overall_mae = df.groupby('Model')['MAE'].mean().sort_values()
    
    print(f"\n📊 OVERALL RANKING (Average MAE across datasets):")
    print("-" * 50)
    
    for i, (model, avg_mae) in enumerate(overall_mae.items(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{emoji} {model:<15} | Avg MAE: {avg_mae:.3f}")
    
    # Consistency analysis
    print(f"\n📊 CONSISTENCY ANALYSIS (Lower std = More consistent):")
    print("-" * 60)
    
    consistency = df.groupby('Model')['MAE_std'].mean().sort_values()
    
    for model, avg_std in consistency.items():
        print(f"{model:<15} | Avg MAE Std: {avg_std:.3f}")
    
    # Domain-specific best models
    print(f"\n📊 DOMAIN-SPECIFIC WINNERS:")
    print("-" * 40)
    
    for dataset in df['Dataset'].unique():
        best_model = df[df['Dataset'] == dataset].loc[df[df['Dataset'] == dataset]['MAE'].idxmin(), 'Model']
        best_mae = df[df['Dataset'] == dataset]['MAE'].min()
        print(f"{dataset:<20} | {best_model} (MAE: {best_mae:.3f})")
    
    # Statistical summary
    print(f"\n📊 STATISTICAL SUMMARY:")
    print("-" * 30)
    print(df.groupby('Model')[['MAE', 'RMSE', 'sMAPE']].agg(['mean', 'std']).round(3))

def main():
    """Main analysis function"""
    
    print("📊 STANDARDIZED MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    # Load results
    df = load_standardized_results()
    
    # Create summary table
    create_standardized_summary_table(df)
    
    # Overall analysis
    overall_analysis(df)
    
    # Create visualizations
    create_standardized_visualizations(df)
    
    # Save results
    os.makedirs("standardized_results", exist_ok=True)
    df.to_csv("standardized_results/complete_comparison.csv", index=False)
    
    # Create final summary
    summary = []
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset].sort_values('MAE')
        for i, (_, row) in enumerate(dataset_df.iterrows(), 1):
            summary.append({
                'Dataset': dataset,
                'Rank': i,
                'Model': row['Model'],
                'MAE': row['MAE'],
                'RMSE': row['RMSE'],
                'sMAPE': row['sMAPE'],
                'Configuration': 'Standardized'
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("standardized_results/ranking_summary.csv", index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"  ✅ standardized_results/complete_comparison.csv")
    print(f"  ✅ standardized_results/ranking_summary.csv") 
    print(f"  ✅ standardized_comparison/ (visualizations)")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"   Fair comparison with standardized configuration")
    print("="*60)

if __name__ == "__main__":
    main()