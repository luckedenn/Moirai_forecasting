#!/usr/bin/env python
"""
Script untuk menjalankan semua model dengan konfigurasi yang seragam dan ringan
Digunakan untuk perbandingan fair antar model
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from light_config import STANDARD_CONFIG, LIGHT_TRAINING_CONFIG, EVAL_CONFIG

def run_command(command, description):
    """Jalankan command dan tampilkan progress"""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    start_time = time.time()
    try:
        # Set environment to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        print(f"[SUCCESS] Duration: {duration:.2f}s")
        if result.stdout:
            print("Output:")
            print(result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"[FAILED] Duration: {duration:.2f}s")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def show_config_summary():
    """Tampilkan ringkasan konfigurasi yang akan digunakan"""
    print("="*60)
    print("KONFIGURASI STANDAR RINGAN")
    print("="*60)
    
    print("\nDATASET CONFIGURATIONS:")
    for dataset_name, config in STANDARD_CONFIG.items():
        print(f"\n  {dataset_name.upper()}:")
        print(f"     - Prediction Length: {config['pred_len']}")
        print(f"     - Context Length: {config['context_len']}")
        print(f"     - Frequency: {config['freq']}")
        print(f"     - N-shots: {config['n_shots']}")
        print(f"     - Max Windows: {config['max_windows']}")
    
    print(f"\nTRAINING CONFIGURATIONS:")
    print(f"  LSTM: epochs={LIGHT_TRAINING_CONFIG['lstm']['epochs']}, " +
          f"batch_size={LIGHT_TRAINING_CONFIG['lstm']['batch_size']}, " +
          f"hidden_size={LIGHT_TRAINING_CONFIG['lstm']['hidden_size']}")
    
    print(f"  ARIMA: max_p/q={LIGHT_TRAINING_CONFIG['arima']['max_p']}, " +
          f"maxiter={LIGHT_TRAINING_CONFIG['arima']['maxiter']}, " +
          f"fast_mode={LIGHT_TRAINING_CONFIG['arima']['fast_mode']}")
    
    print(f"  Moirai: batch_size={LIGHT_TRAINING_CONFIG['moirai']['batch_size']}, " +
          f"num_samples={LIGHT_TRAINING_CONFIG['moirai']['num_samples']}")
    
    print(f"\nEVALUATION CONFIG:")
    print(f"  - Test fraction: {EVAL_CONFIG['test_fraction']}")
    print(f"  - Min windows: {EVAL_CONFIG['min_test_windows']}")
    print(f"  - Max windows: {EVAL_CONFIG['max_test_windows']}")

def main():
    """Main function untuk menjalankan semua model"""
    
    print("STANDARDIZED MODEL COMPARISON")
    print("Konfigurasi seragam dan ringan untuk fair comparison")
    print("="*60)
    
    # Show configuration summary
    show_config_summary()
    
    # Confirm before running
    print(f"\n Jalankan semua model dengan konfigurasi ini? [y/N]: ", end="")
    confirm = input().lower().strip()
    
    if confirm not in ['y', 'yes']:
        print(" Dibatalkan oleh user")
        return
    
    # List of models to run
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
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each model
    for i, model in enumerate(models, 1):
        print(f"\n  RUNNING MODEL {i}/{len(models)}: {model['name']}")
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
    
    # Summary
    total_duration = time.time() - total_start_time
    successful = len([r for r in results if r['success']])
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    print(f"Successful: {successful}/{len(models)} models")
    print(f"Failed: {len(models) - successful}/{len(models)} models")
    
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "[SUCCESS]" if result['success'] else "[FAILED]"
        print(f"  {result['model']:<20} | {status} | {result['timestamp']}")
    
    if successful == len(models):
        print(f"\n[SUCCESS] ALL MODELS COMPLETED SUCCESSFULLY!")
        print(f"Ready for standardized comparison analysis")
        
        # Suggest next steps
        print(f"\nNEXT STEPS:")
        print(f"  1. Run: python analysis_standardized_results.py")
        print(f"  2. Check individual model result directories")
        print(f"  3. Generate comparison tables and plots")
        
    elif successful > 0:
        print(f"\n[PARTIAL] PARTIAL SUCCESS - {successful} models completed")
        print(f"Check failed models and retry if needed")
        
    else:
        print(f"\n[FAILED] ALL MODELS FAILED")
        print(f"Check configuration and dependencies")
        
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()