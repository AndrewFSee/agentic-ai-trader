"""
Test improved ML pipeline with regime features and increased model capacity.

This script tests the improvements on defensive stocks that showed promise.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Import v2 versions
from config_v2 import CONFIG_V2
from feature_engineering_v2 import engineer_features_v2
from data_collection import fetch_daily_bars, collect_all_stocks
from train_models import split_data, train_all_models
from backtest import backtest_all_models


def run_improved_pipeline(test_stocks=None):
    """
    Run improved ML pipeline with new features.
    
    Args:
        test_stocks: List of stock symbols to test. If None, uses all from config.
    """
    config = CONFIG_V2.copy()
    
    # Override with test stocks if provided
    if test_stocks:
        config['symbols'] = test_stocks
    
    print("="*80)
    print("IMPROVED ML PIPELINE TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Symbols: {config['symbols']}")
    print(f"  Date range: {config['start_date']} to {config['end_date']}")
    print(f"  Horizons: {config['prediction_horizons']}")
    print(f"  Models: {list(config['models'].keys())}")
    print(f"\nNew features:")
    print(f"  - Regime detection (HMM + Wasserstein)")
    print(f"  - Feature interactions")
    print(f"  - Polynomial terms")
    print(f"\nModel improvements:")
    print(f"  - XGBoost: 500 trees (was 100), depth 8 (was 6)")
    print(f"  - Random Forest: 300 trees (was 100), depth 15 (was 10)")
    print(f"  - Removed excess regularization")
    print("\n" + "="*80 + "\n")
    
    # Step 1: Collect data
    print("\n[1/4] COLLECTING DATA")
    print("-"*80)
    stock_data = {}
    for symbol in config['symbols']:
        try:
            df = fetch_daily_bars(symbol, config['start_date'], config['end_date'])
            if df is not None and len(df) > 0:
                stock_data[symbol] = df
                print(f"[OK] {symbol}: {len(df)} days")
            else:
                print(f"[FAIL] {symbol}: No data")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
    
    if not stock_data:
        print("\n❌ No data collected. Exiting.")
        return None
    
    # Step 2: Engineer features (v2 with regime and interactions)
    print("\n[2/4] ENGINEERING FEATURES (V2)")
    print("-"*80)
    feature_data = {}
    for symbol, df in stock_data.items():
        try:
            df_features = engineer_features_v2(df, symbol, config)
            
            # Drop rows with NaN in target variables
            for horizon in config['prediction_horizons']:
                df_features = df_features.dropna(subset=[f'target_{horizon}d', f'target_direction_{horizon}d'])
            
            if len(df_features) > 100:
                feature_data[symbol] = df_features
                print(f"[OK] {symbol}: {len(df_features)} samples with features")
            else:
                print(f"[FAIL] {symbol}: Insufficient data after feature engineering")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    if not feature_data:
        print("\n❌ No features engineered. Exiting.")
        return None
    
    # Step 3: Train models
    print("\n[3/4] TRAINING MODELS")
    print("-"*80)
    all_results = []
    
    for symbol, df_features in feature_data.items():
        for horizon in config['prediction_horizons']:
            print(f"\n{symbol} - {horizon}d prediction:")
            print("-" * 40)
            
            try:
                # Get feature columns (exclude OHLCV, timestamp, targets)
                from feature_engineering import get_feature_columns
                feature_cols = get_feature_columns(df_features)
                
                # Train models
                results = train_all_models(
                    symbol, 
                    df_features, 
                    feature_cols,
                    horizon
                )
                
                # Print training results
                print(f"\nTraining Results (Validation Set):")
                for model_name, model_data in results['models'].items():
                    metrics = model_data['metrics']
                    print(f"  {model_data['name']}:")
                    print(f"    Accuracy: {metrics['accuracy']:.3f}")
                    print(f"    ROC-AUC:  {metrics['roc_auc']:.3f}")
                    print(f"    F1:       {metrics['f1']:.3f}")
                
                all_results.append({
                    'symbol': symbol,
                    'horizon': horizon,
                    'results': results
                })
                
            except Exception as e:
                print(f"[ERROR] Training {symbol} {horizon}d: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 4: Backtest
    print("\n[4/4] BACKTESTING")
    print("-"*80)
    backtest_results = []
    
    for result_data in all_results:
        symbol = result_data['symbol']
        horizon = result_data['horizon']
        results = result_data['results']
        
        print(f"\n{symbol} - {horizon}d:")
        print("-" * 40)
        
        try:
            comparison = backtest_all_models(results, config)
            
            # Print comparison
            print("\nBacktest Results:")
            for _, row in comparison.iterrows():
                model = row['model']
                ret = row['total_return']
                sharpe = row['sharpe_ratio']
                if model == 'Buy-and-Hold':
                    print(f"  {model:20s}: {ret:>7.2%}  Sharpe: {sharpe:>5.2f}")
                else:
                    diff = ret - comparison[comparison['model'] == 'Buy-and-Hold']['total_return'].iloc[0]
                    symbol_str = "+" if diff > 0 else "-"
                    print(f"  {model:20s}: {ret:>7.2%}  Sharpe: {sharpe:>5.2f}  ({symbol_str} {diff:>+6.2%} vs B&H)")
            
            backtest_results.append({
                'symbol': symbol,
                'horizon': horizon,
                'comparison': comparison
            })
            
        except Exception as e:
            print(f"[ERROR] Backtesting {symbol} {horizon}d: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if backtest_results:
        # Aggregate by model
        model_performance = {}
        for result in backtest_results:
            for _, row in result['comparison'].iterrows():
                model = row['model']
                if model == 'Buy-and-Hold':
                    continue
                
                if model not in model_performance:
                    model_performance[model] = {
                        'returns': [],
                        'sharpes': [],
                        'beats_bh': 0,
                        'total': 0
                    }
                
                # Find B&H for comparison
                bh_return = result['comparison'][result['comparison']['model'] == 'Buy-and-Hold']['total_return'].iloc[0]
                
                model_performance[model]['returns'].append(row['total_return'])
                model_performance[model]['sharpes'].append(row['sharpe_ratio'])
                model_performance[model]['total'] += 1
                if row['total_return'] > bh_return:
                    model_performance[model]['beats_bh'] += 1
        
        print("\nPerformance by Model:")
        print("-"*80)
        for model, perf in sorted(model_performance.items(), 
                                  key=lambda x: np.mean(x[1]['sharpes']), 
                                  reverse=True):
            mean_ret = np.mean(perf['returns'])
            mean_sharpe = np.mean(perf['sharpes'])
            win_rate = (perf['beats_bh'] / perf['total']) * 100
            
            print(f"\n{model}:")
            print(f"  Mean Return:     {mean_ret:.2%}")
            print(f"  Mean Sharpe:     {mean_sharpe:.2f}")
            print(f"  Beats B&H:       {win_rate:.1f}% ({perf['beats_bh']}/{perf['total']})")
        
        # Compare to baseline
        print("\n" + "="*80)
        print("COMPARISON TO BASELINE (Original Pipeline)")
        print("="*80)
        print("\nBaseline (from previous run):")
        print("  Logistic:  Mean Sharpe 0.68, Beats B&H 52%")
        print("  XGBoost:   Mean Sharpe 0.66, Beats B&H 54%")
        print("\nImproved (current run):")
        for model in ['Logistic Regression', 'XGBoost']:
            if model in model_performance:
                perf = model_performance[model]
                mean_sharpe = np.mean(perf['sharpes'])
                win_rate = (perf['beats_bh'] / perf['total']) * 100
                print(f"  {model.split()[0]:10s} Mean Sharpe {mean_sharpe:.2f}, Beats B&H {win_rate:.0f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("../results/ml_models") / f"improved_pipeline_results_{timestamp}.json"
    
    save_data = {
        "timestamp": timestamp,
        "config": {
            "symbols": config['symbols'],
            "horizons": config['prediction_horizons'],
            "models": list(config['models'].keys())
        },
        "results": [
            {
                "symbol": r['symbol'],
                "horizon": r['horizon'],
                "comparison": r['comparison'].to_dict('records')
            }
            for r in backtest_results
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\n" + "="*80)
    
    return backtest_results


if __name__ == "__main__":
    # Test on defensive stocks that showed promise
    test_stocks = ["PG", "JNJ", "COST", "KO"]
    
    print("\nTesting on defensive stocks that showed 80% win rate in baseline...")
    print("This should take about 15-30 minutes.\n")
    
    results = run_improved_pipeline(test_stocks)
    
    if results:
        print("\n[SUCCESS] Test complete!")
    else:
        print("\n[FAILED] Test failed!")

