"""
Main pipeline orchestrator for ML-based return prediction.

This script:
1. Collects historical data for a universe of stocks
2. Engineers features from price, volume, and technical indicators
3. Trains multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
4. Backtests models against buy-and-hold
5. Generates comprehensive results across stocks and prediction horizons
"""
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List

from config import (
    STOCK_UNIVERSE, PREDICTION_HORIZONS, RESULTS_DIR, VERBOSE
)
from data_collection import collect_all_stocks, get_stock_category
from feature_engineering import engineer_features, get_feature_columns
from train_models import train_all_models
from backtest import backtest_all_models


def run_full_pipeline(symbols: List[str] = None, horizons: List[int] = None):
    """
    Run complete ML pipeline for specified symbols and horizons.
    
    Args:
        symbols: List of stock tickers (if None, uses all from STOCK_UNIVERSE)
        horizons: List of prediction horizons (if None, uses PREDICTION_HORIZONS)
    """
    # Default to all stocks and horizons
    if symbols is None:
        symbols = [s for stocks in STOCK_UNIVERSE.values() for s in stocks]
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    
    print("="*80)
    print("ML RETURN PREDICTION PIPELINE")
    print("="*80)
    print(f"Stocks: {len(symbols)}")
    print(f"Horizons: {horizons}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Collect data
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)
    all_data = collect_all_stocks()
    spy_df = all_data.get("SPY")
    
    # Step 2: Engineer features and train models
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING & MODEL TRAINING")
    print("="*80)
    
    all_results = []
    
    for symbol in symbols:
        if symbol not in all_data or symbol == "SPY":
            print(f"\nWARNING: Skipping {symbol} (no data)")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING: {symbol}")
        print(f"{'='*80}")
        
        df = all_data[symbol]
        category = get_stock_category(symbol)
        
        # Engineer features
        df_features = engineer_features(symbol, df, spy_df)
        feature_cols = get_feature_columns(df_features)
        
        # Train and backtest for each horizon
        for horizon in horizons:
            try:
                # Train models
                training_results = train_all_models(symbol, df_features, feature_cols, horizon)
                
                # Backtest
                backtest_results = backtest_all_models(training_results, symbol, horizon)
                
                # Store results
                result = {
                    'symbol': symbol,
                    'category': category,
                    'horizon': horizon,
                    'training_metrics': {
                        model_name: model_data['metrics']
                        for model_name, model_data in training_results['models'].items()
                    },
                    'backtest_metrics': backtest_results['model_results'],
                    'buy_hold': backtest_results['buy_hold'],
                    'best_model': backtest_results['best_model'],
                    'feature_importance': training_results.get('feature_importance', {})
                }
                
                all_results.append(result)
                
            except Exception as e:
                print(f"\nERROR: Processing {symbol} with {horizon}d horizon: {e}")
                continue
    
    # Step 3: Aggregate results
    print("\n" + "="*80)
    print("STEP 3: AGGREGATING RESULTS")
    print("="*80)
    
    aggregate_results = aggregate_cross_stock_results(all_results)
    
    # Step 4: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"ml_pipeline_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'symbols': symbols,
                'horizons': horizons,
                'num_stocks': len(symbols)
            },
            'individual_results': all_results,
            'aggregate': aggregate_results
        }, f, indent=2, default=str)
    
    print(f"\nOK: Results saved to {results_file}")
    
    # Print summary
    print_final_summary(aggregate_results)
    
    return all_results, aggregate_results


def aggregate_cross_stock_results(results: List[Dict]) -> Dict:
    """Aggregate results across all stocks."""
    
    # Group by horizon and model
    by_horizon = {}
    
    for result in results:
        horizon = result['horizon']
        if horizon not in by_horizon:
            by_horizon[horizon] = {
                'models': {},
                'categories': {}
            }
        
        # Aggregate by model
        for backtest in result['backtest_metrics']:
            model_name = backtest['model']
            if model_name not in by_horizon[horizon]['models']:
                by_horizon[horizon]['models'][model_name] = {
                    'total_returns': [],
                    'sharpe_ratios': [],
                    'max_drawdowns': [],
                    'win_rates': []
                }
            
            by_horizon[horizon]['models'][model_name]['total_returns'].append(backtest['total_return'])
            by_horizon[horizon]['models'][model_name]['sharpe_ratios'].append(backtest['sharpe_ratio'])
            by_horizon[horizon]['models'][model_name]['max_drawdowns'].append(backtest['max_drawdown'])
            by_horizon[horizon]['models'][model_name]['win_rates'].append(backtest['win_rate'])
        
        # Aggregate by category
        category = result['category']
        if category not in by_horizon[horizon]['categories']:
            by_horizon[horizon]['categories'][category] = []
        by_horizon[horizon]['categories'][category].append(result['best_model'])
    
    # Calculate statistics
    aggregate = {}
    
    for horizon, data in by_horizon.items():
        aggregate[horizon] = {'models': {}}
        
        for model_name, metrics in data['models'].items():
            aggregate[horizon]['models'][model_name] = {
                'mean_return': np.mean(metrics['total_returns']),
                'median_return': np.median(metrics['total_returns']),
                'std_return': np.std(metrics['total_returns']),
                'mean_sharpe': np.mean(metrics['sharpe_ratios']),
                'median_sharpe': np.median(metrics['sharpe_ratios']),
                'mean_max_dd': np.mean(metrics['max_drawdowns']),
                'mean_win_rate': np.mean(metrics['win_rates']),
                'num_stocks': len(metrics['total_returns'])
            }
    
    return aggregate


def print_final_summary(aggregate: Dict):
    """Print final summary of results."""
    print("\n" + "="*80)
    print("FINAL SUMMARY - CROSS-STOCK PERFORMANCE")
    print("="*80)
    
    for horizon, data in aggregate.items():
        print(f"\n{horizon}-Day Prediction Horizon:")
        print("-" * 80)
        
        # Sort models by mean Sharpe ratio
        models_sorted = sorted(
            data['models'].items(),
            key=lambda x: x[1]['mean_sharpe'],
            reverse=True
        )
        
        print(f"{'Model':<20} {'Mean Return':>12} {'Mean Sharpe':>12} {'Win Rate':>10} {'# Stocks':>10}")
        print("-" * 80)
        
        for model_name, metrics in models_sorted:
            print(f"{model_name:<20} {metrics['mean_return']:>11.2%} "
                  f"{metrics['mean_sharpe']:>12.2f} {metrics['mean_win_rate']:>10.2%} "
                  f"{metrics['num_stocks']:>10}")
        
        # Identify best model
        best_model = models_sorted[0]
        print(f"\n[BEST] Model for {horizon}d: {best_model[0]}")
        print(f"   Mean Sharpe: {best_model[1]['mean_sharpe']:.2f}")
        print(f"   Mean Return: {best_model[1]['mean_return']:.2%}")


def run_quick_test():
    """Run test on all 25 stocks from config."""
    print("Running test on all 25 stocks...")
    # Get all stocks from config
    from config import STOCK_UNIVERSE
    symbols = []
    for category_stocks in STOCK_UNIVERSE.values():
        symbols.extend(category_stocks)
    horizons = [3, 5, 10]  # Test all horizons including new 3-day
    print(f"Testing {len(symbols)} stocks across 3 prediction horizons (3d, 5d, 10d)")
    run_full_pipeline(symbols, horizons)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode
        run_quick_test()
    else:
        # Full pipeline
        run_full_pipeline()
