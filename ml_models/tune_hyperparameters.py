"""
Hyperparameter tuning using Optuna for XGBoost models.
Optimizes for Sharpe ratio on validation set.
"""
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import PREDICTION_HORIZONS, START_DATE_STR, END_DATE_STR
from data_collection import fetch_daily_bars, fetch_spy_data
from feature_engineering import engineer_features, get_feature_columns
from backtest import backtest_model

def calculate_sharpe(returns: pd.Series) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def prepare_data(symbol: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for training."""
    print(f"\nPreparing data for {symbol} ({horizon}d)...")
    
    # Fetch data
    df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
    spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
    
    # Engineer features
    df = engineer_features(symbol, df, spy_df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    target_col = f'target_direction_{horizon}d'
    
    # Drop NaN
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Train/val/test split (60/20/20)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Features: {len(feature_cols)}")
    
    return train_df, val_df, test_df, feature_cols, target_col

def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame, 
              feature_cols: list, target_col: str, symbol: str, horizon: int) -> float:
    """Objective function for Optuna optimization."""
    
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # Train model
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # Backtest on validation set
    y_pred = model.predict(X_val)
    
    backtest_result = backtest_model(val_df, y_pred, 'XGBoost', symbol, horizon)
    sharpe = backtest_result['sharpe_ratio']
    
    return sharpe

def tune_hyperparameters(symbol: str, horizon: int, n_trials: int = 50) -> Dict:
    """Run hyperparameter tuning for a specific symbol and horizon."""
    print("="*80)
    print(f"HYPERPARAMETER TUNING: {symbol} - {horizon}d")
    print("="*80)
    
    # Prepare data
    train_df, val_df, test_df, feature_cols, target_col = prepare_data(symbol, horizon)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Optimize
    print(f"\nRunning {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, train_df, val_df, feature_cols, target_col, symbol, horizon),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Best parameters
    best_params = study.best_params
    best_sharpe = study.best_value
    
    print(f"\n{'='*80}")
    print(f"BEST HYPERPARAMETERS (Sharpe: {best_sharpe:.3f})")
    print(f"{'='*80}")
    for param, value in best_params.items():
        print(f"  {param:20s}: {value}")
    
    # Train final model with best params
    print(f"\nTraining final model with best parameters...")
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    final_params = {**best_params, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, verbose=False)
    
    # Test performance
    y_pred_test = final_model.predict(X_test)
    
    test_result = backtest_model(test_df, y_pred_test, 'XGBoost', symbol, horizon)
    test_sharpe = test_result['sharpe_ratio']
    test_return = test_result['total_return']
    
    print(f"\n{'='*80}")
    print(f"TEST SET PERFORMANCE")
    print(f"{'='*80}")
    print(f"  Sharpe Ratio:  {test_sharpe:.3f}")
    print(f"  Total Return:  {test_return*100:.2f}%")
    print(f"  Max Drawdown:  {test_result['max_drawdown']*100:.2f}%")
    
    return {
        'symbol': symbol,
        'horizon': horizon,
        'best_params': best_params,
        'val_sharpe': best_sharpe,
        'test_sharpe': test_sharpe,
        'test_return': test_return,
        'test_result': test_result,
        'study': study
    }

def tune_portfolio(symbols: list, horizons: list = [5, 10], n_trials: int = 50):
    """Tune hyperparameters for multiple symbols and horizons."""
    print("="*80)
    print("PORTFOLIO HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Symbols: {symbols}")
    print(f"Horizons: {horizons}")
    print(f"Trials per symbol/horizon: {n_trials}")
    print()
    
    results = []
    
    for symbol in symbols:
        for horizon in horizons:
            result = tune_hyperparameters(symbol, horizon, n_trials)
            results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path('results/ml_models') / f'hyperparameter_tuning_{timestamp}.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert study objects to serializable format
    save_results = []
    for r in results:
        save_r = r.copy()
        save_r.pop('study')  # Can't serialize Optuna study
        save_results.append(save_r)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'symbols': symbols,
            'horizons': horizons,
            'n_trials': n_trials,
            'results': save_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Symbol':<10} {'Horizon':<10} {'Val Sharpe':<12} {'Test Sharpe':<12} {'Test Return':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['symbol']:<10} {r['horizon']:<10} {r['val_sharpe']:<12.3f} {r['test_sharpe']:<12.3f} {r['test_return']*100:<12.2f}%")
    
    return results

if __name__ == "__main__":
    # Test on 3 symbols first
    test_symbols = ['AAPL', 'NVDA', 'JPM']
    
    print("="*80)
    print("XGBoost Hyperparameter Tuning with Optuna")
    print("="*80)
    print("\nThis will optimize XGBoost hyperparameters for maximum Sharpe ratio.")
    print("Using pruned feature set (141 features).")
    print()
    
    # Run tuning
    results = tune_portfolio(
        symbols=test_symbols,
        horizons=[5, 10],  # Focus on 5d and 10d (best performing)
        n_trials=50  # 50 trials per symbol/horizon
    )
    
    print("\n✓ Hyperparameter tuning complete!")
    print("→ Review results and apply best parameters to full 25-stock training")
