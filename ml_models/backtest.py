"""
Backtest module for evaluating ML models against buy-and-hold.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from datetime import datetime

from config import (
    TRANSACTION_COST, INITIAL_CAPITAL, POSITION_SIZE,
    STOP_LOSS, TAKE_PROFIT, RESULTS_DIR, VERBOSE
)


def calculate_returns(prices: pd.Series, positions: pd.Series) -> pd.Series:
    """Calculate strategy returns given positions."""
    returns = prices.pct_change()
    strategy_returns = positions.shift(1) * returns
    return strategy_returns


def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
    """Calculate performance metrics."""
    # Remove NaN
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'num_trades': 0
        }
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized return
    n_days = len(returns)
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe ratio
    excess_returns = returns - 0.0  # Assuming 0% risk-free rate
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Number of trades (position changes)
    num_trades = 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'num_days': n_days
    }
    
    # Add alpha and beta if benchmark provided
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0)
        
        # Beta
        covariance = returns.cov(benchmark_returns)
        benchmark_var = benchmark_returns.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Alpha
        benchmark_total = (1 + benchmark_returns).prod() - 1
        benchmark_ann = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        alpha = annualized_return - beta * benchmark_ann
        
        metrics['beta'] = beta
        metrics['alpha'] = alpha
        metrics['benchmark_return'] = benchmark_total
    
    return metrics


def backtest_model(test_df: pd.DataFrame, y_pred: np.ndarray, 
                  model_name: str, symbol: str, horizon: int) -> Dict:
    """
    Backtest a model's predictions.
    
    Args:
        test_df: Test DataFrame with prices
        y_pred: Model predictions (0 or 1)
        model_name: Name of the model
        symbol: Stock ticker
        horizon: Prediction horizon
    
    Returns:
        Dictionary with backtest results
    """
    # Create positions based on predictions
    # 1 = long, 0 = cash
    positions = pd.Series(y_pred, index=test_df.index)
    
    # Apply transaction costs
    position_changes = positions.diff().fillna(0)
    transaction_costs = position_changes.abs() * TRANSACTION_COST
    
    # Calculate returns
    prices = test_df['close']
    returns = calculate_returns(prices, positions)
    returns = returns - transaction_costs  # Subtract costs
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    # Add model info
    metrics['model'] = model_name
    metrics['symbol'] = symbol
    metrics['horizon'] = horizon
    
    return metrics


def backtest_buy_and_hold(test_df: pd.DataFrame, symbol: str) -> Dict:
    """Backtest buy-and-hold strategy."""
    prices = test_df['close']
    returns = prices.pct_change().dropna()
    
    # Apply single transaction cost at entry
    returns.iloc[0] -= TRANSACTION_COST
    
    metrics = calculate_metrics(returns)
    metrics['model'] = 'Buy-and-Hold'
    metrics['symbol'] = symbol
    metrics['num_trades'] = 1  # Single buy at start
    
    return metrics


def compare_strategies(model_results: List[Dict], buy_hold_result: Dict) -> pd.DataFrame:
    """Compare all strategies and create summary table."""
    all_results = model_results + [buy_hold_result]
    df = pd.DataFrame(all_results)
    
    # Calculate improvement over buy-and-hold
    bh_return = buy_hold_result['total_return']
    df['improvement_vs_bh'] = (df['total_return'] - bh_return) / abs(bh_return) if bh_return != 0 else 0
    
    # Sort by Sharpe ratio
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    return df


def backtest_all_models(results: Dict, symbol: str, horizon: int) -> Dict:
    """
    Backtest all trained models and compare to buy-and-hold.
    
    Args:
        results: Training results from train_all_models()
        symbol: Stock ticker
        horizon: Prediction horizon
    
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {symbol} - {horizon}-day prediction")
    print(f"{'='*80}")
    
    test_df = results['data_splits']['test_df']
    X_test = results['data_splits']['X_test']
    y_test = results['data_splits']['y_test']
    
    model_results = []
    
    # Backtest each model
    for model_name, model_data in results['models'].items():
        print(f"\nBacktesting {model_data['metrics']['model']}...")
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get predictions
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Backtest
        metrics = backtest_model(test_df, y_pred, model_data['metrics']['model'], 
                                symbol, horizon)
        
        if VERBOSE:
            print(f"  Total Return:     {metrics['total_return']:.2%}")
            print(f"  Annualized:       {metrics['annualized_return']:.2%}")
            print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
            print(f"  Win Rate:         {metrics['win_rate']:.2%}")
        
        model_results.append(metrics)
    
    # Backtest buy-and-hold
    print("\nBacktesting Buy-and-Hold...")
    buy_hold_metrics = backtest_buy_and_hold(test_df, symbol)
    
    if VERBOSE:
        print(f"  Total Return:     {buy_hold_metrics['total_return']:.2%}")
        print(f"  Annualized:       {buy_hold_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio:     {buy_hold_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {buy_hold_metrics['max_drawdown']:.2%}")
    
    # Compare strategies
    comparison_df = compare_strategies(model_results, buy_hold_metrics)
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    print(comparison_df[['model', 'total_return', 'annualized_return', 'sharpe_ratio', 
                         'max_drawdown', 'improvement_vs_bh']].to_string(index=False))
    
    # Determine best model
    best_model = comparison_df.iloc[0]
    print(f"\nBEST MODEL: {best_model['model']}")
    print(f"   Sharpe Ratio: {best_model['sharpe_ratio']:.2f}")
    print(f"   Total Return: {best_model['total_return']:.2%}")
    print(f"   vs Buy-Hold:  {best_model['improvement_vs_bh']:+.2%}")
    
    return {
        'model_results': model_results,
        'buy_hold': buy_hold_metrics,
        'comparison': comparison_df,
        'best_model': best_model.to_dict()
    }


if __name__ == "__main__":
    # Test backtesting
    from data_collection import fetch_daily_bars, fetch_spy_data
    from feature_engineering import engineer_features, get_feature_columns
    from train_models import train_all_models
    from config import START_DATE_STR, END_DATE_STR
    
    print("Testing backtest pipeline...")
    
    # Fetch and engineer features
    symbol = "AAPL"
    df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
    spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
    df_features = engineer_features(symbol, df, spy_df)
    feature_cols = get_feature_columns(df_features)
    
    # Train models
    training_results = train_all_models(symbol, df_features, feature_cols, 5)
    
    # Backtest
    backtest_results = backtest_all_models(training_results, symbol, 5)
