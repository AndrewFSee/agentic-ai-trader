#!/usr/bin/env python3
"""
Backtest Rolling HMM Against Buy-and-Hold
=========================================

Tests trading strategy based on rolling-window HMM regimes
against simple buy-and-hold.

Strategy:
- Train HMM on rolling window
- Go LONG in Bullish regime
- Go SHORT or CASH in Bearish regime  
- NEUTRAL in Sideways regime
- Rebalance quarterly when model retrains

Comparison metrics:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
"""

import warnings
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import argparse
from models.rolling_hmm_regime_detection import RollingWindowHMM


def calculate_metrics(returns: pd.Series) -> Dict:
    """Calculate performance metrics."""
    total_return = (1 + returns).prod() - 1
    
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(returns)
    }


def backtest_hmm_strategy(
    symbol: str,
    training_window: int = 756,
    retrain_freq: int = 63,
    persistence: float = 0.90,
    use_volatility_label: bool = True
) -> Dict:
    """
    Backtest HMM-based regime trading strategy.
    
    Args:
        symbol: Stock ticker
        training_window: Days for HMM training
        retrain_freq: Days between retraining
        persistence: State persistence parameter
        use_volatility_label: If True, label states by volatility instead of return
    
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING HMM STRATEGY: {symbol}")
    print(f"{'='*70}")
    
    # Initialize detector
    detector = RollingWindowHMM(
        symbol=symbol,
        training_window_days=training_window,
        retrain_frequency_days=retrain_freq,
        persistence_strength=persistence
    )
    
    # Fetch data
    detector.fetch_data(lookback_days=1500)
    detector.calculate_features()
    
    full_data = detector.features
    
    print(f"Backtest period: {full_data['date'].iloc[0].date()} to {full_data['date'].iloc[-1].date()}")
    print(f"Total days: {len(full_data)}")
    
    # We need enough data for initial training
    if len(full_data) < training_window + 252:  # Need at least 1 year after training
        raise ValueError(f"Insufficient data: {len(full_data)} days")
    
    # Walk forward through time, retraining periodically
    backtest_results = []
    last_train_idx = training_window
    
    # Initial training
    print(f"\n{'='*70}")
    print(f"INITIAL TRAINING")
    print(f"{'='*70}")
    
    detector_instance = RollingWindowHMM(
        symbol=symbol,
        training_window_days=training_window,
        persistence_strength=persistence,
        random_state=42
    )
    detector_instance.features = full_data
    detector_instance.data = pd.DataFrame()
    
    train_start = 0
    train_end = training_window
    detector_instance.train_on_window(window_start_idx=train_start, window_end_idx=train_end)
    
    # Use volatility-based labeling if specified
    if use_volatility_label:
        # Relabel by volatility instead of return
        train_window_data = full_data.iloc[train_start:train_end]
        regime_vols = {}
        for state in range(3):
            mask = detector_instance.model.predict(
                detector_instance.scaler.transform(train_window_data.iloc[:, 3:].values)
            ) == state
            regime_vols[state] = train_window_data.loc[mask, 'log_return'].std()
        
        # Sort by volatility: 0=low vol, 1=med vol, 2=high vol
        sorted_states = sorted(regime_vols.items(), key=lambda x: x[1])
        vol_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
        
        # Override return-based mapping with volatility-based
        detector_instance.regime_mapping = vol_mapping
        
        print(f"Using volatility-based regime labeling:")
        for old, new in vol_mapping.items():
            vol = regime_vols[old]
            print(f"  State {old} → Regime {new} (vol={vol*np.sqrt(252):.1%})")
    
    current_model = detector_instance
    
    # Simulate trading from end of training window forward
    for i in range(train_end, len(full_data)):
        # Check if we need to retrain
        if i - last_train_idx >= retrain_freq:
            print(f"\nRetraining at index {i} ({full_data['date'].iloc[i].date()})")
            
            # Retrain on new window
            new_detector = RollingWindowHMM(
                symbol=symbol,
                training_window_days=training_window,
                persistence_strength=persistence,
                random_state=42
            )
            new_detector.features = full_data
            new_detector.data = pd.DataFrame()
            
            new_train_start = max(0, i - training_window)
            new_detector.train_on_window(window_start_idx=new_train_start, window_end_idx=i)
            
            # Apply volatility labeling
            if use_volatility_label:
                train_data = full_data.iloc[new_train_start:i]
                regime_vols = {}
                for state in range(3):
                    mask = new_detector.model.predict(
                        new_detector.scaler.transform(train_data.iloc[:, 3:].values)
                    ) == state
                    regime_vols[state] = train_data.loc[mask, 'log_return'].std()
                
                sorted_states = sorted(regime_vols.items(), key=lambda x: x[1])
                vol_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
                new_detector.regime_mapping = vol_mapping
            
            current_model = new_detector
            last_train_idx = i
        
        # Get regime for this day using current model
        obs = full_data.iloc[i:i+1].iloc[:, 3:].values
        obs_scaled = current_model.scaler.transform(obs)
        raw_state = current_model.model.predict(obs_scaled)[0]
        regime = current_model.regime_mapping[raw_state]
        
        # Get actual return for this day
        actual_return = full_data.iloc[i]['log_return']
        
        # Trading logic based on regime
        if regime == 2:  # Bullish/High vol
            strategy_return = actual_return  # Long position
            position = 'LONG'
        elif regime == 0:  # Bearish/Low vol
            strategy_return = 0  # Cash position (or could be -actual_return for short)
            position = 'CASH'
        else:  # Sideways/Med vol
            strategy_return = actual_return * 0.5  # Reduced position
            position = 'HALF'
        
        backtest_results.append({
            'date': full_data.iloc[i]['date'],
            'close': full_data.iloc[i]['close'],
            'regime': regime,
            'position': position,
            'actual_return': actual_return,
            'strategy_return': strategy_return
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(backtest_results)
    
    # Calculate metrics
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")
    
    # Buy and hold
    bh_returns = results_df['actual_return']
    bh_metrics = calculate_metrics(bh_returns)
    
    # Strategy
    strat_returns = pd.Series(results_df['strategy_return'].values)
    strat_metrics = calculate_metrics(strat_returns)
    
    print(f"\n{'Buy & Hold':<20} | {'HMM Strategy':<20}")
    print(f"{'-'*42}")
    print(f"{'Total Return:':<20} | {bh_metrics['total_return']:>7.2%} | {strat_metrics['total_return']:>7.2%}")
    print(f"{'Annual Return:':<20} | {bh_metrics['annual_return']:>7.2%} | {strat_metrics['annual_return']:>7.2%}")
    print(f"{'Sharpe Ratio:':<20} | {bh_metrics['sharpe_ratio']:>7.2f} | {strat_metrics['sharpe_ratio']:>7.2f}")
    print(f"{'Max Drawdown:':<20} | {bh_metrics['max_drawdown']:>7.2%} | {strat_metrics['max_drawdown']:>7.2%}")
    print(f"{'Win Rate:':<20} | {bh_metrics['win_rate']:>7.2%} | {strat_metrics['win_rate']:>7.2%}")
    
    # Regime breakdown
    print(f"\n{'='*70}")
    print(f"REGIME BREAKDOWN")
    print(f"{'='*70}")
    
    regime_names = {0: "Low Vol/Bearish", 1: "Med Vol/Sideways", 2: "High Vol/Bullish"}
    for regime_num in [0, 1, 2]:
        regime_mask = results_df['regime'] == regime_num
        days = regime_mask.sum()
        pct = days / len(results_df) * 100
        avg_return = results_df.loc[regime_mask, 'actual_return'].mean() * 252 * 100
        
        print(f"\n{regime_names[regime_num]}:")
        print(f"  Days: {days} ({pct:.1f}%)")
        print(f"  Avg annual return: {avg_return:+.2f}%")
    
    # Outperformance
    outperformance = strat_metrics['total_return'] - bh_metrics['total_return']
    print(f"\n{'='*70}")
    if outperformance > 0:
        print(f"✅ HMM Strategy OUTPERFORMED by {outperformance:+.2%}")
    else:
        print(f"❌ HMM Strategy UNDERPERFORMED by {outperformance:+.2%}")
    print(f"{'='*70}")
    
    return {
        'symbol': symbol,
        'backtest_period': {
            'start': str(results_df['date'].iloc[0].date()),
            'end': str(results_df['date'].iloc[-1].date()),
            'days': len(results_df)
        },
        'buy_hold': bh_metrics,
        'hmm_strategy': strat_metrics,
        'outperformance': float(outperformance),
        'results_df': results_df
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest HMM strategy vs buy-hold')
    parser.add_argument('--symbol', type=str, default='AAPL')
    parser.add_argument('--training-window', type=int, default=756)
    parser.add_argument('--retrain-freq', type=int, default=63)
    parser.add_argument('--persistence', type=float, default=0.90)
    parser.add_argument('--volatility-label', action='store_true',
                       help='Use volatility-based labeling instead of return-based')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"HMM REGIME TRADING BACKTEST")
    print(f"{'#'*70}")
    print(f"\nConfiguration:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Training window: {args.training_window} days")
    print(f"  Retrain frequency: {args.retrain_freq} days")
    print(f"  Persistence: {args.persistence}")
    print(f"  Labeling: {'Volatility-based' if args.volatility_label else 'Return-based'}")
    
    results = backtest_hmm_strategy(
        symbol=args.symbol,
        training_window=args.training_window,
        retrain_freq=args.retrain_freq,
        persistence=args.persistence,
        use_volatility_label=args.volatility_label
    )
    
    print(f"\n{'='*70}")
    print(f"✓ Backtest complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
