#!/usr/bin/env python3
"""
Final Comparison: Paper-Faithful Wasserstein vs Rolling HMM
============================================================

Direct head-to-head comparison on same stocks with same features.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.paper_wasserstein_regime_detection import (
    RollingPaperWassersteinDetector,
    fetch_polygon_bars,
    calculate_features
)

from models.rolling_hmm_regime_detection import RollingWindowHMM


def compare_on_stock(
    symbol: str,
    n_regimes: int = 3,
    window_size: int = 20,
    training_window: int = 500,
    retrain_freq: int = 126,
    high_vol_fraction: float = 0.5,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31"
):
    """Compare Paper Wasserstein vs HMM on a single stock."""
    
    print(f"\n{'='*70}")
    print(f"Comparing on {symbol}")
    print(f"{'='*70}")
    
    # Fetch data
    df = fetch_polygon_bars(symbol, start_date, end_date)
    df = calculate_features(df, window=window_size)
    
    # Split
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train: {len(train_df)} days, Test: {len(test_df)} days")
    
    feature_cols = ['realized_vol', 'trend_strength', 'volume_momentum']
    
    # ========================================================================
    # Paper Wasserstein
    # ========================================================================
    print("\n1. PAPER-FAITHFUL WASSERSTEIN")
    print("-" * 70)
    
    wass_detector = RollingPaperWassersteinDetector(
        n_regimes=n_regimes,
        window_size=window_size,
        training_window_days=training_window,
        retrain_frequency_days=retrain_freq,
        feature_cols=feature_cols
    )
    
    wass_detector.train_on_window(train_df, train_df.index[-1], verbose=True)
    
    wass_predictions = wass_detector.predict_forward_rolling(
        df,
        test_df.index[window_size],
        test_df.index[-1],
        verbose=False
    )
    
    # Backtest Wasserstein
    aligned_wass = wass_predictions.reindex(test_df.index, method='ffill').astype(float).fillna(0.0)
    wass_positions = aligned_wass.copy()
    for i in range(n_regimes - 1):
        wass_positions = wass_positions.replace(float(i), 1.0)
    wass_positions = wass_positions.replace(float(n_regimes - 1), high_vol_fraction)
    
    wass_returns = test_df['returns'] * wass_positions
    wass_returns = wass_returns.dropna()
    
    wass_sharpe = np.sqrt(252) * wass_returns.mean() / wass_returns.std()
    wass_cumret = (1 + wass_returns).cumprod()
    wass_dd = ((wass_cumret - wass_cumret.expanding().max()) / wass_cumret.expanding().max()).min()
    wass_total_return = wass_cumret.iloc[-1] - 1
    
    print(f"  Predictions: {len(wass_predictions)}")
    print(f"  Sharpe: {wass_sharpe:.3f}")
    print(f"  Total Return: {wass_total_return:.2%}")
    print(f"  Max Drawdown: {wass_dd:.2%}")
    
    # ========================================================================
    # Rolling HMM
    # ========================================================================
    print("\n2. ROLLING HMM")
    print("-" * 70)
    
    hmm_detector = RollingWindowHMM(
        n_regimes=n_regimes,
        training_window_days=training_window,
        retrain_frequency_days=retrain_freq,
        persistence_prior=0.8,
        feature_columns=feature_cols
    )
    
    # Extract feature arrays
    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values
    
    # Train HMM
    hmm_detector.train_on_window(train_features, train_df.index[-1])
    
    # Predict on test set (use feature array method)
    hmm_predictions_list = []
    for i in range(len(test_df)):
        # Get features up to current point
        current_features = test_features[:i+1]
        pred = hmm_detector.predict_forward_filter(current_features)
        hmm_predictions_list.append(pred['most_likely_state'])
    
    # Create Series with proper index
    hmm_predictions = pd.Series(hmm_predictions_list, index=test_df.index)
    aligned_hmm = hmm_predictions.astype(float).fillna(0.0)
    
    # Convert to positions
    hmm_positions = aligned_hmm.copy()
    for i in range(n_regimes - 1):
        hmm_positions = hmm_positions.replace(float(i), 1.0)
    hmm_positions = hmm_positions.replace(float(n_regimes - 1), high_vol_fraction)
    
    hmm_returns = test_df['returns'] * hmm_positions
    hmm_returns = hmm_returns.dropna()
    
    hmm_sharpe = np.sqrt(252) * hmm_returns.mean() / hmm_returns.std()
    hmm_cumret = (1 + hmm_returns).cumprod()
    hmm_dd = ((hmm_cumret - hmm_cumret.expanding().max()) / hmm_cumret.expanding().max()).min()
    hmm_total_return = hmm_cumret.iloc[-1] - 1
    
    print(f"  Predictions: {len(hmm_predictions)}")
    print(f"  Sharpe: {hmm_sharpe:.3f}")
    print(f"  Total Return: {hmm_total_return:.2%}")
    print(f"  Max Drawdown: {hmm_dd:.2%}")
    
    # ========================================================================
    # Buy and Hold
    # ========================================================================
    print("\n3. BUY AND HOLD")
    print("-" * 70)
    
    bh_returns = test_df['returns'].dropna()
    bh_sharpe = np.sqrt(252) * bh_returns.mean() / bh_returns.std()
    bh_cumret = (1 + bh_returns).cumprod()
    bh_dd = ((bh_cumret - bh_cumret.expanding().max()) / bh_cumret.expanding().max()).min()
    bh_total_return = bh_cumret.iloc[-1] - 1
    
    print(f"  Sharpe: {bh_sharpe:.3f}")
    print(f"  Total Return: {bh_total_return:.2%}")
    print(f"  Max Drawdown: {bh_dd:.2%}")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'Paper Wass':<15} {'HMM':<15} {'B&H':<15} {'Winner'}")
    print("-" * 80)
    
    # Sharpe
    sharpe_winner = "Paper Wass" if wass_sharpe > max(hmm_sharpe, bh_sharpe) else ("HMM" if hmm_sharpe > bh_sharpe else "B&H")
    print(f"{'Sharpe Ratio':<25} {wass_sharpe:<15.3f} {hmm_sharpe:<15.3f} {bh_sharpe:<15.3f} {sharpe_winner}")
    
    # Return
    ret_winner = "Paper Wass" if wass_total_return > max(hmm_total_return, bh_total_return) else ("HMM" if hmm_total_return > bh_total_return else "B&H")
    print(f"{'Total Return':<25} {wass_total_return:<15.2%} {hmm_total_return:<15.2%} {bh_total_return:<15.2%} {ret_winner}")
    
    # Drawdown (less negative is better)
    dd_winner = "Paper Wass" if wass_dd > max(hmm_dd, bh_dd) else ("HMM" if hmm_dd > bh_dd else "B&H")
    print(f"{'Max Drawdown':<25} {wass_dd:<15.2%} {hmm_dd:<15.2%} {bh_dd:<15.2%} {dd_winner}")
    
    # Improvements over B&H
    print(f"\n{'Improvements over B&H':<25} {'Paper Wass':<15} {'HMM':<15}")
    print("-" * 60)
    print(f"{'Sharpe Delta':<25} {wass_sharpe - bh_sharpe:<+15.3f} {hmm_sharpe - bh_sharpe:<+15.3f}")
    print(f"{'Return Delta':<25} {wass_total_return - bh_total_return:<+15.2%} {hmm_total_return - bh_total_return:<+15.2%}")
    print(f"{'Drawdown Delta':<25} {wass_dd - bh_dd:<+15.2%} {hmm_dd - bh_dd:<+15.2%}")
    
    # Score
    wass_score = sum([
        wass_sharpe > hmm_sharpe,
        wass_sharpe > bh_sharpe,
        wass_total_return > hmm_total_return,
        wass_total_return > bh_total_return,
        wass_dd > hmm_dd,
        wass_dd > bh_dd
    ])
    
    hmm_score = sum([
        hmm_sharpe > wass_sharpe,
        hmm_sharpe > bh_sharpe,
        hmm_total_return > wass_total_return,
        hmm_total_return > bh_total_return,
        hmm_dd > wass_dd,
        hmm_dd > bh_dd
    ])
    
    print(f"\n{'Overall Score (out of 6)':<25} {wass_score:<15} {hmm_score:<15}")
    print(f"{'WINNER':<25} {'PAPER WASS' if wass_score > hmm_score else ('HMM' if hmm_score > wass_score else 'TIE')}")
    
    return {
        'symbol': symbol,
        'paper_wass': {
            'sharpe': float(wass_sharpe),
            'total_return': float(wass_total_return),
            'max_dd': float(wass_dd),
            'n_predictions': len(wass_predictions)
        },
        'hmm': {
            'sharpe': float(hmm_sharpe),
            'total_return': float(hmm_total_return),
            'max_dd': float(hmm_dd),
            'n_predictions': len(hmm_predictions)
        },
        'buy_hold': {
            'sharpe': float(bh_sharpe),
            'total_return': float(bh_total_return),
            'max_dd': float(bh_dd)
        },
        'scores': {
            'paper_wass': int(wass_score),
            'hmm': int(hmm_score)
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--output", default=None)
    
    args = parser.parse_args()
    
    result = compare_on_stock(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"final_comparison_{args.symbol}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
