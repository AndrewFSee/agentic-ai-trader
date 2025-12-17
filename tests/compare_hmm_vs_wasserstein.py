#!/usr/bin/env python3
"""
Compare Rolling Wasserstein vs Rolling HMM
===========================================

Test both approaches on the same data using the winning feature combination:
- realized_vol + trend_strength + volume_momentum

Compare:
1. Sharpe ratio improvement
2. Drawdown reduction
3. Label stability (if applicable)
4. Computational efficiency
"""

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys

# Import both detectors
from models.rolling_hmm_regime_detection import RollingWindowHMM
from rolling_wasserstein_regime_detection import RollingWassersteinRegimeDetector
from polygon import RESTClient
import os

def fetch_data(symbol: str, days_back: int = 1200) -> pd.DataFrame:
    """Fetch data from Polygon"""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    client = RESTClient(api_key)
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days_back)
    
    print(f"Fetching {symbol} data...")
    
    aggs = []
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=1,
        timespan="day",
        from_=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        limit=50000
    ):
        aggs.append(a)
    
    df = pd.DataFrame([{
        'timestamp': a.timestamp,
        'open': a.open,
        'high': a.high,
        'low': a.low,
        'close': a.close,
        'volume': a.volume
    } for a in aggs])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"  Got {len(df)} days")
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute winning features: realized_vol, trend_strength, volume_momentum"""
    df = df.copy()
    
    df['return'] = df['close'].pct_change()
    
    # Winning features from our test
    df['realized_vol'] = df['return'].rolling(20).std()
    
    sma_short = df['close'].rolling(10).mean()
    sma_long = df['close'].rolling(50).mean()
    df['trend_strength'] = (sma_short - sma_long) / df['close']
    
    avg_volume = df['volume'].rolling(20).mean()
    df['volume_momentum'] = (df['volume'] - avg_volume) / (avg_volume + 1e-8)
    
    return df.dropna().reset_index(drop=True)

def backtest_strategy(
    df: pd.DataFrame,
    feature_cols: list,
    detector_type: str,  # 'hmm' or 'wasserstein'
    training_window: int = 500,
    retrain_freq: int = 126,
    persistence: float = 0.80,
    window_size: int = 20  # For Wasserstein only
) -> dict:
    """Backtest a regime-based strategy"""
    
    print(f"\n{'='*70}")
    print(f"BACKTESTING {detector_type.upper()}")
    print(f"{'='*70}")
    
    # Initialize detector
    if detector_type == 'hmm':
        detector = RollingWindowHMM(
            n_states=3,
            feature_columns=feature_cols,
            training_window_days=training_window,
            persistence_prior=persistence
        )
    else:  # wasserstein
        detector = RollingWassersteinRegimeDetector(
            n_regimes=3,
            feature_columns=feature_cols,
            window_size=window_size,
            training_window_days=training_window,
            retrain_frequency_days=retrain_freq
        )
    
    # Prepare data
    df_clean = df[['timestamp', 'close', 'return'] + feature_cols].copy()
    
    # Train initial model
    train_end = training_window
    train_features = df_clean.iloc[:train_end][feature_cols].values
    
    detector.train_on_window(train_features, df_clean['timestamp'].iloc[train_end-1])
    
    # Backtest
    results = []
    last_retrain = train_end
    
    for i in range(train_end, len(df_clean)):
        # Retrain if needed
        if i - last_retrain >= retrain_freq:
            window_start = max(0, i - training_window)
            retrain_features = df_clean.iloc[window_start:i][feature_cols].values
            detector.train_on_window(retrain_features, df_clean['timestamp'].iloc[i-1])
            last_retrain = i
        
        # Predict
        if detector_type == 'hmm':
            current_features = df_clean.iloc[max(0, i-20):i+1][feature_cols].values
        else:  # wasserstein needs at least window_size
            current_features = df_clean.iloc[max(0, i-window_size):i+1][feature_cols].values
        
        try:
            pred = detector.predict_forward_filter(current_features)
            regime = pred['most_likely_state']
            confidence = pred['confidence']
        except:
            regime = 1
            confidence = 0.33
        
        results.append({
            'date': df_clean.iloc[i]['timestamp'],
            'close': df_clean.iloc[i]['close'],
            'return': df_clean.iloc[i]['return'],
            'regime': regime,
            'confidence': confidence
        })
    
    bt_df = pd.DataFrame(results)
    
    # Label by volatility
    for state in range(3):
        mask = bt_df['regime'] == state
        if mask.sum() > 0:
            bt_df.loc[mask, 'regime_vol'] = bt_df.loc[mask, 'return'].std()
    
    vol_map = bt_df.groupby('regime')['regime_vol'].first().sort_values().reset_index()
    vol_map['new_regime'] = range(len(vol_map))
    regime_mapping = dict(zip(vol_map['regime'], vol_map['new_regime']))
    bt_df['regime'] = bt_df['regime'].map(regime_mapping)
    
    # Trading strategy
    bt_df['position'] = 0.0
    bt_df.loc[bt_df['regime'] == 0, 'position'] = 0.0   # Low vol = Cash
    bt_df.loc[bt_df['regime'] == 1, 'position'] = 0.5   # Med vol = Half
    bt_df.loc[bt_df['regime'] == 2, 'position'] = 1.0   # High vol = Full
    
    bt_df['strategy_return'] = bt_df['position'].shift(1) * bt_df['return']
    bt_df['strategy_return'] = bt_df['strategy_return'].fillna(0)
    
    # Compute metrics
    total_days = len(bt_df)
    years = total_days / 252
    
    bh_total = (1 + bt_df['return']).prod() - 1
    bh_annual = (1 + bh_total) ** (1/years) - 1
    bh_sharpe = bt_df['return'].mean() / bt_df['return'].std() * np.sqrt(252)
    bh_cum = (1 + bt_df['return']).cumprod()
    bh_dd = (bh_cum / bh_cum.cummax() - 1).min()
    
    strat_total = (1 + bt_df['strategy_return']).prod() - 1
    strat_annual = (1 + strat_total) ** (1/years) - 1
    strat_sharpe = bt_df['strategy_return'].mean() / bt_df['strategy_return'].std() * np.sqrt(252)
    strat_cum = (1 + bt_df['strategy_return']).cumprod()
    strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
    
    # Regime stats
    regime_dist = bt_df['regime'].value_counts().to_dict()
    
    results = {
        'detector_type': detector_type,
        'total_days': total_days,
        'years': years,
        'bh_total_return': bh_total,
        'bh_annual_return': bh_annual,
        'bh_sharpe': bh_sharpe,
        'bh_max_drawdown': bh_dd,
        'strat_total_return': strat_total,
        'strat_annual_return': strat_annual,
        'strat_sharpe': strat_sharpe,
        'strat_max_drawdown': strat_dd,
        'sharpe_improvement': strat_sharpe - bh_sharpe,
        'drawdown_improvement': strat_dd - bh_dd,
        'return_diff': strat_total - bh_total,
        'regime_distribution': regime_dist
    }
    
    return results

def print_comparison(hmm_results: dict, wass_results: dict, symbol: str):
    """Print side-by-side comparison"""
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: Rolling HMM vs Rolling Wasserstein - {symbol}")
    print(f"{'='*100}")
    
    print(f"\n{'Metric':<30} {'HMM':<25} {'Wasserstein':<25} {'Winner'}")
    print("-" * 100)
    
    # Sharpe improvement
    hmm_sharpe = hmm_results['sharpe_improvement']
    wass_sharpe = wass_results['sharpe_improvement']
    winner = "HMM" if hmm_sharpe > wass_sharpe else "Wasserstein"
    print(f"{'Sharpe Improvement':<30} {hmm_sharpe:>+.4f}{'':<20} {wass_sharpe:>+.4f}{'':<20} {winner}")
    
    # Drawdown improvement
    hmm_dd = hmm_results['drawdown_improvement']
    wass_dd = wass_results['drawdown_improvement']
    winner = "HMM" if hmm_dd > wass_dd else "Wasserstein"
    print(f"{'Drawdown Improvement':<30} {hmm_dd:>+.2%}{'':<20} {wass_dd:>+.2%}{'':<20} {winner}")
    
    # Return difference
    hmm_ret = hmm_results['return_diff']
    wass_ret = wass_results['return_diff']
    winner = "HMM" if hmm_ret > wass_ret else "Wasserstein"
    print(f"{'Return Difference':<30} {hmm_ret:>+.2%}{'':<20} {wass_ret:>+.2%}{'':<20} {winner}")
    
    # Absolute Sharpe
    print(f"\n{'Absolute Sharpe Ratios:':<30}")
    print(f"{'  Buy & Hold':<30} {hmm_results['bh_sharpe']:.4f}{'':<20} {wass_results['bh_sharpe']:.4f}")
    print(f"{'  Strategy':<30} {hmm_results['strat_sharpe']:.4f}{'':<20} {wass_results['strat_sharpe']:.4f}")
    
    print(f"\n{'Absolute Max Drawdowns:':<30}")
    print(f"{'  Buy & Hold':<30} {hmm_results['bh_max_drawdown']:.2%}{'':<20} {wass_results['bh_max_drawdown']:.2%}")
    print(f"{'  Strategy':<30} {hmm_results['strat_max_drawdown']:.2%}{'':<20} {wass_results['strat_max_drawdown']:.2%}")
    
    print(f"\n{'='*100}")
    print(f"OVERALL WINNER: ", end="")
    
    # Score system: Sharpe (most important), Drawdown, Returns
    hmm_score = 0
    wass_score = 0
    
    if hmm_sharpe > wass_sharpe:
        hmm_score += 3
    else:
        wass_score += 3
    
    if hmm_dd > wass_dd:
        hmm_score += 2
    else:
        wass_score += 2
    
    if hmm_ret > wass_ret:
        hmm_score += 1
    else:
        wass_score += 1
    
    if hmm_score > wass_score:
        print(f"Rolling HMM (Score: {hmm_score} vs {wass_score})")
    elif wass_score > hmm_score:
        print(f"Rolling Wasserstein (Score: {wass_score} vs {hmm_score})")
    else:
        print(f"TIE (Score: {hmm_score} vs {wass_score})")
    
    print(f"{'='*100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Rolling HMM vs Wasserstein")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--training-window', type=int, default=500, help='Training window days')
    parser.add_argument('--retrain-freq', type=int, default=126, help='Retrain frequency')
    parser.add_argument('--persistence', type=float, default=0.80, help='HMM persistence')
    parser.add_argument('--window-size', type=int, default=20, help='Wasserstein window size')
    
    args = parser.parse_args()
    
    print("="*100)
    print("ROLLING HMM vs ROLLING WASSERSTEIN COMPARISON")
    print("="*100)
    print(f"Symbol: {args.symbol}")
    print(f"Features: realized_vol + trend_strength + volume_momentum (winning combo)")
    print(f"Training Window: {args.training_window} days")
    print(f"Retrain Frequency: {args.retrain_freq} days")
    print()
    
    # Fetch and prepare data
    df = fetch_data(args.symbol, days_back=1200)
    df = compute_features(df)
    
    feature_cols = ['realized_vol', 'trend_strength', 'volume_momentum']
    
    # Run HMM backtest
    hmm_results = backtest_strategy(
        df, feature_cols, 'hmm',
        training_window=args.training_window,
        retrain_freq=args.retrain_freq,
        persistence=args.persistence
    )
    
    # Run Wasserstein backtest
    wass_results = backtest_strategy(
        df, feature_cols, 'wasserstein',
        training_window=args.training_window,
        retrain_freq=args.retrain_freq,
        window_size=args.window_size
    )
    
    # Compare
    print_comparison(hmm_results, wass_results, args.symbol)
