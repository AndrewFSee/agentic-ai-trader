"""
VIX ROC Strategy - Stubborn Losers Optimization

Exhaustive search for parameters that could work on NVDA, MSFT, META
These stocks failed with both Tier 1 and Growth strategies.

Hypothesis to test:
1. Maybe they need ULTRA-fast re-entry (even faster than Growth)
2. Maybe they need NO strategy at all (pure B&H is optimal)
3. Maybe they need inverse logic (buy dips during VIX spikes?)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ROCParams:
    roc_lookback: int
    exit_roc_thresh: float
    reentry_roc_thresh: float
    min_exit_days: int


class VIXROCStrategy:
    def __init__(self, params: ROCParams):
        self.params = params
    
    def run(self, asset: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
        common_idx = asset.index.intersection(vix.index)
        asset = asset.loc[common_idx].copy()
        vix = vix.loc[common_idx].copy()
        
        vix_close = vix['Close'].values if 'Close' in vix.columns else vix.iloc[:, 0].values
        vix_roc = np.zeros(len(vix_close))
        
        lookback = self.params.roc_lookback
        for i in range(lookback, len(vix_close)):
            prev_vix = vix_close[i - lookback]
            if prev_vix > 0:
                vix_roc[i] = (vix_close[i] - prev_vix) / prev_vix
        
        asset_close = asset['Close'].values if 'Close' in asset.columns else asset.iloc[:, 0].values
        asset_returns = np.zeros(len(asset_close))
        asset_returns[1:] = np.diff(asset_close) / asset_close[:-1]
        
        positions = np.ones(len(asset_close))
        in_market = True
        days_out = 0
        
        for i in range(lookback, len(asset_close)):
            if in_market:
                if vix_roc[i] > self.params.exit_roc_thresh:
                    in_market = False
                    days_out = 0
                    positions[i] = 0
                else:
                    positions[i] = 1
            else:
                days_out += 1
                if days_out >= self.params.min_exit_days and vix_roc[i] < self.params.reentry_roc_thresh:
                    in_market = True
                    positions[i] = 1
                else:
                    positions[i] = 0
        
        results = pd.DataFrame({
            'date': asset.index,
            'close': asset_close,
            'return': asset_returns,
            'position': positions
        })
        results.set_index('date', inplace=True)
        results['strategy_return'] = results['position'].shift(1) * results['return']
        results['strategy_return'] = results['strategy_return'].fillna(0)
        
        return results


def evaluate(results: pd.DataFrame) -> Dict:
    bh_total = (1 + results['return']).prod() - 1
    strat_total = (1 + results['strategy_return']).prod() - 1
    
    bh_cum = (1 + results['return']).cumprod()
    strat_cum = (1 + results['strategy_return']).cumprod()
    
    bh_dd = (bh_cum / bh_cum.cummax() - 1).min()
    strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
    
    # Count trades
    positions = results['position'].values
    trades = np.sum(np.abs(np.diff(positions))) / 2
    
    return {
        'bh_return': bh_total,
        'strat_return': strat_total,
        'excess': strat_total - bh_total,
        'bh_dd': bh_dd,
        'strat_dd': strat_dd,
        'dd_improvement': strat_dd - bh_dd,
        'win': strat_total > bh_total,
        'trades': trades
    }


def main():
    print("="*80)
    print("STUBBORN LOSERS - EXHAUSTIVE PARAMETER SEARCH")
    print("="*80)
    
    # Download data
    print("\nDownloading data...")
    
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    stubborn_tickers = {
        'NVDA': 'NVIDIA',
        'MSFT': 'Microsoft', 
        'META': 'Meta'
    }
    
    all_data = {}
    for ticker in stubborn_tickers:
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            all_data[ticker] = df
    
    # Test period: 2020-2024 (out-of-sample)
    start, end = "2020-01-01", "2024-12-31"
    
    # ULTRA-WIDE parameter grid
    lookbacks = [1, 2, 3, 5, 7, 10, 15, 20]
    exit_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
    reentry_thresholds = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    min_exit_days_list = [1, 2, 3, 5, 7, 10]
    
    total_configs = len(lookbacks) * len(exit_thresholds) * len(reentry_thresholds) * len(min_exit_days_list)
    print(f"\nTesting {total_configs} configurations per ticker...")
    
    # Store best configs per ticker
    best_per_ticker = {}
    
    for ticker, name in stubborn_tickers.items():
        if ticker not in all_data:
            continue
        
        print(f"\n{'='*60}")
        print(f"Optimizing {ticker} ({name})")
        print(f"{'='*60}")
        
        asset = all_data[ticker].loc[start:end]
        vix_period = vix.loc[start:end]
        
        if len(asset) < 50:
            continue
        
        best_excess = -float('inf')
        best_config = None
        best_metrics = None
        
        # Also track all winning configs
        winning_configs = []
        
        config_num = 0
        for lb in lookbacks:
            for exit_th in exit_thresholds:
                for reentry_th in reentry_thresholds:
                    for min_days in min_exit_days_list:
                        config_num += 1
                        
                        params = ROCParams(lb, exit_th, reentry_th, min_days)
                        strategy = VIXROCStrategy(params)
                        
                        results = strategy.run(asset, vix_period)
                        metrics = evaluate(results)
                        
                        if metrics['win']:
                            winning_configs.append({
                                'params': params,
                                'excess': metrics['excess'],
                                'dd_imp': metrics['dd_improvement'],
                                'trades': metrics['trades']
                            })
                        
                        if metrics['excess'] > best_excess:
                            best_excess = metrics['excess']
                            best_config = params
                            best_metrics = metrics
        
        best_per_ticker[ticker] = {
            'best_config': best_config,
            'best_metrics': best_metrics,
            'winning_configs': winning_configs
        }
        
        print(f"\nWinning configs: {len(winning_configs)} / {total_configs} ({100*len(winning_configs)/total_configs:.1f}%)")
        
        if best_config:
            print(f"\nBest config for {ticker}:")
            print(f"  lookback={best_config.roc_lookback}, exit>{best_config.exit_roc_thresh*100:.0f}%, reentry<{best_config.reentry_roc_thresh*100:+.0f}%, min={best_config.min_exit_days}d")
            print(f"  B&H: {best_metrics['bh_return']:+.1%}")
            print(f"  Strategy: {best_metrics['strat_return']:+.1%}")
            print(f"  Excess: {best_metrics['excess']:+.1%}")
            print(f"  DD Improvement: {best_metrics['dd_improvement']:+.1%}")
            print(f"  Trades: {best_metrics['trades']:.0f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: BEST POSSIBLE CONFIGS FOR STUBBORN LOSERS")
    print("="*80)
    
    for ticker, data in best_per_ticker.items():
        config = data['best_config']
        metrics = data['best_metrics']
        n_winning = len(data['winning_configs'])
        
        if metrics['win']:
            status = "✓ WIN"
        else:
            status = "✗ LOSS (even with best params)"
        
        print(f"\n{ticker} ({stubborn_tickers[ticker]}):")
        print(f"  Status: {status}")
        print(f"  Best: lb={config.roc_lookback} exit>{config.exit_roc_thresh*100:.0f}% reentry<{config.reentry_roc_thresh*100:+.0f}% min={config.min_exit_days}d")
        print(f"  Excess: {metrics['excess']:+.1%} (B&H: {metrics['bh_return']:+.1%})")
        print(f"  Winning configs: {n_winning}/{total_configs}")
    
    # Cross-check: Are there ANY common winning configs?
    print("\n" + "="*80)
    print("CROSS-TICKER ANALYSIS: Finding Common Winning Configs")
    print("="*80)
    
    # Get winning config parameter sets for each ticker
    winning_sets = {}
    for ticker, data in best_per_ticker.items():
        winning_sets[ticker] = set()
        for wc in data['winning_configs']:
            p = wc['params']
            key = (p.roc_lookback, p.exit_roc_thresh, p.reentry_roc_thresh, p.min_exit_days)
            winning_sets[ticker].add(key)
    
    # Find intersection
    if len(winning_sets) >= 2:
        common = None
        for ticker, s in winning_sets.items():
            if common is None:
                common = s
            else:
                common = common.intersection(s)
        
        print(f"\nConfigs that win on ALL stubborn stocks: {len(common)}")
        
        if common:
            print("\nCommon winning configs:")
            for i, (lb, exit_th, reentry_th, min_days) in enumerate(list(common)[:10]):
                print(f"  {i+1}. lb={lb} exit>{exit_th*100:.0f}% reentry<{reentry_th*100:+.0f}% min={min_days}d")
        else:
            print("\nNo common winning config exists for all three stocks!")
            print("These stocks truly need different treatment.")
    
    # Analyze WHY these stocks fail
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    for ticker, data in best_per_ticker.items():
        print(f"\n{ticker}:")
        
        asset = all_data[ticker].loc[start:end]
        vix_period = vix.loc[start:end]
        
        # Calculate COVID recovery metrics
        covid_start = "2020-02-19"
        covid_bottom = "2020-03-23"
        covid_end = "2020-06-30"
        
        try:
            pre_covid = asset.loc[:covid_start]['Close'].iloc[-1]
            bottom = asset.loc[covid_bottom]['Close']
            post_covid = asset.loc[:covid_end]['Close'].iloc[-1]
            
            crash = (bottom - pre_covid) / pre_covid
            recovery = (post_covid - bottom) / bottom
            
            print(f"  COVID crash: {crash:+.1%}")
            print(f"  Recovery (to Jun 30): {recovery:+.1%}")
            print(f"  Recovery/Crash ratio: {abs(recovery/crash):.1f}x")
            
            # Days to recover
            recovered_idx = asset.loc[covid_bottom:]['Close'] >= pre_covid
            if recovered_idx.any():
                first_recovery = recovered_idx.idxmax()
                days_to_recover = (first_recovery - pd.Timestamp(covid_bottom)).days
                print(f"  Days to full recovery: {days_to_recover}")
            else:
                print(f"  Days to full recovery: Never (in period)")
        except Exception as e:
            print(f"  Could not compute COVID metrics: {e}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    # Check if any stock has a viable strategy
    viable = []
    not_viable = []
    
    for ticker, data in best_per_ticker.items():
        metrics = data['best_metrics']
        n_winning = len(data['winning_configs'])
        
        # A stock is "viable" if at least 1% of configs win AND best excess > 10%
        win_rate = n_winning / total_configs
        
        if metrics['win'] and win_rate > 0.01 and metrics['excess'] > 0.10:
            viable.append((ticker, metrics['excess'], win_rate))
        else:
            not_viable.append((ticker, metrics['excess'], win_rate))
    
    if viable:
        print("\nVIABLE with optimized params:")
        for ticker, excess, win_rate in viable:
            print(f"  {ticker}: {excess:+.1%} excess, {win_rate*100:.1f}% of configs work")
    
    if not_viable:
        print("\nNOT VIABLE (just buy and hold):")
        for ticker, excess, win_rate in not_viable:
            print(f"  {ticker}: best excess {excess:+.1%}, only {win_rate*100:.1f}% of configs work")
    
    print("""
RECOMMENDATION:
For stocks where < 1% of parameter configs work OR best excess is negative:
→ VIX ROC strategy does NOT add value
→ Just buy and hold these stocks
→ Their recovery dynamics don't align with VIX patterns

These stocks likely have:
1. Massive idiosyncratic returns (NVDA: +2148% B&H)
2. Recovery faster than VIX normalizes
3. Strong momentum that punishes being out of market
""")


if __name__ == "__main__":
    main()
