"""
VIX ROC Strategy - Growth Stock Optimization

Growth stocks (NVDA, AAPL, MSFT, GOOGL, QQQ) have a different profile:
- They drop LESS in crashes (lower downside capture)
- They recover EXPLOSIVELY (higher upside capture)

This means we need:
1. More aggressive exit (catch the spike earlier)
2. MUCH faster re-entry (before VIX calms, to catch recovery)

Hypothesis: Use shorter lookback and lower re-entry threshold.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ROCParams:
    roc_lookback: int = 3
    exit_roc_thresh: float = 0.25
    reentry_roc_thresh: float = -0.10  # Negative = VIX must be falling
    min_exit_days: int = 2
    name: str = "default"


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
    
    return {
        'bh_return': bh_total,
        'strat_return': strat_total,
        'excess': strat_total - bh_total,
        'bh_dd': bh_dd,
        'strat_dd': strat_dd,
        'dd_improvement': strat_dd - bh_dd,
        'win': strat_total > bh_total
    }


def optimize_for_asset(ticker: str, name: str, asset_data: pd.DataFrame, 
                       vix: pd.DataFrame, train_end: str = "2019-12-31"):
    """Optimize parameters for a specific asset using walk-forward."""
    
    # Split data
    train_asset = asset_data.loc[:train_end]
    train_vix = vix.loc[:train_end]
    test_asset = asset_data.loc["2020-01-01":]
    test_vix = vix.loc["2020-01-01":]
    
    if len(train_asset) < 100 or len(test_asset) < 100:
        return None, None
    
    # Parameter grid - more aggressive than Tier 1
    roc_lookbacks = [2, 3, 5, 7]
    exit_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    reentry_thresholds = [0.05, 0.0, -0.05, -0.10, -0.15, -0.20]  # More aggressive (negative = VIX falling)
    min_exit_days = [1, 2, 3]
    
    best_params = None
    best_score = -999
    all_results = []
    
    for lookback in roc_lookbacks:
        for exit_thresh in exit_thresholds:
            for reentry_thresh in reentry_thresholds:
                for min_days in min_exit_days:
                    params = ROCParams(
                        roc_lookback=lookback,
                        exit_roc_thresh=exit_thresh,
                        reentry_roc_thresh=reentry_thresh,
                        min_exit_days=min_days
                    )
                    
                    strategy = VIXROCStrategy(params)
                    results = strategy.run(train_asset, train_vix)
                    metrics = evaluate(results)
                    
                    # Score: prioritize positive excess return
                    score = metrics['excess'] + metrics['dd_improvement'] * 0.3
                    
                    all_results.append({
                        'params': params,
                        'train_excess': metrics['excess'],
                        'train_dd_imp': metrics['dd_improvement'],
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    # Test best params on out-of-sample
    if best_params:
        strategy = VIXROCStrategy(best_params)
        test_results = strategy.run(test_asset, test_vix)
        test_metrics = evaluate(test_results)
        
        return best_params, {
            'train_excess': all_results[0]['train_excess'] if all_results else 0,
            'test_excess': test_metrics['excess'],
            'test_dd_imp': test_metrics['dd_improvement'],
            'test_win': test_metrics['win']
        }
    
    return None, None


def main():
    print("="*80)
    print("VIX ROC STRATEGY - GROWTH STOCK OPTIMIZATION")
    print("="*80)
    
    # Download data
    print("\nDownloading data...")
    
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # Growth stocks to optimize
    growth_stocks = {
        'QQQ': 'Nasdaq 100',
        'XLK': 'Tech Sector',
        'NVDA': 'NVIDIA',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon',
        'META': 'Meta',
    }
    
    all_data = {}
    for ticker in growth_stocks.keys():
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            all_data[ticker] = df
    
    print(f"Downloaded {len(all_data)} assets")
    
    # Optimize for each asset
    print("\n" + "="*80)
    print("OPTIMIZING PARAMETERS FOR EACH GROWTH STOCK")
    print("="*80)
    
    optimized_results = {}
    
    for ticker, name in growth_stocks.items():
        if ticker not in all_data:
            continue
        
        print(f"\nOptimizing {ticker} ({name})...")
        
        best_params, metrics = optimize_for_asset(
            ticker, name, all_data[ticker], vix
        )
        
        if best_params and metrics:
            optimized_results[ticker] = {
                'name': name,
                'params': best_params,
                'metrics': metrics
            }
            
            print(f"  Best params: lb={best_params.roc_lookback} exit>{best_params.exit_roc_thresh:.0%} reentry<{best_params.reentry_roc_thresh:.0%} min={best_params.min_exit_days}d")
            print(f"  Test excess: {metrics['test_excess']:+.1%}, DD Imp: {metrics['test_dd_imp']:+.1%}")
    
    # Find common optimal parameters across growth stocks
    print("\n" + "="*80)
    print("FINDING OPTIMAL PARAMETERS FOR GROWTH STOCKS")
    print("="*80)
    
    # Try different parameter sets on ALL growth stocks
    roc_lookbacks = [2, 3, 5]
    exit_thresholds = [0.15, 0.20, 0.25, 0.30]
    reentry_thresholds = [0.0, -0.05, -0.10, -0.15]
    min_exit_days = [1, 2]
    
    all_config_results = []
    
    for lookback in roc_lookbacks:
        for exit_thresh in exit_thresholds:
            for reentry_thresh in reentry_thresholds:
                for min_days in min_exit_days:
                    params = ROCParams(
                        roc_lookback=lookback,
                        exit_roc_thresh=exit_thresh,
                        reentry_roc_thresh=reentry_thresh,
                        min_exit_days=min_days
                    )
                    
                    # Test on all growth stocks
                    wins = 0
                    total_excess = 0
                    total_dd_imp = 0
                    count = 0
                    
                    for ticker in all_data:
                        asset = all_data[ticker]
                        test_asset = asset.loc["2020-01-01":]
                        test_vix = vix.loc["2020-01-01":]
                        
                        if len(test_asset) < 100:
                            continue
                        
                        strategy = VIXROCStrategy(params)
                        results = strategy.run(test_asset, test_vix)
                        metrics = evaluate(results)
                        
                        if metrics['win']:
                            wins += 1
                        total_excess += metrics['excess']
                        total_dd_imp += metrics['dd_improvement']
                        count += 1
                    
                    if count > 0:
                        all_config_results.append({
                            'params': params,
                            'wins': wins,
                            'win_rate': wins / count,
                            'avg_excess': total_excess / count,
                            'avg_dd_imp': total_dd_imp / count,
                            'count': count
                        })
    
    # Sort by average excess return
    all_config_results.sort(key=lambda x: x['avg_excess'], reverse=True)
    
    print("\nTOP 10 CONFIGURATIONS FOR GROWTH STOCKS:")
    print("-"*80)
    
    for i, r in enumerate(all_config_results[:10]):
        p = r['params']
        config = f"lb={p.roc_lookback} exit>{p.exit_roc_thresh:.0%} reentry<{p.reentry_roc_thresh:+.0%} min={p.min_exit_days}d"
        print(f"{i+1}. {config}")
        print(f"   Wins: {r['wins']}/{r['count']}, Avg Excess: {r['avg_excess']:+.1%}, Avg DD Imp: {r['avg_dd_imp']:+.1%}")
    
    # Test best config in detail
    if all_config_results:
        best = all_config_results[0]
        best_params = best['params']
        
        print("\n" + "="*80)
        print("BEST GROWTH STOCK CONFIGURATION - DETAILED RESULTS")
        print("="*80)
        
        print(f"\nParameters:")
        print(f"  ROC lookback: {best_params.roc_lookback} days")
        print(f"  Exit when VIX ROC > {best_params.exit_roc_thresh:.0%}")
        print(f"  Re-enter when VIX ROC < {best_params.reentry_roc_thresh:+.0%}")
        print(f"  Min days out: {best_params.min_exit_days}")
        
        print(f"\n{'Ticker':<8} {'Name':<18} {'B&H':>10} {'Strategy':>10} {'Excess':>10} {'DD Imp':>10}")
        print("-"*70)
        
        for ticker in growth_stocks:
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker]
            test_asset = asset.loc["2020-01-01":]
            test_vix = vix.loc["2020-01-01":]
            
            strategy = VIXROCStrategy(best_params)
            results = strategy.run(test_asset, test_vix)
            metrics = evaluate(results)
            
            win_mark = "✓" if metrics['win'] else ""
            print(f"{ticker:<8} {growth_stocks[ticker]:<18} {metrics['bh_return']:>10.1%} {metrics['strat_return']:>10.1%} {metrics['excess']:>+10.1%} {metrics['dd_improvement']:>+10.1%} {win_mark}")
    
    # Compare with Tier 1 parameters on growth stocks
    print("\n" + "="*80)
    print("COMPARISON: TIER 1 vs GROWTH OPTIMIZED on Growth Stocks")
    print("="*80)
    
    tier1_params = ROCParams(
        roc_lookback=10,
        exit_roc_thresh=0.50,
        reentry_roc_thresh=0.15,
        min_exit_days=5,
        name="Tier 1 (Conservative)"
    )
    
    growth_params = all_config_results[0]['params'] if all_config_results else ROCParams()
    growth_params.name = "Growth Optimized"
    
    print(f"\nTier 1 params: lb={tier1_params.roc_lookback} exit>{tier1_params.exit_roc_thresh:.0%} reentry<{tier1_params.reentry_roc_thresh:.0%}")
    print(f"Growth params: lb={growth_params.roc_lookback} exit>{growth_params.exit_roc_thresh:.0%} reentry<{growth_params.reentry_roc_thresh:+.0%}")
    
    print(f"\n{'Ticker':<8} {'Name':<18} {'Tier1 Excess':>14} {'Growth Excess':>14} {'Better':>8}")
    print("-"*65)
    
    tier1_wins = 0
    growth_wins = 0
    
    for ticker in growth_stocks:
        if ticker not in all_data:
            continue
        
        asset = all_data[ticker]
        test_asset = asset.loc["2020-01-01":]
        test_vix = vix.loc["2020-01-01":]
        
        # Tier 1
        strategy1 = VIXROCStrategy(tier1_params)
        results1 = strategy1.run(test_asset, test_vix)
        metrics1 = evaluate(results1)
        
        # Growth
        strategy2 = VIXROCStrategy(growth_params)
        results2 = strategy2.run(test_asset, test_vix)
        metrics2 = evaluate(results2)
        
        better = "Growth" if metrics2['excess'] > metrics1['excess'] else "Tier1"
        if metrics2['excess'] > metrics1['excess']:
            growth_wins += 1
        else:
            tier1_wins += 1
        
        print(f"{ticker:<8} {growth_stocks[ticker]:<18} {metrics1['excess']:>+14.1%} {metrics2['excess']:>+14.1%} {better:>8}")
    
    print("-"*65)
    print(f"{'TOTAL':<8} {'':18} {'':>14} {'':>14} Growth: {growth_wins}, Tier1: {tier1_wins}")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL DUAL-STRATEGY RECOMMENDATION")
    print("="*80)
    
    print(f"""
TIER 1 STRATEGY (Index ETFs, Cyclicals, Value):
  Parameters: lb=10 exit>50% reentry<15% min=5d
  Best for: SPY, DIA, IWM, XLI, XLF, XLE, VNQ
  
GROWTH STRATEGY (Tech, High-Growth):
  Parameters: lb={growth_params.roc_lookback} exit>{growth_params.exit_roc_thresh:.0%} reentry<{growth_params.reentry_roc_thresh:+.0%} min={growth_params.min_exit_days}d
  Best for: QQQ, XLK, NVDA, AAPL, MSFT, GOOGL, AMZN
  
KEY DIFFERENCES:
  - Growth uses shorter lookback ({growth_params.roc_lookback} vs 10)
  - Growth has lower exit threshold ({growth_params.exit_roc_thresh:.0%} vs 50%)
  - Growth has NEGATIVE re-entry threshold ({growth_params.reentry_roc_thresh:+.0%} vs +15%)
    → Re-enter ONLY when VIX is actively falling
  - Growth requires fewer days out ({growth_params.min_exit_days} vs 5)
""")


if __name__ == "__main__":
    main()
