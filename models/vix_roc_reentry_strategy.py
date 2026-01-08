"""
VIX ROC Re-Entry Strategy

Key insight: The CHANGE in VIX matters more than the absolute level.
- Exit when VIX ROC spikes (fear accelerating)
- Re-enter when VIX ROC falls (fear decelerating), even if VIX is still high

This should capture recovery earlier than waiting for VIX to fall.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ROCParams:
    """Parameters for VIX ROC strategy."""
    roc_lookback: int = 5          # Days for ROC calculation
    exit_roc_thresh: float = 0.25   # Exit if VIX ROC > this (25% increase)
    reentry_roc_thresh: float = 0.0 # Re-enter if VIX ROC < this (VIX falling or stable)
    min_exit_days: int = 2          # Minimum days to stay out before re-entry check


class VIXROCStrategy:
    """
    Strategy based on VIX Rate of Change.
    
    Logic:
    1. Fully invested by default
    2. Exit when VIX ROC > exit_thresh (VIX spiking)
    3. Re-enter when VIX ROC < reentry_thresh (VIX stabilizing/falling)
    """
    
    def __init__(self, params: ROCParams):
        self.params = params
    
    def run(self, spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
        """Run the strategy and return daily positions."""
        
        # Align dates
        common_idx = spy.index.intersection(vix.index)
        spy = spy.loc[common_idx].copy()
        vix = vix.loc[common_idx].copy()
        
        # Calculate VIX ROC
        vix_close = vix['Close'].values if 'Close' in vix.columns else vix.iloc[:, 0].values
        vix_roc = np.zeros(len(vix_close))
        
        for i in range(self.params.roc_lookback, len(vix_close)):
            prev_vix = vix_close[i - self.params.roc_lookback]
            if prev_vix > 0:
                vix_roc[i] = (vix_close[i] - prev_vix) / prev_vix
        
        # Calculate SPY returns
        spy_close = spy['Close'].values if 'Close' in spy.columns else spy.iloc[:, 0].values
        spy_returns = np.zeros(len(spy_close))
        spy_returns[1:] = np.diff(spy_close) / spy_close[:-1]
        
        # Run strategy
        positions = np.ones(len(spy_close))
        in_market = True
        days_out = 0
        exit_reason = None
        
        for i in range(self.params.roc_lookback, len(spy_close)):
            current_roc = vix_roc[i]
            current_vix = vix_close[i]
            
            if in_market:
                # Check for exit signal: VIX ROC spike
                if current_roc > self.params.exit_roc_thresh:
                    in_market = False
                    days_out = 0
                    exit_reason = f"VIX ROC {current_roc:.1%} > {self.params.exit_roc_thresh:.0%}"
                    positions[i] = 0
                else:
                    positions[i] = 1
            else:
                days_out += 1
                
                # Check for re-entry: VIX ROC falls below threshold
                if days_out >= self.params.min_exit_days:
                    if current_roc < self.params.reentry_roc_thresh:
                        in_market = True
                        positions[i] = 1
                    else:
                        positions[i] = 0
                else:
                    positions[i] = 0
        
        # Build results DataFrame
        results = pd.DataFrame({
            'date': spy.index,
            'spy_close': spy_close,
            'spy_return': spy_returns,
            'vix': vix_close,
            'vix_roc': vix_roc,
            'position': positions
        })
        results.set_index('date', inplace=True)
        
        # Calculate strategy returns
        results['strategy_return'] = results['position'].shift(1) * results['spy_return']
        results['strategy_return'] = results['strategy_return'].fillna(0)
        
        return results


def evaluate_strategy(results: pd.DataFrame, label: str) -> Dict:
    """Calculate performance metrics."""
    
    spy_total = (1 + results['spy_return']).prod() - 1
    strat_total = (1 + results['strategy_return']).prod() - 1
    
    # Drawdowns
    spy_cum = (1 + results['spy_return']).cumprod()
    strat_cum = (1 + results['strategy_return']).cumprod()
    
    spy_dd = (spy_cum / spy_cum.cummax() - 1).min()
    strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
    
    excess = strat_total - spy_total
    dd_improvement = strat_dd - spy_dd  # Less negative = better
    
    return {
        'label': label,
        'spy_return': spy_total,
        'strat_return': strat_total,
        'excess': excess,
        'spy_dd': spy_dd,
        'strat_dd': strat_dd,
        'dd_improvement': dd_improvement,
        'win': excess > 0
    }


def run_test_periods(params: ROCParams) -> List[Dict]:
    """Test across multiple periods."""
    
    print(f"Downloading data...")
    spy = yf.download("SPY", start="2015-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01", progress=False)
    
    # Handle multi-index
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    periods = [
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("Full 2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    strategy = VIXROCStrategy(params)
    results_list = []
    
    for name, start, end in periods:
        spy_period = spy.loc[start:end]
        vix_period = vix.loc[start:end]
        
        if len(spy_period) < 50:
            continue
        
        results = strategy.run(spy_period, vix_period)
        metrics = evaluate_strategy(results, name)
        results_list.append(metrics)
    
    return results_list


def analyze_covid_trades(params: ROCParams):
    """Detailed analysis of COVID period trades."""
    
    print("\n" + "="*70)
    print("DETAILED COVID PERIOD ANALYSIS")
    print("="*70)
    
    spy = yf.download("SPY", start="2020-01-01", end="2020-06-30", progress=False)
    vix = yf.download("^VIX", start="2020-01-01", end="2020-06-30", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    strategy = VIXROCStrategy(params)
    results = strategy.run(spy, vix)
    
    # Find entry/exit points
    position_changes = results['position'].diff()
    exits = results[position_changes == -1]
    entries = results[position_changes == 1]
    
    print(f"\nStrategy Parameters:")
    print(f"  ROC lookback: {params.roc_lookback} days")
    print(f"  Exit when VIX ROC > {params.exit_roc_thresh:.0%}")
    print(f"  Re-enter when VIX ROC < {params.reentry_roc_thresh:.0%}")
    print(f"  Min days out: {params.min_exit_days}")
    
    print(f"\nEXIT signals:")
    for date, row in exits.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: VIX={row['vix']:.1f}, ROC={row['vix_roc']:.1%}, SPY={row['spy_close']:.2f}")
    
    print(f"\nRE-ENTRY signals:")
    for date, row in entries.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: VIX={row['vix']:.1f}, ROC={row['vix_roc']:.1%}, SPY={row['spy_close']:.2f}")
    
    # Performance
    spy_total = (1 + results['spy_return']).prod() - 1
    strat_total = (1 + results['strategy_return']).prod() - 1
    
    print(f"\nJan-Jun 2020 Performance:")
    print(f"  B&H: {spy_total:.1%}")
    print(f"  Strategy: {strat_total:.1%}")
    print(f"  Excess: {strat_total - spy_total:+.1%}")
    
    return results


def grid_search():
    """Find optimal parameters."""
    
    print("="*70)
    print("VIX ROC RE-ENTRY STRATEGY - PARAMETER GRID SEARCH")
    print("="*70)
    
    # Parameter grid
    roc_lookbacks = [3, 5, 10]
    exit_thresholds = [0.15, 0.25, 0.35, 0.50]
    reentry_thresholds = [0.10, 0.05, 0.0, -0.05, -0.10]
    min_exit_days = [1, 2, 3]
    
    best_result = None
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
                    
                    results = run_test_periods(params)
                    
                    if len(results) < 4:
                        continue
                    
                    # Score: number of wins + excess return on full period
                    wins = sum(1 for r in results if r['win'])
                    full_excess = next((r['excess'] for r in results if 'Full' in r['label']), -1)
                    full_dd_imp = next((r['dd_improvement'] for r in results if 'Full' in r['label']), 0)
                    
                    # Score combines wins, return, and DD improvement
                    score = wins + full_excess * 10 + full_dd_imp * 5
                    
                    config_str = f"lb={lookback} exit>{exit_thresh:.0%} reentry<{reentry_thresh:.0%} min={min_days}d"
                    
                    all_results.append({
                        'config': config_str,
                        'params': params,
                        'wins': wins,
                        'full_excess': full_excess,
                        'full_dd_imp': full_dd_imp,
                        'score': score,
                        'results': results
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_result = all_results[-1]
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)
    
    for i, r in enumerate(all_results[:10]):
        print(f"\n{i+1}. {r['config']}")
        print(f"   Wins: {r['wins']}/4, Full Excess: {r['full_excess']:+.1%}, DD Imp: {r['full_dd_imp']:+.1%}")
        
        for period in r['results']:
            win_marker = "✓" if period['win'] else ""
            print(f"   {period['label']}: {period['strat_return']:.1%} vs {period['spy_return']:.1%} = {period['excess']:+.1%} {win_marker}")
    
    return best_result


def compare_with_baseline():
    """Compare ROC strategy with previous best (VIX level-based)."""
    
    print("\n" + "="*70)
    print("COMPARISON: ROC STRATEGY vs PREVIOUS BEST")
    print("="*70)
    
    # Best ROC params (from grid search or manual tuning)
    roc_params = ROCParams(
        roc_lookback=5,
        exit_roc_thresh=0.25,  # Exit when VIX up 25% in 5 days
        reentry_roc_thresh=0.0,  # Re-enter when VIX stops rising
        min_exit_days=2
    )
    
    results = run_test_periods(roc_params)
    
    print(f"\nVIX ROC Strategy (exit>{roc_params.exit_roc_thresh:.0%}, reentry<{roc_params.reentry_roc_thresh:.0%}):")
    
    wins = 0
    for r in results:
        win_marker = "✓" if r['win'] else ""
        if r['win']:
            wins += 1
        print(f"  {r['label']}: {r['strat_return']:.1%} vs {r['spy_return']:.1%} = {r['excess']:+.1%} {win_marker}")
        print(f"    DD: {r['strat_dd']:.1%} vs {r['spy_dd']:.1%} = {r['dd_improvement']:+.1%}")
    
    print(f"\nTotal wins: {wins}/4")


if __name__ == "__main__":
    # Run grid search to find best params
    best = grid_search()
    
    # Detailed analysis of best config on COVID period
    if best:
        print("\n" + "="*70)
        print("BEST CONFIGURATION DETAILED ANALYSIS")
        print("="*70)
        analyze_covid_trades(best['params'])
    
    # Also test a reasonable default
    compare_with_baseline()
