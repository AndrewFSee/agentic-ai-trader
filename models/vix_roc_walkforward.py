"""
VIX ROC Strategy - Walk-Forward Test

Proper methodology:
1. Optimize parameters on 2010-2019 (training)
2. Test on 2020-2024 (out-of-sample)

This avoids the overfitting we saw when optimizing on 2020-2024.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ROCParams:
    roc_lookback: int = 3
    exit_roc_thresh: float = 0.25
    reentry_roc_thresh: float = 0.10
    min_exit_days: int = 3


class VIXROCStrategy:
    def __init__(self, params: ROCParams):
        self.params = params
    
    def run(self, spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
        common_idx = spy.index.intersection(vix.index)
        spy = spy.loc[common_idx].copy()
        vix = vix.loc[common_idx].copy()
        
        vix_close = vix['Close'].values if 'Close' in vix.columns else vix.iloc[:, 0].values
        vix_roc = np.zeros(len(vix_close))
        
        for i in range(self.params.roc_lookback, len(vix_close)):
            prev_vix = vix_close[i - self.params.roc_lookback]
            if prev_vix > 0:
                vix_roc[i] = (vix_close[i] - prev_vix) / prev_vix
        
        spy_close = spy['Close'].values if 'Close' in spy.columns else spy.iloc[:, 0].values
        spy_returns = np.zeros(len(spy_close))
        spy_returns[1:] = np.diff(spy_close) / spy_close[:-1]
        
        positions = np.ones(len(spy_close))
        in_market = True
        days_out = 0
        
        for i in range(self.params.roc_lookback, len(spy_close)):
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
            'date': spy.index,
            'spy_close': spy_close,
            'spy_return': spy_returns,
            'vix': vix_close,
            'vix_roc': vix_roc,
            'position': positions
        })
        results.set_index('date', inplace=True)
        results['strategy_return'] = results['position'].shift(1) * results['spy_return']
        results['strategy_return'] = results['strategy_return'].fillna(0)
        
        return results


def evaluate(results: pd.DataFrame) -> Dict:
    spy_total = (1 + results['spy_return']).prod() - 1
    strat_total = (1 + results['strategy_return']).prod() - 1
    
    spy_cum = (1 + results['spy_return']).cumprod()
    strat_cum = (1 + results['strategy_return']).cumprod()
    
    spy_dd = (spy_cum / spy_cum.cummax() - 1).min()
    strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
    
    return {
        'spy_return': spy_total,
        'strat_return': strat_total,
        'excess': strat_total - spy_total,
        'spy_dd': spy_dd,
        'strat_dd': strat_dd,
        'dd_improvement': strat_dd - spy_dd,
        'win': strat_total > spy_total
    }


def optimize_on_training(spy_train, vix_train):
    """Find best parameters on training data (2010-2019)."""
    
    best_params = None
    best_score = -999
    
    roc_lookbacks = [3, 5, 7, 10]
    exit_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    reentry_thresholds = [0.15, 0.10, 0.05, 0.0, -0.05, -0.10]
    min_exit_days = [1, 2, 3, 5]
    
    print("Optimizing on training data (2010-2019)...")
    
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
                    results = strategy.run(spy_train, vix_train)
                    metrics = evaluate(results)
                    
                    # Score: prioritize positive excess return with good DD
                    score = metrics['excess'] + metrics['dd_improvement'] * 0.5
                    
                    all_results.append({
                        'params': params,
                        'score': score,
                        'metrics': metrics
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    # Sort and show top 5
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 5 configurations on training data:")
    for i, r in enumerate(all_results[:5]):
        p = r['params']
        m = r['metrics']
        print(f"  {i+1}. lb={p.roc_lookback} exit>{p.exit_roc_thresh:.0%} reentry<{p.reentry_roc_thresh:.0%} min={p.min_exit_days}d")
        print(f"      Return: {m['strat_return']:.1%} vs {m['spy_return']:.1%} = {m['excess']:+.1%}, DD: {m['dd_improvement']:+.1%}")
    
    return best_params, all_results[:5]


def main():
    print("="*70)
    print("VIX ROC STRATEGY - WALK-FORWARD TEST")
    print("="*70)
    
    # Download all data
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2010-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # Split into training (2010-2019) and test (2020-2024)
    spy_train = spy.loc["2010-01-01":"2019-12-31"]
    vix_train = vix.loc["2010-01-01":"2019-12-31"]
    
    spy_test = spy.loc["2020-01-01":"2024-12-31"]
    vix_test = vix.loc["2020-01-01":"2024-12-31"]
    
    print(f"\nTraining period: 2010-2019 ({len(spy_train)} days)")
    print(f"Test period: 2020-2024 ({len(spy_test)} days)")
    
    # Optimize on training
    best_params, top5 = optimize_on_training(spy_train, vix_train)
    
    print("\n" + "="*70)
    print("BEST PARAMETERS FROM TRAINING:")
    print("="*70)
    print(f"  ROC lookback: {best_params.roc_lookback} days")
    print(f"  Exit when VIX ROC > {best_params.exit_roc_thresh:.0%}")
    print(f"  Re-enter when VIX ROC < {best_params.reentry_roc_thresh:.0%}")
    print(f"  Min days out: {best_params.min_exit_days}")
    
    # Test on out-of-sample
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE TEST (2020-2024)")
    print("="*70)
    
    strategy = VIXROCStrategy(best_params)
    
    test_periods = [
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2021 Bull", "2021-01-01", "2021-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("2024 Bull", "2024-01-01", "2024-12-31"),
        ("Full 2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    wins = 0
    for name, start, end in test_periods:
        spy_period = spy_test.loc[start:end]
        vix_period = vix_test.loc[start:end]
        
        if len(spy_period) < 50:
            continue
        
        results = strategy.run(spy_period, vix_period)
        metrics = evaluate(results)
        
        if "Full" not in name and metrics['win']:
            wins += 1
        
        win_marker = "✓" if metrics['win'] else ""
        print(f"\n{name}:")
        print(f"  Return: {metrics['strat_return']:.1%} vs {metrics['spy_return']:.1%} = {metrics['excess']:+.1%} {win_marker}")
        print(f"  Max DD: {metrics['strat_dd']:.1%} vs {metrics['spy_dd']:.1%} = {metrics['dd_improvement']:+.1%}")
    
    print(f"\nOut-of-sample wins: {wins}/5")
    
    # Also test top 5 from training on test set
    print("\n" + "="*70)
    print("TOP 5 TRAINING CONFIGS ON TEST SET")
    print("="*70)
    
    for i, r in enumerate(top5):
        p = r['params']
        strategy = VIXROCStrategy(p)
        results = strategy.run(spy_test, vix_test)
        metrics = evaluate(results)
        
        win_marker = "✓" if metrics['win'] else ""
        config = f"lb={p.roc_lookback} exit>{p.exit_roc_thresh:.0%} reentry<{p.reentry_roc_thresh:.0%} min={p.min_exit_days}d"
        print(f"\n{i+1}. {config}")
        print(f"   Training: {r['metrics']['excess']:+.1%} excess, {r['metrics']['dd_improvement']:+.1%} DD")
        print(f"   Test:     {metrics['excess']:+.1%} excess, {metrics['dd_improvement']:+.1%} DD {win_marker}")
    
    # Compare with the "overfitted" 2020-2024 optimal
    print("\n" + "="*70)
    print("COMPARISON: WALK-FORWARD vs OVERFITTED PARAMS")
    print("="*70)
    
    overfitted_params = ROCParams(
        roc_lookback=3,
        exit_roc_thresh=0.25,
        reentry_roc_thresh=0.10,
        min_exit_days=3
    )
    
    print(f"\nWalk-forward params (trained 2010-2019):")
    print(f"  lb={best_params.roc_lookback} exit>{best_params.exit_roc_thresh:.0%} reentry<{best_params.reentry_roc_thresh:.0%} min={best_params.min_exit_days}d")
    
    print(f"\nOverfitted params (optimized on 2020-2024):")
    print(f"  lb={overfitted_params.roc_lookback} exit>{overfitted_params.exit_roc_thresh:.0%} reentry<{overfitted_params.reentry_roc_thresh:.0%} min={overfitted_params.min_exit_days}d")
    
    # Test both on 2020-2024
    wf_strategy = VIXROCStrategy(best_params)
    of_strategy = VIXROCStrategy(overfitted_params)
    
    wf_results = wf_strategy.run(spy_test, vix_test)
    of_results = of_strategy.run(spy_test, vix_test)
    
    wf_metrics = evaluate(wf_results)
    of_metrics = evaluate(of_results)
    
    print(f"\n2020-2024 Performance:")
    print(f"  Walk-forward: {wf_metrics['strat_return']:.1%} vs {wf_metrics['spy_return']:.1%} = {wf_metrics['excess']:+.1%}")
    print(f"  Overfitted:   {of_metrics['strat_return']:.1%} vs {of_metrics['spy_return']:.1%} = {of_metrics['excess']:+.1%}")


if __name__ == "__main__":
    main()
