"""
VIX ROC Strategy - Out-of-Sample Validation

Test if the best parameters from grid search hold up on:
1. Earlier periods (2015-2019) - training period
2. Different entry/exit timing
3. Monthly robustness
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
    """Parameters for VIX ROC strategy."""
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


def main():
    print("="*70)
    print("VIX ROC STRATEGY - OUT-OF-SAMPLE VALIDATION")
    print("="*70)
    
    # Best params from grid search
    best_params = ROCParams(
        roc_lookback=3,
        exit_roc_thresh=0.25,
        reentry_roc_thresh=0.10,
        min_exit_days=3
    )
    
    print(f"\nBest Parameters:")
    print(f"  ROC lookback: {best_params.roc_lookback} days")
    print(f"  Exit when VIX ROC > {best_params.exit_roc_thresh:.0%}")
    print(f"  Re-enter when VIX ROC < {best_params.reentry_roc_thresh:.0%}")
    print(f"  Min days out: {best_params.min_exit_days}")
    
    # Download all data
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2010-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    strategy = VIXROCStrategy(best_params)
    
    # Test on multiple periods
    periods = [
        # Out-of-sample (before grid search period)
        ("2010-2014 (OOS)", "2010-01-01", "2014-12-31"),
        ("2015-2019 (OOS)", "2015-01-01", "2019-12-31"),
        
        # In-sample (grid search period)
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2021 Bull", "2021-01-01", "2021-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("2024 Bull", "2024-01-01", "2024-12-31"),
        
        # Full periods
        ("Full 2010-2019 (OOS)", "2010-01-01", "2019-12-31"),
        ("Full 2020-2024 (IS)", "2020-01-01", "2024-12-31"),
        ("Full 2010-2024", "2010-01-01", "2024-12-31"),
    ]
    
    print("\n" + "="*70)
    print("PERIOD-BY-PERIOD RESULTS")
    print("="*70)
    
    oos_wins = 0
    oos_count = 0
    is_wins = 0
    is_count = 0
    
    for name, start, end in periods:
        spy_period = spy.loc[start:end]
        vix_period = vix.loc[start:end]
        
        if len(spy_period) < 50:
            continue
        
        results = strategy.run(spy_period, vix_period)
        metrics = evaluate(results)
        
        is_oos = "(OOS)" in name
        if is_oos:
            oos_count += 1
            if metrics['win']:
                oos_wins += 1
        elif "Full" not in name:
            is_count += 1
            if metrics['win']:
                is_wins += 1
        
        win_marker = "✓" if metrics['win'] else ""
        print(f"\n{name}:")
        print(f"  Return: {metrics['strat_return']:.1%} vs {metrics['spy_return']:.1%} = {metrics['excess']:+.1%} {win_marker}")
        print(f"  Max DD: {metrics['strat_dd']:.1%} vs {metrics['spy_dd']:.1%} = {metrics['dd_improvement']:+.1%}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOut-of-Sample (2010-2019): {oos_wins}/{oos_count} wins")
    print(f"In-Sample (2020-2024): {is_wins}/{is_count} wins")
    
    # Year-by-year analysis
    print("\n" + "="*70)
    print("YEAR-BY-YEAR BREAKDOWN")
    print("="*70)
    
    yearly_results = []
    for year in range(2010, 2025):
        spy_year = spy.loc[f"{year}-01-01":f"{year}-12-31"]
        vix_year = vix.loc[f"{year}-01-01":f"{year}-12-31"]
        
        if len(spy_year) < 50:
            continue
        
        results = strategy.run(spy_year, vix_year)
        metrics = evaluate(results)
        
        yearly_results.append({
            'year': year,
            **metrics
        })
    
    print(f"\n{'Year':<6} {'B&H':>8} {'Strategy':>10} {'Excess':>10} {'Win':>5}")
    print("-" * 45)
    
    wins = 0
    for r in yearly_results:
        win_marker = "✓" if r['win'] else ""
        if r['win']:
            wins += 1
        print(f"{r['year']:<6} {r['spy_return']:>8.1%} {r['strat_return']:>10.1%} {r['excess']:>+10.1%} {win_marker:>5}")
    
    print("-" * 45)
    print(f"Total wins: {wins}/{len(yearly_results)}")
    
    # Count by market regime
    bear_years = [2011, 2015, 2018, 2020, 2022]  # Years with significant drawdowns
    bull_years = [y for y in range(2010, 2025) if y not in bear_years]
    
    bear_wins = sum(1 for r in yearly_results if r['year'] in bear_years and r['win'])
    bear_count = sum(1 for r in yearly_results if r['year'] in bear_years)
    
    bull_wins = sum(1 for r in yearly_results if r['year'] in bull_years and r['win'])
    bull_count = sum(1 for r in yearly_results if r['year'] in bull_years)
    
    print(f"\nBy market regime:")
    print(f"  Bear/Volatile years: {bear_wins}/{bear_count} wins")
    print(f"  Bull years: {bull_wins}/{bull_count} wins")
    
    # Trade statistics
    print("\n" + "="*70)
    print("TRADE STATISTICS (Full Period 2010-2024)")
    print("="*70)
    
    full_results = strategy.run(spy, vix)
    position_changes = full_results['position'].diff()
    
    exits = full_results[position_changes == -1]
    entries = full_results[position_changes == 1]
    
    print(f"\nTotal exits: {len(exits)}")
    print(f"Total re-entries: {len(entries)}")
    
    # Analyze time out of market
    out_periods = []
    in_market = True
    exit_date = None
    
    for date, row in full_results.iterrows():
        if in_market and row['position'] == 0:
            in_market = False
            exit_date = date
        elif not in_market and row['position'] == 1:
            in_market = True
            if exit_date:
                days_out = (date - exit_date).days
                out_periods.append(days_out)
    
    if out_periods:
        print(f"\nTime out of market:")
        print(f"  Average: {np.mean(out_periods):.1f} days")
        print(f"  Median: {np.median(out_periods):.1f} days")
        print(f"  Min: {min(out_periods)} days")
        print(f"  Max: {max(out_periods)} days")
    
    # Total time in market
    pct_in_market = full_results['position'].mean()
    print(f"\nTime in market: {pct_in_market:.1%}")


if __name__ == "__main__":
    main()
