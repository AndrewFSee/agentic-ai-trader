"""
VIX ROC Strategy - Final Dual Strategy Validation

Validate both strategies across all time periods:
1. Tier 1 (Conservative): lb=10 exit>50% reentry<15% min=5d
2. Growth (Aggressive): lb=2 exit>20% reentry<0% min=1d
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ROCParams:
    roc_lookback: int
    exit_roc_thresh: float
    reentry_roc_thresh: float
    min_exit_days: int
    name: str = ""


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


def main():
    print("="*80)
    print("VIX ROC DUAL-STRATEGY VALIDATION")
    print("="*80)
    
    # Define strategies
    tier1_params = ROCParams(
        roc_lookback=10,
        exit_roc_thresh=0.50,
        reentry_roc_thresh=0.15,
        min_exit_days=5,
        name="Tier 1 (Conservative)"
    )
    
    growth_params = ROCParams(
        roc_lookback=2,
        exit_roc_thresh=0.20,
        reentry_roc_thresh=0.0,
        min_exit_days=1,
        name="Growth (Aggressive)"
    )
    
    # Download data
    print("\nDownloading data...")
    
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # All assets
    tier1_assets = {
        'SPY': 'S&P 500',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'XLI': 'Industrials',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'VNQ': 'Real Estate',
    }
    
    growth_assets = {
        'QQQ': 'Nasdaq 100',
        'XLK': 'Tech Sector',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
    }
    
    all_tickers = {**tier1_assets, **growth_assets}
    
    all_data = {}
    for ticker in all_tickers:
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            all_data[ticker] = df
    
    # Test periods
    periods = [
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("Full 2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    # Test Tier 1 strategy on Tier 1 assets
    print("\n" + "="*80)
    print("TIER 1 STRATEGY on TIER 1 ASSETS")
    print("="*80)
    
    tier1_strategy = VIXROCStrategy(tier1_params)
    
    for period_name, start, end in periods:
        print(f"\n{period_name}:")
        wins = 0
        total = 0
        for ticker, name in tier1_assets.items():
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker].loc[start:end]
            vix_period = vix.loc[start:end]
            
            if len(asset) < 50:
                continue
            
            results = tier1_strategy.run(asset, vix_period)
            metrics = evaluate(results)
            
            total += 1
            if metrics['win']:
                wins += 1
            
            win_mark = "✓" if metrics['win'] else ""
            print(f"  {ticker:<5} {metrics['excess']:+6.1%} {win_mark}")
        
        print(f"  --- Wins: {wins}/{total}")
    
    # Test Growth strategy on Growth assets
    print("\n" + "="*80)
    print("GROWTH STRATEGY on GROWTH ASSETS")
    print("="*80)
    
    growth_strategy = VIXROCStrategy(growth_params)
    
    for period_name, start, end in periods:
        print(f"\n{period_name}:")
        wins = 0
        total = 0
        for ticker, name in growth_assets.items():
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker].loc[start:end]
            vix_period = vix.loc[start:end]
            
            if len(asset) < 50:
                continue
            
            results = growth_strategy.run(asset, vix_period)
            metrics = evaluate(results)
            
            total += 1
            if metrics['win']:
                wins += 1
            
            win_mark = "✓" if metrics['win'] else ""
            print(f"  {ticker:<5} {metrics['excess']:+6.1%} {win_mark}")
        
        print(f"  --- Wins: {wins}/{total}")
    
    # Summary comparison table
    print("\n" + "="*80)
    print("FULL 2020-2024 SUMMARY")
    print("="*80)
    
    print(f"\n{'Asset':<6} {'Type':<8} {'Best Strategy':<20} {'B&H':>10} {'Strategy':>10} {'Excess':>10} {'DD Imp':>10}")
    print("-"*85)
    
    start, end = "2020-01-01", "2024-12-31"
    
    overall_wins = 0
    overall_total = 0
    
    for ticker in all_tickers:
        if ticker not in all_data:
            continue
        
        asset = all_data[ticker].loc[start:end]
        vix_period = vix.loc[start:end]
        
        if len(asset) < 50:
            continue
        
        # Determine asset type and best strategy
        if ticker in tier1_assets:
            asset_type = "Tier1"
            best_strategy = tier1_strategy
            strategy_name = "Tier 1"
        else:
            asset_type = "Growth"
            best_strategy = growth_strategy
            strategy_name = "Growth"
        
        results = best_strategy.run(asset, vix_period)
        metrics = evaluate(results)
        
        overall_total += 1
        if metrics['win']:
            overall_wins += 1
        
        win_mark = "✓" if metrics['win'] else ""
        print(f"{ticker:<6} {asset_type:<8} {strategy_name:<20} {metrics['bh_return']:>10.1%} {metrics['strat_return']:>10.1%} {metrics['excess']:>+10.1%} {metrics['dd_improvement']:>+10.1%} {win_mark}")
    
    print("-"*85)
    print(f"OVERALL WINS: {overall_wins}/{overall_total} ({100*overall_wins/overall_total:.0f}%)")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL PRODUCTION RECOMMENDATION")
    print("="*80)
    
    print("""
TWO-TIER VIX ROC STRATEGY:

┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1 STRATEGY (Conservative)                                         │
│ Parameters: lookback=10, exit>50%, reentry<15%, min_days=5             │
│ Assets: SPY, DIA, IWM, XLI, XLF, XLE, VNQ                              │
│ Philosophy: Fewer trades, stay out longer, catch major crashes         │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ GROWTH STRATEGY (Aggressive)                                            │
│ Parameters: lookback=2, exit>20%, reentry<0%, min_days=1               │
│ Assets: QQQ, XLK, AAPL, MSFT, GOOGL, AMZN, NVDA, META                  │
│ Philosophy: Quick trades, re-enter fast when VIX falling               │
└────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
- Growth stocks recover faster than VIX calms down
- Tier 1 assets recover in sync with VIX normalization
- Using the WRONG strategy on an asset type destroys alpha

IMPLEMENTATION:
1. Classify asset at runtime (Tier1 vs Growth)
2. Apply corresponding strategy parameters
3. Never use Growth params on Tier1 assets (too many whipsaws)
4. Never use Tier1 params on Growth assets (misses recovery)
""")


if __name__ == "__main__":
    main()
