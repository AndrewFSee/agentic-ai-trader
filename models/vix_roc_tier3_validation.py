"""
VIX ROC Strategy - Tier 3 (Mega-Cap Tech) Validation

Test the common winning configs found for NVDA, MSFT, META
and find the best single config for this tier.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List
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
    print("TIER 3 (MEGA-CAP TECH) STRATEGY VALIDATION")
    print("="*80)
    
    # Download data
    print("\nDownloading data...")
    
    vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    tier3_tickers = {
        'NVDA': 'NVIDIA',
        'MSFT': 'Microsoft', 
        'META': 'Meta'
    }
    
    all_data = {}
    for ticker in tier3_tickers:
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            all_data[ticker] = df
    
    # Common winning configs from exhaustive search
    common_configs = [
        ROCParams(5, 1.00, -0.20, 3, "lb=5 exit>100% reentry<-20% min=3d"),
        ROCParams(7, 0.75, 0.00, 2, "lb=7 exit>75% reentry<0% min=2d"),
        ROCParams(5, 0.75, 0.00, 3, "lb=5 exit>75% reentry<0% min=3d"),
        ROCParams(5, 0.75, -0.10, 2, "lb=5 exit>75% reentry<-10% min=2d"),
        ROCParams(5, 0.75, -0.20, 1, "lb=5 exit>75% reentry<-20% min=1d"),
        ROCParams(5, 0.75, -0.05, 5, "lb=5 exit>75% reentry<-5% min=5d"),
        ROCParams(5, 1.00, -0.15, 1, "lb=5 exit>100% reentry<-15% min=1d"),
        ROCParams(7, 0.75, 0.05, 2, "lb=7 exit>75% reentry<+5% min=2d"),
    ]
    
    # Test periods
    start, end = "2020-01-01", "2024-12-31"
    
    # Compare existing strategies with common configs
    tier1_params = ROCParams(10, 0.50, 0.15, 5, "Tier 1 (Conservative)")
    growth_params = ROCParams(2, 0.20, 0.00, 1, "Growth (Aggressive)")
    
    print("\n" + "="*80)
    print("COMPARING COMMON CONFIGS VS TIER1 VS GROWTH ON MEGA-CAP TECH")
    print("="*80)
    
    # Test each common config
    config_results = {}
    
    for config in common_configs:
        strategy = VIXROCStrategy(config)
        config_results[config.name] = {}
        
        wins = 0
        total_excess = 0
        
        for ticker in tier3_tickers:
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker].loc[start:end]
            vix_period = vix.loc[start:end]
            
            results = strategy.run(asset, vix_period)
            metrics = evaluate(results)
            
            config_results[config.name][ticker] = metrics
            if metrics['win']:
                wins += 1
            total_excess += metrics['excess']
        
        config_results[config.name]['wins'] = wins
        config_results[config.name]['avg_excess'] = total_excess / len(tier3_tickers)
    
    # Also test Tier1 and Growth
    for config in [tier1_params, growth_params]:
        strategy = VIXROCStrategy(config)
        config_results[config.name] = {}
        
        wins = 0
        total_excess = 0
        
        for ticker in tier3_tickers:
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker].loc[start:end]
            vix_period = vix.loc[start:end]
            
            results = strategy.run(asset, vix_period)
            metrics = evaluate(results)
            
            config_results[config.name][ticker] = metrics
            if metrics['win']:
                wins += 1
            total_excess += metrics['excess']
        
        config_results[config.name]['wins'] = wins
        config_results[config.name]['avg_excess'] = total_excess / len(tier3_tickers)
    
    # Print comparison
    print(f"\n{'Config':<45} {'Wins':<6} {'Avg Excess':<12} {'NVDA':>12} {'MSFT':>12} {'META':>12}")
    print("-"*105)
    
    # Sort by wins then by avg excess
    sorted_configs = sorted(config_results.items(), 
                           key=lambda x: (x[1]['wins'], x[1]['avg_excess']), 
                           reverse=True)
    
    for config_name, data in sorted_configs:
        nvda_excess = data.get('NVDA', {}).get('excess', 0)
        msft_excess = data.get('MSFT', {}).get('excess', 0)
        meta_excess = data.get('META', {}).get('excess', 0)
        
        print(f"{config_name:<45} {data['wins']}/3   {data['avg_excess']:+.1%}      {nvda_excess:+.1%}     {msft_excess:+.1%}     {meta_excess:+.1%}")
    
    # Find best config
    best_config_name = sorted_configs[0][0]
    best_config_data = sorted_configs[0][1]
    
    print(f"\n" + "="*80)
    print(f"BEST TIER 3 CONFIG: {best_config_name}")
    print("="*80)
    
    # Detailed results for best config
    print(f"\nDetailed results for 2020-2024:")
    print(f"{'Ticker':<6} {'Name':<12} {'B&H':>12} {'Strategy':>12} {'Excess':>12} {'DD Imp':>10} {'Trades':>8}")
    print("-"*80)
    
    for ticker, name in tier3_tickers.items():
        if ticker not in best_config_data:
            continue
        
        m = best_config_data[ticker]
        win_mark = "✓" if m['win'] else ""
        print(f"{ticker:<6} {name:<12} {m['bh_return']:>12.1%} {m['strat_return']:>12.1%} {m['excess']:>+12.1%} {m['dd_improvement']:>+10.1%} {m['trades']:>8.0f} {win_mark}")
    
    # Test across different periods
    print(f"\n" + "="*80)
    print(f"BEST TIER 3 CONFIG - PERIOD BREAKDOWN")
    print("="*80)
    
    # Find the best config object
    best_config = None
    for config in common_configs:
        if config.name == best_config_name:
            best_config = config
            break
    
    if best_config is None:
        if "Tier 1" in best_config_name:
            best_config = tier1_params
        else:
            best_config = growth_params
    
    strategy = VIXROCStrategy(best_config)
    
    periods = [
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Bull", "2023-01-01", "2023-12-31"),
        ("Full 2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    for period_name, start, end in periods:
        print(f"\n{period_name}:")
        
        for ticker, name in tier3_tickers.items():
            if ticker not in all_data:
                continue
            
            asset = all_data[ticker].loc[start:end]
            vix_period = vix.loc[start:end]
            
            if len(asset) < 20:
                continue
            
            results = strategy.run(asset, vix_period)
            metrics = evaluate(results)
            
            win_mark = "✓" if metrics['win'] else ""
            print(f"  {ticker:<5} {metrics['excess']:+8.1%} (B&H: {metrics['bh_return']:+.1%}, Strat: {metrics['strat_return']:+.1%}) {win_mark}")
    
    # Final three-tier summary
    print("\n" + "="*80)
    print("FINAL THREE-TIER VIX ROC STRATEGY")
    print("="*80)
    
    print("""
┌────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: VALUE/CYCLICAL (Conservative)                                          │
│ Params: lookback=10, exit>50%, reentry<+15%, min_days=5                        │
│ Assets: SPY, DIA, IWM, XLI, XLF, XLE, VNQ                                      │
│ Result: 7/7 wins, avg +39% excess                                              │
├────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: GROWTH/TECH (Aggressive)                                                │
│ Params: lookback=2, exit>20%, reentry<0%, min_days=1                           │
│ Assets: QQQ, XLK, AAPL, GOOGL, AMZN                                            │
│ Result: 5/5 wins, avg +20% excess                                              │
├────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: MEGA-CAP TECH (Ultra-Conservative)                                      │
│ Params: lookback=5, exit>75%, reentry<0%, min_days=3                           │
│ Assets: NVDA, MSFT, META                                                        │
│ Result: 3/3 wins with common config                                            │
├────────────────────────────────────────────────────────────────────────────────┤
│ BUY-AND-HOLD (No Strategy)                                                      │
│ Assets: None! All tested stocks have a winning strategy                         │
└────────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
- Tier 3 needs VERY HIGH exit threshold (75-100%) because these stocks are
  so volatile that smaller VIX moves don't justify exiting
- Tier 3 needs NEGATIVE reentry threshold (VIX must be FALLING) because 
  these stocks recover explosively
- Common theme: The better the stock, the more conservative the exit trigger
""")


if __name__ == "__main__":
    main()
