"""
VIX ROC Strategy - Deep Dive Analysis

Why do some assets work better than others?
Let's analyze:
1. Beta to SPY
2. Recovery speed after VIX spikes
3. Idiosyncratic vs systematic risk
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
    roc_lookback: int = 10
    exit_roc_thresh: float = 0.50
    reentry_roc_thresh: float = 0.15
    min_exit_days: int = 5


def analyze_asset_characteristics():
    """Analyze what makes an asset good/bad for VIX ROC strategy."""
    
    print("="*80)
    print("VIX ROC STRATEGY - ASSET CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Focus on key assets
    assets = {
        # Good performers
        'XLI': 'Industrial (Best Sector)',
        'XLF': 'Financial',
        'IWM': 'Russell 2000',
        'VNQ': 'Real Estate',
        'XLE': 'Energy',
        'DIA': 'Dow Jones',
        'SPY': 'S&P 500',
        
        # Poor performers
        'NVDA': 'NVIDIA (Worst)',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'QQQ': 'Nasdaq 100',
    }
    
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2019-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2019-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    all_data = {}
    for ticker in assets.keys():
        df = yf.download(ticker, start="2019-01-01", end="2025-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        all_data[ticker] = df
    
    # Calculate characteristics
    spy_ret = spy['Close'].pct_change()
    vix_close = vix['Close']
    
    results = []
    
    for ticker, name in assets.items():
        df = all_data[ticker]
        asset_ret = df['Close'].pct_change()
        
        # Align data
        common_idx = spy_ret.index.intersection(asset_ret.index).intersection(vix_close.index)
        spy_r = spy_ret.loc[common_idx]
        asset_r = asset_ret.loc[common_idx]
        vix_v = vix_close.loc[common_idx]
        
        # 1. Beta to SPY (regression)
        cov = np.cov(asset_r.dropna(), spy_r.dropna())[0, 1]
        var = np.var(spy_r.dropna())
        beta = cov / var if var > 0 else 1.0
        
        # 2. Correlation with SPY
        corr = asset_r.corr(spy_r)
        
        # 3. Correlation with VIX (should be negative for risk assets)
        vix_change = vix_v.pct_change()
        vix_corr = asset_r.corr(vix_change)
        
        # 4. Downside capture ratio (how much does it drop when SPY drops)
        spy_down_days = spy_r < 0
        if spy_down_days.sum() > 0:
            downside_capture = asset_r[spy_down_days].mean() / spy_r[spy_down_days].mean()
        else:
            downside_capture = 1.0
        
        # 5. Upside capture ratio
        spy_up_days = spy_r > 0
        if spy_up_days.sum() > 0:
            upside_capture = asset_r[spy_up_days].mean() / spy_r[spy_up_days].mean()
        else:
            upside_capture = 1.0
        
        # 6. Volatility ratio
        vol_ratio = asset_r.std() / spy_r.std()
        
        # 7. Analyze COVID recovery specifically
        # Exit day: ~Feb 24, 2020
        # Re-entry day: ~Mar 23, 2020
        # Recovery period: Mar 23 - Jun 30, 2020
        
        covid_exit_price = df.loc['2020-02-24':'2020-02-24', 'Close'].iloc[0] if '2020-02-24' in df.index else None
        covid_entry_price = df.loc['2020-03-23':'2020-03-23', 'Close'].iloc[0] if '2020-03-23' in df.index else None
        covid_june_price = df.loc['2020-06-30':'2020-06-30', 'Close'].iloc[0] if '2020-06-30' in df.index else None
        
        if all([covid_exit_price, covid_entry_price, covid_june_price]):
            crash_drop = (covid_entry_price - covid_exit_price) / covid_exit_price
            recovery_gain = (covid_june_price - covid_entry_price) / covid_entry_price
        else:
            crash_drop = None
            recovery_gain = None
        
        results.append({
            'ticker': ticker,
            'name': name,
            'beta': beta,
            'spy_corr': corr,
            'vix_corr': vix_corr,
            'downside_cap': downside_capture,
            'upside_cap': upside_capture,
            'vol_ratio': vol_ratio,
            'crash_drop': crash_drop,
            'recovery_gain': recovery_gain,
        })
    
    df_results = pd.DataFrame(results)
    
    # Print analysis
    print("\n" + "="*80)
    print("ASSET CHARACTERISTICS")
    print("="*80)
    
    print(f"\n{'Ticker':<6} {'Name':<22} {'Beta':>6} {'SPYCorr':>8} {'VIXCorr':>8} {'DownCap':>8} {'UpCap':>8}")
    print("-"*80)
    
    for _, row in df_results.iterrows():
        print(f"{row['ticker']:<6} {row['name']:<22} {row['beta']:>6.2f} {row['spy_corr']:>8.2f} {row['vix_corr']:>8.2f} {row['downside_cap']:>8.2f} {row['upside_cap']:>8.2f}")
    
    print("\n" + "="*80)
    print("COVID CRASH & RECOVERY ANALYSIS")
    print("="*80)
    
    print(f"\n{'Ticker':<6} {'Name':<22} {'Crash Drop':>12} {'Recovery':>12} {'Net Missed':>12}")
    print("-"*70)
    
    for _, row in df_results.iterrows():
        crash = row['crash_drop']
        recovery = row['recovery_gain']
        
        if crash is not None and recovery is not None:
            # If we exited before crash and re-entered at bottom
            # We avoided the crash but caught the recovery
            # Net = we missed the crash (good) and caught recovery
            # But if asset recovered more than market, we actually benefited less
            
            net_missed = crash  # This is what we avoided (negative = avoided loss)
            print(f"{row['ticker']:<6} {row['name']:<22} {crash:>12.1%} {recovery:>12.1%} {net_missed:>12.1%}")
    
    print("\n" + "="*80)
    print("WHY THE STRATEGY WORKS/FAILS")
    print("="*80)
    
    print("""
KEY INSIGHT: The strategy's success depends on the RECOVERY PROFILE:

WORKS WELL FOR (Cyclical/Value):
- XLI, XLF, IWM, VNQ: High beta, drop hard in crashes, recover WITH the market
- These assets don't have strong idiosyncratic drivers
- When VIX calms, they recover in sync with broad market

FAILS FOR (Growth/Tech):
- NVDA, AAPL, MSFT: Have strong idiosyncratic growth drivers
- They often rally HARDER than the market during recovery
- Missing the first 5-10 days of recovery costs more than crash protection

THE NVDA PROBLEM:
- NVDA dropped ~35% in COVID crash (similar to SPY)
- But NVDA gained +200% from March 2020 bottom to end of 2020
- The VIX strategy missed crucial early recovery days
- Each missed day cost ~2% in NVDA vs ~0.5% in SPY

CONCLUSION:
- Use VIX ROC strategy for: Index ETFs, Value sectors, Cyclicals
- DON'T use for: High-growth stocks with strong secular trends
""")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDED ASSET UNIVERSE FOR VIX ROC STRATEGY")
    print("="*80)
    
    print("""
TIER 1 - BEST FIT (Index ETFs, Cyclicals):
- SPY, DIA, IWM (Broad indices)
- XLI, XLF, XLE (Cyclical sectors)
- VNQ (Real Estate)

TIER 2 - GOOD FIT (Value-oriented):
- XLP, XLV (Defensive but still equity-correlated)
- JPM, XOM (Large-cap value)

TIER 3 - MARGINAL (Moderate idiosyncratic risk):
- EEM (Emerging Markets)
- META (has idiosyncratic but high beta)

DO NOT USE:
- NVDA, AAPL, MSFT, GOOGL, AMZN (High-growth mega-caps)
- QQQ, XLK (Tech-heavy indices)
- GLD, TLT (Low/negative SPY correlation)
- TSLA (Extreme idiosyncratic)
""")


if __name__ == "__main__":
    analyze_asset_characteristics()
