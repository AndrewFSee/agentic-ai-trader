"""
VIX ROC Strategy - Multi-Asset Testing

Test the walk-forward VIX ROC strategy on:
1. Major S&P 500 large caps (AAPL, MSFT, GOOGL, AMZN, NVDA, etc.)
2. Sector ETFs (XLK, XLF, XLE, XLV, etc.)
3. Other major indices (QQQ, IWM, DIA)
4. Less correlated assets (GLD, TLT, EEM)

Hypothesis: Assets highly correlated with SPY should benefit most
from VIX-based signals since VIX measures S&P 500 fear.
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
    roc_lookback: int = 10
    exit_roc_thresh: float = 0.50
    reentry_roc_thresh: float = 0.15
    min_exit_days: int = 5


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


def calculate_correlation(asset: pd.DataFrame, spy: pd.DataFrame) -> float:
    """Calculate return correlation with SPY."""
    common_idx = asset.index.intersection(spy.index)
    if len(common_idx) < 50:
        return 0.0
    
    asset_close = asset.loc[common_idx, 'Close'] if 'Close' in asset.columns else asset.loc[common_idx].iloc[:, 0]
    spy_close = spy.loc[common_idx, 'Close'] if 'Close' in spy.columns else spy.loc[common_idx].iloc[:, 0]
    
    asset_ret = asset_close.pct_change().dropna()
    spy_ret = spy_close.pct_change().dropna()
    
    common = asset_ret.index.intersection(spy_ret.index)
    if len(common) < 50:
        return 0.0
    
    return asset_ret.loc[common].corr(spy_ret.loc[common])


def main():
    print("="*80)
    print("VIX ROC STRATEGY - MULTI-ASSET TESTING")
    print("="*80)
    
    # Walk-forward optimal parameters
    params = ROCParams(
        roc_lookback=10,
        exit_roc_thresh=0.50,
        reentry_roc_thresh=0.15,
        min_exit_days=5
    )
    
    print(f"\nStrategy Parameters (Walk-Forward Optimized on 2010-2019):")
    print(f"  Exit when VIX ROC > {params.exit_roc_thresh:.0%}")
    print(f"  Re-enter when VIX ROC < {params.reentry_roc_thresh:.0%}")
    print(f"  Lookback: {params.roc_lookback} days, Min out: {params.min_exit_days} days")
    
    # Assets to test
    assets = {
        # Major indices
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq 100 ETF',
        'IWM': 'Russell 2000 ETF',
        'DIA': 'Dow Jones ETF',
        
        # Mega-cap tech
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
        'TSLA': 'Tesla',
        
        # Other large caps
        'JPM': 'JPMorgan',
        'V': 'Visa',
        'JNJ': 'Johnson & Johnson',
        'UNH': 'UnitedHealth',
        'XOM': 'Exxon Mobil',
        'WMT': 'Walmart',
        'PG': 'Procter & Gamble',
        
        # Sector ETFs
        'XLK': 'Tech Sector',
        'XLF': 'Financial Sector',
        'XLE': 'Energy Sector',
        'XLV': 'Healthcare Sector',
        'XLI': 'Industrial Sector',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        
        # Less correlated assets
        'GLD': 'Gold',
        'TLT': 'Long-term Treasuries',
        'EEM': 'Emerging Markets',
        'VNQ': 'Real Estate',
    }
    
    # Download data
    print("\nDownloading data...")
    
    # Get VIX and SPY first
    vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01", progress=False)
    spy_data = yf.download("SPY", start="2015-01-01", end="2025-01-01", progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    
    # Download all assets
    all_data = {}
    for ticker in assets.keys():
        try:
            df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > 100:
                all_data[ticker] = df
        except Exception as e:
            print(f"  Failed to download {ticker}: {e}")
    
    print(f"Downloaded {len(all_data)} assets")
    
    # Test each asset
    strategy = VIXROCStrategy(params)
    results_list = []
    
    # Test periods
    test_periods = [
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("Full 2020-2024", "2020-01-01", "2024-12-31"),
    ]
    
    print("\n" + "="*80)
    print("TESTING EACH ASSET")
    print("="*80)
    
    for ticker, name in assets.items():
        if ticker not in all_data:
            continue
        
        df = all_data[ticker]
        corr = calculate_correlation(df, spy_data)
        
        result_entry = {
            'ticker': ticker,
            'name': name,
            'spy_corr': corr,
        }
        
        # Test on each period
        for period_name, start, end in test_periods:
            asset_period = df.loc[start:end]
            vix_period = vix.loc[start:end]
            
            if len(asset_period) < 50:
                result_entry[f'{period_name}_excess'] = None
                result_entry[f'{period_name}_dd_imp'] = None
                continue
            
            results = strategy.run(asset_period, vix_period)
            metrics = evaluate(results)
            
            result_entry[f'{period_name}_excess'] = metrics['excess']
            result_entry[f'{period_name}_dd_imp'] = metrics['dd_improvement']
            result_entry[f'{period_name}_win'] = metrics['win']
        
        results_list.append(result_entry)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by SPY correlation
    results_df = results_df.sort_values('spy_corr', ascending=False)
    
    # Print results tables
    print("\n" + "="*80)
    print("RESULTS BY SPY CORRELATION (Highest to Lowest)")
    print("="*80)
    
    print(f"\n{'Ticker':<8} {'Name':<22} {'SPY Corr':>10} {'COVID':>10} {'2022':>10} {'Full':>10}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        covid = row.get('2020 COVID_excess')
        bear = row.get('2022 Bear_excess')
        full = row.get('Full 2020-2024_excess')
        
        covid_str = f"{covid:+.1%}" if covid is not None else "N/A"
        bear_str = f"{bear:+.1%}" if bear is not None else "N/A"
        full_str = f"{full:+.1%}" if full is not None else "N/A"
        
        print(f"{row['ticker']:<8} {row['name']:<22} {row['spy_corr']:>10.2f} {covid_str:>10} {bear_str:>10} {full_str:>10}")
    
    # Analyze by correlation buckets
    print("\n" + "="*80)
    print("ANALYSIS BY CORRELATION BUCKET")
    print("="*80)
    
    def get_bucket(corr):
        if corr >= 0.8:
            return "High (≥0.8)"
        elif corr >= 0.6:
            return "Medium (0.6-0.8)"
        elif corr >= 0.4:
            return "Low (0.4-0.6)"
        else:
            return "Very Low (<0.4)"
    
    results_df['corr_bucket'] = results_df['spy_corr'].apply(get_bucket)
    
    for bucket in ["High (≥0.8)", "Medium (0.6-0.8)", "Low (0.4-0.6)", "Very Low (<0.4)"]:
        bucket_df = results_df[results_df['corr_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue
        
        print(f"\n{bucket} Correlation Assets ({len(bucket_df)} assets):")
        
        # Calculate averages
        covid_avg = bucket_df['2020 COVID_excess'].dropna().mean()
        bear_avg = bucket_df['2022 Bear_excess'].dropna().mean()
        full_avg = bucket_df['Full 2020-2024_excess'].dropna().mean()
        
        covid_wins = bucket_df['2020 COVID_win'].dropna().sum()
        bear_wins = bucket_df['2022 Bear_win'].dropna().sum()
        full_wins = bucket_df['Full 2020-2024_win'].dropna().sum()
        
        n = len(bucket_df)
        
        print(f"  COVID 2020:    Avg excess {covid_avg:+.1%}, Wins: {int(covid_wins)}/{n}")
        print(f"  2022 Bear:     Avg excess {bear_avg:+.1%}, Wins: {int(bear_wins)}/{n}")
        print(f"  Full 2020-24:  Avg excess {full_avg:+.1%}, Wins: {int(full_wins)}/{n}")
    
    # Best and worst performers
    print("\n" + "="*80)
    print("TOP 10 PERFORMERS (Full 2020-2024)")
    print("="*80)
    
    top10 = results_df.dropna(subset=['Full 2020-2024_excess']).nlargest(10, 'Full 2020-2024_excess')
    
    for _, row in top10.iterrows():
        excess = row['Full 2020-2024_excess']
        dd_imp = row.get('Full 2020-2024_dd_imp', 0)
        print(f"  {row['ticker']:<6} {row['name']:<22} Corr: {row['spy_corr']:.2f}  Excess: {excess:+.1%}  DD Imp: {dd_imp:+.1%}")
    
    print("\n" + "="*80)
    print("BOTTOM 5 PERFORMERS (Full 2020-2024)")
    print("="*80)
    
    bottom5 = results_df.dropna(subset=['Full 2020-2024_excess']).nsmallest(5, 'Full 2020-2024_excess')
    
    for _, row in bottom5.iterrows():
        excess = row['Full 2020-2024_excess']
        dd_imp = row.get('Full 2020-2024_dd_imp', 0)
        print(f"  {row['ticker']:<6} {row['name']:<22} Corr: {row['spy_corr']:.2f}  Excess: {excess:+.1%}  DD Imp: {dd_imp:+.1%}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_assets = len(results_df)
    
    full_wins = results_df['Full 2020-2024_win'].dropna().sum()
    full_count = results_df['Full 2020-2024_win'].dropna().count()
    
    covid_wins = results_df['2020 COVID_win'].dropna().sum()
    covid_count = results_df['2020 COVID_win'].dropna().count()
    
    bear_wins = results_df['2022 Bear_win'].dropna().sum()
    bear_count = results_df['2022 Bear_win'].dropna().count()
    
    print(f"\nAssets tested: {total_assets}")
    print(f"\nWin rates:")
    print(f"  2020 COVID:    {int(covid_wins)}/{int(covid_count)} ({100*covid_wins/covid_count:.0f}%)")
    print(f"  2022 Bear:     {int(bear_wins)}/{int(bear_count)} ({100*bear_wins/bear_count:.0f}%)")
    print(f"  Full 2020-24:  {int(full_wins)}/{int(full_count)} ({100*full_wins/full_count:.0f}%)")
    
    # Correlation with strategy performance
    valid_data = results_df.dropna(subset=['Full 2020-2024_excess'])
    if len(valid_data) > 5:
        perf_corr = valid_data['spy_corr'].corr(valid_data['Full 2020-2024_excess'])
        print(f"\nCorrelation between SPY correlation and strategy excess return: {perf_corr:.3f}")
    
    return results_df


if __name__ == "__main__":
    results = main()
