"""
BOCPD with Wavelet Multi-Resolution - Full Backtest

Best configuration from testing:
- Wavelet: db6
- Keep 4 detail levels (remove only highest frequency D1)
- BOCPD with hazard_rate=1/30

Results: 98.2% return, 1.37 Sharpe, -9.6% MaxDD
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD
import pywt


def wavelet_multiresolution_signal(returns: np.ndarray, wavelet: str = 'db6',
                                    keep_levels: int = 4) -> np.ndarray:
    """
    Create multi-resolution wavelet signal by keeping specified detail levels.
    
    Args:
        returns: Raw return series
        wavelet: Wavelet type
        keep_levels: Number of detail levels to keep (from finest)
    
    Returns:
        Reconstructed signal with high-frequency noise removed
    """
    # Decompose
    level = 5
    coeffs = pywt.wavedec(returns, wavelet, mode='per', level=level)
    
    # Keep only specified number of detail levels
    modified_coeffs = [coeffs[0]]  # Always keep approximation
    for i in range(1, len(coeffs)):
        level_from_top = i
        if level_from_top <= keep_levels:
            modified_coeffs.append(coeffs[i])
        else:
            modified_coeffs.append(np.zeros_like(coeffs[i]))
    
    # Reconstruct
    reconstructed = pywt.waverec(modified_coeffs, wavelet, mode='per')
    if len(reconstructed) > len(returns):
        reconstructed = reconstructed[:len(returns)]
    
    return reconstructed


def full_backtest():
    """Run comprehensive backtest on wavelet-BOCPD strategy."""
    
    print("=" * 80)
    print("WAVELET + BOCPD STRATEGY - COMPREHENSIVE BACKTEST")
    print("=" * 80)
    
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    # Download SPY
    spy = yf.download("SPY", period="5y", progress=False)
    if spy.empty:
        print("Failed to download")
        return
    
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    returns = close.pct_change().dropna()
    dates = returns.index
    raw_returns = returns.values
    
    print(f"Data: {len(returns)} observations")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    n = len(raw_returns)
    n_years = n / 252
    
    # =========================================================================
    # Test different level configurations
    # =========================================================================
    print("\n" + "=" * 80)
    print("GRID SEARCH: Wavelet Levels x Hazard Rates")
    print("=" * 80)
    
    results = []
    
    for keep_levels in [3, 4, 5]:
        for hazard in [1/20, 1/25, 1/30, 1/40]:
            # Create wavelet signal
            signal = wavelet_multiresolution_signal(raw_returns, 'db6', keep_levels)
            signal_std = (signal - np.mean(signal)) / np.std(signal)
            
            # Run BOCPD
            detector = BOCPD(hazard_rate=hazard, mu0=0.0, kappa0=0.1,
                            alpha0=2.0, beta0=1.0)
            
            positions = np.zeros(n)
            regime_obs = []
            regime_start = 0
            
            for t in range(n):
                detector.update(signal_std[t])
                regime_obs.append(signal[t])
                
                if t > 0:
                    maps = detector.map_run_lengths
                    if len(maps) >= 2:
                        if maps[-2] > 3 and maps[-1] <= 2 and (t - regime_start) >= 3:
                            regime_start = t
                            regime_obs = [signal[t]]
                
                if len(regime_obs) >= 2:
                    regime_mean = np.mean(regime_obs[-10:])
                    positions[t] = 1 if regime_mean > 0 else 0
            
            pos_shifted = np.roll(positions, 1)
            pos_shifted[0] = 0
            strat_ret = pos_shifted * raw_returns
            
            # Metrics
            strat_total = (1 + strat_ret).prod() - 1
            strat_vol = strat_ret.std() * np.sqrt(252)
            strat_annual = (1 + strat_total) ** (252 / n) - 1
            strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
            
            strat_cum = np.cumprod(1 + strat_ret)
            strat_peak = np.maximum.accumulate(strat_cum)
            strat_dd = (strat_cum / strat_peak - 1).min()
            
            cps = detector.detect_change_points(method='map_drop', min_spacing=5)
            n_trades = np.sum(np.diff(positions) != 0)
            
            results.append({
                'levels': keep_levels,
                'hazard': hazard,
                'return': strat_total,
                'sharpe': strat_sharpe,
                'max_dd': strat_dd,
                'n_cps': len(cps),
                'trades': n_trades,
                'positions': positions,
                'strat_ret': strat_ret
            })
    
    # Buy & hold benchmark
    bench_total = (1 + raw_returns).prod() - 1
    bench_vol = raw_returns.std() * np.sqrt(252)
    bench_annual = (1 + bench_total) ** (252 / n) - 1
    bench_sharpe = bench_annual / bench_vol
    bench_cum = np.cumprod(1 + raw_returns)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    print(f"\n{'Levels':>8} {'Hazard':>10} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'CPs':>6}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        hz_str = f"1/{int(1/r['hazard'])}"
        print(f"{r['levels']:>8} {hz_str:>10} {r['return']:>9.1%} "
              f"{r['sharpe']:>8.2f} {r['max_dd']:>9.1%} {r['n_cps']:>6}")
    
    print(f"\n{'B&H':>8} {'--':>10} {bench_total:>9.1%} {bench_sharpe:>8.2f} {bench_dd:>9.1%} {'--':>6}")
    
    # =========================================================================
    # Best configuration analysis
    # =========================================================================
    best = max(results, key=lambda x: x['sharpe'])
    
    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION: Levels={best['levels']}, Hazard=1/{int(1/best['hazard'])}")
    print("=" * 80)
    
    print(f"\n  Total Return:     {best['return']:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"  Sharpe Ratio:     {best['sharpe']:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"  Max Drawdown:     {best['max_dd']:>7.1%} vs {bench_dd:>7.1%}")
    print(f"  Change Points:    {best['n_cps']:>7} ({best['n_cps']/n_years:.1f}/year)")
    print(f"  Trades:           {int(best['trades']):>7}")
    
    # =========================================================================
    # Yearly breakdown
    # =========================================================================
    print("\n" + "=" * 80)
    print("YEARLY PERFORMANCE (Best Config)")
    print("=" * 80)
    
    # Align with dates
    strat_returns_series = pd.Series(best['strat_ret'], index=dates)
    bench_returns_series = pd.Series(raw_returns, index=dates)
    
    print(f"\n{'Year':>6} {'Strategy':>12} {'B&H':>12} {'Outperform':>12}")
    print("-" * 50)
    
    for year in sorted(strat_returns_series.index.year.unique()):
        strat_yr = strat_returns_series[strat_returns_series.index.year == year]
        bench_yr = bench_returns_series[bench_returns_series.index.year == year]
        
        strat_ret_yr = (1 + strat_yr).prod() - 1
        bench_ret_yr = (1 + bench_yr).prod() - 1
        outperf = strat_ret_yr - bench_ret_yr
        
        print(f"{year:>6} {strat_ret_yr:>11.1%} {bench_ret_yr:>11.1%} {outperf:>11.1%}")
    
    # =========================================================================
    # Drawdown analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("DRAWDOWN ANALYSIS")
    print("=" * 80)
    
    strat_cum = np.cumprod(1 + best['strat_ret'])
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd_series = strat_cum / strat_peak - 1
    
    bench_cum = np.cumprod(1 + raw_returns)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd_series = bench_cum / bench_peak - 1
    
    # Top 5 drawdowns
    print("\nTop 5 Drawdowns (Strategy):")
    strat_dd_df = pd.Series(strat_dd_series, index=dates)
    for i, (idx, dd) in enumerate(strat_dd_df.nsmallest(5).items(), 1):
        print(f"  {i}. {dd:.1%} on {idx.strftime('%Y-%m-%d')}")
    
    print("\nTop 5 Drawdowns (Buy & Hold):")
    bench_dd_df = pd.Series(bench_dd_series, index=dates)
    for i, (idx, dd) in enumerate(bench_dd_df.nsmallest(5).items(), 1):
        print(f"  {i}. {dd:.1%} on {idx.strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Trade statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRADE STATISTICS")
    print("=" * 80)
    
    positions = best['positions']
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    
    # Identify trades
    entries = np.where((pos_shifted == 0) & (np.roll(pos_shifted, -1) == 1))[0]
    exits = np.where((pos_shifted == 1) & (np.roll(pos_shifted, -1) == 0))[0]
    
    time_in_market = (pos_shifted == 1).sum() / n
    
    print(f"\n  Time in Market:   {time_in_market:>7.1%}")
    print(f"  Entry Signals:    {len(entries):>7}")
    print(f"  Exit Signals:     {len(exits):>7}")
    
    # Return during in-market vs out-of-market periods
    in_market_ret = raw_returns[pos_shifted == 1]
    out_market_ret = raw_returns[pos_shifted == 0]
    
    if len(in_market_ret) > 0:
        print(f"\n  Mean Daily Return (In Market):  {in_market_ret.mean()*100:>6.3f}%")
    if len(out_market_ret) > 0:
        print(f"  Mean Daily Return (Out Market): {out_market_ret.mean()*100:>6.3f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"""
    The Wavelet + BOCPD strategy uses multi-resolution wavelet decomposition
    to filter out high-frequency noise from returns before detecting regime changes.
    
    Best Configuration:
      - Wavelet: Daubechies 6 (db6)
      - Keep detail levels: {best['levels']} (remove only finest scale noise)
      - Hazard rate: 1/{int(1/best['hazard'])}
      
    Performance:
      - Beat Buy & Hold:     {best['return'] - bench_total:>+7.1%}
      - Sharpe improvement:  {best['sharpe'] - bench_sharpe:>+7.2f}
      - DD reduction:        {bench_dd - best['max_dd']:>+7.1%}
      
    Trade frequency: {best['n_cps']/n_years:.1f} change points/year
    """)


if __name__ == "__main__":
    full_backtest()
