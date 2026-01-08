"""
BOCPD on Different Price Signals

Compare what BOCPD detects when run on:
1. Raw daily returns (very noisy)
2. Cumulative log prices (trend detection)
3. Rolling mean returns (momentum)
4. Volatility (abs returns)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD
from typing import List


def run_comparison():
    """Compare BOCPD on different price signals."""
    
    print("=" * 70)
    print("BOCPD ON DIFFERENT PRICE SIGNALS")
    print("=" * 70)
    
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
    
    returns = close.pct_change().dropna().values
    prices = close.values[1:]  # Align with returns
    dates = close.index[1:]
    
    print(f"Data: {len(returns)} observations")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Signal 1: Raw Returns (what we were doing)
    # =========================================================================
    print("\n" + "-" * 70)
    print("SIGNAL 1: Raw Daily Returns")
    print("-" * 70)
    print("This tries to detect mean shifts in returns (bullish/bearish)")
    
    returns_std = (returns - np.mean(returns)) / np.std(returns)
    
    detector = BOCPD(hazard_rate=1/100, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in returns_std:
        detector.update(x)
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=20)
    print(f"\nChange points detected: {len(cps)}")
    print("Problem: Daily returns are ~N(0, σ) - almost no persistent mean shift!")
    print(f"Return mean: {returns.mean():.6f}, std: {returns.std():.4f}")
    print("Signal-to-noise ratio is terrible for detecting 'bullish' vs 'bearish'")
    
    # =========================================================================
    # Signal 2: Log Prices (Trend Detection)
    # =========================================================================
    print("\n" + "-" * 70)
    print("SIGNAL 2: Log Prices (Trend Detection)")
    print("-" * 70)
    print("This detects changes in the TREND (slope of log prices)")
    
    log_prices = np.log(prices)
    # Use first differences of log prices = returns, but we want to detect trend changes
    # Instead, use the log price level, standardized
    log_prices_std = (log_prices - np.mean(log_prices)) / np.std(log_prices)
    
    detector = BOCPD(hazard_rate=1/100, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in log_prices_std:
        detector.update(x)
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=20)
    print(f"\nChange points detected: {len(cps)}")
    print("Note: Log prices have a strong trend, so BOCPD sees constant 'change'")
    
    if cps:
        print("\nFirst 10 change points:")
        for cp in cps[:10]:
            if cp < len(dates):
                print(f"  {dates[cp-1].strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Signal 3: Rolling Momentum (Smoothed Returns)
    # =========================================================================
    print("\n" + "-" * 70)
    print("SIGNAL 3: Rolling 20-day Momentum")
    print("-" * 70)
    print("This detects shifts in medium-term momentum")
    
    window = 20
    rolling_mean = pd.Series(returns).rolling(window).mean().dropna().values
    rolling_std = (rolling_mean - np.mean(rolling_mean)) / np.std(rolling_mean)
    dates_rolling = dates[window-1:]
    
    detector = BOCPD(hazard_rate=1/100, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in rolling_std:
        detector.update(x)
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=20)
    print(f"\nChange points detected: {len(cps)}")
    
    if cps:
        print("\nFirst 15 change points:")
        for cp in cps[:15]:
            if cp < len(dates_rolling):
                # Get context
                idx = cp - 1
                momentum_before = rolling_mean[max(0, idx-5):idx].mean() if idx > 0 else 0
                momentum_after = rolling_mean[idx:min(len(rolling_mean), idx+5)].mean()
                direction = "↑ Bullish" if momentum_after > momentum_before else "↓ Bearish"
                print(f"  {dates_rolling[idx].strftime('%Y-%m-%d')}: {direction} (mom: {momentum_before*100:.2f}% -> {momentum_after*100:.2f}%)")
    
    # =========================================================================
    # Signal 4: Volatility (what worked best)
    # =========================================================================
    print("\n" + "-" * 70)
    print("SIGNAL 4: Realized Volatility (20-day)")
    print("-" * 70)
    print("This detects volatility regime changes")
    
    rolling_vol = pd.Series(returns).rolling(window).std().dropna().values * np.sqrt(252)
    vol_std = (rolling_vol - np.mean(rolling_vol)) / np.std(rolling_vol)
    
    detector = BOCPD(hazard_rate=1/50, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in vol_std:
        detector.update(x)
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=20)
    print(f"\nChange points detected: {len(cps)}")
    
    if cps:
        print("\nFirst 15 change points:")
        for cp in cps[:15]:
            if cp < len(dates_rolling):
                idx = cp - 1
                vol_at_cp = rolling_vol[idx] * 100
                regime = "HIGH VOL" if vol_at_cp > 15 else "LOW VOL"
                print(f"  {dates_rolling[idx].strftime('%Y-%m-%d')}: {regime} ({vol_at_cp:.1f}% annualized)")
    
    # =========================================================================
    # Backtest: Momentum-based strategy
    # =========================================================================
    print("\n" + "=" * 70)
    print("BACKTEST: Momentum Regime Strategy")
    print("=" * 70)
    
    # Reset detector for online simulation
    detector = BOCPD(hazard_rate=1/100, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions = np.zeros(len(rolling_mean))
    current_position = 1  # Start long
    regime_start = 0
    observations_in_regime = []
    
    for t in range(len(rolling_std)):
        detector.update(rolling_std[t])
        observations_in_regime.append(rolling_mean[t])
        
        # Check for regime change
        if t > 0 and len(detector.map_run_lengths) > 1:
            prev_map = detector.map_run_lengths[-2]
            curr_map = detector.map_run_lengths[-1]
            
            if prev_map > 3 and curr_map <= 2 and (t - regime_start) >= 10:
                regime_start = t
                observations_in_regime = [rolling_mean[t]]
        
        # Classify regime after 5 observations
        if len(observations_in_regime) >= 5:
            recent_momentum = np.mean(observations_in_regime[-5:])
            current_position = 1 if recent_momentum > 0 else 0
        
        positions[t] = current_position
    
    # Calculate returns (use raw returns aligned with rolling window)
    returns_aligned = returns[window-1:]
    
    # Shift positions by 1
    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0
    
    strategy_returns = positions_shifted * returns_aligned
    
    # Metrics
    strat_total = (1 + strategy_returns).prod() - 1
    bench_total = (1 + returns_aligned).prod() - 1
    
    strat_vol = strategy_returns.std() * np.sqrt(252)
    bench_vol = returns_aligned.std() * np.sqrt(252)
    
    n_days = len(strategy_returns)
    strat_annual = (1 + strat_total) ** (252 / n_days) - 1
    bench_annual = (1 + bench_total) ** (252 / n_days) - 1
    
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
    
    # Max drawdown
    strat_cum = np.cumprod(1 + strategy_returns)
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum / strat_peak - 1).min()
    
    bench_cum = np.cumprod(1 + returns_aligned)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    time_in_market = (positions_shifted != 0).sum() / len(positions_shifted)
    
    print(f"\n{'Metric':<25} {'Strategy':>12} {'Buy & Hold':>12}")
    print("-" * 50)
    print(f"{'Total Return':<25} {strat_total:>11.1%} {bench_total:>11.1%}")
    print(f"{'Annual Return':<25} {strat_annual:>11.1%} {bench_annual:>11.1%}")
    print(f"{'Sharpe Ratio':<25} {strat_sharpe:>12.2f} {bench_sharpe:>12.2f}")
    print(f"{'Max Drawdown':<25} {strat_dd:>11.1%} {bench_dd:>11.1%}")
    print(f"{'Volatility (ann)':<25} {strat_vol:>11.1%} {bench_vol:>11.1%}")
    print(f"{'Time in Market':<25} {time_in_market:>11.1%} {'100.0%':>12}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
BOCPD can detect different types of change points depending on the input signal:

1. RAW RETURNS: Too noisy - daily returns have near-zero mean, huge variance
   Not useful for bullish/bearish regime detection

2. LOG PRICES: Detects trend breaks, but non-stationary (always trending)
   Need to use returns or detrended prices

3. ROLLING MOMENTUM: Detects shifts in medium-term momentum
   Better signal-to-noise ratio, useful for trend-following

4. VOLATILITY: Detects volatility regime changes
   Clear regimes (low vol / high vol), works well

For PRICE/TREND change points, use Rolling Momentum (Signal 3).
For VOLATILITY change points, use Realized Vol (Signal 4).
    """)


if __name__ == "__main__":
    run_comparison()
