"""
BOCPD Momentum Strategy - Improved Backtest

The key insight: BOCPD tells us WHEN a regime change happens,
but we also get the regime's CHARACTERISTICS (mean, std) which
we should use for position sizing and direction.

Also test: Long/Short vs Long-Only, and regime filtering.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD


def run_comprehensive_backtest():
    """Run comprehensive backtest with multiple strategies."""
    
    print("=" * 80)
    print("BOCPD MOMENTUM STRATEGY - COMPREHENSIVE BACKTEST")
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
    
    # =========================================================================
    # Strategy 1: EMA-5 momentum with regime-based position sizing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 1: EMA-5 with Regime-Based Signals")
    print("=" * 80)
    
    window = 5
    hazard = 1/20
    signal = returns.ewm(span=window).mean().values
    signal_std = (signal - np.mean(signal)) / np.std(signal)
    
    detector = BOCPD(hazard_rate=hazard, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions = np.zeros(len(signal))
    regime_start = 0
    regime_obs = []
    
    for t in range(len(signal)):
        detector.update(signal_std[t])
        regime_obs.append(signal[t])  # Original (not standardized) signal
        
        # Detect regime change
        if t > 0:
            maps = detector.map_run_lengths
            if len(maps) >= 2:
                prev_map = maps[-2]
                curr_map = maps[-1]
                
                if prev_map > 3 and curr_map <= 2 and (t - regime_start) >= 3:
                    regime_start = t
                    regime_obs = [signal[t]]
        
        # Position based on regime mean
        if len(regime_obs) >= 2:
            regime_mean = np.mean(regime_obs[-10:])  # Last 10 obs or less
            
            # Simple: go long if recent momentum is positive
            if regime_mean > 0:
                positions[t] = 1
            else:
                positions[t] = 0  # Cash
    
    # Shift positions and calculate returns
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    
    strat_ret = pos_shifted * raw_returns
    
    print_metrics("EMA-5 Long-Only", strat_ret, raw_returns, pos_shifted)
    
    # =========================================================================
    # Strategy 2: Long/Short with confidence threshold
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 2: Long/Short with Confidence")
    print("=" * 80)
    
    detector2 = BOCPD(hazard_rate=hazard, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions2 = np.zeros(len(signal))
    regime_obs2 = []
    regime_start2 = 0
    
    for t in range(len(signal)):
        detector2.update(signal_std[t])
        regime_obs2.append(signal[t])
        
        # Detect regime change
        if t > 0:
            maps = detector2.map_run_lengths
            if len(maps) >= 2:
                if maps[-2] > 3 and maps[-1] <= 2 and (t - regime_start2) >= 3:
                    regime_start2 = t
                    regime_obs2 = [signal[t]]
        
        # Long/Short based on regime mean
        if len(regime_obs2) >= 2:
            regime_mean = np.mean(regime_obs2[-10:])
            regime_std_local = np.std(regime_obs2[-10:]) if len(regime_obs2) >= 3 else np.std(signal[:t+1])
            
            # Confidence: how strong is the signal relative to noise?
            z = regime_mean / regime_std_local if regime_std_local > 0 else 0
            
            if z > 0.5:
                positions2[t] = 1  # Strong bullish
            elif z < -0.5:
                positions2[t] = -1  # Strong bearish
            else:
                positions2[t] = 0  # No signal
    
    pos2_shifted = np.roll(positions2, 1)
    pos2_shifted[0] = 0
    
    strat_ret2 = pos2_shifted * raw_returns
    
    print_metrics("EMA-5 Long/Short", strat_ret2, raw_returns, pos2_shifted)
    
    # =========================================================================
    # Strategy 3: Trend-following with volatility filter
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 3: Trend-Following with Vol Filter")
    print("=" * 80)
    
    # Two signals: momentum + volatility
    mom_signal = returns.ewm(span=10).mean().values
    vol_signal = returns.rolling(20).std().fillna(0.01).values
    vol_threshold = np.percentile(vol_signal[20:], 75)  # High vol = top 25%
    
    mom_std = (mom_signal - np.mean(mom_signal)) / np.std(mom_signal)
    
    detector3 = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions3 = np.zeros(len(mom_signal))
    regime_obs3 = []
    regime_start3 = 0
    
    for t in range(len(mom_signal)):
        detector3.update(mom_std[t])
        regime_obs3.append(mom_signal[t])
        
        # Detect regime change
        if t > 0:
            maps = detector3.map_run_lengths
            if len(maps) >= 2:
                if maps[-2] > 3 and maps[-1] <= 2 and (t - regime_start3) >= 5:
                    regime_start3 = t
                    regime_obs3 = [mom_signal[t]]
        
        if len(regime_obs3) >= 3:
            regime_mean = np.mean(regime_obs3[-10:])
            
            # In high vol: reduce position or go to cash
            if vol_signal[t] > vol_threshold:
                positions3[t] = 0.5 if regime_mean > 0 else 0
            else:
                positions3[t] = 1 if regime_mean > 0 else 0
    
    pos3_shifted = np.roll(positions3, 1)
    pos3_shifted[0] = 0
    
    strat_ret3 = pos3_shifted * raw_returns
    
    print_metrics("Trend + Vol Filter", strat_ret3, raw_returns, pos3_shifted)
    
    # =========================================================================
    # Strategy 4: Multi-timeframe (fast + slow)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 4: Multi-Timeframe Momentum")
    print("=" * 80)
    
    # Fast signal (5-day) and slow signal (20-day)
    fast_mom = returns.ewm(span=5).mean().values
    slow_mom = returns.ewm(span=20).mean().values
    
    fast_std = (fast_mom - np.mean(fast_mom)) / np.std(fast_mom)
    slow_std = (slow_mom - np.mean(slow_mom)) / np.std(slow_mom)
    
    detector_fast = BOCPD(hazard_rate=1/15, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    detector_slow = BOCPD(hazard_rate=1/50, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions4 = np.zeros(len(fast_mom))
    fast_regime_mean = 0
    slow_regime_mean = 0
    fast_obs = []
    slow_obs = []
    fast_start = 0
    slow_start = 0
    
    for t in range(len(fast_mom)):
        detector_fast.update(fast_std[t])
        detector_slow.update(slow_std[t])
        
        fast_obs.append(fast_mom[t])
        slow_obs.append(slow_mom[t])
        
        # Fast regime changes
        if t > 0:
            fmaps = detector_fast.map_run_lengths
            if len(fmaps) >= 2 and fmaps[-2] > 3 and fmaps[-1] <= 2:
                fast_start = t
                fast_obs = [fast_mom[t]]
        
        # Slow regime changes
        if t > 0:
            smaps = detector_slow.map_run_lengths
            if len(smaps) >= 2 and smaps[-2] > 3 and smaps[-1] <= 2:
                slow_start = t
                slow_obs = [slow_mom[t]]
        
        if len(fast_obs) >= 2 and len(slow_obs) >= 2:
            fast_regime_mean = np.mean(fast_obs[-5:])
            slow_regime_mean = np.mean(slow_obs[-10:])
            
            # Only go long when both fast and slow agree
            if fast_regime_mean > 0 and slow_regime_mean > 0:
                positions4[t] = 1
            elif fast_regime_mean < 0 and slow_regime_mean < 0:
                positions4[t] = -0.5  # Partial short on agreement
            else:
                positions4[t] = 0  # Mixed signals = cash
    
    pos4_shifted = np.roll(positions4, 1)
    pos4_shifted[0] = 0
    
    strat_ret4 = pos4_shifted * raw_returns
    
    print_metrics("Multi-TF Momentum", strat_ret4, raw_returns, pos4_shifted)
    
    # =========================================================================
    # Strategy 5: Adaptive hazard rate based on realized volatility
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 5: Adaptive BOCPD (Vol-Adjusted Hazard)")
    print("=" * 80)
    
    mom_signal5 = returns.ewm(span=10).mean().values
    vol_5 = returns.rolling(20).std().fillna(0.01).values
    mom_std5 = (mom_signal5 - np.mean(mom_signal5)) / np.std(mom_signal5)
    
    vol_median = np.median(vol_5[20:])
    
    positions5 = np.zeros(len(mom_signal5))
    regime_obs5 = []
    regime_start5 = 0
    
    # We can't dynamically change hazard, but we can adjust detection threshold
    detector5 = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    for t in range(len(mom_signal5)):
        detector5.update(mom_std5[t])
        regime_obs5.append(mom_signal5[t])
        
        # Adaptive min_spacing based on volatility
        vol_ratio = vol_5[t] / vol_median if vol_median > 0 else 1
        adaptive_spacing = max(3, int(5 / vol_ratio))  # Faster in high vol
        
        # Detect regime change
        if t > 0:
            maps = detector5.map_run_lengths
            if len(maps) >= 2:
                if maps[-2] > 3 and maps[-1] <= 2 and (t - regime_start5) >= adaptive_spacing:
                    regime_start5 = t
                    regime_obs5 = [mom_signal5[t]]
        
        if len(regime_obs5) >= 2:
            regime_mean = np.mean(regime_obs5[-10:])
            positions5[t] = 1 if regime_mean > 0 else 0
    
    pos5_shifted = np.roll(positions5, 1)
    pos5_shifted[0] = 0
    
    strat_ret5 = pos5_shifted * raw_returns
    
    print_metrics("Adaptive BOCPD", strat_ret5, raw_returns, pos5_shifted)
    
    # =========================================================================
    # Strategy 6: Use prob_short_run instead of MAP drop
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 6: P(r<=k) Detection Method")
    print("=" * 80)
    
    mom_signal6 = returns.ewm(span=5).mean().values
    mom_std6 = (mom_signal6 - np.mean(mom_signal6)) / np.std(mom_signal6)
    
    detector6 = BOCPD(hazard_rate=1/20, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions6 = np.zeros(len(mom_signal6))
    regime_obs6 = []
    regime_start6 = 0
    
    for t in range(len(mom_signal6)):
        result = detector6.update(mom_std6[t], short_threshold=3)
        regime_obs6.append(mom_signal6[t])
        
        # Use prob_short_run for detection
        if t > 0 and result.prob_short_run > 0.8 and (t - regime_start6) >= 3:
            regime_start6 = t
            regime_obs6 = [mom_signal6[t]]
        
        if len(regime_obs6) >= 2:
            regime_mean = np.mean(regime_obs6[-10:])
            positions6[t] = 1 if regime_mean > 0 else 0
    
    pos6_shifted = np.roll(positions6, 1)
    pos6_shifted[0] = 0
    
    strat_ret6 = pos6_shifted * raw_returns
    
    print_metrics("P(r<=3) Method", strat_ret6, raw_returns, pos6_shifted)
    
    # =========================================================================
    # Compare all strategies in a summary table
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    strategies = [
        ("EMA-5 Long-Only", strat_ret, pos_shifted),
        ("EMA-5 Long/Short", strat_ret2, pos2_shifted),
        ("Trend + Vol Filter", strat_ret3, pos3_shifted),
        ("Multi-TF Momentum", strat_ret4, pos4_shifted),
        ("Adaptive BOCPD", strat_ret5, pos5_shifted),
        ("P(r<=3) Method", strat_ret6, pos6_shifted),
        ("Buy & Hold", raw_returns, np.ones(len(raw_returns)))
    ]
    
    print(f"\n{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
    print("-" * 60)
    
    for name, rets, pos in strategies:
        total_ret = (1 + rets).prod() - 1
        vol = rets.std() * np.sqrt(252)
        annual_ret = (1 + total_ret) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0
        sharpe = annual_ret / vol if vol > 0 else 0
        
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        max_dd = (cum / peak - 1).min()
        
        n_trades = int(np.sum(np.diff(pos) != 0)) if len(pos) > 1 else 0
        
        print(f"{name:<20} {total_ret:>9.1%} {sharpe:>8.2f} {max_dd:>9.1%} {n_trades:>8}")


def print_metrics(name: str, strat_ret: np.ndarray, bench_ret: np.ndarray, positions: np.ndarray):
    """Print detailed metrics for a strategy."""
    
    strat_total = (1 + strat_ret).prod() - 1
    bench_total = (1 + bench_ret).prod() - 1
    
    strat_vol = strat_ret.std() * np.sqrt(252)
    bench_vol = bench_ret.std() * np.sqrt(252)
    
    n_days = len(strat_ret)
    strat_annual = (1 + strat_total) ** (252 / n_days) - 1 if n_days > 0 else 0
    bench_annual = (1 + bench_total) ** (252 / n_days) - 1 if n_days > 0 else 0
    
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
    
    strat_cum = np.cumprod(1 + strat_ret)
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum / strat_peak - 1).min()
    
    bench_cum = np.cumprod(1 + bench_ret)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    time_long = (positions > 0).sum() / len(positions)
    time_short = (positions < 0).sum() / len(positions)
    n_trades = np.sum(np.diff(positions) != 0)
    
    print(f"\n  {name}:")
    print(f"    Total Return:     {strat_total:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"    Annualized:       {strat_annual:>7.1%} vs {bench_annual:>7.1%}")
    print(f"    Volatility:       {strat_vol:>7.1%} vs {bench_vol:>7.1%}")
    print(f"    Sharpe Ratio:     {strat_sharpe:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"    Max Drawdown:     {strat_dd:>7.1%} vs {bench_dd:>7.1%}")
    print(f"    Time Long:        {time_long:>7.1%}")
    print(f"    Time Short:       {time_short:>7.1%}")
    print(f"    Trades:           {int(n_trades):>7}")


if __name__ == "__main__":
    run_comprehensive_backtest()
