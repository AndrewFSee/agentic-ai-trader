"""
BOCPD Dual-Regime Strategy

Use TWO BOCPD detectors:
1. Volatility regime detector (high/low vol)
2. Momentum regime detector (bullish/bearish)

Position sizing:
- Full position when: bullish + low vol
- Reduced position when: bullish + high vol
- Zero when: bearish
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD


def dual_regime_backtest():
    """Test dual-regime strategy."""
    
    print("=" * 80)
    print("BOCPD DUAL-REGIME STRATEGY")
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
    # Create signals
    # =========================================================================
    
    # Momentum signal: EMA of returns
    mom_ema5 = returns.ewm(span=5).mean().values
    mom_ema10 = returns.ewm(span=10).mean().values
    
    # Volatility signal: rolling realized vol
    vol_20d = returns.rolling(20).std().fillna(returns.std()).values * np.sqrt(252)
    
    # Standardize signals
    mom5_std = (mom_ema5 - np.mean(mom_ema5)) / np.std(mom_ema5)
    mom10_std = (mom_ema10 - np.mean(mom_ema10)) / np.std(mom_ema10)
    vol_std = (vol_20d - np.mean(vol_20d)) / np.std(vol_20d)
    
    # =========================================================================
    # BOCPD Detectors
    # =========================================================================
    
    # Momentum detector (faster, for trend changes)
    mom_detector = BOCPD(hazard_rate=1/20, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    # Volatility detector (slower, for regime changes)
    vol_detector = BOCPD(hazard_rate=1/50, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    # =========================================================================
    # Run strategy
    # =========================================================================
    
    n = len(raw_returns)
    positions = np.zeros(n)
    
    mom_regime_obs = []
    vol_regime_obs = []
    mom_regime_start = 0
    vol_regime_start = 0
    
    mom_regime_mean = 0
    vol_regime_mean = 0
    
    regime_log = []
    
    for t in range(n):
        # Update detectors
        mom_detector.update(mom5_std[t])
        vol_detector.update(vol_std[t])
        
        mom_regime_obs.append(mom_ema5[t])
        vol_regime_obs.append(vol_20d[t])
        
        # Check for momentum regime change
        if t > 0 and len(mom_detector.map_run_lengths) >= 2:
            if mom_detector.map_run_lengths[-2] > 3 and mom_detector.map_run_lengths[-1] <= 2:
                if (t - mom_regime_start) >= 3:
                    mom_regime_start = t
                    mom_regime_obs = [mom_ema5[t]]
        
        # Check for volatility regime change
        if t > 0 and len(vol_detector.map_run_lengths) >= 2:
            if vol_detector.map_run_lengths[-2] > 3 and vol_detector.map_run_lengths[-1] <= 2:
                if (t - vol_regime_start) >= 5:
                    vol_regime_start = t
                    vol_regime_obs = [vol_20d[t]]
        
        # Calculate regime characteristics
        if len(mom_regime_obs) >= 2:
            mom_regime_mean = np.mean(mom_regime_obs[-10:])
        
        if len(vol_regime_obs) >= 2:
            vol_regime_mean = np.mean(vol_regime_obs[-10:])
        
        # Classify regimes
        is_bullish = mom_regime_mean > 0
        vol_median = np.median(vol_20d[:max(1, t)])
        is_high_vol = vol_regime_mean > vol_median * 1.2  # 20% above median
        
        # Position sizing based on dual regime
        if is_bullish:
            if is_high_vol:
                positions[t] = 0.5  # Reduced position in high vol
            else:
                positions[t] = 1.0  # Full position in low vol
        else:
            positions[t] = 0.0  # Cash when bearish
        
        if t % 252 == 0:  # Log annually
            regime_log.append({
                'date': dates[t].strftime('%Y-%m-%d'),
                'mom_mean': mom_regime_mean,
                'vol_mean': vol_regime_mean,
                'is_bullish': is_bullish,
                'is_high_vol': is_high_vol,
                'position': positions[t]
            })
    
    # Shift positions
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    
    strat_ret = pos_shifted * raw_returns
    
    # =========================================================================
    # Metrics
    # =========================================================================
    print("\n" + "-" * 60)
    print("DUAL-REGIME STRATEGY RESULTS")
    print("-" * 60)
    
    strat_total = (1 + strat_ret).prod() - 1
    bench_total = (1 + raw_returns).prod() - 1
    
    strat_vol = strat_ret.std() * np.sqrt(252)
    bench_vol = raw_returns.std() * np.sqrt(252)
    
    strat_annual = (1 + strat_total) ** (252 / n) - 1
    bench_annual = (1 + bench_total) ** (252 / n) - 1
    
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
    
    strat_cum = np.cumprod(1 + strat_ret)
    bench_cum = np.cumprod(1 + raw_returns)
    
    strat_peak = np.maximum.accumulate(strat_cum)
    bench_peak = np.maximum.accumulate(bench_cum)
    
    strat_dd = (strat_cum / strat_peak - 1).min()
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    time_full = (pos_shifted == 1).sum() / n
    time_reduced = (pos_shifted == 0.5).sum() / n
    time_cash = (pos_shifted == 0).sum() / n
    n_trades = np.sum(np.diff(positions) != 0)
    
    print(f"\nTotal Return:      {strat_total:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"Annualized:        {strat_annual:>7.1%} vs {bench_annual:>7.1%}")
    print(f"Volatility:        {strat_vol:>7.1%} vs {bench_vol:>7.1%}")
    print(f"Sharpe Ratio:      {strat_sharpe:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"Max Drawdown:      {strat_dd:>7.1%} vs {bench_dd:>7.1%}")
    print(f"\nTime Full (1.0):   {time_full:>7.1%}")
    print(f"Time Reduced (0.5): {time_reduced:>7.1%}")
    print(f"Time Cash (0.0):   {time_cash:>7.1%}")
    print(f"Number of Trades:  {int(n_trades):>7}")
    
    # =========================================================================
    # Now test a simpler approach: just use BOCPD for timing, not regime
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALTERNATIVE: BOCPD Change Point Timing Strategy")
    print("=" * 80)
    
    # Idea: Stay long, but exit on change point if momentum is negative
    
    detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    positions2 = np.ones(n)  # Start long
    
    current_pos = 1
    change_points = []
    
    for t in range(n):
        result = detector.update(mom10_std[t])
        
        # Detect change point
        is_change = False
        if t > 0 and len(detector.map_run_lengths) >= 2:
            if detector.map_run_lengths[-2] > 3 and detector.map_run_lengths[-1] <= 2:
                is_change = True
                change_points.append(t)
        
        if is_change:
            # On change point, look at recent momentum to decide
            recent_mom = np.mean(mom_ema10[max(0, t-5):t+1]) if t >= 5 else mom_ema10[t]
            
            if recent_mom > 0:
                current_pos = 1  # Stay/go long
            else:
                current_pos = 0  # Go to cash
        
        positions2[t] = current_pos
    
    pos2_shifted = np.roll(positions2, 1)
    pos2_shifted[0] = 1  # Start long
    
    strat_ret2 = pos2_shifted * raw_returns
    
    strat_total2 = (1 + strat_ret2).prod() - 1
    strat_vol2 = strat_ret2.std() * np.sqrt(252)
    strat_annual2 = (1 + strat_total2) ** (252 / n) - 1
    strat_sharpe2 = strat_annual2 / strat_vol2 if strat_vol2 > 0 else 0
    
    strat_cum2 = np.cumprod(1 + strat_ret2)
    strat_peak2 = np.maximum.accumulate(strat_cum2)
    strat_dd2 = (strat_cum2 / strat_peak2 - 1).min()
    
    print(f"\nChange Point Timing Strategy:")
    print(f"  Total Return:    {strat_total2:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"  Sharpe Ratio:    {strat_sharpe2:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"  Max Drawdown:    {strat_dd2:>7.1%} vs {bench_dd:>7.1%}")
    print(f"  Change Points:   {len(change_points)}")
    print(f"  Trades:          {int(np.sum(np.diff(positions2) != 0))}")
    
    # =========================================================================
    # Compare with simple moving average crossover baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: Simple EMA Crossover (no BOCPD)")
    print("=" * 80)
    
    ema_fast = close.ewm(span=10).mean().values[1:]  # Align with returns
    ema_slow = close.ewm(span=50).mean().values[1:]
    
    positions_sma = np.where(ema_fast > ema_slow, 1, 0)
    pos_sma_shifted = np.roll(positions_sma, 1)
    pos_sma_shifted[0] = 0
    
    strat_ret_sma = pos_sma_shifted * raw_returns
    
    strat_total_sma = (1 + strat_ret_sma).prod() - 1
    strat_vol_sma = strat_ret_sma.std() * np.sqrt(252)
    strat_annual_sma = (1 + strat_total_sma) ** (252 / n) - 1
    strat_sharpe_sma = strat_annual_sma / strat_vol_sma if strat_vol_sma > 0 else 0
    
    strat_cum_sma = np.cumprod(1 + strat_ret_sma)
    strat_peak_sma = np.maximum.accumulate(strat_cum_sma)
    strat_dd_sma = (strat_cum_sma / strat_peak_sma - 1).min()
    
    print(f"\nEMA 10/50 Crossover:")
    print(f"  Total Return:    {strat_total_sma:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"  Sharpe Ratio:    {strat_sharpe_sma:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"  Max Drawdown:    {strat_dd_sma:>7.1%} vs {bench_dd:>7.1%}")
    print(f"  Trades:          {int(np.sum(np.diff(positions_sma) != 0))}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Strategy':<30} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 60)
    print(f"{'Dual-Regime BOCPD':<30} {strat_total:>7.1%} {strat_sharpe:>8.2f} {strat_dd:>7.1%}")
    print(f"{'Change Point Timing':<30} {strat_total2:>7.1%} {strat_sharpe2:>8.2f} {strat_dd2:>7.1%}")
    print(f"{'EMA 10/50 Crossover':<30} {strat_total_sma:>7.1%} {strat_sharpe_sma:>8.2f} {strat_dd_sma:>7.1%}")
    print(f"{'Buy & Hold':<30} {bench_total:>7.1%} {bench_sharpe:>8.2f} {bench_dd:>7.1%}")


if __name__ == "__main__":
    dual_regime_backtest()
