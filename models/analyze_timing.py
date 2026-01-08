"""
Analyze BOCPD timing - is it catching structural breaks too late?
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from alpha_regime_detector_v2 import AlphaRegimeDetectorV2, FeatureConfig, BOCPDConfig, RegimeConfig, OverlayConfig

def analyze_covid_timing():
    """Detailed day-by-day analysis of COVID crash timing."""
    
    print("Downloading SPY data...")
    spy = yf.download('SPY', start='2020-01-01', end='2020-05-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    detector = AlphaRegimeDetectorV2(
        bocpd_config=BOCPDConfig(hazard_rate=1/50),  # More sensitive
        overlay_config=OverlayConfig()
    )
    
    results = detector.process_dataframe('SPY', spy)
    close_prices = spy['Close'].iloc[1:].values
    daily_rets = spy['Close'].pct_change().iloc[1:].values
    
    print("\n" + "=" * 95)
    print("COVID CRASH TIMELINE - Is BOCPD too late?")
    print("=" * 95)
    print(f"{'Date':<12} {'Close':>7} {'Ret':>7} {'Pos':>5} {'ERL':>6} {'Risk':>5} {'Trans':>5}  Notes")
    print("-" * 95)
    
    covid_start = '2020-02-18'
    covid_end = '2020-03-25'
    mask = (results.index >= covid_start) & (results.index <= covid_end)
    
    total_loss_bh = 0
    total_loss_strat = 0
    
    for idx in results.index[mask]:
        loc = results.index.get_loc(idx)
        row = results.iloc[loc]
        close = close_prices[loc]
        ret = daily_rets[loc] * 100
        pos = row['position_scalar']
        erl = row['expected_run_length']
        risk = max(row['risk_cp'], row['risk_rl'])
        trans = 'YES' if row['in_transition'] else 'no'
        
        # Note: position at t applies to return at t+1 (1-day lag)
        # So for analysis, we look at yesterday's position vs today's return
        if loc > 0:
            prev_pos = results.iloc[loc-1]['position_scalar']
            strat_ret = ret * prev_pos
        else:
            prev_pos = 1.0
            strat_ret = ret
        
        notes = ''
        if abs(ret) > 3: 
            notes += ' *** BIG MOVE ***'
        if ret < -5:
            notes += ' CRASH DAY'
        if pos < 0.5: 
            notes += ' SEVERE'
        elif pos < 0.9: 
            notes += ' reduced'
        
        # Track cumulative loss
        if ret < 0:
            total_loss_bh += ret
            total_loss_strat += strat_ret
        
        date_str = idx.strftime("%Y-%m-%d")
        print(f"{date_str:<12} {close:7.1f} {ret:+6.1f}% {pos:5.2f} {erl:6.1f} {risk:5.2f} {trans:>5}  {notes}")
    
    print("-" * 95)
    print(f"\nTotal downside captured by B&H: {total_loss_bh:.1f}%")
    print(f"Total downside captured by Strategy: {total_loss_strat:.1f}%")
    print(f"Downside reduction: {(1 - total_loss_strat/total_loss_bh)*100:.1f}%")
    
    # Key insight
    print("\n" + "=" * 95)
    print("KEY TIMING ANALYSIS")
    print("=" * 95)
    
    # Find first big down day
    first_crash_day = None
    for idx in results.index[mask]:
        loc = results.index.get_loc(idx)
        ret = daily_rets[loc] * 100
        if ret < -3:
            first_crash_day = idx
            first_crash_ret = ret
            break
    
    # Find when position first reduced
    first_reduced_day = None
    for idx in results.index[mask]:
        loc = results.index.get_loc(idx)
        pos = results.iloc[loc]['position_scalar']
        if pos < 0.95:
            first_reduced_day = idx
            first_reduced_pos = pos
            break
    
    if first_crash_day and first_reduced_day:
        print(f"\nFirst big crash day (>3%): {first_crash_day.strftime('%Y-%m-%d')} ({first_crash_ret:.1f}%)")
        print(f"First position reduction: {first_reduced_day.strftime('%Y-%m-%d')} (pos={first_reduced_pos:.2f})")
        
        days_late = (first_reduced_day - first_crash_day).days
        print(f"\nTIMING GAP: Position reduced {days_late} days AFTER first crash day")
        
        if days_late > 0:
            print("\n*** DIAGNOSIS: BOCPD is REACTIVE, not PREDICTIVE ***")
            print("The model detects the regime change AFTER the volatility spike,")
            print("which means position reduction happens AFTER the big down days.")
    
    return results


def compute_actual_edge():
    """Compute actual edge with proper timing."""
    
    print("\n" + "=" * 95)
    print("ACTUAL EDGE COMPUTATION (Full Period)")
    print("=" * 95)
    
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    detector = AlphaRegimeDetectorV2(
        bocpd_config=BOCPDConfig(hazard_rate=1/50),
        overlay_config=OverlayConfig()
    )
    
    results = detector.process_dataframe('SPY', spy)
    
    # Properly lagged signals: position[t-1] applies to return[t]
    signals = results['position_scalar'].shift(1).fillna(1.0)
    returns = spy['Close'].pct_change().iloc[1:]
    
    # Align
    aligned = pd.DataFrame({
        'return': returns.values,
        'signal': signals.values
    }, index=results.index)
    
    strat_returns = aligned['return'] * aligned['signal']
    
    # Compute metrics
    def metrics(rets):
        total = (1 + rets).prod() - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / vol if vol > 0 else 0
        
        cum = (1 + rets).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        return total * 100, sharpe, max_dd * 100
    
    bh_total, bh_sharpe, bh_dd = metrics(aligned['return'])
    st_total, st_sharpe, st_dd = metrics(strat_returns)
    
    print(f"\n{'Metric':<20} {'Strategy':>12} {'Buy & Hold':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Total Return':<20} {st_total:>11.1f}% {bh_total:>11.1f}% {st_total-bh_total:>+11.1f}%")
    print(f"{'Sharpe Ratio':<20} {st_sharpe:>12.2f} {bh_sharpe:>12.2f} {st_sharpe-bh_sharpe:>+12.2f}")
    print(f"{'Max Drawdown':<20} {st_dd:>11.1f}% {bh_dd:>11.1f}% {st_dd-bh_dd:>+11.1f}%")
    
    avg_pos = aligned['signal'].mean()
    time_reduced = (aligned['signal'] < 1.0).mean() * 100
    
    print(f"\n{'Avg Position':<20} {avg_pos:>12.2f}")
    print(f"{'Time Reduced':<20} {time_reduced:>11.1f}%")
    
    # The truth
    print("\n" + "=" * 95)
    print("VERDICT")
    print("=" * 95)
    
    if st_total < bh_total and st_dd >= bh_dd - 2:
        print("""
BOCPD as currently implemented provides NO EDGE because:

1. It's REACTIVE: Detects regime changes AFTER they happen
   - By the time ERL drops, the big down days have already occurred
   - Position reduction is 1-5 days LATE

2. It's TOO SLOW: The Normal-Inverse-Gamma conjugate prior updates
   gradually, requiring multiple observations to shift the posterior

3. It MISSES THE POINT: We don't need to detect THAT a crash happened,
   we need to detect it's ABOUT to happen (or at least same-day)

POSSIBLE FIXES:
- Use faster features (intraday volatility, VIX, put/call ratio)
- Use leading indicators instead of lagging (credit spreads, etc.)
- Increase hazard rate dramatically (but causes false positives)
- Accept this is a TAIL RISK tool, not an alpha generator
""")
    
    return aligned


if __name__ == "__main__":
    results = analyze_covid_timing()
    aligned = compute_actual_edge()
