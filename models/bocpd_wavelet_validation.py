"""
Wavelet + BOCPD Strategy - Out-of-Sample Validation & Forward Bias Audit

Goals:
1. Split data into train (2021-2023) and test (2024-2025) periods
2. Tune parameters on train only
3. Evaluate on test (truly out-of-sample)
4. Audit for any forward-looking bias
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD
import pywt


def audit_forward_bias():
    """
    AUDIT: Check for forward-looking bias in the strategy.
    
    Common sources of bias:
    1. Using future data in wavelet decomposition
    2. Using future data in standardization
    3. Using future data in BOCPD updates
    4. Position taken before signal is available
    """
    
    print("=" * 80)
    print("FORWARD BIAS AUDIT")
    print("=" * 80)
    
    print("""
    AUDIT CHECKLIST:
    
    1. WAVELET DECOMPOSITION
       - pywt.wavedec() operates on the ENTIRE array at once
       - This means the wavelet coefficients at time t use data from t+1, t+2, ...
       - THIS IS FORWARD-LOOKING BIAS!
       
       FIX: Use online/causal wavelet filtering or expanding window
    
    2. STANDARDIZATION
       - (signal - mean) / std uses the ENTIRE dataset mean/std
       - At time t, we don't know future mean/std
       - THIS IS FORWARD-LOOKING BIAS!
       
       FIX: Use expanding window standardization
    
    3. BOCPD UPDATE
       - BOCPD.update(x) is truly online - only uses past data
       - ✓ NO BIAS HERE
    
    4. POSITION TIMING
       - We shift positions by 1 (trade next day)
       - ✓ NO BIAS HERE (if implemented correctly)
    
    5. REGIME MEAN CALCULATION
       - We use observations in current regime
       - Need to verify we only use past observations
       - CHECK REQUIRED
    
    VERDICT: The current implementation HAS forward-looking bias from:
             - Wavelet decomposition (batch processing)
             - Standardization (full-sample mean/std)
    """)
    
    return ["wavelet_decomposition", "standardization"]


def online_wavelet_filter(returns: np.ndarray, wavelet: str = 'db6',
                          level: int = 5, keep_levels: int = 3) -> np.ndarray:
    """
    Online wavelet filtering using expanding window.
    
    At each time t, we only use data up to time t.
    This is computationally expensive but eliminates forward bias.
    """
    n = len(returns)
    filtered = np.zeros(n)
    
    # Minimum samples needed for wavelet decomposition at given level
    min_samples = 2 ** level
    
    for t in range(n):
        if t < min_samples:
            # Not enough data for wavelet - use raw value
            filtered[t] = returns[t]
        else:
            # Use data up to time t only
            window = returns[:t+1]
            
            try:
                # Decompose
                coeffs = pywt.wavedec(window, wavelet, mode='per', level=level)
                
                # Keep only specified detail levels
                modified_coeffs = [coeffs[0]]
                for i in range(1, len(coeffs)):
                    if i <= keep_levels:
                        modified_coeffs.append(coeffs[i])
                    else:
                        modified_coeffs.append(np.zeros_like(coeffs[i]))
                
                # Reconstruct
                reconstructed = pywt.waverec(modified_coeffs, wavelet, mode='per')
                
                # Take the last value (current time)
                filtered[t] = reconstructed[min(t, len(reconstructed)-1)]
                
            except Exception:
                filtered[t] = returns[t]
    
    return filtered


def online_standardize(signal: np.ndarray, min_window: int = 20) -> np.ndarray:
    """
    Online standardization using expanding window.
    
    At each time t, standardize using only data up to time t.
    """
    n = len(signal)
    standardized = np.zeros(n)
    
    for t in range(n):
        if t < min_window:
            # Not enough history - use raw value
            standardized[t] = signal[t]
        else:
            # Use expanding window up to time t
            window = signal[:t+1]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                standardized[t] = (signal[t] - mean) / std
            else:
                standardized[t] = 0
    
    return standardized


def run_unbiased_backtest():
    """Run backtest with no forward-looking bias."""
    
    print("\n" + "=" * 80)
    print("UNBIASED BACKTEST (Online Processing)")
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
    
    n = len(raw_returns)
    n_years = n / 252
    
    print(f"Data: {n} observations")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Method 1: Fully online wavelet (expensive but unbiased)
    # =========================================================================
    print("\n[Method 1: Fully Online Wavelet]")
    print("  Processing (this takes a while)...")
    
    # This is too slow for full dataset, so we'll approximate
    # by using rolling windows instead
    
    # =========================================================================
    # Method 2: Rolling window wavelet (faster, still unbiased)
    # =========================================================================
    print("\n[Method 2: Rolling Window Wavelet]")
    
    window_size = 252  # 1 year of data
    min_samples = 64   # Minimum for wavelet
    
    filtered = np.zeros(n)
    
    for t in range(n):
        start = max(0, t - window_size + 1)
        window = raw_returns[start:t+1]
        
        if len(window) < min_samples:
            filtered[t] = raw_returns[t]
        else:
            try:
                coeffs = pywt.wavedec(window, 'db6', mode='per', level=5)
                
                # Keep 3 detail levels (best from earlier)
                modified_coeffs = [coeffs[0]]
                for i in range(1, len(coeffs)):
                    if i <= 3:
                        modified_coeffs.append(coeffs[i])
                    else:
                        modified_coeffs.append(np.zeros_like(coeffs[i]))
                
                reconstructed = pywt.waverec(modified_coeffs, 'db6', mode='per')
                filtered[t] = reconstructed[-1]  # Last value = current time
                
            except Exception:
                filtered[t] = raw_returns[t]
    
    # Online standardization
    standardized = online_standardize(filtered, min_window=50)
    
    # Run BOCPD (already online by design)
    detector = BOCPD(hazard_rate=1/20, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions = np.zeros(n)
    regime_obs = []
    regime_start = 0
    
    for t in range(n):
        detector.update(standardized[t])
        regime_obs.append(filtered[t])
        
        # Detect regime change
        if t > 0:
            maps = detector.map_run_lengths
            if len(maps) >= 2:
                if maps[-2] > 3 and maps[-1] <= 2 and (t - regime_start) >= 3:
                    regime_start = t
                    regime_obs = [filtered[t]]
        
        # Position based on regime mean (using only past observations)
        if len(regime_obs) >= 2:
            regime_mean = np.mean(regime_obs[-10:])
            positions[t] = 1 if regime_mean > 0 else 0
    
    # Shift positions (trade next day)
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    
    strat_ret = pos_shifted * raw_returns
    
    # Metrics
    strat_total = (1 + strat_ret).prod() - 1
    bench_total = (1 + raw_returns).prod() - 1
    
    strat_vol = strat_ret.std() * np.sqrt(252)
    bench_vol = raw_returns.std() * np.sqrt(252)
    
    strat_annual = (1 + strat_total) ** (252 / n) - 1
    bench_annual = (1 + bench_total) ** (252 / n) - 1
    
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
    
    strat_cum = np.cumprod(1 + strat_ret)
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum / strat_peak - 1).min()
    
    bench_cum = np.cumprod(1 + raw_returns)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=5)
    n_trades = np.sum(np.diff(positions) != 0)
    
    print(f"\n  UNBIASED RESULTS (Rolling Window Wavelet):")
    print(f"    Total Return:     {strat_total:>7.1%} vs {bench_total:>7.1%} (B&H)")
    print(f"    Sharpe Ratio:     {strat_sharpe:>7.2f} vs {bench_sharpe:>7.2f}")
    print(f"    Max Drawdown:     {strat_dd:>7.1%} vs {bench_dd:>7.1%}")
    print(f"    Change Points:    {len(cps):>7} ({len(cps)/n_years:.1f}/year)")
    print(f"    Trades:           {int(n_trades):>7}")
    
    return {
        'strat_total': strat_total,
        'strat_sharpe': strat_sharpe,
        'strat_dd': strat_dd,
        'bench_total': bench_total,
        'bench_sharpe': bench_sharpe,
        'bench_dd': bench_dd,
        'positions': positions,
        'strat_ret': strat_ret,
        'dates': dates
    }


def walk_forward_validation():
    """
    Walk-forward validation:
    - Train on 2021-2023 (tune parameters)
    - Test on 2024-2025 (out-of-sample)
    """
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    # Download SPY with specific dates
    spy = yf.download("SPY", start="2021-01-01", end="2026-01-07", progress=False)
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
    
    # Split: Train (2021-2023), Test (2024-2025)
    train_end = "2023-12-31"
    train_mask = dates <= train_end
    test_mask = dates > train_end
    
    train_returns = raw_returns[train_mask]
    test_returns = raw_returns[test_mask]
    train_dates = dates[train_mask]
    test_dates = dates[test_mask]
    
    print(f"\nTrain: {len(train_returns)} days ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')})")
    print(f"Test:  {len(test_returns)} days ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")
    
    # =========================================================================
    # Step 1: Tune parameters on TRAIN data only
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 1: Parameter Tuning on TRAIN Data (2021-2023)")
    print("-" * 60)
    
    best_sharpe = -np.inf
    best_params = None
    
    window_size = 252
    min_samples = 64
    
    for keep_levels in [2, 3, 4]:
        for hazard in [1/15, 1/20, 1/25, 1/30]:
            # Rolling window wavelet (unbiased)
            n_train = len(train_returns)
            filtered = np.zeros(n_train)
            
            for t in range(n_train):
                start = max(0, t - window_size + 1)
                window = train_returns[start:t+1]
                
                if len(window) < min_samples:
                    filtered[t] = train_returns[t]
                else:
                    try:
                        coeffs = pywt.wavedec(window, 'db6', mode='per', level=5)
                        modified_coeffs = [coeffs[0]]
                        for i in range(1, len(coeffs)):
                            if i <= keep_levels:
                                modified_coeffs.append(coeffs[i])
                            else:
                                modified_coeffs.append(np.zeros_like(coeffs[i]))
                        reconstructed = pywt.waverec(modified_coeffs, 'db6', mode='per')
                        filtered[t] = reconstructed[-1]
                    except:
                        filtered[t] = train_returns[t]
            
            # Online standardization
            standardized = online_standardize(filtered, min_window=50)
            
            # BOCPD
            detector = BOCPD(hazard_rate=hazard, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
            
            positions = np.zeros(n_train)
            regime_obs = []
            regime_start = 0
            
            for t in range(n_train):
                detector.update(standardized[t])
                regime_obs.append(filtered[t])
                
                if t > 0 and len(detector.map_run_lengths) >= 2:
                    if detector.map_run_lengths[-2] > 3 and detector.map_run_lengths[-1] <= 2:
                        if (t - regime_start) >= 3:
                            regime_start = t
                            regime_obs = [filtered[t]]
                
                if len(regime_obs) >= 2:
                    regime_mean = np.mean(regime_obs[-10:])
                    positions[t] = 1 if regime_mean > 0 else 0
            
            pos_shifted = np.roll(positions, 1)
            pos_shifted[0] = 0
            strat_ret = pos_shifted * train_returns
            
            strat_total = (1 + strat_ret).prod() - 1
            strat_vol = strat_ret.std() * np.sqrt(252)
            strat_annual = (1 + strat_total) ** (252 / n_train) - 1
            strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
            
            if strat_sharpe > best_sharpe:
                best_sharpe = strat_sharpe
                best_params = {'keep_levels': keep_levels, 'hazard': hazard}
    
    print(f"\n  Best parameters (train):")
    print(f"    Keep levels: {best_params['keep_levels']}")
    print(f"    Hazard rate: 1/{int(1/best_params['hazard'])}")
    print(f"    Train Sharpe: {best_sharpe:.2f}")
    
    # =========================================================================
    # Step 2: Apply to TEST data (out-of-sample)
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 2: Out-of-Sample Test (2024-2025)")
    print("-" * 60)
    
    keep_levels = best_params['keep_levels']
    hazard = best_params['hazard']
    
    # Need to run on FULL data but only evaluate test period
    # This simulates real-time trading where we have history
    
    n_full = len(raw_returns)
    filtered_full = np.zeros(n_full)
    
    for t in range(n_full):
        start = max(0, t - window_size + 1)
        window = raw_returns[start:t+1]
        
        if len(window) < min_samples:
            filtered_full[t] = raw_returns[t]
        else:
            try:
                coeffs = pywt.wavedec(window, 'db6', mode='per', level=5)
                modified_coeffs = [coeffs[0]]
                for i in range(1, len(coeffs)):
                    if i <= keep_levels:
                        modified_coeffs.append(coeffs[i])
                    else:
                        modified_coeffs.append(np.zeros_like(coeffs[i]))
                reconstructed = pywt.waverec(modified_coeffs, 'db6', mode='per')
                filtered_full[t] = reconstructed[-1]
            except:
                filtered_full[t] = raw_returns[t]
    
    standardized_full = online_standardize(filtered_full, min_window=50)
    
    detector = BOCPD(hazard_rate=hazard, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    
    positions_full = np.zeros(n_full)
    regime_obs = []
    regime_start = 0
    
    for t in range(n_full):
        detector.update(standardized_full[t])
        regime_obs.append(filtered_full[t])
        
        if t > 0 and len(detector.map_run_lengths) >= 2:
            if detector.map_run_lengths[-2] > 3 and detector.map_run_lengths[-1] <= 2:
                if (t - regime_start) >= 3:
                    regime_start = t
                    regime_obs = [filtered_full[t]]
        
        if len(regime_obs) >= 2:
            regime_mean = np.mean(regime_obs[-10:])
            positions_full[t] = 1 if regime_mean > 0 else 0
    
    pos_shifted_full = np.roll(positions_full, 1)
    pos_shifted_full[0] = 0
    strat_ret_full = pos_shifted_full * raw_returns
    
    # Extract test period only
    test_strat_ret = strat_ret_full[test_mask]
    test_positions = pos_shifted_full[test_mask]
    
    # Test metrics
    test_strat_total = (1 + test_strat_ret).prod() - 1
    test_bench_total = (1 + test_returns).prod() - 1
    
    n_test = len(test_returns)
    test_strat_vol = test_strat_ret.std() * np.sqrt(252)
    test_bench_vol = test_returns.std() * np.sqrt(252)
    
    test_strat_annual = (1 + test_strat_total) ** (252 / n_test) - 1
    test_bench_annual = (1 + test_bench_total) ** (252 / n_test) - 1
    
    test_strat_sharpe = test_strat_annual / test_strat_vol if test_strat_vol > 0 else 0
    test_bench_sharpe = test_bench_annual / test_bench_vol if test_bench_vol > 0 else 0
    
    test_strat_cum = np.cumprod(1 + test_strat_ret)
    test_strat_peak = np.maximum.accumulate(test_strat_cum)
    test_strat_dd = (test_strat_cum / test_strat_peak - 1).min()
    
    test_bench_cum = np.cumprod(1 + test_returns)
    test_bench_peak = np.maximum.accumulate(test_bench_cum)
    test_bench_dd = (test_bench_cum / test_bench_peak - 1).min()
    
    n_trades = np.sum(np.diff(test_positions) != 0)
    
    print(f"\n  OUT-OF-SAMPLE RESULTS (2024-2025):")
    print(f"    Total Return:     {test_strat_total:>7.1%} vs {test_bench_total:>7.1%} (B&H)")
    print(f"    Sharpe Ratio:     {test_strat_sharpe:>7.2f} vs {test_bench_sharpe:>7.2f}")
    print(f"    Max Drawdown:     {test_strat_dd:>7.1%} vs {test_bench_dd:>7.1%}")
    print(f"    Trades:           {int(n_trades):>7}")
    
    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: In-Sample vs Out-of-Sample")
    print("=" * 80)
    
    # Calculate train period metrics
    train_strat_ret = strat_ret_full[train_mask]
    train_strat_total = (1 + train_strat_ret).prod() - 1
    train_bench_total = (1 + train_returns).prod() - 1
    
    n_train = len(train_returns)
    train_strat_vol = train_strat_ret.std() * np.sqrt(252)
    train_strat_annual = (1 + train_strat_total) ** (252 / n_train) - 1
    train_strat_sharpe = train_strat_annual / train_strat_vol if train_strat_vol > 0 else 0
    
    train_bench_vol = train_returns.std() * np.sqrt(252)
    train_bench_annual = (1 + train_bench_total) ** (252 / n_train) - 1
    train_bench_sharpe = train_bench_annual / train_bench_vol if train_bench_vol > 0 else 0
    
    print(f"\n{'Period':<20} {'Strategy':>12} {'B&H':>12} {'Strat Sharpe':>14} {'B&H Sharpe':>12}")
    print("-" * 75)
    print(f"{'Train (2021-2023)':<20} {train_strat_total:>11.1%} {train_bench_total:>11.1%} "
          f"{train_strat_sharpe:>14.2f} {train_bench_sharpe:>12.2f}")
    print(f"{'Test (2024-2025)':<20} {test_strat_total:>11.1%} {test_bench_total:>11.1%} "
          f"{test_strat_sharpe:>14.2f} {test_bench_sharpe:>12.2f}")
    
    # Yearly breakdown
    print("\n  Yearly Performance:")
    strat_ret_series = pd.Series(strat_ret_full, index=dates)
    bench_ret_series = pd.Series(raw_returns, index=dates)
    
    print(f"\n  {'Year':>6} {'Strategy':>12} {'B&H':>12} {'Outperform':>12} {'Note':>15}")
    print("  " + "-" * 60)
    
    for year in sorted(strat_ret_series.index.year.unique()):
        strat_yr = strat_ret_series[strat_ret_series.index.year == year]
        bench_yr = bench_ret_series[bench_ret_series.index.year == year]
        
        strat_ret_yr = (1 + strat_yr).prod() - 1
        bench_ret_yr = (1 + bench_yr).prod() - 1
        outperf = strat_ret_yr - bench_ret_yr
        
        note = "(train)" if year <= 2023 else "(TEST)"
        print(f"  {year:>6} {strat_ret_yr:>11.1%} {bench_ret_yr:>11.1%} {outperf:>11.1%} {note:>15}")


def main():
    """Run full validation."""
    
    # Step 1: Audit for bias
    biases = audit_forward_bias()
    
    # Step 2: Run unbiased backtest
    results = run_unbiased_backtest()
    
    # Step 3: Walk-forward validation
    walk_forward_validation()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    The original backtest had forward-looking bias from:
    1. Batch wavelet decomposition (used entire array)
    2. Full-sample standardization (used future mean/std)
    
    After fixing these issues with:
    1. Rolling window wavelet (252-day lookback)
    2. Expanding window standardization
    
    The strategy's performance is more realistic but still needs
    to be evaluated on the out-of-sample period.
    """)


if __name__ == "__main__":
    main()
