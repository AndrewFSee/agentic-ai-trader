"""
Advanced Wavelet Analysis for BOCPD

Try different approaches:
1. Very low threshold (minimal denoising)
2. Multi-resolution: use wavelet detail coefficients for regime detection
3. Wavelet decomposition levels separately
4. Hybrid: combine raw with denoised
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD
import pywt


def test_advanced_wavelets():
    """Test advanced wavelet approaches."""
    
    print("=" * 80)
    print("ADVANCED WAVELET ANALYSIS FOR BOCPD")
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
    n_years = len(returns) / 252
    
    # =========================================================================
    # Approach 1: Very low thresholds (5-10% of max)
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 1: Minimal Denoising (Low Thresholds)")
    print("=" * 80)
    
    results = []
    
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for wavelet in ['db4', 'db6', 'sym6']:
            coefficients = pywt.wavedec(raw_returns, wavelet, mode='per')
            threshold = thresh * np.abs(raw_returns).max()
            coefficients[1:] = [pywt.threshold(c, value=threshold, mode='soft') 
                               for c in coefficients[1:]]
            denoised = pywt.waverec(coefficients, wavelet, mode='per')
            if len(denoised) > len(raw_returns):
                denoised = denoised[:len(raw_returns)]
            
            denoised_std = (denoised - np.mean(denoised)) / np.std(denoised)
            
            detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                            alpha0=2.0, beta0=1.0)
            for x in denoised_std:
                detector.update(x)
            
            cps = detector.detect_change_points(method='map_drop', min_spacing=5)
            noise_red = 1 - (np.std(denoised) / np.std(raw_returns))
            
            results.append({
                'method': f'{wavelet}-{thresh}',
                'wavelet': wavelet,
                'threshold': thresh,
                'n_cps': len(cps),
                'cps_per_year': len(cps) / n_years,
                'noise_reduction': noise_red,
                'denoised': denoised,
                'detector': detector
            })
    
    print(f"\n{'Method':<15} {'CPs':>6} {'CPs/Yr':>10} {'Noise Red':>12}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: x['cps_per_year'], reverse=True):
        print(f"{r['method']:<15} {r['n_cps']:>6} {r['cps_per_year']:>10.1f} "
              f"{r['noise_reduction']:>11.1%}")
    
    # =========================================================================
    # Approach 2: Multi-Resolution - Use Detail Coefficients
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 2: Multi-Resolution (Detail Coefficients)")
    print("=" * 80)
    
    wavelet = 'db6'
    
    # Decompose into multiple levels
    coeffs = pywt.wavedec(raw_returns, wavelet, mode='per', level=5)
    
    print(f"\nWavelet decomposition levels ({wavelet}):")
    print(f"  cA5 (Approximation): {len(coeffs[0])} coefficients")
    for i, d in enumerate(coeffs[1:], 1):
        level = len(coeffs) - i
        print(f"  cD{level} (Detail):      {len(d)} coefficients")
    
    # Reconstruct at different detail levels
    print("\n  Testing signals from different levels:")
    
    level_results = []
    
    for keep_levels in [1, 2, 3, 4, 5]:  # How many detail levels to keep
        # Zero out detail coefficients above keep_levels
        modified_coeffs = [coeffs[0]]  # Keep approximation
        for i in range(1, len(coeffs)):
            level_from_top = i
            if level_from_top <= keep_levels:
                modified_coeffs.append(coeffs[i])  # Keep this detail
            else:
                modified_coeffs.append(np.zeros_like(coeffs[i]))  # Zero out
        
        reconstructed = pywt.waverec(modified_coeffs, wavelet, mode='per')
        if len(reconstructed) > len(raw_returns):
            reconstructed = reconstructed[:len(raw_returns)]
        
        rec_std = (reconstructed - np.mean(reconstructed)) / np.std(reconstructed)
        
        detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                        alpha0=2.0, beta0=1.0)
        for x in rec_std:
            detector.update(x)
        
        cps = detector.detect_change_points(method='map_drop', min_spacing=5)
        
        level_results.append({
            'levels': keep_levels,
            'n_cps': len(cps),
            'cps_per_year': len(cps) / n_years,
            'signal': reconstructed,
            'detector': detector
        })
        
        print(f"    Keep {keep_levels} detail levels: {len(cps)} CPs ({len(cps)/n_years:.1f}/year)")
    
    # =========================================================================
    # Approach 3: Use only high-frequency detail coefficients
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 3: High-Frequency Details Only (Noise = Signal)")
    print("=" * 80)
    
    print("\n  Testing individual detail levels:")
    
    for detail_idx in range(1, min(5, len(coeffs))):
        # Reconstruct using only one detail level
        modified_coeffs = [np.zeros_like(coeffs[0])]  # Zero approximation
        for i in range(1, len(coeffs)):
            if i == detail_idx:
                modified_coeffs.append(coeffs[i])
            else:
                modified_coeffs.append(np.zeros_like(coeffs[i]))
        
        reconstructed = pywt.waverec(modified_coeffs, wavelet, mode='per')
        if len(reconstructed) > len(raw_returns):
            reconstructed = reconstructed[:len(raw_returns)]
        
        # Skip if all zeros
        if np.std(reconstructed) < 1e-10:
            continue
        
        rec_std = (reconstructed - np.mean(reconstructed)) / np.std(reconstructed)
        
        detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                        alpha0=2.0, beta0=1.0)
        for x in rec_std:
            detector.update(x)
        
        cps = detector.detect_change_points(method='map_drop', min_spacing=5)
        level = len(coeffs) - detail_idx
        print(f"    Detail D{level} only: {len(cps)} CPs ({len(cps)/n_years:.1f}/year)")
    
    # =========================================================================
    # Approach 4: Stationary Wavelet Transform (non-decimated)
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 4: Stationary Wavelet Transform (SWT)")
    print("=" * 80)
    
    # SWT requires length to be multiple of 2^level
    level = 4
    padded_len = 2**level * (len(raw_returns) // (2**level) + 1)
    padded = np.pad(raw_returns, (0, padded_len - len(raw_returns)), mode='constant')
    
    # SWT decomposition
    swt_coeffs = pywt.swt(padded, 'db4', level=level)
    
    print(f"\n  SWT decomposition (level {level}):")
    
    for i, (cA, cD) in enumerate(swt_coeffs):
        # Use the detail coefficients at this level
        detail_signal = cD[:len(raw_returns)]
        
        if np.std(detail_signal) < 1e-10:
            continue
        
        detail_std = (detail_signal - np.mean(detail_signal)) / np.std(detail_signal)
        
        detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                        alpha0=2.0, beta0=1.0)
        for x in detail_std:
            detector.update(x)
        
        cps = detector.detect_change_points(method='map_drop', min_spacing=5)
        print(f"    SWT Detail level {i+1}: {len(cps)} CPs ({len(cps)/n_years:.1f}/year)")
    
    # =========================================================================
    # Approach 5: Hybrid Signal (raw + wavelet-smoothed)
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 5: Hybrid Signals")
    print("=" * 80)
    
    # Light denoising
    coefficients = pywt.wavedec(raw_returns, 'db4', mode='per')
    threshold = 0.05 * np.abs(raw_returns).max()
    coefficients[1:] = [pywt.threshold(c, value=threshold, mode='soft') 
                       for c in coefficients[1:]]
    light_denoised = pywt.waverec(coefficients, 'db4', mode='per')
    if len(light_denoised) > len(raw_returns):
        light_denoised = light_denoised[:len(raw_returns)]
    
    print("\n  Testing hybrid approaches:")
    
    # Different weighted combinations
    for alpha in [0.3, 0.5, 0.7]:
        hybrid = alpha * raw_returns + (1 - alpha) * light_denoised
        hybrid_std = (hybrid - np.mean(hybrid)) / np.std(hybrid)
        
        detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                        alpha0=2.0, beta0=1.0)
        for x in hybrid_std:
            detector.update(x)
        
        cps = detector.detect_change_points(method='map_drop', min_spacing=5)
        print(f"    {alpha:.0%} raw + {1-alpha:.0%} denoised: {len(cps)} CPs ({len(cps)/n_years:.1f}/year)")
    
    # =========================================================================
    # Approach 6: Wavelet on EMA-smoothed returns
    # =========================================================================
    print("\n" + "=" * 80)
    print("APPROACH 6: Wavelet on EMA-Smoothed Returns")
    print("=" * 80)
    
    for ema_span in [5, 10]:
        ema_returns = pd.Series(raw_returns).ewm(span=ema_span).mean().values
        
        for thresh in [0.1, 0.3]:
            coefficients = pywt.wavedec(ema_returns, 'db4', mode='per')
            threshold = thresh * np.abs(ema_returns).max()
            coefficients[1:] = [pywt.threshold(c, value=threshold, mode='soft') 
                               for c in coefficients[1:]]
            denoised = pywt.waverec(coefficients, 'db4', mode='per')
            if len(denoised) > len(raw_returns):
                denoised = denoised[:len(raw_returns)]
            
            denoised_std = (denoised - np.mean(denoised)) / np.std(denoised)
            
            detector = BOCPD(hazard_rate=1/25, mu0=0.0, kappa0=0.1, 
                            alpha0=2.0, beta0=1.0)
            for x in denoised_std:
                detector.update(x)
            
            cps = detector.detect_change_points(method='map_drop', min_spacing=5)
            print(f"    EMA-{ema_span} + wavelet(t={thresh}): {len(cps)} CPs ({len(cps)/n_years:.1f}/year)")
    
    # =========================================================================
    # Best approach: Run full backtest
    # =========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST: Best Wavelet Configurations")
    print("=" * 80)
    
    # Select promising configs from results
    good_configs = [r for r in results if 8 <= r['cps_per_year'] <= 40]
    good_configs.sort(key=lambda x: x['cps_per_year'], reverse=True)
    
    if not good_configs:
        # Fall back to level results
        good_configs = [r for r in level_results if 8 <= r['cps_per_year'] <= 40]
    
    if not good_configs:
        # Use what we have
        good_configs = sorted(results, key=lambda x: abs(x['cps_per_year'] - 15))[:3]
    
    for config in good_configs[:3]:
        if 'denoised' in config:
            backtest_strategy(raw_returns, config['denoised'], 
                            config['detector'], config['method'])
        elif 'signal' in config:
            backtest_strategy(raw_returns, config['signal'],
                            config['detector'], f"Level-{config['levels']}")


def backtest_strategy(raw_returns: np.ndarray, signal: np.ndarray,
                      detector: BOCPD, name: str):
    """Backtest a momentum strategy."""
    
    n = len(raw_returns)
    positions = np.zeros(n)
    
    regime_obs = []
    regime_start = 0
    
    for t in range(n):
        regime_obs.append(signal[t])
        
        if t > 0 and len(detector.map_run_lengths) > t:
            maps = detector.map_run_lengths
            if t < len(maps) and t > 0:
                if maps[t-1] > 3 and maps[t] <= 2 and (t - regime_start) >= 3:
                    regime_start = t
                    regime_obs = [signal[t]]
        
        if len(regime_obs) >= 2:
            regime_mean = np.mean(regime_obs[-10:])
            positions[t] = 1 if regime_mean > 0 else 0
    
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    
    strat_ret = pos_shifted * raw_returns
    
    strat_total = (1 + strat_ret).prod() - 1
    bench_total = (1 + raw_returns).prod() - 1
    
    strat_vol = strat_ret.std() * np.sqrt(252)
    strat_annual = (1 + strat_total) ** (252 / n) - 1
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    
    bench_vol = raw_returns.std() * np.sqrt(252)
    bench_annual = (1 + bench_total) ** (252 / n) - 1
    bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
    
    strat_cum = np.cumprod(1 + strat_ret)
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd = (strat_cum / strat_peak - 1).min()
    
    bench_cum = np.cumprod(1 + raw_returns)
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum / bench_peak - 1).min()
    
    n_trades = np.sum(np.diff(positions) != 0)
    cps = detector.detect_change_points(method='map_drop', min_spacing=5)
    
    print(f"\n  {name}:")
    print(f"    CPs: {len(cps)} ({len(cps)/(n/252):.1f}/year)")
    print(f"    Return:  {strat_total:>6.1%} vs {bench_total:>6.1%} | "
          f"Sharpe: {strat_sharpe:.2f} vs {bench_sharpe:.2f} | "
          f"MaxDD: {strat_dd:>6.1%} vs {bench_dd:>6.1%}")


if __name__ == "__main__":
    test_advanced_wavelets()
