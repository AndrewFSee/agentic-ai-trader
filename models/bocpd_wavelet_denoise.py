"""
Wavelet Denoising for BOCPD

Following Jansen's "Machine Learning for Algorithmic Trading":
- Use wavelet decomposition to denoise returns
- Apply thresholding to remove noise coefficients
- Reconstruct signal using inverse wavelet transform
- Feed denoised signal to BOCPD for change point detection
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("pywavelets not installed. Run: pip install PyWavelets")


def wavelet_denoise(signal: np.ndarray, wavelet: str = 'db6', 
                    threshold_scale: float = 0.5, mode: str = 'soft') -> np.ndarray:
    """
    Denoise signal using wavelet thresholding.
    
    Args:
        signal: Input signal to denoise
        wavelet: Wavelet type (e.g., 'db6', 'haar', 'sym8')
        threshold_scale: Scale factor for threshold (higher = more smoothing)
        mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised signal
    """
    # Decompose signal
    coefficients = pywt.wavedec(signal, wavelet, mode='per')
    
    # Calculate threshold
    threshold = threshold_scale * np.abs(signal).max()
    
    # Apply thresholding to detail coefficients (not approximation)
    coefficients[1:] = [pywt.threshold(c, value=threshold, mode=mode) 
                        for c in coefficients[1:]]
    
    # Reconstruct signal
    reconstructed = pywt.waverec(coefficients, wavelet, mode='per')
    
    # Handle length mismatch (waverec can add 1 sample)
    if len(reconstructed) > len(signal):
        reconstructed = reconstructed[:len(signal)]
    
    return reconstructed


def test_wavelet_denoising():
    """Test wavelet denoising with different parameters."""
    
    print("=" * 80)
    print("WAVELET DENOISING FOR BOCPD")
    print("=" * 80)
    
    if not HAS_PYWT:
        return
    
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
    # Show available wavelets
    # =========================================================================
    print("\n[Available Wavelet Families]")
    print(pywt.families(short=False))
    
    # =========================================================================
    # Test different wavelets and thresholds
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTING WAVELET PARAMETERS")
    print("=" * 80)
    
    wavelets = ['db4', 'db6', 'db8', 'sym6', 'sym8', 'coif3', 'haar']
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    results = []
    
    for wavelet in wavelets:
        for thresh in thresholds:
            try:
                # Denoise returns
                denoised = wavelet_denoise(raw_returns, wavelet=wavelet, 
                                          threshold_scale=thresh)
                
                # Standardize for BOCPD
                denoised_std = (denoised - np.mean(denoised)) / np.std(denoised)
                
                # Run BOCPD
                detector = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, 
                                alpha0=2.0, beta0=1.0)
                for x in denoised_std:
                    detector.update(x)
                
                cps = detector.detect_change_points(method='map_drop', min_spacing=5)
                cps_per_year = len(cps) / n_years
                
                # Calculate signal-to-noise improvement
                raw_std = np.std(raw_returns)
                denoised_std_val = np.std(denoised)
                noise_reduction = 1 - (denoised_std_val / raw_std)
                
                results.append({
                    'wavelet': wavelet,
                    'threshold': thresh,
                    'n_cps': len(cps),
                    'cps_per_year': cps_per_year,
                    'noise_reduction': noise_reduction,
                    'denoised': denoised,
                    'detector': detector
                })
                
            except Exception as e:
                print(f"  {wavelet}, thresh={thresh}: ERROR - {e}")
    
    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'Wavelet':<10} {'Thresh':>8} {'CPs':>6} {'CPs/Yr':>10} {'Noise Red':>12}")
    print("-" * 50)
    
    for r in sorted(results, key=lambda x: x['cps_per_year'], reverse=True):
        print(f"{r['wavelet']:<10} {r['threshold']:>8.1f} {r['n_cps']:>6} "
              f"{r['cps_per_year']:>10.1f} {r['noise_reduction']:>11.1%}")
    
    # =========================================================================
    # Find candidates with 10-30 CPs/year
    # =========================================================================
    print("\n" + "=" * 80)
    print("BEST CANDIDATES (Target: 10-30 CPs/year)")
    print("=" * 80)
    
    good_results = [r for r in results if 10 <= r['cps_per_year'] <= 35]
    good_results.sort(key=lambda x: abs(x['cps_per_year'] - 20))
    
    print(f"\n{'Wavelet':<10} {'Thresh':>8} {'CPs/Yr':>10} {'Noise Red':>12}")
    print("-" * 45)
    for r in good_results[:10]:
        print(f"{r['wavelet']:<10} {r['threshold']:>8.1f} "
              f"{r['cps_per_year']:>10.1f} {r['noise_reduction']:>11.1%}")
    
    # =========================================================================
    # Compare raw vs denoised BOCPD
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Raw Returns vs Wavelet-Denoised")
    print("=" * 80)
    
    # Raw returns BOCPD
    raw_std = (raw_returns - np.mean(raw_returns)) / np.std(raw_returns)
    detector_raw = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in raw_std:
        detector_raw.update(x)
    cps_raw = detector_raw.detect_change_points(method='map_drop', min_spacing=5)
    
    print(f"\nRaw Returns:         {len(cps_raw)} CPs ({len(cps_raw)/n_years:.1f}/year)")
    
    # Best wavelet result
    if good_results:
        best = good_results[0]
        print(f"Best Wavelet ({best['wavelet']}, {best['threshold']}): "
              f"{best['n_cps']} CPs ({best['cps_per_year']:.1f}/year)")
    
    # =========================================================================
    # Backtest top 3 wavelet configurations
    # =========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST: Top Wavelet Configurations")
    print("=" * 80)
    
    for i, r in enumerate(good_results[:3]):
        backtest_wavelet_strategy(
            raw_returns, 
            r['denoised'],
            r['detector'],
            f"{r['wavelet']}-{r['threshold']}"
        )
    
    # =========================================================================
    # Also test EMA-smoothed version for comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("BASELINE: EMA-10 Smoothing (no wavelets)")
    print("-" * 60)
    
    ema_signal = pd.Series(raw_returns).ewm(span=10).mean().values
    ema_std = (ema_signal - np.mean(ema_signal)) / np.std(ema_signal)
    
    detector_ema = BOCPD(hazard_rate=1/30, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0)
    for x in ema_std:
        detector_ema.update(x)
    
    backtest_wavelet_strategy(raw_returns, ema_signal, detector_ema, "EMA-10")
    
    return results


def backtest_wavelet_strategy(raw_returns: np.ndarray, signal: np.ndarray,
                              detector: BOCPD, name: str):
    """Backtest a strategy using wavelet-denoised signals."""
    
    n = len(raw_returns)
    positions = np.zeros(n)
    
    regime_obs = []
    regime_start = 0
    
    for t in range(n):
        regime_obs.append(signal[t])
        
        # Check for regime change
        if t > 0 and len(detector.map_run_lengths) > t:
            maps = detector.map_run_lengths
            if t < len(maps) and t > 0:
                if maps[t-1] > 3 and maps[t] <= 2 and (t - regime_start) >= 3:
                    regime_start = t
                    regime_obs = [signal[t]]
        
        # Position based on regime mean
        if len(regime_obs) >= 2:
            regime_mean = np.mean(regime_obs[-10:])
            positions[t] = 1 if regime_mean > 0 else 0
    
    # Shift positions
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
    
    n_trades = np.sum(np.diff(positions) != 0)
    time_in_market = (pos_shifted > 0).sum() / n
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=5)
    
    print(f"\n  {name}:")
    print(f"    Change Points:    {len(cps):>5} ({len(cps)/(n/252):.1f}/year)")
    print(f"    Total Return:     {strat_total:>6.1%} vs {bench_total:>6.1%} (B&H)")
    print(f"    Sharpe Ratio:     {strat_sharpe:>6.2f} vs {bench_sharpe:>6.2f}")
    print(f"    Max Drawdown:     {strat_dd:>6.1%} vs {bench_dd:>6.1%}")
    print(f"    Time in Market:   {time_in_market:>6.1%}")
    print(f"    Trades:           {int(n_trades):>5}")


def visualize_wavelets():
    """Visualize different wavelet denoising results."""
    
    print("\n" + "=" * 80)
    print("WAVELET VISUALIZATION (Signal Comparison)")
    print("=" * 80)
    
    if not HAS_PYWT:
        return
    
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    # Download SPY
    spy = yf.download("SPY", period="1y", progress=False)
    if spy.empty:
        print("Failed to download")
        return
    
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    returns = close.pct_change().dropna().values
    
    print(f"\nComparing denoising methods on last 252 trading days:")
    print(f"\n{'Method':<25} {'Std Dev':>12} {'Noise Red':>12} {'Max':>10} {'Min':>10}")
    print("-" * 70)
    
    print(f"{'Raw Returns':<25} {np.std(returns):>12.5f} {'--':>12} "
          f"{np.max(returns):>10.4f} {np.min(returns):>10.4f}")
    
    for wavelet in ['db4', 'db6', 'sym6', 'haar']:
        for thresh in [0.3, 0.5, 0.7]:
            denoised = wavelet_denoise(returns, wavelet=wavelet, threshold_scale=thresh)
            noise_red = 1 - (np.std(denoised) / np.std(returns))
            name = f"{wavelet} (t={thresh})"
            print(f"{name:<25} {np.std(denoised):>12.5f} {noise_red:>11.1%} "
                  f"{np.max(denoised):>10.4f} {np.min(denoised):>10.4f}")


if __name__ == "__main__":
    test_wavelet_denoising()
    visualize_wavelets()
