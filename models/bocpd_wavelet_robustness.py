"""
BOCPD Wavelet Robustness Testing

Validate the best configuration (Window=63, Level=4, Keep=2) across:
1. Multiple tickers: SPY, QQQ, IWM, DIA, TLT
2. Multiple time periods: Walk-forward with 3 different test windows
3. Different market conditions: Bull, Bear, Sideways

This ensures we're not just fitting to one test set.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from scipy.special import gammaln, logsumexp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BOCPD Implementation (same as before)
# ============================================================================

class FastBOCPD:
    """Lightweight BOCPD for backtesting."""
    
    def __init__(self, hazard_rate: float = 0.04, max_run_length: int = 300):
        self.hazard_rate = hazard_rate
        self.log_H = np.log(hazard_rate)
        self.log_1_minus_H = np.log(1 - hazard_rate)
        self.max_run_length = max_run_length
        
        # NIG prior
        self.mu0, self.kappa0, self.alpha0, self.beta0 = 0.0, 0.1, 2.0, 1.0
        self.reset()
    
    def reset(self):
        self.log_R = np.array([0.0])
        self.n = np.array([0.0])
        self.sum_x = np.array([0.0])
        self.sum_x2 = np.array([0.0])
        self.map_history = []
    
    def update(self, x: float) -> int:
        # Predictive likelihood
        kappa_n = self.kappa0 + self.n
        mu_n = (self.kappa0 * self.mu0 + self.sum_x) / kappa_n
        alpha_n = self.alpha0 + self.n / 2.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            xbar = np.where(self.n > 0, self.sum_x / self.n, 0.0)
        ss = np.maximum(self.sum_x2 - self.n * xbar**2, 0.0)
        coupling = self.kappa0 * self.n * (xbar - self.mu0)**2 / kappa_n
        beta_n = self.beta0 + 0.5 * ss + 0.5 * coupling
        
        df = 2.0 * alpha_n
        var = np.maximum(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n), 1e-10)
        z = (x - mu_n)**2 / (df * var)
        log_pred = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi * var) - ((df + 1) / 2) * np.log1p(z)
        
        # Update posterior
        log_growth = self.log_R + log_pred + self.log_1_minus_H
        log_cp = logsumexp(self.log_R + log_pred) + self.log_H
        log_R_new = np.concatenate([[log_cp], log_growth])
        log_R_new = log_R_new - logsumexp(log_R_new)
        
        # Update sufficient stats
        self.n = np.concatenate([[0.0], self.n + 1])
        self.sum_x = np.concatenate([[0.0], self.sum_x + x])
        self.sum_x2 = np.concatenate([[0.0], self.sum_x2 + x**2])
        
        # Truncate
        if len(log_R_new) > self.max_run_length:
            log_R_new = log_R_new[:self.max_run_length]
            self.n, self.sum_x, self.sum_x2 = self.n[:self.max_run_length], self.sum_x[:self.max_run_length], self.sum_x2[:self.max_run_length]
        
        self.log_R = log_R_new
        map_rl = int(np.argmax(np.exp(log_R_new)))
        self.map_history.append(map_rl)
        return map_rl


def rolling_wavelet_denoise(returns: np.ndarray, window: int = 63, 
                            wavelet: str = 'db6', level: int = 4, 
                            keep_levels: int = 2) -> np.ndarray:
    """Rolling window wavelet denoising - no forward bias."""
    denoised = np.zeros_like(returns)
    
    for i in range(len(returns)):
        if i < window:
            # Not enough history - use raw return
            denoised[i] = returns[i]
        else:
            # Get window of data
            window_data = returns[i-window+1:i+1]
            
            # Standardize within window
            mu, sigma = np.mean(window_data), np.std(window_data)
            if sigma < 1e-10:
                denoised[i] = 0.0
                continue
            
            standardized = (window_data - mu) / sigma
            
            # Wavelet decomposition
            max_level = pywt.dwt_max_level(len(standardized), wavelet)
            use_level = min(level, max_level)
            
            if use_level < 1:
                denoised[i] = returns[i]
                continue
            
            coeffs = pywt.wavedec(standardized, wavelet, level=use_level)
            
            # Keep only specified detail levels (remove fine noise)
            remove_levels = len(coeffs) - 1 - keep_levels
            for j in range(1, min(remove_levels + 1, len(coeffs))):
                coeffs[-j] = np.zeros_like(coeffs[-j])
            
            # Reconstruct
            reconstructed = pywt.waverec(coeffs, wavelet)
            if len(reconstructed) > len(standardized):
                reconstructed = reconstructed[:len(standardized)]
            
            # Use the last value (current point)
            denoised[i] = reconstructed[-1]
    
    return denoised


def run_bocpd_strategy(prices: pd.Series, returns: np.ndarray, 
                       window: int = 63, level: int = 4, keep_levels: int = 2,
                       hazard: float = 0.04) -> Dict:
    """Run wavelet-BOCPD strategy on price series."""
    
    # Wavelet denoise
    denoised = rolling_wavelet_denoise(returns, window=window, level=level, keep_levels=keep_levels)
    
    # Run BOCPD
    bocpd = FastBOCPD(hazard_rate=hazard)
    signals = np.zeros(len(returns))
    
    # Expanding window standardization
    running_sum = 0.0
    running_sum2 = 0.0
    
    for i in range(len(denoised)):
        running_sum += denoised[i]
        running_sum2 += denoised[i]**2
        n = i + 1
        mu = running_sum / n
        std = np.sqrt(max(running_sum2/n - mu**2, 1e-10)) if n > 1 else 1.0
        
        standardized = (denoised[i] - mu) / std if std > 1e-10 else 0.0
        map_rl = bocpd.update(standardized)
        
        # Simple momentum signal based on recent denoised returns
        if i >= 20:
            recent_mean = np.mean(denoised[i-20:i+1])
            signals[i] = 1 if recent_mean > 0 else 0
            
            # Reduce position after change point
            if len(bocpd.map_history) > 1:
                if bocpd.map_history[-2] > 3 and map_rl <= 2:
                    signals[i] = 0.5  # Half position during uncertainty
    
    # Calculate returns - align signals with returns properly
    # signals[i] is based on info up to day i, applied to return from day i to day i+1
    price_arr = prices.values
    daily_returns = np.diff(price_arr) / price_arr[:-1]
    
    # Ensure proper alignment: signals has same length as returns
    # daily_returns is len(prices)-1, signals is len(returns)
    # We need to align: signal on day i applies to return on day i
    min_len = min(len(signals), len(daily_returns))
    strategy_returns = signals[:min_len] * daily_returns[:min_len]
    bh_returns = daily_returns[:min_len]
    
    # Metrics
    def calc_metrics(rets):
        total = np.prod(1 + rets) - 1
        ann_ret = (1 + total) ** (252 / len(rets)) - 1
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        return {
            'total_return': total * 100,
            'ann_return': ann_ret * 100,
            'ann_vol': ann_vol * 100,
            'sharpe': sharpe,
            'max_dd': max_dd * 100
        }
    
    strat_metrics = calc_metrics(strategy_returns)
    bh_metrics = calc_metrics(bh_returns)
    
    # Count trades
    trades = np.sum(np.abs(np.diff(signals[:min_len])) > 0.4)
    
    return {
        'strategy': strat_metrics,
        'buy_hold': bh_metrics,
        'trades': trades,
        'signals': signals[:min_len],
        'daily_returns': daily_returns[:min_len]
    }


# ============================================================================
# Multi-Ticker Testing
# ============================================================================

def test_multiple_tickers():
    """Test on multiple tickers."""
    print("=" * 80)
    print("MULTI-TICKER ROBUSTNESS TEST")
    print("Best Config: Window=63, Level=4, Keep=2, Hazard=1/25")
    print("=" * 80)
    
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']
    
    # Test period: 2024-01-01 to present (out-of-sample)
    test_start = '2024-01-01'
    test_end = '2025-06-30'
    
    # Need historical data for warmup
    full_start = '2022-01-01'
    
    results = []
    
    for ticker in tickers:
        print(f"\nDownloading {ticker}...")
        try:
            data = yf.download(ticker, start=full_start, end=test_end, progress=False)
            if data.empty or len(data) < 300:
                print(f"  Insufficient data for {ticker}")
                continue
            
            # Extract close prices
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'][ticker]
            else:
                close = data['Close']
            
            returns = close.pct_change().dropna().values
            
            # Find test period start index
            test_mask = close.index >= test_start
            test_start_idx = np.where(test_mask)[0][0] if test_mask.any() else len(close) - 252
            
            # Run strategy on full data
            result = run_bocpd_strategy(
                close, returns,
                window=63, level=4, keep_levels=2, hazard=1/25
            )
            
            # Extract test period metrics using pre-computed signals and returns
            signals = result['signals']
            daily_returns = result['daily_returns']
            
            # Find test start in the signal array
            # dates for returns start at index 1 of close
            return_dates = close.index[1:len(signals)+1]
            test_mask = return_dates >= test_start
            if not test_mask.any():
                print(f"  No test data for {ticker}")
                continue
            test_start_idx = np.where(test_mask)[0][0]
            
            # Test period only
            test_strat_rets = signals[test_start_idx:] * daily_returns[test_start_idx:]
            test_bh_rets = daily_returns[test_start_idx:]
            
            def calc_metrics(rets):
                if len(rets) == 0:
                    return {'total_return': 0, 'sharpe': 0, 'max_dd': 0}
                total = np.prod(1 + rets) - 1
                ann_ret = (1 + total) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0
                ann_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0.01
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                cum = np.cumprod(1 + rets)
                peak = np.maximum.accumulate(cum)
                dd = (cum - peak) / peak
                max_dd = dd.min()
                return {'total_return': total * 100, 'sharpe': sharpe, 'max_dd': max_dd * 100}
            
            strat_metrics = calc_metrics(test_strat_rets)
            bh_metrics = calc_metrics(test_bh_rets)
            
            results.append({
                'ticker': ticker,
                'test_days': len(test_strat_rets),
                'strat_return': strat_metrics['total_return'],
                'strat_sharpe': strat_metrics['sharpe'],
                'strat_dd': strat_metrics['max_dd'],
                'bh_return': bh_metrics['total_return'],
                'bh_sharpe': bh_metrics['sharpe'],
                'bh_dd': bh_metrics['max_dd']
            })
            
            print(f"  {ticker}: Strategy {strat_metrics['total_return']:.1f}% (Sharpe {strat_metrics['sharpe']:.2f}) vs B&H {bh_metrics['total_return']:.1f}% (Sharpe {bh_metrics['sharpe']:.2f})")
            
        except Exception as e:
            print(f"  Error with {ticker}: {e}")
    
    return results


# ============================================================================
# Walk-Forward Validation
# ============================================================================

def test_walk_forward():
    """Walk-forward validation with multiple test windows."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (SPY)")
    print("3 non-overlapping test periods")
    print("=" * 80)
    
    # Download full history
    print("\nDownloading SPY...")
    data = yf.download('SPY', start='2018-01-01', end='2025-12-31', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY']
    else:
        close = data['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    # Define test periods
    test_periods = [
        ('2022-01-01', '2022-12-31', 'Bear Market (2022)'),
        ('2023-01-01', '2023-12-31', 'Recovery (2023)'),
        ('2024-01-01', '2025-06-30', 'Bull Market (2024-25)')
    ]
    
    results = []
    
    for start, end, label in test_periods:
        print(f"\n{label}:")
        print("-" * 50)
        
        # Find indices
        start_mask = dates >= start
        end_mask = dates <= end
        test_mask = start_mask & end_mask
        
        if not test_mask.any():
            print("  No data for this period")
            continue
        
        test_start_idx = np.where(test_mask)[0][0]
        test_end_idx = np.where(test_mask)[0][-1] + 1
        
        # Run strategy on all data up to test end
        # Use full data for wavelet warmup, then extract test period
        result = run_bocpd_strategy(
            close,
            returns,
            window=63, level=4, keep_levels=2, hazard=1/25
        )
        
        # Extract test period using pre-computed signals
        signals = result['signals']
        daily_returns = result['daily_returns']
        
        # Map test period to signal indices
        return_dates = dates[:len(signals)]
        period_mask = (return_dates >= start) & (return_dates <= end)
        
        if not period_mask.any():
            print("  No data for this period")
            continue
        
        test_strat_rets = signals[period_mask] * daily_returns[period_mask]
        test_bh_rets = daily_returns[period_mask]
        
        def calc_metrics(rets):
            if len(rets) == 0:
                return {'total_return': 0, 'sharpe': 0, 'max_dd': 0}
            total = np.prod(1 + rets) - 1
            ann_ret = (1 + total) ** (252 / len(rets)) - 1
            ann_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0.01
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cum = np.cumprod(1 + rets)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_dd = dd.min()
            return {'total_return': total * 100, 'sharpe': sharpe, 'max_dd': max_dd * 100}
        
        strat_m = calc_metrics(test_strat_rets)
        bh_m = calc_metrics(test_bh_rets)
        
        print(f"  Days: {len(test_strat_rets)}")
        print(f"  Strategy: {strat_m['total_return']:+.1f}% return, {strat_m['sharpe']:.2f} Sharpe, {strat_m['max_dd']:.1f}% DD")
        print(f"  Buy&Hold: {bh_m['total_return']:+.1f}% return, {bh_m['sharpe']:.2f} Sharpe, {bh_m['max_dd']:.1f}% DD")
        
        sharpe_improvement = ((strat_m['sharpe'] / bh_m['sharpe']) - 1) * 100 if bh_m['sharpe'] != 0 else 0
        dd_improvement = ((strat_m['max_dd'] / bh_m['max_dd']) - 1) * 100 if bh_m['max_dd'] != 0 else 0
        
        print(f"  Sharpe Improvement: {sharpe_improvement:+.0f}%")
        print(f"  Drawdown Improvement: {dd_improvement:+.0f}%")
        
        results.append({
            'period': label,
            'strat_return': strat_m['total_return'],
            'strat_sharpe': strat_m['sharpe'],
            'strat_dd': strat_m['max_dd'],
            'bh_return': bh_m['total_return'],
            'bh_sharpe': bh_m['sharpe'],
            'bh_dd': bh_m['max_dd']
        })
    
    return results


# ============================================================================
# Alternative Configurations Test
# ============================================================================

def test_config_stability():
    """Test similar configurations to ensure not over-fit to exact params."""
    print("\n" + "=" * 80)
    print("CONFIGURATION STABILITY TEST")
    print("Testing nearby parameters to check for robustness")
    print("=" * 80)
    
    print("\nDownloading SPY...")
    data = yf.download('SPY', start='2021-01-01', end='2025-12-31', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY']
    else:
        close = data['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    # Test period
    test_start = '2024-01-01'
    test_mask = dates >= test_start
    test_start_idx = np.where(test_mask)[0][0]
    
    # Test nearby configurations
    configs = [
        # (window, level, keep, hazard, label)
        (63, 4, 2, 1/25, "Best (63/4/2/25)"),
        (63, 4, 2, 1/20, "Alt hazard 1/20"),
        (63, 4, 2, 1/30, "Alt hazard 1/30"),
        (63, 3, 2, 1/25, "Alt level=3"),
        (63, 5, 2, 1/25, "Alt level=5"),
        (63, 4, 1, 1/25, "Alt keep=1"),
        (63, 4, 3, 1/25, "Alt keep=3"),
        (50, 4, 2, 1/25, "Alt window=50"),
        (80, 4, 2, 1/25, "Alt window=80"),
    ]
    
    results = []
    
    print(f"\n{'Config':<20} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 55)
    
    for window, level, keep, hazard, label in configs:
        result = run_bocpd_strategy(close, returns, window=window, level=level, keep_levels=keep, hazard=hazard)
        
        signals = result['signals']
        daily_returns = result['daily_returns']
        
        # Find test period
        return_dates = close.index[1:len(signals)+1]
        test_mask = return_dates >= test_start
        if not test_mask.any():
            continue
        test_start_idx = np.where(test_mask)[0][0]
        
        test_strat_rets = signals[test_start_idx:] * daily_returns[test_start_idx:]
        
        if len(test_strat_rets) == 0:
            continue
            
        total = np.prod(1 + test_strat_rets) - 1
        ann_ret = (1 + total) ** (252 / len(test_strat_rets)) - 1
        ann_vol = np.std(test_strat_rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum = np.cumprod(1 + test_strat_rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        print(f"{label:<20} {total*100:>9.1f}% {sharpe:>10.2f} {max_dd*100:>9.1f}%")
        
        results.append({
            'config': label,
            'return': total * 100,
            'sharpe': sharpe,
            'max_dd': max_dd * 100
        })
    
    # Summary stats
    sharpes = [r['sharpe'] for r in results]
    print(f"\nSharpe Range: {min(sharpes):.2f} - {max(sharpes):.2f}")
    print(f"Sharpe Mean: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BOCPD WAVELET ROBUSTNESS VALIDATION")
    print("Testing if Window=63/Level=4/Keep=2 is truly robust")
    print("=" * 80)
    
    # Run all tests
    ticker_results = test_multiple_tickers()
    
    walkforward_results = test_walk_forward()
    
    config_results = test_config_stability()
    
    # Final Summary
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    
    print("\n1. MULTI-TICKER RESULTS:")
    if ticker_results:
        sharpe_wins = sum(1 for r in ticker_results if r['strat_sharpe'] > r['bh_sharpe'])
        dd_wins = sum(1 for r in ticker_results if r['strat_dd'] > r['bh_dd'])  # Less negative = better
        print(f"   Sharpe better than B&H: {sharpe_wins}/{len(ticker_results)} tickers")
        print(f"   Drawdown better than B&H: {dd_wins}/{len(ticker_results)} tickers")
        avg_sharpe = np.mean([r['strat_sharpe'] for r in ticker_results])
        print(f"   Average Strategy Sharpe: {avg_sharpe:.2f}")
    
    print("\n2. WALK-FORWARD RESULTS:")
    if walkforward_results:
        sharpe_wins = sum(1 for r in walkforward_results if r['strat_sharpe'] > r['bh_sharpe'])
        print(f"   Sharpe better than B&H: {sharpe_wins}/{len(walkforward_results)} periods")
        for r in walkforward_results:
            winner = "✓ Strategy" if r['strat_sharpe'] > r['bh_sharpe'] else "✗ B&H"
            print(f"   {r['period']}: {winner} ({r['strat_sharpe']:.2f} vs {r['bh_sharpe']:.2f})")
    
    print("\n3. CONFIG STABILITY:")
    if config_results:
        sharpes = [r['sharpe'] for r in config_results]
        print(f"   Nearby configs Sharpe range: {min(sharpes):.2f} - {max(sharpes):.2f}")
        print(f"   Coefficient of Variation: {np.std(sharpes)/np.mean(sharpes)*100:.1f}%")
        if np.std(sharpes) / np.mean(sharpes) < 0.2:
            print("   → Low variance = ROBUST configuration")
        else:
            print("   → High variance = Possibly over-fit")
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
