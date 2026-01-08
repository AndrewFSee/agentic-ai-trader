"""
BOCPD Wavelet Sanity Check

The Sharpe ratios from robustness.py are suspiciously high (3-5).
This script performs rigorous sanity checks:
1. Verify no forward bias in signal generation
2. Apply realistic transaction costs
3. Check signal-to-return alignment
4. Compare against simple baselines
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from scipy.special import gammaln, logsumexp
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class FastBOCPD:
    """Lightweight BOCPD for backtesting."""
    
    def __init__(self, hazard_rate: float = 0.04, max_run_length: int = 300):
        self.hazard_rate = hazard_rate
        self.log_H = np.log(hazard_rate)
        self.log_1_minus_H = np.log(1 - hazard_rate)
        self.max_run_length = max_run_length
        self.mu0, self.kappa0, self.alpha0, self.beta0 = 0.0, 0.1, 2.0, 1.0
        self.reset()
    
    def reset(self):
        self.log_R = np.array([0.0])
        self.n = np.array([0.0])
        self.sum_x = np.array([0.0])
        self.sum_x2 = np.array([0.0])
        self.map_history = []
    
    def update(self, x: float) -> int:
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
        
        log_growth = self.log_R + log_pred + self.log_1_minus_H
        log_cp = logsumexp(self.log_R + log_pred) + self.log_H
        log_R_new = np.concatenate([[log_cp], log_growth])
        log_R_new = log_R_new - logsumexp(log_R_new)
        
        self.n = np.concatenate([[0.0], self.n + 1])
        self.sum_x = np.concatenate([[0.0], self.sum_x + x])
        self.sum_x2 = np.concatenate([[0.0], self.sum_x2 + x**2])
        
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
    """Rolling window wavelet denoising."""
    denoised = np.zeros_like(returns)
    
    for i in range(len(returns)):
        if i < window:
            denoised[i] = returns[i]
        else:
            window_data = returns[i-window+1:i+1]
            mu, sigma = np.mean(window_data), np.std(window_data)
            if sigma < 1e-10:
                denoised[i] = 0.0
                continue
            standardized = (window_data - mu) / sigma
            max_level = pywt.dwt_max_level(len(standardized), wavelet)
            use_level = min(level, max_level)
            if use_level < 1:
                denoised[i] = returns[i]
                continue
            coeffs = pywt.wavedec(standardized, wavelet, level=use_level)
            remove_levels = len(coeffs) - 1 - keep_levels
            for j in range(1, min(remove_levels + 1, len(coeffs))):
                coeffs[-j] = np.zeros_like(coeffs[-j])
            reconstructed = pywt.waverec(coeffs, wavelet)
            if len(reconstructed) > len(standardized):
                reconstructed = reconstructed[:len(standardized)]
            denoised[i] = reconstructed[-1]
    
    return denoised


def calc_metrics(returns: np.ndarray, trading_days_per_year: int = 252) -> Dict:
    """Calculate performance metrics."""
    if len(returns) == 0:
        return {'total': 0, 'ann_ret': 0, 'ann_vol': 0, 'sharpe': 0, 'max_dd': 0}
    
    total = np.prod(1 + returns) - 1
    years = len(returns) / trading_days_per_year
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = np.std(returns) * np.sqrt(trading_days_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return {
        'total': total * 100,
        'ann_ret': ann_ret * 100,
        'ann_vol': ann_vol * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100
    }


def run_strategy_with_audit(close: pd.Series, returns: np.ndarray,
                            window: int = 63, level: int = 4, 
                            keep_levels: int = 2, hazard: float = 0.04,
                            transaction_cost: float = 0.001) -> Dict:
    """
    Run strategy with explicit forward-bias prevention and transaction costs.
    
    CRITICAL: Signal on day t can only use data up to and including day t.
    Signal on day t is applied to return from day t to day t+1.
    """
    n = len(returns)
    
    # Wavelet denoise - rolling window, no forward bias
    denoised = rolling_wavelet_denoise(returns, window=window, level=level, keep_levels=keep_levels)
    
    # BOCPD processing
    bocpd = FastBOCPD(hazard_rate=hazard)
    
    # Signal array: signal[t] is the position held from day t to day t+1
    signals = np.zeros(n)
    
    # Expanding window standardization for BOCPD input
    running_sum = 0.0
    running_sum2 = 0.0
    
    for t in range(n):
        # Update standardization with data up to and including t
        running_sum += denoised[t]
        running_sum2 += denoised[t]**2
        count = t + 1
        mu = running_sum / count
        var = running_sum2/count - mu**2 if count > 1 else 1.0
        std = np.sqrt(max(var, 1e-10))
        
        standardized = (denoised[t] - mu) / std if std > 1e-10 else 0.0
        
        # Update BOCPD with standardized observation
        map_rl = bocpd.update(standardized)
        
        # Generate signal using only data up to t
        if t >= 20:
            # Momentum signal: look at recent denoised returns (all available up to t)
            recent = denoised[t-20:t+1]  # Days t-20 through t
            recent_mean = np.mean(recent)
            signals[t] = 1.0 if recent_mean > 0 else 0.0
            
            # Reduce position on change point detection
            if len(bocpd.map_history) > 1:
                if bocpd.map_history[-2] > 3 and map_rl <= 2:
                    signals[t] = 0.5
    
    # Calculate strategy returns
    # Return from day t to t+1 is applied to signal[t]
    # returns[t] = (price[t+1] - price[t]) / price[t] but we have returns calculated from close prices
    # Actually, returns[t] = close[t+1]/close[t] - 1 where close[0] is first day after we start
    # Let's be very explicit:
    
    # close has dates, returns[i] = (close[i+1] - close[i]) / close[i]
    # So returns[i] is the return from day i to day i+1
    # Signal[i] should be the position held overnight from day i to day i+1
    # This is what we want: signal based on data up to day i, applied to return from i to i+1
    
    # Wait - there's an off-by-one issue. Let me think carefully:
    # - returns array: returns[0] = day0->day1, returns[1] = day1->day2, etc.
    # - signals array: signals[0] = position after seeing day0 return
    # 
    # The issue: when we compute signals[t], we're using denoised[t] which is based on returns[t]
    # But returns[t] is the return from day t to day t+1.
    # That means when we're at the END of day t+1, we know returns[t].
    # So signals[t] is actually known at end of day t+1.
    # 
    # To apply signals[t] to returns[t], we'd need to know returns[t] before it happens.
    # This is forward bias!
    #
    # CORRECT APPROACH:
    # - At end of day t, we know returns[0:t] (returns up to yesterday)
    # - We compute signal for overnight position (day t to day t+1)
    # - This signal should use denoised[0:t], not including today's return!
    
    # Let's fix this properly:
    signals_corrected = np.zeros(n)
    bocpd2 = FastBOCPD(hazard_rate=hazard)
    running_sum = 0.0
    running_sum2 = 0.0
    
    for t in range(n):
        # At end of day t+1, we know returns[t]
        # For position from day t+1 to t+2, we use returns[0:t+1]
        
        # Generate signal for TOMORROW using data available TODAY
        if t >= 21:
            # We know returns[0:t] (up to yesterday)
            # Use that to generate signal for today's position
            
            # First update BOCPD with yesterday's data
            prev_denoised = denoised[t-1]
            running_sum += prev_denoised
            running_sum2 += prev_denoised**2
            count = t
            mu = running_sum / count
            var = running_sum2/count - mu**2 if count > 1 else 1.0
            std = np.sqrt(max(var, 1e-10))
            standardized = (prev_denoised - mu) / std if std > 1e-10 else 0.0
            map_rl = bocpd2.update(standardized)
            
            # Signal based on data up to t-1
            recent = denoised[t-21:t]  # Days t-21 through t-1
            recent_mean = np.mean(recent)
            signals_corrected[t] = 1.0 if recent_mean > 0 else 0.0
            
            if len(bocpd2.map_history) > 1:
                if bocpd2.map_history[-2] > 3 and map_rl <= 2:
                    signals_corrected[t] = 0.5
    
    # Now signals_corrected[t] uses only data up to t-1, applied to returns[t]
    strategy_returns = signals_corrected * returns
    bh_returns = returns
    
    # Add transaction costs
    position_changes = np.abs(np.diff(np.concatenate([[0], signals_corrected])))
    tc_drag = position_changes * transaction_cost
    strategy_returns_net = strategy_returns - tc_drag
    
    # Count trades
    trades = np.sum(position_changes > 0.1)
    
    return {
        'signals': signals_corrected,
        'strategy_gross': calc_metrics(strategy_returns),
        'strategy_net': calc_metrics(strategy_returns_net),
        'buy_hold': calc_metrics(bh_returns),
        'trades': int(trades),
        'avg_position': np.mean(signals_corrected),
        'time_in_market': np.mean(signals_corrected > 0)
    }


def main():
    print("=" * 80)
    print("BOCPD WAVELET SANITY CHECK")
    print("Verifying no forward bias + realistic transaction costs")
    print("=" * 80)
    
    # Download data
    print("\nDownloading SPY...")
    data = yf.download('SPY', start='2021-01-01', end='2025-12-31', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY']
    else:
        close = data['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    print(f"Data: {len(returns)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Run with audit
    result = run_strategy_with_audit(
        close, returns,
        window=63, level=4, keep_levels=2, hazard=1/25,
        transaction_cost=0.001  # 10 bps round-trip
    )
    
    # Test period analysis
    test_start = '2024-01-01'
    test_mask = dates >= test_start
    test_idx = np.where(test_mask)[0]
    
    if len(test_idx) == 0:
        print("No test data!")
        return
    
    test_start_idx = test_idx[0]
    
    test_signals = result['signals'][test_start_idx:]
    test_returns = returns[test_start_idx:]
    test_strat = test_signals * test_returns
    
    # Transaction costs in test period
    test_pos_changes = np.abs(np.diff(np.concatenate([[0], test_signals])))
    test_tc = test_pos_changes * 0.001
    test_strat_net = test_strat - test_tc
    
    print("\n" + "=" * 80)
    print(f"TEST PERIOD: {test_start} to {dates[-1].strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    print("\n[Strategy Metrics - GROSS (no costs)]")
    strat_gross = calc_metrics(test_strat)
    print(f"  Total Return: {strat_gross['total']:.1f}%")
    print(f"  Ann. Return:  {strat_gross['ann_ret']:.1f}%")
    print(f"  Ann. Vol:     {strat_gross['ann_vol']:.1f}%")
    print(f"  Sharpe:       {strat_gross['sharpe']:.2f}")
    print(f"  Max Drawdown: {strat_gross['max_dd']:.1f}%")
    
    print("\n[Strategy Metrics - NET (10 bps per trade)]")
    strat_net = calc_metrics(test_strat_net)
    print(f"  Total Return: {strat_net['total']:.1f}%")
    print(f"  Ann. Return:  {strat_net['ann_ret']:.1f}%")
    print(f"  Sharpe:       {strat_net['sharpe']:.2f}")
    print(f"  Max Drawdown: {strat_net['max_dd']:.1f}%")
    
    print("\n[Buy & Hold Metrics]")
    bh = calc_metrics(test_returns)
    print(f"  Total Return: {bh['total']:.1f}%")
    print(f"  Ann. Return:  {bh['ann_ret']:.1f}%")
    print(f"  Sharpe:       {bh['sharpe']:.2f}")
    print(f"  Max Drawdown: {bh['max_dd']:.1f}%")
    
    print("\n[Trading Activity]")
    trades_in_test = np.sum(test_pos_changes > 0.1)
    print(f"  Trades: {int(trades_in_test)}")
    print(f"  Time in Market: {np.mean(test_signals > 0)*100:.1f}%")
    print(f"  Avg Position: {np.mean(test_signals):.2f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON TO SIMPLE BASELINES")
    print("=" * 80)
    
    # Simple 20-day momentum baseline
    momentum_signals = np.zeros(len(returns))
    for t in range(21, len(returns)):
        momentum_signals[t] = 1.0 if np.mean(returns[t-21:t]) > 0 else 0.0
    
    test_mom = momentum_signals[test_start_idx:] * test_returns
    mom_changes = np.abs(np.diff(np.concatenate([[0], momentum_signals[test_start_idx:]])))
    test_mom_net = test_mom - mom_changes * 0.001
    
    print("\n[Simple 20-day Momentum (no wavelet, no BOCPD)]")
    mom_metrics = calc_metrics(test_mom_net)
    print(f"  Total Return: {mom_metrics['total']:.1f}%")
    print(f"  Sharpe:       {mom_metrics['sharpe']:.2f}")
    print(f"  Trades:       {int(np.sum(mom_changes > 0.1))}")
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    sharpe_diff = strat_net['sharpe'] - bh['sharpe']
    if strat_net['sharpe'] > 2.5:
        print("⚠️  WARNING: Sharpe > 2.5 is suspicious. Check for remaining bias.")
    elif strat_net['sharpe'] > bh['sharpe']:
        print(f"✓ Strategy beats B&H by {sharpe_diff:.2f} Sharpe (net of costs)")
    else:
        print(f"✗ Strategy underperforms B&H by {-sharpe_diff:.2f} Sharpe")
    
    if strat_net['max_dd'] > bh['max_dd']:  # Less negative = better
        print(f"✓ Strategy has better drawdown: {strat_net['max_dd']:.1f}% vs {bh['max_dd']:.1f}%")
    
    if strat_net['sharpe'] > mom_metrics['sharpe']:
        print(f"✓ Wavelet-BOCPD beats simple momentum by {strat_net['sharpe'] - mom_metrics['sharpe']:.2f} Sharpe")
    else:
        print(f"✗ Simple momentum matches or beats Wavelet-BOCPD")


if __name__ == "__main__":
    main()
