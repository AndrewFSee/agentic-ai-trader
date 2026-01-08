"""
BOCPD Wavelet - CORRECTED Robustness Test

With proper forward-bias prevention:
- Signal at time t uses only data from 0 to t-1
- Signal applied to return from t to t+1
- Transaction costs included
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from scipy.special import gammaln, logsumexp
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FastBOCPD:
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
            self.n = self.n[:self.max_run_length]
            self.sum_x = self.sum_x[:self.max_run_length]
            self.sum_x2 = self.sum_x2[:self.max_run_length]
        
        self.log_R = log_R_new
        map_rl = int(np.argmax(np.exp(log_R_new)))
        self.map_history.append(map_rl)
        return map_rl


def rolling_wavelet_denoise(returns: np.ndarray, window: int = 63, 
                            wavelet: str = 'db6', level: int = 4, 
                            keep_levels: int = 2) -> np.ndarray:
    """Rolling window wavelet - uses data up to and including current point."""
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


def calc_metrics(returns: np.ndarray) -> Dict:
    if len(returns) == 0:
        return {'total': 0, 'sharpe': 0, 'max_dd': 0}
    total = np.prod(1 + returns) - 1
    years = len(returns) / 252
    ann_ret = (1 + total) ** (1/years) - 1 if years > 0 else 0
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return {'total': total*100, 'sharpe': sharpe, 'max_dd': max_dd*100}


def run_unbiased_strategy(returns: np.ndarray, window: int = 63, 
                          level: int = 4, keep_levels: int = 2,
                          hazard: float = 0.04, lookback: int = 20,
                          transaction_cost: float = 0.001) -> Dict:
    """
    Unbiased strategy: signal[t] uses only data[0:t], applied to return[t].
    """
    n = len(returns)
    
    # Pre-compute denoised returns (this is fine - we'll use them correctly)
    denoised = rolling_wavelet_denoise(returns, window=window, level=level, keep_levels=keep_levels)
    
    # Signals - signal[t] uses data up to t-1 only
    signals = np.zeros(n)
    bocpd = FastBOCPD(hazard_rate=hazard)
    
    # Running stats for BOCPD standardization
    running_sum = 0.0
    running_sum2 = 0.0
    
    # Warmup period
    warmup = max(window, lookback + 1)
    
    for t in range(warmup, n):
        # Use data from yesterday (t-1) to make signal for today
        # BOCPD sees denoised[t-1]
        obs = denoised[t-1]
        running_sum += obs
        running_sum2 += obs**2
        count = t - warmup + 1
        mu = running_sum / count
        var = running_sum2/count - mu**2 if count > 1 else 1.0
        std = np.sqrt(max(var, 1e-10))
        standardized = (obs - mu) / std if std > 1e-10 else 0.0
        
        map_rl = bocpd.update(standardized)
        
        # Momentum signal using data up to t-1
        recent = denoised[t-lookback-1:t]  # lookback days ending at t-1
        if len(recent) > 0:
            signals[t] = 1.0 if np.mean(recent) > 0 else 0.0
            
            # Reduce on change point
            if len(bocpd.map_history) > 1:
                if bocpd.map_history[-2] > 3 and map_rl <= 2:
                    signals[t] = 0.5
    
    # Strategy returns
    strat_returns = signals * returns
    
    # Transaction costs
    pos_changes = np.abs(np.diff(np.concatenate([[0], signals])))
    tc = pos_changes * transaction_cost
    strat_returns_net = strat_returns - tc
    
    trades = int(np.sum(pos_changes > 0.1))
    
    return {
        'signals': signals,
        'strat_gross': calc_metrics(strat_returns),
        'strat_net': calc_metrics(strat_returns_net),
        'bh': calc_metrics(returns),
        'trades': trades
    }


def test_multi_ticker():
    """Test on multiple tickers with corrected methodology."""
    print("=" * 80)
    print("MULTI-TICKER TEST (Unbiased + Transaction Costs)")
    print("=" * 80)
    
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']
    test_start = '2024-01-01'
    
    results = []
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        try:
            data = yf.download(ticker, start='2021-01-01', end='2025-12-31', progress=False)
            if data.empty:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'][ticker]
            else:
                close = data['Close']
            
            returns = close.pct_change().dropna().values
            dates = close.index[1:]
            
            # Run unbiased strategy
            result = run_unbiased_strategy(returns, window=63, level=4, keep_levels=2, hazard=1/25)
            
            # Extract test period
            test_mask = dates >= test_start
            if not test_mask.any():
                continue
            test_idx = np.where(test_mask)[0]
            test_start_idx = test_idx[0]
            
            signals = result['signals'][test_start_idx:]
            test_returns = returns[test_start_idx:]
            
            test_strat = signals * test_returns
            pos_changes = np.abs(np.diff(np.concatenate([[0], signals])))
            test_strat_net = test_strat - pos_changes * 0.001
            
            strat_m = calc_metrics(test_strat_net)
            bh_m = calc_metrics(test_returns)
            
            print(f"  Strategy: {strat_m['total']:+.1f}% (Sharpe {strat_m['sharpe']:.2f}, DD {strat_m['max_dd']:.1f}%)")
            print(f"  Buy&Hold: {bh_m['total']:+.1f}% (Sharpe {bh_m['sharpe']:.2f}, DD {bh_m['max_dd']:.1f}%)")
            
            results.append({
                'ticker': ticker,
                'strat_return': strat_m['total'],
                'strat_sharpe': strat_m['sharpe'],
                'strat_dd': strat_m['max_dd'],
                'bh_return': bh_m['total'],
                'bh_sharpe': bh_m['sharpe'],
                'bh_dd': bh_m['max_dd']
            })
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def test_walk_forward():
    """Walk-forward on multiple periods."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD TEST (SPY, Unbiased + Transaction Costs)")
    print("=" * 80)
    
    data = yf.download('SPY', start='2018-01-01', end='2025-12-31', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY']
    else:
        close = data['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    # Run full strategy
    result = run_unbiased_strategy(returns, window=63, level=4, keep_levels=2, hazard=1/25)
    
    test_periods = [
        ('2020-01-01', '2020-12-31', 'COVID Year (2020)'),
        ('2021-01-01', '2021-12-31', 'Bull Market (2021)'),
        ('2022-01-01', '2022-12-31', 'Bear Market (2022)'),
        ('2023-01-01', '2023-12-31', 'Recovery (2023)'),
        ('2024-01-01', '2025-12-31', 'Recent (2024-25)')
    ]
    
    results = []
    
    for start, end, label in test_periods:
        mask = (dates >= start) & (dates <= end)
        if not mask.any():
            continue
        
        signals = result['signals'][mask]
        period_returns = returns[mask]
        
        strat_rets = signals * period_returns
        pos_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        strat_net = strat_rets - pos_changes * 0.001
        
        strat_m = calc_metrics(strat_net)
        bh_m = calc_metrics(period_returns)
        
        print(f"\n{label}:")
        print(f"  Strategy: {strat_m['total']:+.1f}% (Sharpe {strat_m['sharpe']:.2f}, DD {strat_m['max_dd']:.1f}%)")
        print(f"  Buy&Hold: {bh_m['total']:+.1f}% (Sharpe {bh_m['sharpe']:.2f}, DD {bh_m['max_dd']:.1f}%)")
        
        results.append({
            'period': label,
            'strat_sharpe': strat_m['sharpe'],
            'bh_sharpe': bh_m['sharpe'],
            'strat_dd': strat_m['max_dd'],
            'bh_dd': bh_m['max_dd']
        })
    
    return results


def test_simple_baselines():
    """Compare against simple baselines."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (SPY 2024-25)")
    print("=" * 80)
    
    data = yf.download('SPY', start='2021-01-01', end='2025-12-31', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']['SPY']
    else:
        close = data['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    test_start = '2024-01-01'
    test_mask = dates >= test_start
    test_idx = np.where(test_mask)[0][0]
    test_returns = returns[test_idx:]
    
    # 1. Wavelet-BOCPD
    result = run_unbiased_strategy(returns, window=63, level=4, keep_levels=2, hazard=1/25)
    wavelet_signals = result['signals'][test_idx:]
    wavelet_strat = wavelet_signals * test_returns
    wavelet_tc = np.abs(np.diff(np.concatenate([[0], wavelet_signals]))) * 0.001
    wavelet_net = calc_metrics(wavelet_strat - wavelet_tc)
    wavelet_trades = np.sum(np.abs(np.diff(wavelet_signals)) > 0.1)
    
    # 2. Simple 20-day momentum (unbiased)
    mom_signals = np.zeros(len(returns))
    for t in range(21, len(returns)):
        mom_signals[t] = 1.0 if np.mean(returns[t-21:t]) > 0 else 0.0
    mom_test = mom_signals[test_idx:] * test_returns
    mom_tc = np.abs(np.diff(np.concatenate([[0], mom_signals[test_idx:]]))) * 0.001
    mom_net = calc_metrics(mom_test - mom_tc)
    mom_trades = np.sum(np.abs(np.diff(mom_signals[test_idx:])) > 0.1)
    
    # 3. SMA crossover (50/200)
    sma50 = np.zeros(len(returns))
    sma200 = np.zeros(len(returns))
    for t in range(200, len(returns)):
        sma50[t] = np.mean(returns[t-50:t])
        sma200[t] = np.mean(returns[t-200:t])
    sma_signals = np.zeros(len(returns))
    for t in range(201, len(returns)):
        sma_signals[t] = 1.0 if sma50[t-1] > sma200[t-1] else 0.0
    sma_test = sma_signals[test_idx:] * test_returns
    sma_tc = np.abs(np.diff(np.concatenate([[0], sma_signals[test_idx:]]))) * 0.001
    sma_net = calc_metrics(sma_test - sma_tc)
    sma_trades = np.sum(np.abs(np.diff(sma_signals[test_idx:])) > 0.1)
    
    # 4. Buy & Hold
    bh_m = calc_metrics(test_returns)
    
    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>10}")
    print("-" * 70)
    print(f"{'Wavelet-BOCPD':<25} {wavelet_net['total']:>9.1f}% {wavelet_net['sharpe']:>10.2f} {wavelet_net['max_dd']:>9.1f}% {int(wavelet_trades):>10}")
    print(f"{'20-day Momentum':<25} {mom_net['total']:>9.1f}% {mom_net['sharpe']:>10.2f} {mom_net['max_dd']:>9.1f}% {int(mom_trades):>10}")
    print(f"{'SMA 50/200 Crossover':<25} {sma_net['total']:>9.1f}% {sma_net['sharpe']:>10.2f} {sma_net['max_dd']:>9.1f}% {int(sma_trades):>10}")
    print(f"{'Buy & Hold':<25} {bh_m['total']:>9.1f}% {bh_m['sharpe']:>10.2f} {bh_m['max_dd']:>9.1f}% {'0':>10}")
    
    return {
        'wavelet': wavelet_net,
        'momentum': mom_net,
        'sma': sma_net,
        'bh': bh_m
    }


if __name__ == "__main__":
    print("=" * 80)
    print("CORRECTED BOCPD WAVELET ROBUSTNESS TEST")
    print("With proper forward-bias prevention + transaction costs")
    print("=" * 80)
    
    ticker_results = test_multi_ticker()
    wf_results = test_walk_forward()
    baseline_results = test_simple_baselines()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n1. MULTI-TICKER (2024-25 test period):")
    if ticker_results:
        sharpe_wins = sum(1 for r in ticker_results if r['strat_sharpe'] > r['bh_sharpe'])
        dd_wins = sum(1 for r in ticker_results if r['strat_dd'] > r['bh_dd'])
        print(f"   Sharpe wins: {sharpe_wins}/{len(ticker_results)}")
        print(f"   Drawdown wins: {dd_wins}/{len(ticker_results)}")
    
    print("\n2. WALK-FORWARD (SPY, multiple years):")
    if wf_results:
        for r in wf_results:
            winner = "Strategy" if r['strat_sharpe'] > r['bh_sharpe'] else "B&H"
            print(f"   {r['period']}: {winner} wins ({r['strat_sharpe']:.2f} vs {r['bh_sharpe']:.2f})")
    
    print("\n3. HONEST VERDICT:")
    print("   The Wavelet-BOCPD strategy, when implemented WITHOUT forward bias")
    print("   and WITH transaction costs, primarily provides DRAWDOWN PROTECTION")
    print("   rather than alpha generation. Simple momentum often beats it.")
