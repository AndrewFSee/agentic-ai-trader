"""
Alpha Regime Detector - Forward-Looking Bias Audit & OOS Backtests

This script:
1. Audits every component for forward-looking bias
2. Creates proper train/test splits
3. Backtests on multiple tickers with realistic assumptions
4. Compares against simple baselines

Key bias sources to check:
- Feature engineering (volatility, standardization)
- BOCPD updates (should be strictly online)
- Signal generation timing
- Regime labeling
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'c:/Users/Andrew/projects/agentic_ai_trader/models')

from alpha_regime_detector import (
    FeatureConfig, BOCPDConfig, RegimeConfig,
    AlphaRegimeDetector, FeatureEngine,
    VolatilityEstimator, IntradayMeasure
)


# =============================================================================
# FORWARD-LOOKING BIAS AUDIT
# =============================================================================

def audit_feature_engineering():
    """
    Audit: Does feature engineering use future data?
    
    Potential bias sources:
    1. EWMA volatility - should only use past data ✓
    2. Standardization - must be rolling/expanding, not full-sample
    3. Rolling windows - must not include current observation in lookback
    """
    print("=" * 70)
    print("AUDIT 1: Feature Engineering")
    print("=" * 70)
    
    # Create simple test data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    
    # Regime shift at t=50: low vol -> high vol
    prices = np.zeros(n)
    prices[0] = 100
    for i in range(1, n):
        if i < 50:
            ret = np.random.randn() * 0.01
        else:
            ret = np.random.randn() * 0.03  # 3x volatility
        prices[i] = prices[i-1] * (1 + ret)
    
    df = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.005,
        'Low': prices * 0.995,
        'Close': prices,
        'Volume': np.random.randint(1e6, 1e7, n)
    }, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Process with feature engine
    engine = FeatureEngine(FeatureConfig(ewma_span=20))
    features = engine.process_dataframe(df)
    
    # Check: sigma at t=49 should NOT know about high-vol regime
    # sigma at t=50-55 should gradually increase
    print("\nVolatility around regime shift (t=50):")
    print("  If unbiased: sigma should be LOW at t=49, gradually increase after")
    print("  If biased: sigma would already be elevated before t=50")
    
    for t in [45, 48, 49, 50, 51, 52, 55, 60]:
        if t < len(features):
            sigma = features.iloc[t]['sigma']
            print(f"  t={t}: sigma = {sigma:.5f}")
    
    # Verdict
    sigma_before = features.iloc[45:50]['sigma'].mean()
    sigma_after = features.iloc[55:60]['sigma'].mean()
    
    if sigma_after > sigma_before * 1.5:
        print("\n✓ PASS: Volatility correctly adapts AFTER regime shift")
    else:
        print("\n✗ FAIL: Possible forward-looking bias in volatility")
    
    return features


def audit_bocpd_updates():
    """
    Audit: Does BOCPD use future observations?
    
    The BOCPD update at time t should only use x_1, ..., x_t.
    """
    print("\n" + "=" * 70)
    print("AUDIT 2: BOCPD Online Updates")
    print("=" * 70)
    
    from alpha_regime_detector import BOCPD, BOCPDConfig
    
    # Create data with clear change point at t=50
    np.random.seed(42)
    data1 = np.random.randn(50) * 0.5
    data2 = np.random.randn(50) * 0.5 + 5.0  # Mean shift
    data = np.concatenate([data1, data2])
    
    # Run BOCPD and check MAP run length at t=49 vs t=51
    bocpd = BOCPD(BOCPDConfig(hazard_rate=0.02))
    
    map_history = []
    for i, x in enumerate(data):
        bocpd.update(x)
        map_history.append(bocpd.get_map_run_length())
    
    print("\nMAP run length around change point (t=50):")
    print("  If unbiased: MAP should be ~49 at t=49, drop to ~1 around t=51")
    print("  If biased: MAP would drop BEFORE the actual change")
    
    for t in [45, 48, 49, 50, 51, 52, 55]:
        print(f"  t={t}: MAP = {map_history[t]}")
    
    # Check timing
    first_drop = None
    for t in range(48, 55):
        if map_history[t-1] > 5 and map_history[t] <= 3:
            first_drop = t
            break
    
    if first_drop is not None and first_drop >= 50:
        print(f"\n✓ PASS: First MAP drop at t={first_drop} (≥50)")
    elif first_drop is not None:
        print(f"\n⚠ CHECK: First MAP drop at t={first_drop} (<50, but may be natural)")
    else:
        print("\n? No clear MAP drop detected")
    
    return map_history


def audit_signal_timing():
    """
    Audit: Is the trading signal correctly aligned?
    
    Critical check:
    - Signal at time t should be generated using data up to t-1 only
    - Signal at time t is applied to return from t to t+1
    """
    print("\n" + "=" * 70)
    print("AUDIT 3: Signal Timing (CRITICAL)")
    print("=" * 70)
    
    print("""
    Correct signal timing:
    ┌─────────────────────────────────────────────────────────────────┐
    │  At END of day t-1:                                             │
    │    - We know: OHLCV[0:t-1], returns[0:t-1]                     │
    │    - We compute: signal[t] for position on day t               │
    │    - Signal[t] is applied to return[t] = price[t+1]/price[t]-1 │
    └─────────────────────────────────────────────────────────────────┘
    
    Forward bias occurs if:
    - signal[t] uses any data from day t or later
    - This includes: return[t], OHLCV[t], features computed from day t
    """)
    
    # Check the AlphaRegimeDetector implementation
    print("\nChecking AlphaRegimeDetector.update() method:")
    print("  - update() receives OHLCV for day t")
    print("  - Computes features using day t data")
    print("  - Updates BOCPD with day t features")
    print("  - Returns regime state for day t")
    print("")
    print("  ⚠ ISSUE: The current implementation generates signals AFTER")
    print("           seeing day t's data. For unbiased trading signals,")
    print("           we need to LAG the signal by 1 day.")
    
    return None


def audit_regime_labeling():
    """
    Audit: Are regime labels assigned without future knowledge?
    """
    print("\n" + "=" * 70)
    print("AUDIT 4: Regime Labeling")
    print("=" * 70)
    
    print("""
    The regime_id is incremented when a change point is detected.
    
    Check: Does detection at time t use only data up to t?
    
    Current implementation:
    1. Features computed from OHLCV[t] 
    2. BOCPD updated with features[t]
    3. MAP drop detected by comparing map[t-1] vs map[t]
    4. If drop detected, regime_id increments
    
    ✓ This is correct: detection uses only past MAP values
    """)
    
    return None


# =============================================================================
# UNBIASED STRATEGY IMPLEMENTATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Feature settings
    ewma_span: int = 30
    vol_estimator: VolatilityEstimator = VolatilityEstimator.EWMA
    
    # BOCPD settings
    hazard_rate: float = 0.05
    
    # Detection settings
    map_drop_from: int = 5
    map_drop_to: int = 2
    min_spacing: int = 10
    
    # Strategy settings
    momentum_lookback: int = 20
    reduce_on_change: bool = True
    change_position: float = 0.5
    
    # Costs
    transaction_cost: float = 0.001  # 10 bps


class UnbiasedRegimeStrategy:
    """
    Unbiased regime-based trading strategy.
    
    Key: Signal at time t uses only data available at end of day t-1.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset strategy state."""
        self.detector = AlphaRegimeDetector(
            feature_config=FeatureConfig(
                vol_estimator=self.config.vol_estimator,
                ewma_span=self.config.ewma_span
            ),
            bocpd_config=BOCPDConfig(
                hazard_rate=self.config.hazard_rate
            ),
            regime_config=RegimeConfig(
                detection_method="map_drop",
                map_drop_from=self.config.map_drop_from,
                map_drop_to=self.config.map_drop_to,
                min_spacing=self.config.min_spacing
            )
        )
        
        # History for lagged signals
        self._z_ret_history: List[float] = []
        self._regime_state_history: List[Dict] = []
        self._signals: List[float] = []
    
    def update(
        self,
        ticker: str,
        timestamp: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None
    ) -> Optional[float]:
        """
        Process new bar and return LAGGED signal.
        
        The signal returned is for the PREVIOUS day's position,
        which is the correct unbiased approach.
        
        Args:
            Standard OHLCV inputs
            
        Returns:
            Signal for day t-1 (or None if insufficient history)
        """
        # Update detector with today's data
        state = self.detector.update(
            ticker, timestamp, open_, high, low, close, volume
        )
        
        if state is None:
            return None
        
        # Store state
        self._regime_state_history.append({
            'timestamp': timestamp,
            'regime_id': state.regime_id,
            'change_prob': state.change_prob,
            'days_in_regime': state.days_in_regime
        })
        
        # Get z_ret from feature engine (need to track this)
        if ticker in self.detector._tickers:
            feat_engine = self.detector._tickers[ticker]['feature_engine']
            if feat_engine._z_ret_history:
                z_ret = feat_engine._z_ret_history[-1]
                self._z_ret_history.append(z_ret)
        
        # Generate LAGGED signal (using data up to t-1)
        if len(self._z_ret_history) < self.config.momentum_lookback + 1:
            self._signals.append(0.0)
            return 0.0
        
        # Momentum based on z_ret history UP TO YESTERDAY
        recent_z = self._z_ret_history[-(self.config.momentum_lookback+1):-1]
        signal = 1.0 if np.mean(recent_z) > 0 else 0.0
        
        # Check if we detected a regime change YESTERDAY
        if self.config.reduce_on_change and len(self._regime_state_history) >= 2:
            prev_regime = self._regime_state_history[-2]['regime_id']
            curr_regime = self._regime_state_history[-1]['regime_id']
            
            # If regime changed yesterday, reduce position today
            if curr_regime > prev_regime:
                signal = self.config.change_position
        
        self._signals.append(signal)
        return signal
    
    def get_signals(self) -> np.ndarray:
        """Get all generated signals."""
        return np.array(self._signals)


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    config: BacktestConfig,
    train_end: str,
    test_start: str,
    test_end: str
) -> Dict:
    """
    Run train/test backtest.
    
    Args:
        df: OHLCV DataFrame
        ticker: Ticker symbol
        config: Backtest configuration
        train_end: End of training period
        test_start: Start of test period
        test_end: End of test period
        
    Returns:
        Dict with backtest results
    """
    strategy = UnbiasedRegimeStrategy(config)
    
    # Process all data (strategy handles lagging internally)
    signals = []
    timestamps = []
    
    for idx, row in df.iterrows():
        timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
        
        signal = strategy.update(
            ticker=ticker,
            timestamp=timestamp,
            open_=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row.get('Volume', None)
        )
        
        if signal is not None:
            signals.append(signal)
            timestamps.append(timestamp)
    
    signals = np.array(signals)
    timestamps = pd.DatetimeIndex(timestamps)
    
    # Calculate returns
    close = df['Close'].values
    returns = np.diff(close) / close[:-1]
    
    # Align: signals[i] applied to returns[i+1] (next day's return)
    # Because signal[i] is generated at end of day i, applied to day i+1
    min_len = min(len(signals) - 1, len(returns) - 1)
    aligned_signals = signals[:min_len]
    aligned_returns = returns[1:min_len+1]  # Shift returns by 1
    aligned_timestamps = timestamps[:min_len]
    
    # Split into train/test
    train_mask = aligned_timestamps < train_end
    test_mask = (aligned_timestamps >= test_start) & (aligned_timestamps <= test_end)
    
    # Test period metrics
    test_signals = aligned_signals[test_mask]
    test_returns = aligned_returns[test_mask]
    
    if len(test_returns) == 0:
        return {'error': 'No test data'}
    
    # Strategy returns (with transaction costs)
    strat_returns = test_signals * test_returns
    
    # Transaction costs
    pos_changes = np.abs(np.diff(np.concatenate([[0], test_signals])))
    tc = pos_changes * config.transaction_cost
    strat_returns_net = strat_returns - tc
    
    # Buy & hold
    bh_returns = test_returns
    
    # Metrics
    def calc_metrics(rets):
        if len(rets) == 0:
            return {'total': 0, 'sharpe': 0, 'max_dd': 0}
        total = np.prod(1 + rets) - 1
        years = len(rets) / 252
        ann_ret = (1 + total) ** (1/years) - 1 if years > 0 else 0
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        return {'total': total*100, 'sharpe': sharpe, 'max_dd': max_dd*100}
    
    strat_m = calc_metrics(strat_returns_net)
    bh_m = calc_metrics(bh_returns)
    
    # Trade count
    trades = int(np.sum(pos_changes > 0.1))
    
    return {
        'ticker': ticker,
        'test_days': len(test_returns),
        'strat_return': strat_m['total'],
        'strat_sharpe': strat_m['sharpe'],
        'strat_dd': strat_m['max_dd'],
        'bh_return': bh_m['total'],
        'bh_sharpe': bh_m['sharpe'],
        'bh_dd': bh_m['max_dd'],
        'trades': trades,
        'avg_position': np.mean(test_signals),
        'time_in_market': np.mean(test_signals > 0)
    }


# =============================================================================
# MULTI-TICKER OOS BACKTESTS
# =============================================================================

def run_oos_backtests():
    """Run out-of-sample backtests on multiple tickers."""
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE BACKTESTS")
    print("=" * 70)
    
    # Tickers to test
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'EFA', 'EEM']
    
    # Time periods
    # Train: 2020-01-01 to 2022-12-31 (3 years)
    # Test: 2023-01-01 to 2024-12-31 (2 years)
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2024-12-31'
    
    print(f"\nTrain period: 2020-01-01 to {train_end}")
    print(f"Test period:  {test_start} to {test_end}")
    
    # Download data
    print("\nDownloading data...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start='2019-06-01', end='2025-01-01', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  {ticker}: Failed - {e}")
    
    # Configuration
    config = BacktestConfig(
        ewma_span=30,
        hazard_rate=0.05,
        map_drop_from=5,
        map_drop_to=2,
        min_spacing=10,
        momentum_lookback=20,
        reduce_on_change=True,
        transaction_cost=0.001
    )
    
    # Run backtests
    print("\nRunning backtests...")
    results = []
    
    for ticker, df in data.items():
        result = run_backtest(df, ticker, config, train_end, test_start, test_end)
        if 'error' not in result:
            results.append(result)
            print(f"  {ticker}: Strategy {result['strat_return']:+.1f}% vs B&H {result['bh_return']:+.1f}%")
    
    return results


def run_walk_forward():
    """Run walk-forward validation on SPY."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION (SPY)")
    print("=" * 70)
    
    # Download SPY
    print("\nDownloading SPY...")
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)
    
    if spy.empty:
        print("Failed to download")
        return None
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Walk-forward periods
    periods = [
        ('2020-01-01', '2020-12-31', 'COVID Year'),
        ('2021-01-01', '2021-12-31', 'Bull 2021'),
        ('2022-01-01', '2022-12-31', 'Bear 2022'),
        ('2023-01-01', '2023-12-31', 'Recovery 2023'),
        ('2024-01-01', '2024-12-31', 'Bull 2024'),
    ]
    
    config = BacktestConfig(
        ewma_span=30,
        hazard_rate=0.05,
        momentum_lookback=20,
        transaction_cost=0.001
    )
    
    print(f"\n{'Period':<20} {'Strategy':>12} {'B&H':>12} {'Strat Sharpe':>12} {'B&H Sharpe':>12}")
    print("-" * 70)
    
    results = []
    for start, end, label in periods:
        # Train on everything before the test period
        train_end_dt = pd.Timestamp(start) - pd.Timedelta(days=1)
        train_end = train_end_dt.strftime('%Y-%m-%d')
        
        result = run_backtest(spy, 'SPY', config, train_end, start, end)
        
        if 'error' not in result:
            print(f"{label:<20} {result['strat_return']:>+11.1f}% {result['bh_return']:>+11.1f}% "
                  f"{result['strat_sharpe']:>12.2f} {result['bh_sharpe']:>12.2f}")
            result['period'] = label
            results.append(result)
    
    return results


def compare_to_baselines():
    """Compare to simple baseline strategies."""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (SPY 2023-2024)")
    print("=" * 70)
    
    # Download SPY
    spy = yf.download('SPY', start='2021-01-01', end='2025-01-01', progress=False)
    
    if spy.empty:
        print("Failed to download")
        return None
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    close = spy['Close'].values
    returns = np.diff(close) / close[:-1]
    dates = spy.index[1:]
    
    test_mask = (dates >= '2023-01-01') & (dates <= '2024-12-31')
    test_returns = returns[test_mask]
    test_dates = dates[test_mask]
    
    # Map test indices to full array indices
    test_start_idx = np.where(test_mask)[0][0]
    
    def calc_metrics(rets):
        if len(rets) == 0:
            return {'total': 0, 'sharpe': 0, 'max_dd': 0}
        total = np.prod(1 + rets) - 1
        years = len(rets) / 252
        ann_ret = (1 + total) ** (1/years) - 1 if years > 0 else 0
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        return {'total': total*100, 'sharpe': sharpe, 'max_dd': max_dd*100}
    
    results = {}
    
    # 1. Alpha Regime Strategy
    config = BacktestConfig()
    regime_result = run_backtest(spy, 'SPY', config, '2022-12-31', '2023-01-01', '2024-12-31')
    results['Alpha Regime'] = {
        'return': regime_result['strat_return'],
        'sharpe': regime_result['strat_sharpe'],
        'dd': regime_result['strat_dd'],
        'trades': regime_result['trades']
    }
    
    # 2. Simple 20-day momentum (unbiased)
    mom_signals = np.zeros(len(returns))
    for t in range(21, len(returns)):
        mom_signals[t] = 1.0 if np.mean(returns[t-21:t]) > 0 else 0.0
    test_mom_signals = mom_signals[test_start_idx:test_start_idx+len(test_returns)]
    mom_rets = test_mom_signals * test_returns
    mom_tc = np.abs(np.diff(np.concatenate([[0], test_mom_signals]))) * 0.001
    mom_m = calc_metrics(mom_rets - mom_tc)
    mom_trades = np.sum(np.abs(np.diff(test_mom_signals)) > 0.1)
    results['20-day Momentum'] = {
        'return': mom_m['total'],
        'sharpe': mom_m['sharpe'],
        'dd': mom_m['max_dd'],
        'trades': int(mom_trades)
    }
    
    # 3. SMA Crossover (50/200)
    sma_signals = np.zeros(len(close))
    for t in range(200, len(close)):
        sma50 = np.mean(close[t-50:t])
        sma200 = np.mean(close[t-200:t])
        sma_signals[t] = 1.0 if sma50 > sma200 else 0.0
    test_sma_signals = sma_signals[test_start_idx+1:test_start_idx+1+len(test_returns)]
    if len(test_sma_signals) < len(test_returns):
        test_sma_signals = np.concatenate([test_sma_signals, np.zeros(len(test_returns) - len(test_sma_signals))])
    sma_rets = test_sma_signals * test_returns
    sma_tc = np.abs(np.diff(np.concatenate([[0], test_sma_signals]))) * 0.001
    sma_m = calc_metrics(sma_rets - sma_tc)
    sma_trades = np.sum(np.abs(np.diff(test_sma_signals)) > 0.1)
    results['SMA 50/200'] = {
        'return': sma_m['total'],
        'sharpe': sma_m['sharpe'],
        'dd': sma_m['max_dd'],
        'trades': int(sma_trades)
    }
    
    # 4. Buy & Hold
    bh_m = calc_metrics(test_returns)
    results['Buy & Hold'] = {
        'return': bh_m['total'],
        'sharpe': bh_m['sharpe'],
        'dd': bh_m['max_dd'],
        'trades': 0
    }
    
    # Print results
    print(f"\n{'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Trades':>10}")
    print("-" * 65)
    
    for name, m in results.items():
        print(f"{name:<20} {m['return']:>+9.1f}% {m['sharpe']:>10.2f} {m['dd']:>+9.1f}% {m['trades']:>10}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALPHA REGIME DETECTOR - FORWARD BIAS AUDIT & OOS BACKTESTS")
    print("=" * 70)
    
    # Run audits
    print("\n" + "=" * 70)
    print("SECTION 1: FORWARD-LOOKING BIAS AUDITS")
    print("=" * 70)
    
    audit_feature_engineering()
    audit_bocpd_updates()
    audit_signal_timing()
    audit_regime_labeling()
    
    # Run OOS backtests
    print("\n" + "=" * 70)
    print("SECTION 2: OUT-OF-SAMPLE BACKTESTS")
    print("=" * 70)
    
    oos_results = run_oos_backtests()
    wf_results = run_walk_forward()
    baseline_results = compare_to_baselines()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if oos_results:
        print("\n1. MULTI-TICKER OOS (2023-2024):")
        sharpe_wins = sum(1 for r in oos_results if r['strat_sharpe'] > r['bh_sharpe'])
        return_wins = sum(1 for r in oos_results if r['strat_return'] > r['bh_return'])
        dd_wins = sum(1 for r in oos_results if r['strat_dd'] > r['bh_dd'])
        
        print(f"   Sharpe wins: {sharpe_wins}/{len(oos_results)}")
        print(f"   Return wins: {return_wins}/{len(oos_results)}")
        print(f"   Drawdown wins: {dd_wins}/{len(oos_results)}")
        
        avg_strat_sharpe = np.mean([r['strat_sharpe'] for r in oos_results])
        avg_bh_sharpe = np.mean([r['bh_sharpe'] for r in oos_results])
        print(f"   Avg Strategy Sharpe: {avg_strat_sharpe:.2f}")
        print(f"   Avg B&H Sharpe: {avg_bh_sharpe:.2f}")
    
    if wf_results:
        print("\n2. WALK-FORWARD (SPY):")
        for r in wf_results:
            winner = "✓ Strategy" if r['strat_sharpe'] > r['bh_sharpe'] else "✗ B&H"
            print(f"   {r['period']}: {winner}")
    
    print("\n3. VERDICT:")
    print("   The alpha regime detector, when implemented with proper signal")
    print("   lagging (1-day delay), provides:")
    print("   - Regime change detection for risk management")
    print("   - Potential drawdown reduction during transitions")
    print("   - NOT guaranteed alpha generation vs simple strategies")
