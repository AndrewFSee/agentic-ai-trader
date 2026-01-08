"""
BOCPD Trading Strategy Backtest

Uses change point detection to generate buy/sell signals:
1. Detect regime changes in real-time (no look-ahead)
2. Classify new regime as bullish/bearish based on early observations
3. Go long in bullish regimes, flat/short in bearish

Key: All signals are generated ONLINE - no future data used.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_returns: np.ndarray
    benchmark_returns: np.ndarray
    positions: np.ndarray
    change_points: List[int]
    regime_labels: List[str]
    
    @property
    def strategy_cumulative(self) -> np.ndarray:
        return np.cumprod(1 + self.strategy_returns) - 1
    
    @property
    def benchmark_cumulative(self) -> np.ndarray:
        return np.cumprod(1 + self.benchmark_returns) - 1
    
    def summary(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        strat = self.strategy_returns
        bench = self.benchmark_returns
        
        # Returns
        strat_total = (1 + strat).prod() - 1
        bench_total = (1 + bench).prod() - 1
        
        # Annualized (assuming daily)
        n_days = len(strat)
        strat_annual = (1 + strat_total) ** (252 / n_days) - 1
        bench_annual = (1 + bench_total) ** (252 / n_days) - 1
        
        # Volatility
        strat_vol = strat.std() * np.sqrt(252)
        bench_vol = bench.std() * np.sqrt(252)
        
        # Sharpe (assuming 0 risk-free rate)
        strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
        bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
        
        # Max drawdown
        strat_cum = np.cumprod(1 + strat)
        strat_peak = np.maximum.accumulate(strat_cum)
        strat_dd = (strat_cum / strat_peak - 1).min()
        
        bench_cum = np.cumprod(1 + bench)
        bench_peak = np.maximum.accumulate(bench_cum)
        bench_dd = (bench_cum / bench_peak - 1).min()
        
        # Win rate
        win_rate = (strat > 0).sum() / len(strat) if len(strat) > 0 else 0
        
        # Time in market
        time_in_market = (self.positions != 0).sum() / len(self.positions)
        
        return {
            'strategy_total_return': strat_total,
            'benchmark_total_return': bench_total,
            'strategy_annual_return': strat_annual,
            'benchmark_annual_return': bench_annual,
            'strategy_volatility': strat_vol,
            'benchmark_volatility': bench_vol,
            'strategy_sharpe': strat_sharpe,
            'benchmark_sharpe': bench_sharpe,
            'strategy_max_dd': strat_dd,
            'benchmark_max_dd': bench_dd,
            'win_rate': win_rate,
            'time_in_market': time_in_market,
            'n_change_points': len(self.change_points),
            'n_trades': np.sum(np.diff(self.positions) != 0)
        }


def classify_regime(
    observations: List[float],
    lookback: int = 5,
    threshold: float = 0.0
) -> str:
    """
    Classify regime as bullish/bearish based on recent observations.
    
    Uses the mean of recent observations to determine direction.
    """
    if len(observations) < lookback:
        recent = observations
    else:
        recent = observations[-lookback:]
    
    if not recent:
        return 'neutral'
    
    mean_return = np.mean(recent)
    
    if mean_return > threshold:
        return 'bullish'
    elif mean_return < -threshold:
        return 'bearish'
    else:
        return 'neutral'


def bocpd_strategy(
    returns: np.ndarray,
    hazard_rate: float = 0.01,
    regime_lookback: int = 5,
    min_regime_length: int = 5,
    long_only: bool = True
) -> BacktestResult:
    """
    BOCPD-based trading strategy.
    
    Logic:
    1. Run BOCPD online to detect regime changes
    2. After a change point, wait for regime_lookback observations
    3. Classify new regime based on mean of those observations
    4. Position: +1 for bullish, 0 or -1 for bearish (depending on long_only)
    
    Args:
        returns: Array of daily returns
        hazard_rate: BOCPD hazard rate (1/expected_regime_length)
        regime_lookback: Observations to use for regime classification
        min_regime_length: Minimum observations before allowing another change
        long_only: If True, go flat in bearish; if False, go short
        
    Returns:
        BacktestResult with strategy performance
    """
    n = len(returns)
    
    # Standardize returns for BOCPD
    returns_std = (returns - np.mean(returns)) / np.std(returns)
    
    # Initialize
    detector = BOCPD(
        hazard_rate=hazard_rate,
        mu0=0.0,
        kappa0=0.1,
        alpha0=2.0,
        beta0=1.0
    )
    
    positions = np.zeros(n)
    change_points = []
    regime_labels = []
    
    current_position = 0  # Start flat
    current_regime = 'neutral'
    regime_start = 0
    observations_in_regime = []
    
    for t in range(n):
        # Update BOCPD with standardized return
        result = detector.update(returns_std[t])
        observations_in_regime.append(returns[t])  # Track raw returns
        
        # Check for regime change
        is_change_point = False
        if t > 0:
            prev_map = detector.map_run_lengths[-2] if len(detector.map_run_lengths) > 1 else 0
            curr_map = detector.map_run_lengths[-1]
            
            # Change detected if MAP drops significantly
            if prev_map > 3 and curr_map <= 2 and (t - regime_start) >= min_regime_length:
                is_change_point = True
                change_points.append(t)
                regime_start = t
                observations_in_regime = [returns[t]]
        
        # Classify regime after enough observations
        if len(observations_in_regime) >= regime_lookback:
            new_regime = classify_regime(observations_in_regime, regime_lookback)
            
            if new_regime != current_regime:
                current_regime = new_regime
                regime_labels.append((t, new_regime))
                
                # Update position
                if new_regime == 'bullish':
                    current_position = 1
                elif new_regime == 'bearish':
                    current_position = -1 if not long_only else 0
                else:
                    current_position = 0
        
        positions[t] = current_position
    
    # Calculate strategy returns (position is for NEXT day's return)
    # Shift positions by 1 to avoid look-ahead
    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0  # No position on first day
    
    strategy_returns = positions_shifted * returns
    
    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=returns,
        positions=positions_shifted,
        change_points=change_points,
        regime_labels=regime_labels
    )


def run_backtest():
    """Run backtest on SPY."""
    print("=" * 70)
    print("BOCPD TRADING STRATEGY BACKTEST")
    print("=" * 70)
    
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    # Download SPY
    print("\nDownloading SPY data...")
    spy = yf.download("SPY", period="10y", progress=False)
    if spy.empty:
        print("Failed to download")
        return
    
    # Get returns
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]
    
    print(f"Data: {len(returns)} daily returns")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Strategy 1: Long-only, moderate regime changes
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 1: Long-Only, Hazard=1/100")
    print("-" * 70)
    
    result1 = bocpd_strategy(
        returns,
        hazard_rate=1/100,  # Expect ~100 day regimes
        regime_lookback=5,
        min_regime_length=10,
        long_only=True
    )
    
    metrics1 = result1.summary()
    print(f"\nPerformance:")
    print(f"  Strategy Total Return:  {metrics1['strategy_total_return']:>8.1%}")
    print(f"  Benchmark Total Return: {metrics1['benchmark_total_return']:>8.1%}")
    print(f"  Strategy Sharpe:        {metrics1['strategy_sharpe']:>8.2f}")
    print(f"  Benchmark Sharpe:       {metrics1['benchmark_sharpe']:>8.2f}")
    print(f"  Strategy Max DD:        {metrics1['strategy_max_dd']:>8.1%}")
    print(f"  Benchmark Max DD:       {metrics1['benchmark_max_dd']:>8.1%}")
    print(f"  Time in Market:         {metrics1['time_in_market']:>8.1%}")
    print(f"  Number of Trades:       {int(metrics1['n_trades']):>8d}")
    print(f"  Change Points Detected: {metrics1['n_change_points']:>8d}")
    
    # =========================================================================
    # Strategy 2: Long-Short
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 2: Long-Short, Hazard=1/100")
    print("-" * 70)
    
    result2 = bocpd_strategy(
        returns,
        hazard_rate=1/100,
        regime_lookback=5,
        min_regime_length=10,
        long_only=False
    )
    
    metrics2 = result2.summary()
    print(f"\nPerformance:")
    print(f"  Strategy Total Return:  {metrics2['strategy_total_return']:>8.1%}")
    print(f"  Benchmark Total Return: {metrics2['benchmark_total_return']:>8.1%}")
    print(f"  Strategy Sharpe:        {metrics2['strategy_sharpe']:>8.2f}")
    print(f"  Benchmark Sharpe:       {metrics2['benchmark_sharpe']:>8.2f}")
    print(f"  Strategy Max DD:        {metrics2['strategy_max_dd']:>8.1%}")
    print(f"  Benchmark Max DD:       {metrics2['benchmark_max_dd']:>8.1%}")
    
    # =========================================================================
    # Strategy 3: Faster regime detection
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 3: Long-Only, Faster Detection (Hazard=1/50)")
    print("-" * 70)
    
    result3 = bocpd_strategy(
        returns,
        hazard_rate=1/50,  # Faster regime changes
        regime_lookback=3,
        min_regime_length=5,
        long_only=True
    )
    
    metrics3 = result3.summary()
    print(f"\nPerformance:")
    print(f"  Strategy Total Return:  {metrics3['strategy_total_return']:>8.1%}")
    print(f"  Benchmark Total Return: {metrics3['benchmark_total_return']:>8.1%}")
    print(f"  Strategy Sharpe:        {metrics3['strategy_sharpe']:>8.2f}")
    print(f"  Benchmark Sharpe:       {metrics3['benchmark_sharpe']:>8.2f}")
    print(f"  Strategy Max DD:        {metrics3['strategy_max_dd']:>8.1%}")
    print(f"  Benchmark Max DD:       {metrics3['benchmark_max_dd']:>8.1%}")
    print(f"  Time in Market:         {metrics3['time_in_market']:>8.1%}")
    print(f"  Number of Trades:       {int(metrics3['n_trades']):>8d}")
    
    # =========================================================================
    # Alternative: Volatility-based regime
    # =========================================================================
    print("\n" + "-" * 70)
    print("STRATEGY 4: Volatility Regime (Exit during high vol)")
    print("-" * 70)
    
    result4 = volatility_regime_strategy(returns, hazard_rate=1/50)
    metrics4 = result4.summary()
    
    print(f"\nPerformance:")
    print(f"  Strategy Total Return:  {metrics4['strategy_total_return']:>8.1%}")
    print(f"  Benchmark Total Return: {metrics4['benchmark_total_return']:>8.1%}")
    print(f"  Strategy Sharpe:        {metrics4['strategy_sharpe']:>8.2f}")
    print(f"  Benchmark Sharpe:       {metrics4['benchmark_sharpe']:>8.2f}")
    print(f"  Strategy Max DD:        {metrics4['strategy_max_dd']:>8.1%}")
    print(f"  Benchmark Max DD:       {metrics4['benchmark_max_dd']:>8.1%}")
    print(f"  Time in Market:         {metrics4['time_in_market']:>8.1%}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Trades':>8}")
    print("-" * 70)
    print(f"{'Buy & Hold':<30} {metrics1['benchmark_total_return']:>9.1%} {metrics1['benchmark_sharpe']:>10.2f} {metrics1['benchmark_max_dd']:>9.1%} {'N/A':>8}")
    print(f"{'1. Long-Only (H=1/100)':<30} {metrics1['strategy_total_return']:>9.1%} {metrics1['strategy_sharpe']:>10.2f} {metrics1['strategy_max_dd']:>9.1%} {int(metrics1['n_trades']):>8}")
    print(f"{'2. Long-Short (H=1/100)':<30} {metrics2['strategy_total_return']:>9.1%} {metrics2['strategy_sharpe']:>10.2f} {metrics2['strategy_max_dd']:>9.1%} {int(metrics2['n_trades']):>8}")
    print(f"{'3. Long-Only (H=1/50)':<30} {metrics3['strategy_total_return']:>9.1%} {metrics3['strategy_sharpe']:>10.2f} {metrics3['strategy_max_dd']:>9.1%} {int(metrics3['n_trades']):>8}")
    print(f"{'4. Volatility Regime':<30} {metrics4['strategy_total_return']:>9.1%} {metrics4['strategy_sharpe']:>10.2f} {metrics4['strategy_max_dd']:>9.1%} {int(metrics4['n_trades']):>8}")
    
    # Show some recent signals
    print("\n" + "-" * 70)
    print("RECENT CHANGE POINTS (Strategy 1)")
    print("-" * 70)
    
    for cp in result1.change_points[-10:]:
        if cp < len(dates):
            # Find regime classification after this CP
            regime = 'unknown'
            for t, label in result1.regime_labels:
                if t >= cp:
                    regime = label
                    break
            print(f"  {dates[cp].strftime('%Y-%m-%d')}: New regime -> {regime}")


def volatility_regime_strategy(
    returns: np.ndarray,
    hazard_rate: float = 0.02
) -> BacktestResult:
    """
    Strategy that exits during high volatility regimes.
    
    Uses absolute returns as volatility proxy.
    """
    n = len(returns)
    
    # Use absolute returns as volatility proxy
    vol_proxy = np.abs(returns)
    vol_std = (vol_proxy - np.mean(vol_proxy)) / np.std(vol_proxy)
    
    detector = BOCPD(
        hazard_rate=hazard_rate,
        mu0=0.0,
        kappa0=0.1,
        alpha0=2.0,
        beta0=1.0
    )
    
    positions = np.zeros(n)
    change_points = []
    regime_labels = []
    
    current_position = 1  # Start long
    current_regime = 'low_vol'
    regime_start = 0
    observations_in_regime = []
    
    for t in range(n):
        result = detector.update(vol_std[t])
        observations_in_regime.append(vol_proxy[t])
        
        # Check for regime change
        if t > 0:
            prev_map = detector.map_run_lengths[-2] if len(detector.map_run_lengths) > 1 else 0
            curr_map = detector.map_run_lengths[-1]
            
            if prev_map > 3 and curr_map <= 2 and (t - regime_start) >= 5:
                change_points.append(t)
                regime_start = t
                observations_in_regime = [vol_proxy[t]]
        
        # Classify volatility regime
        if len(observations_in_regime) >= 3:
            mean_vol = np.mean(observations_in_regime[-5:])
            overall_mean_vol = np.mean(vol_proxy[:t+1]) if t > 0 else mean_vol
            
            if mean_vol > overall_mean_vol * 1.5:
                new_regime = 'high_vol'
            else:
                new_regime = 'low_vol'
            
            if new_regime != current_regime:
                current_regime = new_regime
                regime_labels.append((t, new_regime))
                
                # Exit during high vol
                current_position = 0 if new_regime == 'high_vol' else 1
        
        positions[t] = current_position
    
    # Shift positions
    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0
    
    strategy_returns = positions_shifted * returns
    
    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=returns,
        positions=positions_shifted,
        change_points=change_points,
        regime_labels=regime_labels
    )


if __name__ == "__main__":
    run_backtest()
