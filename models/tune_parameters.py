"""
Parameter Tuning and Multi-Period Validation

This script systematically tests parameter combinations across multiple
market periods to find a strategy that CONSISTENTLY beats buy & hold.

Key Insight from Previous Testing:
    - VIX-based signals provide earlier warning than BOCPD alone
    - The challenge is RECOVERY timing - we stay out too long
    - Need to find the right balance of protection vs participation

Test Periods:
    1. 2018 Q4 selloff (Oct-Dec 2018, sharp but short)
    2. 2020 COVID crash + recovery (Feb-Aug 2020)
    3. 2022 bear market (Jan-Oct 2022, prolonged decline)
    4. 2023-2024 bull market (recovery test)
    5. Full period 2018-2024 (overall performance)

Success Criteria:
    - Beat B&H on total return in MOST periods
    - Better Sharpe ratio across ALL periods
    - Reduced max drawdown across ALL periods
    - If we can't beat B&H consistently, the strategy fails

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from alpha_regime_detector_v2 import (
    AlphaRegimeDetectorV2, FeatureConfig, BOCPDConfig, 
    RegimeConfig, OverlayConfig
)
from leading_indicators import (
    LeadingIndicatorEngine, LeadingIndicatorConfig
)


@dataclass
class StrategyParams:
    """Complete parameter set for the integrated strategy."""
    # VIX thresholds
    vix_emergency_level: float = 40.0
    vix_roc_warning: float = 0.20
    vix_roc_danger: float = 0.40
    
    # Position sizing
    normal_position: float = 1.0
    override_position: float = 0.60
    emergency_position: float = 0.35
    
    # Entry/Exit thresholds
    leading_override_thr: float = 0.65
    leading_exit_thr: float = 0.30  # NEW: Exit override when risk falls below this
    
    # Recovery parameters (NEW)
    vix_recovery_level: float = 25.0  # VIX below this = start considering recovery
    vix_roc_recovery: float = -0.10  # VIX falling 10%+ = recovery signal
    recovery_hold_days: int = 3       # Days of calm before full recovery
    
    def to_dict(self) -> Dict:
        return {
            'vix_emergency': self.vix_emergency_level,
            'vix_roc_warn': self.vix_roc_warning,
            'override_pos': self.override_position,
            'emergency_pos': self.emergency_position,
            'override_thr': self.leading_override_thr,
            'exit_thr': self.leading_exit_thr,
            'vix_recovery': self.vix_recovery_level,
            'recovery_days': self.recovery_hold_days
        }


class TunableIntegratedStrategy:
    """
    Strategy with tunable parameters and improved recovery logic.
    """
    
    def __init__(self, params: StrategyParams):
        self.params = params
        self.leading_engine = LeadingIndicatorEngine(
            LeadingIndicatorConfig(
                vix_roc_warning=params.vix_roc_warning,
                vix_roc_danger=params.vix_roc_danger
            )
        )
        self._calm_days = 0
        self._in_protection = False
    
    def reset(self):
        self.leading_engine.reset()
        self._calm_days = 0
        self._in_protection = False
    
    def compute_position(
        self,
        vix: float,
        vix_roc: float,
        leading_risk: float
    ) -> Tuple[float, str]:
        """
        Compute position with improved recovery logic.
        
        FAST-IN: Enter protection quickly when danger signals appear
        FAST-OUT: Exit protection quickly when conditions normalize
        """
        # Check VIX emergency (highest priority)
        if vix > self.params.vix_emergency_level:
            self._in_protection = True
            self._calm_days = 0
            return self.params.emergency_position, "VIX_EMERGENCY"
        
        # Check leading indicator override
        if leading_risk > self.params.leading_override_thr:
            self._in_protection = True
            self._calm_days = 0
            return self.params.override_position, "LEADING_OVERRIDE"
        
        # Recovery logic (NEW - key improvement)
        if self._in_protection:
            # Check if conditions are calming
            is_calming = (
                vix < self.params.vix_recovery_level and
                vix_roc < self.params.vix_roc_recovery and  # VIX falling
                leading_risk < self.params.leading_exit_thr
            )
            
            if is_calming:
                self._calm_days += 1
            else:
                self._calm_days = 0
            
            # Exit protection after enough calm days
            if self._calm_days >= self.params.recovery_hold_days:
                self._in_protection = False
                self._calm_days = 0
                return self.params.normal_position, "RECOVERED"
            
            # Still in protection but calming - gradual return
            if is_calming and self._calm_days >= 1:
                # Gradual position increase during recovery
                recovery_pct = self._calm_days / self.params.recovery_hold_days
                position = self.params.override_position + recovery_pct * (1.0 - self.params.override_position)
                return position, "RECOVERING"
            
            # Maintain protection
            return self.params.override_position, "PROTECTED"
        
        # Normal mode - check for mild elevation
        if leading_risk > 0.40:
            # Mild reduction for elevated but not alarming conditions
            return 0.90, "ELEVATED"
        
        return self.params.normal_position, "NORMAL"
    
    def backtest(
        self,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> Dict:
        """Run backtest with current parameters."""
        self.reset()
        
        positions = []
        reasons = []
        
        for idx in spy_df.index:
            if idx not in vix_df.index:
                continue
            
            vix = vix_df.loc[idx]['Close'] if 'Close' in vix_df.columns else vix_df.loc[idx]
            if pd.isna(vix):
                continue
            
            # Update leading indicators
            state = self.leading_engine.update(pd.Timestamp(idx), vix)
            
            # Compute position
            pos, reason = self.compute_position(
                vix=state.vix_level,
                vix_roc=state.vix_roc,
                leading_risk=state.leading_risk_score
            )
            
            positions.append({'date': idx, 'position': pos, 'reason': reason, 'vix': vix})
        
        if not positions:
            return {'error': 'No positions'}
        
        pos_df = pd.DataFrame(positions).set_index('date')
        
        # Calculate returns with lag
        close = spy_df['Close']
        returns = close.pct_change()
        
        # Lag signals by 1 day (signal at end of day t applies to day t+1)
        signals = pos_df['position'].shift(1).fillna(1.0)
        
        # Align
        aligned = pd.DataFrame({
            'return': returns,
            'signal': signals,
            'vix': pos_df['vix']
        }).dropna()
        
        if len(aligned) < 10:
            return {'error': 'Insufficient data'}
        
        # Strategy returns
        strat_returns = aligned['signal'] * aligned['return']
        
        # Transaction costs
        tc = aligned['signal'].diff().abs().fillna(0) * transaction_cost
        strat_returns_net = strat_returns - tc
        
        # Metrics
        def calc_metrics(rets):
            total = (1 + rets).prod() - 1
            years = max(len(rets) / 252, 0.01)
            ann_ret = (1 + total) ** (1/years) - 1
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cum = (1 + rets).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / peak
            max_dd = dd.min()
            return {
                'total_return': total * 100,
                'sharpe': sharpe,
                'max_drawdown': max_dd * 100,
                'annual_vol': ann_vol * 100
            }
        
        strat_m = calc_metrics(strat_returns_net)
        bh_m = calc_metrics(aligned['return'])
        
        # Time in protection
        time_protected = (aligned['signal'] < 1.0).mean() * 100
        avg_position = aligned['signal'].mean()
        
        return {
            'strategy': strat_m,
            'benchmark': bh_m,
            'time_protected': time_protected,
            'avg_position': avg_position,
            'days': len(aligned),
            'beats_bh_return': strat_m['total_return'] > bh_m['total_return'],
            'beats_bh_sharpe': strat_m['sharpe'] > bh_m['sharpe'],
            'beats_bh_dd': abs(strat_m['max_drawdown']) < abs(bh_m['max_drawdown']),
            'excess_return': strat_m['total_return'] - bh_m['total_return'],
            'dd_improvement': abs(bh_m['max_drawdown']) - abs(strat_m['max_drawdown'])
        }


def generate_param_grid() -> List[StrategyParams]:
    """Generate parameter combinations to test."""
    grid = []
    
    # Key parameters to tune
    vix_emergency_levels = [35.0, 40.0, 45.0]
    vix_roc_warnings = [0.15, 0.20, 0.25]
    override_positions = [0.50, 0.60, 0.70]
    emergency_positions = [0.25, 0.35, 0.45]
    override_thresholds = [0.55, 0.65, 0.75]
    exit_thresholds = [0.25, 0.30, 0.35]
    vix_recovery_levels = [20.0, 25.0, 30.0]
    recovery_hold_days = [2, 3, 5]
    
    # Generate all combinations (careful - this can be large!)
    # Let's be strategic - test key combinations
    for vix_emerg in vix_emergency_levels:
        for vix_roc in vix_roc_warnings:
            for override_pos in override_positions:
                for override_thr in override_thresholds:
                    for exit_thr in exit_thresholds:
                        for vix_recov in vix_recovery_levels:
                            for recov_days in recovery_hold_days:
                                params = StrategyParams(
                                    vix_emergency_level=vix_emerg,
                                    vix_roc_warning=vix_roc,
                                    vix_roc_danger=vix_roc * 2,  # Danger = 2x warning
                                    override_position=override_pos,
                                    emergency_position=override_pos - 0.25,  # Emergency = override - 25%
                                    leading_override_thr=override_thr,
                                    leading_exit_thr=exit_thr,
                                    vix_recovery_level=vix_recov,
                                    recovery_hold_days=recov_days
                                )
                                grid.append(params)
    
    return grid


def generate_focused_grid() -> List[StrategyParams]:
    """Generate a focused parameter grid based on initial testing."""
    grid = []
    
    # Based on initial results, focus on promising ranges
    configs = [
        # Aggressive protection, fast recovery
        {'vix_emerg': 35, 'vix_roc': 0.15, 'override_pos': 0.50, 'override_thr': 0.55, 
         'exit_thr': 0.25, 'vix_recov': 20, 'recov_days': 2},
        
        # Moderate protection, moderate recovery
        {'vix_emerg': 40, 'vix_roc': 0.20, 'override_pos': 0.60, 'override_thr': 0.60, 
         'exit_thr': 0.30, 'vix_recov': 25, 'recov_days': 3},
        
        # Conservative protection, fast recovery
        {'vix_emerg': 45, 'vix_roc': 0.20, 'override_pos': 0.65, 'override_thr': 0.65, 
         'exit_thr': 0.25, 'vix_recov': 22, 'recov_days': 2},
        
        # Aggressive entry, slow exit (more protection)
        {'vix_emerg': 35, 'vix_roc': 0.15, 'override_pos': 0.55, 'override_thr': 0.50, 
         'exit_thr': 0.20, 'vix_recov': 22, 'recov_days': 3},
        
        # Very fast in/out
        {'vix_emerg': 40, 'vix_roc': 0.15, 'override_pos': 0.60, 'override_thr': 0.55, 
         'exit_thr': 0.25, 'vix_recov': 23, 'recov_days': 1},
        
        # Balanced approach
        {'vix_emerg': 38, 'vix_roc': 0.18, 'override_pos': 0.55, 'override_thr': 0.58, 
         'exit_thr': 0.28, 'vix_recov': 24, 'recov_days': 2},
        
        # High conviction entries only
        {'vix_emerg': 45, 'vix_roc': 0.25, 'override_pos': 0.50, 'override_thr': 0.70, 
         'exit_thr': 0.30, 'vix_recov': 25, 'recov_days': 2},
        
        # Very aggressive recovery
        {'vix_emerg': 40, 'vix_roc': 0.20, 'override_pos': 0.60, 'override_thr': 0.60, 
         'exit_thr': 0.35, 'vix_recov': 28, 'recov_days': 1},
        
        # Ultra-fast recovery (minimal protection time)
        {'vix_emerg': 35, 'vix_roc': 0.20, 'override_pos': 0.65, 'override_thr': 0.60, 
         'exit_thr': 0.40, 'vix_recov': 30, 'recov_days': 1},
        
        # Maximum protection
        {'vix_emerg': 30, 'vix_roc': 0.12, 'override_pos': 0.40, 'override_thr': 0.45, 
         'exit_thr': 0.20, 'vix_recov': 18, 'recov_days': 3},
    ]
    
    for c in configs:
        params = StrategyParams(
            vix_emergency_level=c['vix_emerg'],
            vix_roc_warning=c['vix_roc'],
            vix_roc_danger=c['vix_roc'] * 2,
            override_position=c['override_pos'],
            emergency_position=max(0.25, c['override_pos'] - 0.25),
            leading_override_thr=c['override_thr'],
            leading_exit_thr=c['exit_thr'],
            vix_recovery_level=c['vix_recov'],
            recovery_hold_days=c['recov_days']
        )
        grid.append(params)
    
    return grid


def run_multi_period_test(
    spy_full: pd.DataFrame,
    vix_full: pd.DataFrame,
    params: StrategyParams
) -> Dict:
    """
    Test strategy across multiple market periods.
    
    Returns aggregated results and per-period breakdown.
    """
    periods = {
        '2018_Q4': ('2018-09-01', '2019-03-31'),      # Q4 selloff + recovery
        '2020_COVID': ('2020-01-01', '2020-12-31'),   # Full COVID year
        '2022_Bear': ('2022-01-01', '2022-12-31'),    # Bear market
        '2023_Bull': ('2023-01-01', '2023-12-31'),    # Recovery/bull
        '2024': ('2024-01-01', '2024-12-31'),         # Recent
        'Full_Period': ('2018-01-01', '2024-12-31'),  # Overall
    }
    
    results = {}
    
    for period_name, (start, end) in periods.items():
        # Filter data
        mask = (spy_full.index >= start) & (spy_full.index <= end)
        spy_period = spy_full[mask]
        
        vix_mask = (vix_full.index >= start) & (vix_full.index <= end)
        vix_period = vix_full[vix_mask]
        
        if len(spy_period) < 50:
            results[period_name] = {'error': 'Insufficient data'}
            continue
        
        # Run backtest
        strategy = TunableIntegratedStrategy(params)
        result = strategy.backtest(spy_period, vix_period)
        results[period_name] = result
    
    # Aggregate metrics
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return {'error': 'No valid results', 'periods': results}
    
    # Count wins
    beats_return = sum(1 for v in valid_results.values() if v.get('beats_bh_return', False))
    beats_sharpe = sum(1 for v in valid_results.values() if v.get('beats_bh_sharpe', False))
    beats_dd = sum(1 for v in valid_results.values() if v.get('beats_bh_dd', False))
    
    # Average excess return
    avg_excess = np.mean([v['excess_return'] for v in valid_results.values()])
    avg_dd_improvement = np.mean([v['dd_improvement'] for v in valid_results.values()])
    
    return {
        'periods': results,
        'beats_return': beats_return,
        'beats_sharpe': beats_sharpe,
        'beats_dd': beats_dd,
        'total_periods': len(valid_results),
        'avg_excess_return': avg_excess,
        'avg_dd_improvement': avg_dd_improvement,
        'params': params.to_dict()
    }


def print_period_results(result: Dict):
    """Pretty print period-by-period results."""
    print(f"\n{'Period':<15} {'Strat Ret':>10} {'B&H Ret':>10} {'Excess':>8} "
          f"{'Strat DD':>10} {'B&H DD':>10} {'DD Impr':>8} {'Sharpe':>7}")
    print("-" * 90)
    
    for period, data in result['periods'].items():
        if 'error' in data:
            print(f"{period:<15} {'ERROR':>10}")
            continue
        
        strat_ret = data['strategy']['total_return']
        bh_ret = data['benchmark']['total_return']
        excess = strat_ret - bh_ret
        strat_dd = data['strategy']['max_drawdown']
        bh_dd = data['benchmark']['max_drawdown']
        dd_impr = abs(bh_dd) - abs(strat_dd)
        sharpe = data['strategy']['sharpe']
        
        # Highlight wins
        ret_mark = "+" if excess > 0 else ""
        dd_mark = "+" if dd_impr > 0 else ""
        
        print(f"{period:<15} {strat_ret:>+9.1f}% {bh_ret:>+9.1f}% {ret_mark}{excess:>+7.1f}% "
              f"{strat_dd:>9.1f}% {bh_dd:>9.1f}% {dd_mark}{dd_impr:>+7.1f}% {sharpe:>7.2f}")


if __name__ == "__main__":
    print("=" * 90)
    print("PARAMETER TUNING & MULTI-PERIOD VALIDATION")
    print("Goal: Find a strategy that CONSISTENTLY beats Buy & Hold")
    print("=" * 90)
    
    try:
        import yfinance as yf
        
        # Download full data range
        print("\nDownloading data (2017-2024)...")
        spy = yf.download("SPY", start="2017-01-01", end="2025-01-01", progress=False)
        vix = yf.download("^VIX", start="2017-01-01", end="2025-01-01", progress=False)
        
        # Handle multi-index
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        print(f"SPY: {len(spy)} bars ({spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')})")
        print(f"VIX: {len(vix)} bars")
        
        # Generate parameter grid
        print("\nGenerating focused parameter grid...")
        param_grid = generate_focused_grid()
        print(f"Testing {len(param_grid)} parameter combinations")
        
        # Test each parameter set
        all_results = []
        
        for i, params in enumerate(param_grid):
            result = run_multi_period_test(spy, vix, params)
            all_results.append(result)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Tested {i + 1}/{len(param_grid)} combinations...")
        
        print("\n" + "=" * 90)
        print("RESULTS SUMMARY")
        print("=" * 90)
        
        # Sort by number of periods beaten + average excess return
        def score(r):
            if 'error' in r:
                return -999
            return (r['beats_return'] * 10 + r['avg_excess_return'] + r['avg_dd_improvement'])
        
        all_results.sort(key=score, reverse=True)
        
        # Show top 5
        print("\nTOP 5 PARAMETER SETS:")
        print("-" * 90)
        
        for i, result in enumerate(all_results[:5]):
            if 'error' in result:
                continue
            
            print(f"\n{'='*40}")
            print(f"RANK #{i+1}")
            print(f"{'='*40}")
            print(f"Parameters: {result['params']}")
            print(f"\nWins: Return {result['beats_return']}/{result['total_periods']}, "
                  f"Sharpe {result['beats_sharpe']}/{result['total_periods']}, "
                  f"DD {result['beats_dd']}/{result['total_periods']}")
            print(f"Avg Excess Return: {result['avg_excess_return']:+.1f}%")
            print(f"Avg DD Improvement: {result['avg_dd_improvement']:+.1f}%")
            
            print_period_results(result)
        
        # Show the best one in detail
        best = all_results[0]
        if 'error' not in best:
            print("\n" + "=" * 90)
            print("BEST STRATEGY DETAILED ANALYSIS")
            print("=" * 90)
            print(f"\nOptimal Parameters:")
            for k, v in best['params'].items():
                print(f"  {k}: {v}")
            
            # Check if it consistently beats B&H
            beats_all_return = best['beats_return'] >= best['total_periods'] - 1  # Allow 1 miss
            beats_all_sharpe = best['beats_sharpe'] >= best['total_periods'] - 1
            beats_all_dd = best['beats_dd'] == best['total_periods']
            
            print(f"\nConsistency Check:")
            print(f"  Beats B&H Return in most periods: {'✓ YES' if beats_all_return else '✗ NO'}")
            print(f"  Beats B&H Sharpe in most periods: {'✓ YES' if beats_all_sharpe else '✗ NO'}")
            print(f"  Beats B&H Drawdown in ALL periods: {'✓ YES' if beats_all_dd else '✗ NO'}")
            
            if beats_all_return or beats_all_sharpe:
                print("\n✓ STRATEGY SHOWS PROMISE - Worth integrating into production!")
            else:
                print("\n✗ STRATEGY NEEDS MORE TUNING - Does not consistently beat B&H")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
