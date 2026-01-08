"""
Hybrid Strategy: VIX Spike Detection + Minimum Hold Period

Key Insight from Previous Testing:
    - Pure VIX spike strategy whipsaws during volatile periods (goes in/out too fast)
    - But we need aggressive recovery to capture the rebound
    
New Approach:
    1. Enter protection on VIX spike (ROC > threshold AND level > min)
    2. Stay in protection for MINIMUM X days (avoid whipsaw)
    3. Exit protection when VIX level drops below recovery threshold
    4. Weight position by VIX level (higher VIX = lower position)

This should:
    - Avoid the whipsaw during the crash (minimum hold)
    - Still capture the recovery once things calm down
    - Provide better protection during sustained high VIX periods

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HybridParams:
    """Parameters for hybrid strategy."""
    # Entry: VIX spike detection
    vix_roc_spike: float = 0.30      # VIX rising 30%+ = spike
    vix_min_entry: float = 20.0      # Only enter if VIX > this
    
    # Hold period (anti-whipsaw)
    min_hold_days: int = 5           # Stay in protection at least this many days
    
    # Exit: VIX level recovery
    vix_recovery_level: float = 25.0 # Exit protection when VIX drops below this
    
    # Position sizing (VIX-weighted)
    base_protection_pos: float = 0.50  # Base position during protection
    vix_scale_factor: float = 0.01     # Reduce position by 1% per VIX point above 30


class HybridVIXStrategy:
    """
    Hybrid strategy with spike entry, minimum hold, and level-based exit.
    """
    
    def __init__(self, params: HybridParams):
        self.params = params
        self._vix_history: List[float] = []
        self._in_protection = False
        self._protection_days = 0
        self._roc_window = 5
    
    def reset(self):
        self._vix_history = []
        self._in_protection = False
        self._protection_days = 0
    
    def update(self, vix: float) -> Tuple[float, str]:
        """
        Update with new VIX and return position.
        """
        self._vix_history.append(vix)
        if len(self._vix_history) > 50:
            self._vix_history = self._vix_history[-50:]
        
        # Calculate VIX rate of change
        if len(self._vix_history) > self._roc_window:
            prev_vix = self._vix_history[-self._roc_window - 1]
            vix_roc = (vix - prev_vix) / max(prev_vix, 1.0)
        else:
            vix_roc = 0.0
        
        # Check for new spike (entry)
        is_new_spike = (
            not self._in_protection and
            vix > self.params.vix_min_entry and
            vix_roc > self.params.vix_roc_spike
        )
        
        if is_new_spike:
            self._in_protection = True
            self._protection_days = 0
        
        if self._in_protection:
            self._protection_days += 1
            
            # Check for exit conditions
            can_exit = self._protection_days >= self.params.min_hold_days
            should_exit = vix < self.params.vix_recovery_level
            
            if can_exit and should_exit:
                self._in_protection = False
                self._protection_days = 0
                return 1.0, "RECOVERED"
            
            # VIX-weighted position
            base_pos = self.params.base_protection_pos
            if vix > 30:
                # Reduce position further for very high VIX
                reduction = (vix - 30) * self.params.vix_scale_factor
                position = max(0.20, base_pos - reduction)
            else:
                position = base_pos
            
            return position, f"PROTECTED_D{self._protection_days}"
        
        # Normal mode
        return 1.0, "NORMAL"
    
    def backtest(
        self,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> Dict:
        """Run backtest."""
        self.reset()
        
        positions = []
        
        for idx in spy_df.index:
            if idx not in vix_df.index:
                continue
            
            vix = vix_df.loc[idx]['Close'] if 'Close' in vix_df.columns else vix_df.loc[idx]
            if pd.isna(vix):
                continue
            
            pos, reason = self.update(vix)
            positions.append({'date': idx, 'position': pos, 'vix': vix, 'reason': reason})
        
        if not positions:
            return {'error': 'No positions'}
        
        pos_df = pd.DataFrame(positions).set_index('date')
        
        # Calculate returns with lag
        close = spy_df['Close']
        returns = close.pct_change()
        signals = pos_df['position'].shift(1).fillna(1.0)
        
        aligned = pd.DataFrame({
            'return': returns,
            'signal': signals,
            'vix': pos_df['vix']
        }).dropna()
        
        if len(aligned) < 10:
            return {'error': 'Insufficient data'}
        
        # Strategy returns
        strat_returns = aligned['signal'] * aligned['return']
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
                'max_drawdown': max_dd * 100
            }
        
        strat_m = calc_metrics(strat_returns_net)
        bh_m = calc_metrics(aligned['return'])
        
        time_protected = (aligned['signal'] < 1.0).mean() * 100
        
        return {
            'strategy': strat_m,
            'benchmark': bh_m,
            'time_protected': time_protected,
            'days': len(aligned),
            'excess_return': strat_m['total_return'] - bh_m['total_return'],
            'dd_improvement': abs(bh_m['max_drawdown']) - abs(strat_m['max_drawdown'])
        }


def run_hybrid_test():
    """Test hybrid strategy with various parameters."""
    
    print("=" * 90)
    print("HYBRID VIX STRATEGY: Spike Entry + Minimum Hold + Level Exit")
    print("=" * 90)
    
    import yfinance as yf
    
    # Download data
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2017-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2017-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    print(f"Data: {len(spy)} bars")
    
    # Test different parameter combinations
    configs = [
        {'vix_roc': 0.25, 'vix_min': 18, 'min_hold': 3, 'vix_recov': 22, 'base_pos': 0.50},
        {'vix_roc': 0.25, 'vix_min': 20, 'min_hold': 5, 'vix_recov': 25, 'base_pos': 0.50},
        {'vix_roc': 0.30, 'vix_min': 20, 'min_hold': 3, 'vix_recov': 22, 'base_pos': 0.55},
        {'vix_roc': 0.30, 'vix_min': 20, 'min_hold': 5, 'vix_recov': 25, 'base_pos': 0.50},
        {'vix_roc': 0.30, 'vix_min': 20, 'min_hold': 7, 'vix_recov': 25, 'base_pos': 0.45},
        {'vix_roc': 0.35, 'vix_min': 22, 'min_hold': 5, 'vix_recov': 25, 'base_pos': 0.50},
        {'vix_roc': 0.35, 'vix_min': 22, 'min_hold': 7, 'vix_recov': 28, 'base_pos': 0.45},
        {'vix_roc': 0.40, 'vix_min': 25, 'min_hold': 5, 'vix_recov': 25, 'base_pos': 0.50},
        # Try very aggressive recovery
        {'vix_roc': 0.25, 'vix_min': 18, 'min_hold': 2, 'vix_recov': 20, 'base_pos': 0.55},
        # Try longer hold with lower position
        {'vix_roc': 0.30, 'vix_min': 20, 'min_hold': 10, 'vix_recov': 22, 'base_pos': 0.40},
    ]
    
    periods = {
        '2018_Q4': ('2018-09-01', '2019-03-31'),
        '2020_COVID': ('2020-01-01', '2020-12-31'),
        '2022_Bear': ('2022-01-01', '2022-12-31'),
        '2023_Bull': ('2023-01-01', '2023-12-31'),
        'Full_2018_2024': ('2018-01-01', '2024-12-31'),
    }
    
    all_results = []
    
    for config in configs:
        params = HybridParams(
            vix_roc_spike=config['vix_roc'],
            vix_min_entry=config['vix_min'],
            min_hold_days=config['min_hold'],
            vix_recovery_level=config['vix_recov'],
            base_protection_pos=config['base_pos']
        )
        
        period_results = {}
        for period_name, (start, end) in periods.items():
            mask = (spy.index >= start) & (spy.index <= end)
            spy_p = spy[mask]
            vix_mask = (vix.index >= start) & (vix.index <= end)
            vix_p = vix[vix_mask]
            
            if len(spy_p) < 50:
                continue
            
            strategy = HybridVIXStrategy(params)
            result = strategy.backtest(spy_p, vix_p)
            period_results[period_name] = result
        
        all_results.append({
            'config': config,
            'periods': period_results
        })
    
    # Print results
    print("\n" + "=" * 90)
    print("RESULTS BY CONFIGURATION")
    print("=" * 90)
    
    best_score = -999
    best_result = None
    
    for result in all_results:
        config = result['config']
        print(f"\n{'='*70}")
        print(f"ROC>{config['vix_roc']:.0%} VIX>{config['vix_min']} Hold={config['min_hold']}d Recov<{config['vix_recov']} Pos={config['base_pos']}")
        print(f"{'='*70}")
        
        print(f"\n{'Period':<18} {'Strat':>10} {'B&H':>10} {'Excess':>8} {'S_DD':>8} {'B_DD':>8} {'DD_Imp':>8}")
        print("-" * 80)
        
        wins = 0
        total = 0
        valid_results = []
        
        for period_name, data in result['periods'].items():
            if 'error' in data:
                print(f"{period_name:<18} ERROR")
                continue
            
            total += 1
            valid_results.append(data)
            strat_ret = data['strategy']['total_return']
            bh_ret = data['benchmark']['total_return']
            excess = strat_ret - bh_ret
            strat_dd = data['strategy']['max_drawdown']
            bh_dd = data['benchmark']['max_drawdown']
            dd_imp = abs(bh_dd) - abs(strat_dd)
            
            if excess > 0:
                wins += 1
            
            mark = "✓" if excess > 0 else " "
            print(f"{period_name:<18} {strat_ret:>+9.1f}% {bh_ret:>+9.1f}% {excess:>+7.1f}%{mark} "
                  f"{strat_dd:>7.1f}% {bh_dd:>7.1f}% {dd_imp:>+7.1f}%")
        
        print(f"\nWINS: {wins}/{total}")
        
        # Score this configuration
        if valid_results:
            avg_excess = np.mean([r['excess_return'] for r in valid_results])
            avg_dd_imp = np.mean([r['dd_improvement'] for r in valid_results])
            score = wins * 20 + avg_excess + avg_dd_imp * 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
    
    # Show best configuration
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION")
    print("=" * 90)
    
    if best_result:
        print(f"\nParameters: {best_result['config']}")
        
        # Show COVID timeline
        print("\n" + "=" * 70)
        print("COVID TIMELINE (Best Config)")
        print("=" * 70)
        
        params = HybridParams(
            vix_roc_spike=best_result['config']['vix_roc'],
            vix_min_entry=best_result['config']['vix_min'],
            min_hold_days=best_result['config']['min_hold'],
            vix_recovery_level=best_result['config']['vix_recov'],
            base_protection_pos=best_result['config']['base_pos']
        )
        
        mask = (spy.index >= '2020-02-01') & (spy.index <= '2020-05-31')
        spy_covid = spy[mask]
        vix_covid = vix[(vix.index >= '2020-02-01') & (vix.index <= '2020-05-31')]
        
        strategy = HybridVIXStrategy(params)
        strategy.reset()
        
        print(f"\n{'Date':<12} {'VIX':>8} {'Position':>10} {'Reason':<20}")
        print("-" * 55)
        
        for idx in spy_covid.index:
            if idx not in vix_covid.index:
                continue
            vix_val = vix_covid.loc[idx]['Close']
            pos, reason = strategy.update(vix_val)
            
            # Only print during interesting periods
            if pos < 1.0 or 'PROTECTED' in reason or reason == 'RECOVERED':
                date_str = idx.strftime('%Y-%m-%d')
                print(f"{date_str:<12} {vix_val:>8.1f} {pos:>10.2f} {reason:<20}")
        
        # Summary
        print("\n" + "=" * 90)
        print("FINAL VERDICT")
        print("=" * 90)
        
        full_period = best_result['periods'].get('Full_2018_2024', {})
        if full_period and 'error' not in full_period:
            strat_ret = full_period['strategy']['total_return']
            bh_ret = full_period['benchmark']['total_return']
            excess = strat_ret - bh_ret
            dd_imp = full_period['dd_improvement']
            
            print(f"\nFull Period (2018-2024):")
            print(f"  Strategy Return: {strat_ret:+.1f}%")
            print(f"  Buy & Hold:      {bh_ret:+.1f}%")
            print(f"  Excess Return:   {excess:+.1f}%")
            print(f"  DD Improvement:  {dd_imp:+.1f}%")
            
            if excess > 0:
                print(f"\n✓ STRATEGY BEATS BUY & HOLD - Ready for production!")
            else:
                print(f"\n✗ Strategy underperforms by {-excess:.1f}% over full period")
                print("   Consider: This may be acceptable if DD reduction is valuable")
                
                # Calculate risk-adjusted comparison
                sharpe_strat = full_period['strategy']['sharpe']
                sharpe_bh = full_period['benchmark']['sharpe']
                print(f"\n   Sharpe Comparison: Strategy {sharpe_strat:.2f} vs B&H {sharpe_bh:.2f}")
                if sharpe_strat > sharpe_bh:
                    print("   ✓ Better risk-adjusted returns!")


if __name__ == "__main__":
    run_hybrid_test()
