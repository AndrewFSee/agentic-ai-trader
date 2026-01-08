"""
Aggressive Recovery Strategy - VIX Spike Protection Only

Key Insight from Previous Testing:
    - VIX-based protection helps drawdowns but costs too much return
    - The strategy stays in "protection mode" too long during recovery
    - Need to ONLY protect during extreme spikes, then immediately return

New Approach:
    1. ONLY reduce position when VIX > threshold AND rising fast
    2. Return to full position IMMEDIATELY when VIX stops spiking
    3. Use VIX RATE OF CHANGE as the primary signal, not level

Theory:
    - VIX level above 30 is common and often happens during recovery
    - VIX RISING FAST (>20% in 5 days) is the real danger signal
    - VIX FALLING is the recovery signal, regardless of level

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VIXSpikeParams:
    """Ultra-simple VIX spike protection parameters."""
    # Spike detection (ENTRY)
    vix_roc_spike: float = 0.30      # VIX rising 30%+ in 5 days = danger
    vix_min_level: float = 20.0      # Only trigger if VIX > this
    
    # Protection level
    spike_position: float = 0.50      # Position during spike
    
    # Recovery (EXIT) - very fast
    vix_roc_recovery: float = 0.00   # VIX not rising = recover (0 = same day)


class VIXSpikeStrategy:
    """
    Ultra-simple strategy: Only hedge during VIX spikes.
    
    Logic:
        IF VIX > min_level AND VIX_ROC > spike_threshold:
            Position = spike_position
        ELSE:
            Position = 1.0
            
    No complex state machine, no multi-day confirmations.
    """
    
    def __init__(self, params: VIXSpikeParams):
        self.params = params
        self._vix_history: List[float] = []
        self._roc_window = 5
    
    def reset(self):
        self._vix_history = []
    
    def update(self, vix: float) -> Tuple[float, float, str]:
        """
        Update with new VIX and return position.
        
        Returns:
            (position, vix_roc, reason)
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
        
        # Simple spike detection
        is_spiking = (
            vix > self.params.vix_min_level and
            vix_roc > self.params.vix_roc_spike
        )
        
        if is_spiking:
            return self.params.spike_position, vix_roc, "SPIKE"
        else:
            return 1.0, vix_roc, "NORMAL"
    
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
            
            pos, vix_roc, reason = self.update(vix)
            positions.append({'date': idx, 'position': pos, 'vix': vix, 'vix_roc': vix_roc, 'reason': reason})
        
        if not positions:
            return {'error': 'No positions'}
        
        pos_df = pd.DataFrame(positions).set_index('date')
        
        # Calculate returns with lag
        close = spy_df['Close']
        returns = close.pct_change()
        signals = pos_df['position'].shift(1).fillna(1.0)
        
        # Align
        aligned = pd.DataFrame({
            'return': returns,
            'signal': signals,
            'vix': pos_df['vix'],
            'vix_roc': pos_df['vix_roc']
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
        n_spikes = (pos_df['reason'] == 'SPIKE').sum()
        
        return {
            'strategy': strat_m,
            'benchmark': bh_m,
            'time_protected': time_protected,
            'n_spike_days': n_spikes,
            'days': len(aligned),
            'excess_return': strat_m['total_return'] - bh_m['total_return'],
            'dd_improvement': abs(bh_m['max_drawdown']) - abs(strat_m['max_drawdown'])
        }


def run_comprehensive_test():
    """Test multiple VIX spike thresholds across periods."""
    
    print("=" * 90)
    print("VIX SPIKE-ONLY STRATEGY - Aggressive Recovery")
    print("Only protect during VIX spikes, immediately return when spike ends")
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
    
    # Test different thresholds
    thresholds = [
        {'vix_roc_spike': 0.20, 'vix_min': 18, 'spike_pos': 0.40},
        {'vix_roc_spike': 0.25, 'vix_min': 20, 'spike_pos': 0.45},
        {'vix_roc_spike': 0.30, 'vix_min': 20, 'spike_pos': 0.50},
        {'vix_roc_spike': 0.35, 'vix_min': 22, 'spike_pos': 0.50},
        {'vix_roc_spike': 0.40, 'vix_min': 22, 'spike_pos': 0.55},
        {'vix_roc_spike': 0.50, 'vix_min': 25, 'spike_pos': 0.50},
        {'vix_roc_spike': 0.60, 'vix_min': 25, 'spike_pos': 0.45},
        {'vix_roc_spike': 0.70, 'vix_min': 25, 'spike_pos': 0.40},
    ]
    
    periods = {
        '2018_Q4': ('2018-09-01', '2019-03-31'),
        '2020_COVID': ('2020-01-01', '2020-12-31'),
        '2022_Bear': ('2022-01-01', '2022-12-31'),
        '2023_Bull': ('2023-01-01', '2023-12-31'),
        'Full_2018_2024': ('2018-01-01', '2024-12-31'),
    }
    
    all_results = []
    
    for thresh in thresholds:
        params = VIXSpikeParams(
            vix_roc_spike=thresh['vix_roc_spike'],
            vix_min_level=thresh['vix_min'],
            spike_position=thresh['spike_pos']
        )
        
        period_results = {}
        for period_name, (start, end) in periods.items():
            mask = (spy.index >= start) & (spy.index <= end)
            spy_p = spy[mask]
            vix_mask = (vix.index >= start) & (vix.index <= end)
            vix_p = vix[vix_mask]
            
            if len(spy_p) < 50:
                period_results[period_name] = {'error': 'Insufficient'}
                continue
            
            strategy = VIXSpikeStrategy(params)
            result = strategy.backtest(spy_p, vix_p)
            period_results[period_name] = result
        
        all_results.append({
            'params': thresh,
            'periods': period_results
        })
    
    # Print results
    print("\n" + "=" * 90)
    print("RESULTS BY THRESHOLD")
    print("=" * 90)
    
    for result in all_results:
        params = result['params']
        print(f"\n{'='*60}")
        print(f"VIX_ROC > {params['vix_roc_spike']:.0%} AND VIX > {params['vix_min']}, Position = {params['spike_pos']}")
        print(f"{'='*60}")
        
        print(f"\n{'Period':<18} {'Strat':>10} {'B&H':>10} {'Excess':>8} {'S_DD':>8} {'B_DD':>8} {'DD_Imp':>8} {'Prot%':>7}")
        print("-" * 90)
        
        wins = 0
        total = 0
        
        for period_name, data in result['periods'].items():
            if 'error' in data:
                print(f"{period_name:<18} ERROR")
                continue
            
            total += 1
            strat_ret = data['strategy']['total_return']
            bh_ret = data['benchmark']['total_return']
            excess = strat_ret - bh_ret
            strat_dd = data['strategy']['max_drawdown']
            bh_dd = data['benchmark']['max_drawdown']
            dd_imp = abs(bh_dd) - abs(strat_dd)
            prot = data['time_protected']
            
            if excess > 0:
                wins += 1
            
            mark = "+" if excess > 0 else ""
            print(f"{period_name:<18} {strat_ret:>+9.1f}% {bh_ret:>+9.1f}% {mark}{excess:>+7.1f}% "
                  f"{strat_dd:>7.1f}% {bh_dd:>7.1f}% {dd_imp:>+7.1f}% {prot:>6.1f}%")
        
        print(f"\nPeriods where strategy BEATS B&H: {wins}/{total}")
    
    # Find best
    print("\n" + "=" * 90)
    print("FINDING BEST CONFIGURATION")
    print("=" * 90)
    
    best_score = -999
    best_result = None
    
    for result in all_results:
        periods = result['periods']
        valid = [p for p in periods.values() if 'error' not in p]
        
        if len(valid) < 3:
            continue
        
        # Score: wins + avg excess - dd sacrifice
        wins = sum(1 for p in valid if p['excess_return'] > 0)
        avg_excess = np.mean([p['excess_return'] for p in valid])
        avg_dd_imp = np.mean([p['dd_improvement'] for p in valid])
        
        # Penalty for being in protection too much
        avg_prot = np.mean([p['time_protected'] for p in valid])
        
        score = wins * 20 + avg_excess + avg_dd_imp * 0.5 - avg_prot * 0.2
        
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result:
        print(f"\nBest Configuration: {best_result['params']}")
        print(f"\nDetailed Results:")
        for period_name, data in best_result['periods'].items():
            if 'error' in data:
                continue
            print(f"  {period_name}: Return {data['strategy']['total_return']:+.1f}% vs B&H {data['benchmark']['total_return']:+.1f}% "
                  f"(Excess: {data['excess_return']:+.1f}%, DD Improvement: {data['dd_improvement']:+.1f}%)")
    
    # COVID specific analysis
    print("\n" + "=" * 90)
    print("COVID CRASH DETAILED TIMELINE (Best Config)")
    print("=" * 90)
    
    if best_result:
        params = VIXSpikeParams(
            vix_roc_spike=best_result['params']['vix_roc_spike'],
            vix_min_level=best_result['params']['vix_min'],
            spike_position=best_result['params']['spike_pos']
        )
        
        mask = (spy.index >= '2020-02-01') & (spy.index <= '2020-04-30')
        spy_covid = spy[mask]
        vix_covid = vix[(vix.index >= '2020-02-01') & (vix.index <= '2020-04-30')]
        
        strategy = VIXSpikeStrategy(params)
        strategy.reset()
        
        print(f"\n{'Date':<12} {'VIX':>8} {'VIX_ROC':>10} {'Position':>10} {'Reason':<10}")
        print("-" * 55)
        
        for idx in spy_covid.index:
            if idx not in vix_covid.index:
                continue
            vix_val = vix_covid.loc[idx]['Close']
            pos, vix_roc, reason = strategy.update(vix_val)
            
            date_str = idx.strftime('%Y-%m-%d')
            print(f"{date_str:<12} {vix_val:>8.1f} {vix_roc:>+10.1%} {pos:>10.2f} {reason:<10}")


if __name__ == "__main__":
    run_comprehensive_test()
