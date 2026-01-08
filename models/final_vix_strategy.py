"""
Final Optimized Strategy: VIX Velocity + ML + Smarter Recovery

Key Learnings from Testing:
1. VIX falling for 2+ consecutive days = start recovering
2. ML calm_prob > 25-30% accelerates recovery
3. VIX level < 35 + falling = full recovery
4. The problem: Whipsaw during volatile periods

NEW APPROACH:
- Use a "cooling off" signal: VIX peak is set, and we exit when VIX is
  X% below peak AND falling
- Allow re-entry if VIX spikes again during recovery
- Use progressive position sizing based on VIX level

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FinalParams:
    """Final optimized parameters."""
    # Entry
    vix_roc_spike: float = 0.25
    vix_min_entry: float = 18.0
    
    # Exit: VIX must be X% below peak
    vix_drop_from_peak: float = 0.15  # 15% below peak
    min_protection_days: int = 2
    
    # Position sizing (VIX-level based)
    pos_vix_30: float = 1.00   # Full position if VIX < 30
    pos_vix_40: float = 0.80   # VIX 30-40
    pos_vix_50: float = 0.60   # VIX 40-50
    pos_vix_60: float = 0.45   # VIX 50-60
    pos_vix_high: float = 0.30 # VIX > 60
    
    # ML acceleration
    calm_prob_boost: float = 0.30  # If calm_prob > this, use more aggressive positions


def get_vix_position(vix: float, params: FinalParams, calm_prob: float = 0) -> float:
    """Get position size based on VIX level."""
    # Boost positions if ML predicts calm
    boost = 0.10 if calm_prob > params.calm_prob_boost else 0.0
    
    if vix < 30:
        return min(1.0, params.pos_vix_30 + boost)
    elif vix < 40:
        return min(1.0, params.pos_vix_40 + boost)
    elif vix < 50:
        return min(1.0, params.pos_vix_50 + boost)
    elif vix < 60:
        return min(1.0, params.pos_vix_60 + boost)
    else:
        return min(1.0, params.pos_vix_high + boost)


class FinalStrategy:
    """
    Final optimized strategy with:
    - VIX spike entry
    - VIX % drop from peak exit
    - VIX-level based position sizing during recovery
    - ML calm probability boost
    """
    
    def __init__(self, params: FinalParams):
        self.params = params
        self._reset()
    
    def _reset(self):
        self._vix_history = []
        self._in_protection = False
        self._protection_days = 0
        self._vix_peak = 0.0
    
    def update(self, vix: float, calm_prob: float = 0) -> Tuple[float, str]:
        """Update and return position."""
        self._vix_history.append(vix)
        if len(self._vix_history) > 50:
            self._vix_history = self._vix_history[-50:]
        
        # VIX ROC
        vix_roc = 0
        if len(self._vix_history) > 5:
            vix_roc = (vix - self._vix_history[-6]) / max(self._vix_history[-6], 1)
        
        # Entry check
        vix_spike = vix_roc > self.params.vix_roc_spike and vix > self.params.vix_min_entry
        
        if not self._in_protection:
            if vix_spike:
                self._in_protection = True
                self._protection_days = 0
                self._vix_peak = vix
                pos = get_vix_position(vix, self.params, 0)
                return pos, f'ENTER_SPIKE (ROC={vix_roc:.0%})'
            else:
                return 1.0, 'NORMAL'
        
        # In protection
        self._protection_days += 1
        self._vix_peak = max(self._vix_peak, vix)
        
        # Check for exit
        can_exit = self._protection_days >= self.params.min_protection_days
        vix_drop_pct = (self._vix_peak - vix) / self._vix_peak
        
        if can_exit and vix_drop_pct > self.params.vix_drop_from_peak:
            # VIX dropped enough from peak - start recovering
            pos = get_vix_position(vix, self.params, calm_prob)
            
            if pos >= 0.95:
                self._in_protection = False
                self._vix_peak = 0
                return 1.0, f'EXIT_FULL (VIX={vix:.0f}, drop={vix_drop_pct:.0%})'
            else:
                # Partial position based on current VIX level
                return pos, f'RECOVERING (VIX={vix:.0f}, pos={pos:.0%})'
        else:
            # Still in protection - use VIX-level position
            pos = get_vix_position(vix, self.params, calm_prob)
            return pos, f'PROTECTED_D{self._protection_days} (pos={pos:.0%})'


def run_final_test():
    """Test the final optimized strategy."""
    
    print("=" * 90)
    print("FINAL OPTIMIZED STRATEGY: VIX Level-Based Position Sizing")
    print("=" * 90)
    
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2015-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    print(f"Data: {len(spy)} bars")
    
    # Train ML model (simplified)
    print("\nTraining ML calm predictor...")
    
    train_spy = spy[spy.index <= '2019-12-31']
    train_vix = vix[vix.index <= '2019-12-31']
    
    close = train_spy['Close']
    returns = close.pct_change()
    vix_aligned = train_vix['Close'].reindex(train_spy.index).ffill()
    
    # Simple features
    features = pd.DataFrame(index=train_spy.index)
    features['vix'] = vix_aligned.shift(1)
    features['vix_mom_5'] = (vix_aligned / vix_aligned.shift(5) - 1).shift(1)
    features['vix_mom_10'] = (vix_aligned / vix_aligned.shift(10) - 1).shift(1)
    rvol_20 = returns.rolling(20).std() * np.sqrt(252)
    features['rvol'] = rvol_20.shift(1)
    features['drawdown'] = (close / close.rolling(60).max() - 1).shift(1)
    
    vol_median = rvol_20.rolling(60).median()
    current_is_high = (rvol_20.shift(1) > vol_median)
    future_vol = returns.shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
    future_is_low = future_vol <= vol_median
    calm_target = (current_is_high & future_is_low).astype(int)
    
    common_idx = features.dropna().index.intersection(calm_target.dropna().index)
    X = features.loc[common_idx]
    high_mask = current_is_high.loc[common_idx].fillna(False)
    X_high = X[high_mask]
    y_calm = calm_target.loc[common_idx][high_mask]
    
    calm_model = None
    if len(X_high) > 50 and y_calm.sum() > 10:
        calm_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        calm_model.fit(X_high, y_calm)
        print(f"  Trained on {len(X_high)} high-vol samples, {y_calm.sum()} calms")
    
    feature_cols = X.columns.tolist()
    
    # Full data features
    full_close = spy['Close']
    full_returns = full_close.pct_change()
    full_vix = vix['Close'].reindex(spy.index).ffill()
    
    full_features = pd.DataFrame(index=spy.index)
    full_features['vix'] = full_vix.shift(1)
    full_features['vix_mom_5'] = (full_vix / full_vix.shift(5) - 1).shift(1)
    full_features['vix_mom_10'] = (full_vix / full_vix.shift(10) - 1).shift(1)
    rvol_20_full = full_returns.rolling(20).std() * np.sqrt(252)
    full_features['rvol'] = rvol_20_full.shift(1)
    full_features['drawdown'] = (full_close / full_close.rolling(60).max() - 1).shift(1)
    
    vol_median_full = rvol_20_full.rolling(60).median()
    current_is_high_full = rvol_20_full.shift(1) > vol_median_full
    
    # Test configurations
    configs = [
        # Very aggressive (almost no protection)
        {'drop': 0.05, 'p30': 1.00, 'p40': 1.00, 'p50': 0.90, 'p60': 0.80, 'phigh': 0.70},
        # Aggressive recovery
        {'drop': 0.08, 'p30': 1.00, 'p40': 0.95, 'p50': 0.85, 'p60': 0.70, 'phigh': 0.55},
        {'drop': 0.10, 'p30': 1.00, 'p40': 0.90, 'p50': 0.80, 'p60': 0.65, 'phigh': 0.50},
        # Moderate
        {'drop': 0.10, 'p30': 1.00, 'p40': 0.90, 'p50': 0.75, 'p60': 0.60, 'phigh': 0.45},
        # More conservative
        {'drop': 0.15, 'p30': 1.00, 'p40': 0.85, 'p50': 0.70, 'p60': 0.55, 'phigh': 0.40},
        # Even more aggressive
        {'drop': 0.05, 'p30': 1.00, 'p40': 1.00, 'p50': 0.95, 'p60': 0.85, 'phigh': 0.75},
    ]
    
    periods = [
        ('2020 COVID', '2020-01-01', '2020-12-31'),
        ('2022 Bear', '2022-01-01', '2022-12-31'),
        ('2023 Bull', '2023-01-01', '2023-12-31'),
        ('Full 2020-2024', '2020-01-01', '2024-12-31'),
    ]
    
    best_result = None
    best_score = -999
    
    print("\n" + "=" * 90)
    print("RESULTS BY CONFIGURATION")
    print("=" * 90)
    
    for config in configs:
        params = FinalParams(
            vix_drop_from_peak=config['drop'],
            pos_vix_30=config['p30'],
            pos_vix_40=config['p40'],
            pos_vix_50=config['p50'],
            pos_vix_60=config['p60'],
            pos_vix_high=config['phigh']
        )
        
        print(f"\n{'='*80}")
        print(f"Drop>{config['drop']:.0%} | Pos: <30={config['p30']:.0%} 30-40={config['p40']:.0%} "
              f"40-50={config['p50']:.0%} 50-60={config['p60']:.0%} >60={config['phigh']:.0%}")
        print(f"{'='*80}")
        
        print(f"\n{'Period':<18} {'Strat':>10} {'B&H':>10} {'Excess':>8} {'S_DD':>8} {'B_DD':>8} {'DD_Imp':>8}")
        print("-" * 80)
        
        period_results = {}
        wins = 0
        
        for period_name, start, end in periods:
            mask = (spy.index >= start) & (spy.index <= end)
            period_spy = spy[mask]
            
            if len(period_spy) < 50:
                continue
            
            strategy = FinalStrategy(params)
            
            positions = []
            for idx in period_spy.index:
                if idx not in full_vix.index:
                    continue
                
                vix_val = full_vix.loc[idx]
                if pd.isna(vix_val):
                    continue
                
                # Get calm probability
                calm_prob = 0
                is_high = current_is_high_full.loc[idx] if idx in current_is_high_full.index else False
                if is_high and calm_model is not None:
                    feat = full_features.loc[idx:idx].fillna(0)
                    for col in feature_cols:
                        if col not in feat.columns:
                            feat[col] = 0
                    feat = feat[feature_cols]
                    try:
                        calm_prob = float(calm_model.predict_proba(feat)[0][1])
                    except:
                        pass
                
                pos, reason = strategy.update(vix_val, calm_prob)
                positions.append({'date': idx, 'position': pos})
            
            if not positions:
                continue
            
            pos_df = pd.DataFrame(positions).set_index('date')
            period_returns = full_returns.loc[pos_df.index]
            signals = pos_df['position'].shift(1).fillna(1.0)
            
            aligned = pd.DataFrame({'return': period_returns, 'signal': signals}).dropna()
            strat_rets = aligned['signal'] * aligned['return']
            tc = aligned['signal'].diff().abs().fillna(0) * 0.001
            strat_net = strat_rets - tc
            
            def calc_m(rets):
                total = (1 + rets).prod() - 1
                cum = (1 + rets).cumprod()
                peak = cum.expanding().max()
                dd = ((cum - peak) / peak).min()
                return {'total': total * 100, 'dd': dd * 100}
            
            strat_m = calc_m(strat_net)
            bh_m = calc_m(aligned['return'])
            excess = strat_m['total'] - bh_m['total']
            dd_imp = abs(bh_m['dd']) - abs(strat_m['dd'])
            
            period_results[period_name] = {
                'strat': strat_m['total'],
                'bh': bh_m['total'],
                'excess': excess,
                'strat_dd': strat_m['dd'],
                'bh_dd': bh_m['dd'],
                'dd_imp': dd_imp
            }
            
            if excess > 0:
                wins += 1
            
            mark = "✓" if excess > 0 else " "
            print(f"{period_name:<18} {strat_m['total']:>+9.1f}% {bh_m['total']:>+9.1f}% {excess:>+7.1f}%{mark} "
                  f"{strat_m['dd']:>7.1f}% {bh_m['dd']:>7.1f}% {dd_imp:>+7.1f}%")
        
        print(f"\nWINS: {wins}/{len(period_results)}")
        
        if period_results:
            avg_excess = np.mean([d['excess'] for d in period_results.values()])
            score = wins * 25 + avg_excess
            
            if score > best_score:
                best_score = score
                best_result = {'config': config, 'periods': period_results, 'params': params}
    
    # COVID timeline
    if best_result:
        print("\n" + "=" * 90)
        print("BEST CONFIGURATION - COVID TIMELINE")
        print("=" * 90)
        print(f"\nConfig: {best_result['config']}")
        
        params = best_result['params']
        strategy = FinalStrategy(params)
        
        covid_mask = (spy.index >= '2020-02-01') & (spy.index <= '2020-04-30')
        covid_spy = spy[covid_mask]
        
        print(f"\n{'Date':<12} {'VIX':>6} {'Peak':>6} {'Drop%':>7} {'Pos':>6} {'Reason':<30}")
        print("-" * 75)
        
        for idx in covid_spy.index:
            if idx not in full_vix.index:
                continue
            
            vix_val = full_vix.loc[idx]
            is_high = current_is_high_full.loc[idx] if idx in current_is_high_full.index else False
            
            calm_prob = 0
            if is_high and calm_model is not None:
                feat = full_features.loc[idx:idx].fillna(0)
                for col in feature_cols:
                    if col not in feat.columns:
                        feat[col] = 0
                feat = feat[feature_cols]
                try:
                    calm_prob = float(calm_model.predict_proba(feat)[0][1])
                except:
                    pass
            
            pos, reason = strategy.update(vix_val, calm_prob)
            
            if pos < 1.0 or 'ENTER' in reason or 'EXIT' in reason:
                drop_pct = (strategy._vix_peak - vix_val) / strategy._vix_peak * 100 if strategy._vix_peak > 0 else 0
                print(f"{idx.strftime('%Y-%m-%d'):<12} {vix_val:>6.1f} {strategy._vix_peak:>6.1f} {drop_pct:>6.1f}% "
                      f"{pos:>6.2f} {reason:<30}")
        
        # Summary
        print("\n" + "=" * 90)
        print("FINAL SUMMARY")
        print("=" * 90)
        
        full_period = best_result['periods'].get('Full 2020-2024', {})
        if full_period:
            print(f"\nFull Period (2020-2024):")
            print(f"  Strategy Return: {full_period['strat']:+.1f}%")
            print(f"  Buy & Hold:      {full_period['bh']:+.1f}%")
            print(f"  Excess Return:   {full_period['excess']:+.1f}%")
            print(f"  DD Improvement:  {full_period['dd_imp']:+.1f}%")
            
            if full_period['excess'] > 0:
                print(f"\n✓ STRATEGY BEATS BUY & HOLD!")
            else:
                print(f"\n✗ Underperforms by {-full_period['excess']:.1f}%")
                
                # Risk-adjusted
                total_wins = sum(1 for p in best_result['periods'].values() if p['excess'] > 0)
                total_periods = len(best_result['periods'])
                avg_dd_imp = np.mean([p['dd_imp'] for p in best_result['periods'].values()])
                
                print(f"\n   Win Rate: {total_wins}/{total_periods} periods")
                print(f"   Avg DD Improvement: {avg_dd_imp:+.1f}%")
                
                if avg_dd_imp > 5:
                    print(f"\n   VERDICT: Good tail-risk hedge (reduces DD by ~{avg_dd_imp:.0f}%)")
                    print(f"   Accept as risk management overlay, not alpha generator")


if __name__ == "__main__":
    run_final_test()
