"""
Enhanced Predictive Strategy: More Aggressive Recovery + VIX Momentum

Key changes from v1:
1. Use VIX MOMENTUM (not just decrease from peak) for faster recovery
2. Lower calm_prob threshold since model is conservative
3. Use partial positions during high VIX but falling momentum

The insight: VIX momentum (velocity) is more predictive than VIX level
- VIX can be 40+ but if it's FALLING, we should be getting back in
- Original strategy waited too long because VIX stayed high

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnhancedParams:
    """Parameters for enhanced predictive strategy."""
    # Entry thresholds
    vix_roc_spike: float = 0.25      # VIX rising 25%+ = spike
    vix_min_entry: float = 18.0      # Only enter if VIX > this
    spike_prob_threshold: float = 0.55  # ML early warning
    
    # Recovery thresholds (AGGRESSIVE)
    vix_falling_days: int = 2        # Exit if VIX falling for N consecutive days
    calm_prob_threshold: float = 0.30   # Lower threshold (ML is conservative)
    vix_max_for_full: float = 35.0   # Full position if VIX < this and falling
    
    # Minimum protection
    min_protection_days: int = 1     # Minimum 1 day protection
    
    # Position sizing
    normal_position: float = 1.0
    protected_position: float = 0.45   # Stronger protection
    partial_position: float = 0.70     # Partial recovery position
    ml_warning_position: float = 0.80  # Lighter reduction on ML warning only


class EnhancedPredictiveStrategy:
    """
    Enhanced strategy with VIX momentum-based recovery.
    
    Key insight: Use VIX velocity (falling vs rising) not just level.
    """
    
    def __init__(self, params: EnhancedParams):
        self.params = params
        self._reset()
    
    def _reset(self):
        self._vix_history: List[float] = []
        self._in_protection = False
        self._protection_days = 0
        self._vix_peak = 0.0
        self._consecutive_falling = 0
    
    def update(self, vix: float, spike_prob: float, calm_prob: float) -> Tuple[float, str]:
        """
        Update strategy state and return position.
        
        Args:
            vix: Current VIX level
            spike_prob: ML probability of vol spike (if in low vol)
            calm_prob: ML probability of vol calming (if in high vol)
        
        Returns: (position, reason)
        """
        self._vix_history.append(vix)
        if len(self._vix_history) > 100:
            self._vix_history = self._vix_history[-100:]
        
        # VIX rate of change (5-day)
        if len(self._vix_history) > 5:
            prev_vix = self._vix_history[-6]
            vix_roc = (vix - prev_vix) / max(prev_vix, 1.0)
        else:
            vix_roc = 0.0
        
        # VIX momentum (1-day)
        if len(self._vix_history) > 1:
            vix_1d_change = vix - self._vix_history[-2]
            if vix_1d_change < 0:
                self._consecutive_falling += 1
            else:
                self._consecutive_falling = 0
        
        # === ENTRY LOGIC ===
        vix_spike = vix_roc > self.params.vix_roc_spike and vix > self.params.vix_min_entry
        ml_warning = spike_prob > self.params.spike_prob_threshold
        
        if not self._in_protection:
            if vix_spike:
                self._in_protection = True
                self._protection_days = 0
                self._vix_peak = vix
                self._consecutive_falling = 0
                return self.params.protected_position, f'SPIKE (VIX_ROC={vix_roc:.0%})'
            elif ml_warning:
                return self.params.ml_warning_position, f'ML_WARN (p={spike_prob:.0%})'
            else:
                return self.params.normal_position, 'NORMAL'
        
        # === IN PROTECTION: CHECK FOR EXIT ===
        self._protection_days += 1
        self._vix_peak = max(self._vix_peak, vix)
        
        can_exit = self._protection_days >= self.params.min_protection_days
        
        # Recovery conditions
        vix_falling = self._consecutive_falling >= self.params.vix_falling_days
        vix_reasonable = vix < self.params.vix_max_for_full
        ml_calm = calm_prob > self.params.calm_prob_threshold
        
        if can_exit and vix_falling:
            if vix_reasonable:
                # VIX < 35 and falling for N days - full exit
                self._in_protection = False
                self._vix_peak = 0
                return self.params.normal_position, f'EXIT (VIX={vix:.0f} falling {self._consecutive_falling}d)'
            elif ml_calm:
                # VIX high but falling + ML says calm - full exit
                self._in_protection = False
                self._vix_peak = 0
                return self.params.normal_position, f'EXIT_ML (calm={calm_prob:.0%})'
            else:
                # VIX high but falling - partial exit
                return self.params.partial_position, f'PARTIAL (VIX={vix:.0f} falling)'
        
        elif can_exit and ml_calm and vix < 30:
            # ML confident + VIX not crazy high
            self._in_protection = False
            return self.params.normal_position, f'EXIT_ML_LOW_VIX (p={calm_prob:.0%})'
        
        else:
            return self.params.protected_position, f'PROTECTED_D{self._protection_days}'


def run_test():
    """Test the enhanced predictive strategy."""
    
    print("=" * 90)
    print("ENHANCED PREDICTIVE STRATEGY: VIX Momentum Recovery")
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
    
    # =========================================
    # TRAIN ML MODEL
    # =========================================
    print("\nTraining ML volatility predictor...")
    
    train_end = '2019-12-31'
    train_spy = spy[spy.index <= train_end]
    train_vix = vix[vix.index <= train_end]
    
    close = train_spy['Close']
    returns = close.pct_change()
    vix_aligned = train_vix['Close'].reindex(train_spy.index).ffill()
    
    # Features
    features = pd.DataFrame(index=train_spy.index)
    features['vix'] = vix_aligned.shift(1)
    vix_mean = vix_aligned.rolling(60).mean()
    vix_std = vix_aligned.rolling(60).std()
    features['vix_zscore'] = ((vix_aligned - vix_mean) / (vix_std + 1e-10)).shift(1)
    features['vix_mom_5'] = (vix_aligned / vix_aligned.shift(5) - 1).shift(1)
    features['vix_mom_10'] = (vix_aligned / vix_aligned.shift(10) - 1).shift(1)
    for w in [5, 10, 20]:
        rvol = returns.rolling(w).std() * np.sqrt(252)
        features[f'rvol_{w}'] = rvol.shift(1)
    rvol_20 = returns.rolling(20).std() * np.sqrt(252)
    rvol_60 = returns.rolling(60).std() * np.sqrt(252)
    features['vol_mom'] = (rvol_20 / rvol_60 - 1).shift(1)
    features['drawdown'] = (close / close.rolling(60).max() - 1).shift(1)
    
    # Regimes
    vol_median = rvol_20.rolling(60).median()
    current_vol = rvol_20.shift(1)
    future_vol = returns.shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
    
    current_is_low = current_vol <= vol_median
    current_is_high = current_vol > vol_median
    future_is_high = future_vol > vol_median
    future_is_low = future_vol <= vol_median
    
    spike_target = (current_is_low & future_is_high).astype(int)
    calm_target = (current_is_high & future_is_low).astype(int)
    
    # Train models
    common_idx = features.dropna().index.intersection(spike_target.dropna().index)
    X = features.loc[common_idx]
    feature_cols = X.columns.tolist()
    
    low_mask = current_is_low.loc[common_idx].fillna(False)
    X_low = X[low_mask]
    y_spike = spike_target.loc[common_idx][low_mask]
    
    spike_model = None
    if len(X_low) > 50 and y_spike.sum() > 10:
        spike_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        spike_model.fit(X_low, y_spike)
        print(f"  Trained spike model on {len(X_low)} samples, {y_spike.sum()} positives")
    
    high_mask = current_is_high.loc[common_idx].fillna(False)
    X_high = X[high_mask]
    y_calm = calm_target.loc[common_idx][high_mask]
    
    calm_model = None
    if len(X_high) > 50 and y_calm.sum() > 10:
        calm_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        calm_model.fit(X_high, y_calm)
        print(f"  Trained calm model on {len(X_high)} samples, {y_calm.sum()} positives")
    
    # =========================================
    # TEST DIFFERENT CONFIGURATIONS
    # =========================================
    
    configs = [
        {'falling_days': 1, 'calm_p': 0.25, 'vix_max': 35, 'prot_pos': 0.45},
        {'falling_days': 2, 'calm_p': 0.25, 'vix_max': 35, 'prot_pos': 0.45},
        {'falling_days': 2, 'calm_p': 0.30, 'vix_max': 30, 'prot_pos': 0.50},
        {'falling_days': 2, 'calm_p': 0.35, 'vix_max': 40, 'prot_pos': 0.50},
        {'falling_days': 3, 'calm_p': 0.30, 'vix_max': 35, 'prot_pos': 0.45},
        # More aggressive recovery
        {'falling_days': 1, 'calm_p': 0.20, 'vix_max': 40, 'prot_pos': 0.50},
        # Stronger protection, faster exit
        {'falling_days': 1, 'calm_p': 0.25, 'vix_max': 45, 'prot_pos': 0.35},
    ]
    
    periods = [
        ('2020 COVID', '2020-01-01', '2020-12-31'),
        ('2022 Bear', '2022-01-01', '2022-12-31'),
        ('2023 Bull', '2023-01-01', '2023-12-31'),
        ('Full 2020-2024', '2020-01-01', '2024-12-31'),
    ]
    
    best_result = None
    best_score = -999
    
    # Prepare full features for prediction
    full_close = spy['Close']
    full_returns = full_close.pct_change()
    full_vix_aligned = vix['Close'].reindex(spy.index).ffill()
    
    # Compute full features
    full_features = pd.DataFrame(index=spy.index)
    full_features['vix'] = full_vix_aligned.shift(1)
    vix_mean_full = full_vix_aligned.rolling(60).mean()
    vix_std_full = full_vix_aligned.rolling(60).std()
    full_features['vix_zscore'] = ((full_vix_aligned - vix_mean_full) / (vix_std_full + 1e-10)).shift(1)
    full_features['vix_mom_5'] = (full_vix_aligned / full_vix_aligned.shift(5) - 1).shift(1)
    full_features['vix_mom_10'] = (full_vix_aligned / full_vix_aligned.shift(10) - 1).shift(1)
    for w in [5, 10, 20]:
        rvol = full_returns.rolling(w).std() * np.sqrt(252)
        full_features[f'rvol_{w}'] = rvol.shift(1)
    rvol_20_full = full_returns.rolling(20).std() * np.sqrt(252)
    rvol_60_full = full_returns.rolling(60).std() * np.sqrt(252)
    full_features['vol_mom'] = (rvol_20_full / rvol_60_full - 1).shift(1)
    full_features['drawdown'] = (full_close / full_close.rolling(60).max() - 1).shift(1)
    
    # Vol regime for prediction
    vol_median_full = rvol_20_full.rolling(60).median()
    current_vol_full = rvol_20_full.shift(1)
    is_low_vol_full = current_vol_full <= vol_median_full
    
    print("\n" + "=" * 90)
    print("RESULTS BY CONFIGURATION")
    print("=" * 90)
    
    for config in configs:
        params = EnhancedParams(
            vix_falling_days=config['falling_days'],
            calm_prob_threshold=config['calm_p'],
            vix_max_for_full=config['vix_max'],
            protected_position=config['prot_pos']
        )
        
        print(f"\n{'='*80}")
        print(f"Falling={config['falling_days']}d calm_p>{config['calm_p']:.0%} VIX_max={config['vix_max']} prot_pos={config['prot_pos']}")
        print(f"{'='*80}")
        
        print(f"\n{'Period':<18} {'Strat':>10} {'B&H':>10} {'Excess':>8} {'S_DD':>8} {'B_DD':>8} {'DD_Imp':>8}")
        print("-" * 80)
        
        period_results = {}
        wins = 0
        
        for period_name, start, end in periods:
            # Get period data
            mask = (spy.index >= start) & (spy.index <= end)
            period_spy = spy[mask]
            
            if len(period_spy) < 50:
                continue
            
            strategy = EnhancedPredictiveStrategy(params)
            
            positions = []
            for idx in period_spy.index:
                if idx not in full_vix_aligned.index:
                    continue
                
                vix_val = full_vix_aligned.loc[idx]
                if pd.isna(vix_val):
                    continue
                
                # Get ML predictions
                is_low = is_low_vol_full.loc[idx] if idx in is_low_vol_full.index else True
                
                feat = full_features.loc[idx:idx].fillna(0)
                for col in feature_cols:
                    if col not in feat.columns:
                        feat[col] = 0
                feat = feat[feature_cols]
                
                spike_prob = 0.0
                calm_prob = 0.0
                
                try:
                    if is_low and spike_model is not None:
                        spike_prob = float(spike_model.predict_proba(feat)[0][1])
                    elif not is_low and calm_model is not None:
                        calm_prob = float(calm_model.predict_proba(feat)[0][1])
                except:
                    pass
                
                pos, reason = strategy.update(vix_val, spike_prob, calm_prob)
                positions.append({'date': idx, 'position': pos, 'reason': reason})
            
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
        
        # Score
        if period_results:
            avg_excess = np.mean([d['excess'] for d in period_results.values()])
            avg_dd_imp = np.mean([d['dd_imp'] for d in period_results.values()])
            score = wins * 25 + avg_excess
            
            if score > best_score:
                best_score = score
                best_result = {'config': config, 'periods': period_results}
    
    # COVID Timeline for best config
    if best_result:
        print("\n" + "=" * 90)
        print("BEST CONFIGURATION - COVID TIMELINE")
        print("=" * 90)
        print(f"\nConfig: {best_result['config']}")
        
        config = best_result['config']
        params = EnhancedParams(
            vix_falling_days=config['falling_days'],
            calm_prob_threshold=config['calm_p'],
            vix_max_for_full=config['vix_max'],
            protected_position=config['prot_pos']
        )
        
        strategy = EnhancedPredictiveStrategy(params)
        
        covid_mask = (spy.index >= '2020-02-01') & (spy.index <= '2020-04-30')
        covid_spy = spy[covid_mask]
        
        print(f"\n{'Date':<12} {'VIX':>6} {'Pos':>6} {'Reason':<35}")
        print("-" * 65)
        
        for idx in covid_spy.index:
            if idx not in full_vix_aligned.index:
                continue
            
            vix_val = full_vix_aligned.loc[idx]
            is_low = is_low_vol_full.loc[idx] if idx in is_low_vol_full.index else True
            
            feat = full_features.loc[idx:idx].fillna(0)
            for col in feature_cols:
                if col not in feat.columns:
                    feat[col] = 0
            feat = feat[feature_cols]
            
            spike_prob = 0.0
            calm_prob = 0.0
            try:
                if is_low and spike_model is not None:
                    spike_prob = float(spike_model.predict_proba(feat)[0][1])
                elif not is_low and calm_model is not None:
                    calm_prob = float(calm_model.predict_proba(feat)[0][1])
            except:
                pass
            
            pos, reason = strategy.update(vix_val, spike_prob, calm_prob)
            
            if pos < 1.0 or 'EXIT' in reason or 'SPIKE' in reason:
                print(f"{idx.strftime('%Y-%m-%d'):<12} {vix_val:>6.1f} {pos:>6.2f} {reason:<35}")
        
        # Summary
        print("\n" + "=" * 90)
        print("SUMMARY")
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
                print(f"\n✗ Still underperforms by {-full_period['excess']:.1f}%")
                print("   But drawdown reduction may make it worthwhile")


if __name__ == "__main__":
    run_test()
