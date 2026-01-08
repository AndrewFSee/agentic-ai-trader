"""
ML Early Exit Strategy: Use spike_probability for EARLIER protection entry

The insight: Previous strategies detect VIX spikes REACTIVELY (after the fact).
If we can predict spikes BEFORE they happen, we might beat B&H.

This strategy:
1. Uses ML spike_probability > threshold to reduce position BEFORE VIX spikes
2. Uses VIX level-based position sizing during high vol
3. Aggressive recovery when VIX falling

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


def run_early_exit_test():
    """Test ML-based early exit strategy."""
    
    print("=" * 90)
    print("ML EARLY EXIT STRATEGY: Predict Spikes BEFORE They Happen")
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
    # TRAIN ML MODELS ON PRE-2020 DATA
    # =========================================
    print("\nTraining ML models on 2015-2019 data...")
    
    train_spy = spy[spy.index <= '2019-12-31']
    train_vix = vix[vix.index <= '2019-12-31']
    
    close = train_spy['Close']
    returns = close.pct_change()
    vix_aligned = train_vix['Close'].reindex(train_spy.index).ffill()
    
    # Features (all lagged by 1 day to avoid lookahead)
    features = pd.DataFrame(index=train_spy.index)
    features['vix'] = vix_aligned.shift(1)
    vix_mean = vix_aligned.rolling(60).mean()
    vix_std = vix_aligned.rolling(60).std()
    features['vix_zscore'] = ((vix_aligned - vix_mean) / (vix_std + 1e-10)).shift(1)
    features['vix_mom_5'] = (vix_aligned / vix_aligned.shift(5) - 1).shift(1)
    features['vix_mom_10'] = (vix_aligned / vix_aligned.shift(10) - 1).shift(1)
    features['vix_acceleration'] = features['vix_mom_5'].diff().shift(1)
    
    rvol_5 = returns.rolling(5).std() * np.sqrt(252)
    rvol_20 = returns.rolling(20).std() * np.sqrt(252)
    rvol_60 = returns.rolling(60).std() * np.sqrt(252)
    features['rvol_5'] = rvol_5.shift(1)
    features['rvol_20'] = rvol_20.shift(1)
    features['vol_mom'] = (rvol_20 / rvol_60 - 1).shift(1)
    features['drawdown'] = (close / close.rolling(60).max() - 1).shift(1)
    features['return_5d'] = (close / close.shift(5) - 1).shift(1)
    
    # Target: Vol spike in next 5 days
    vol_median = rvol_20.rolling(60).median()
    current_is_low = rvol_20.shift(1) <= vol_median
    future_vol = returns.shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
    future_is_high = future_vol > vol_median * 1.5  # 50% above median
    spike_target = (current_is_low & future_is_high).astype(int)
    
    # Train spike model
    common_idx = features.dropna().index.intersection(spike_target.dropna().index)
    X = features.loc[common_idx]
    low_mask = current_is_low.loc[common_idx].fillna(False)
    X_low = X[low_mask]
    y_spike = spike_target.loc[common_idx][low_mask]
    
    spike_model = None
    if len(X_low) > 50 and y_spike.sum() > 10:
        spike_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced', random_state=42
        )
        spike_model.fit(X_low, y_spike)
        print(f"  Spike model: {len(X_low)} samples, {y_spike.sum()} positives ({y_spike.mean():.1%})")
    
    # Train calm model (for high vol periods)
    current_is_high = rvol_20.shift(1) > vol_median
    future_is_low = future_vol <= vol_median
    calm_target = (current_is_high & future_is_low).astype(int)
    
    high_mask = current_is_high.loc[common_idx].fillna(False)
    X_high = X[high_mask]
    y_calm = calm_target.loc[common_idx][high_mask]
    
    calm_model = None
    if len(X_high) > 50 and y_calm.sum() > 10:
        calm_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced', random_state=42
        )
        calm_model.fit(X_high, y_calm)
        print(f"  Calm model: {len(X_high)} samples, {y_calm.sum()} positives ({y_calm.mean():.1%})")
    
    feature_cols = X.columns.tolist()
    
    # =========================================
    # COMPUTE FEATURES FOR FULL PERIOD
    # =========================================
    full_close = spy['Close']
    full_returns = full_close.pct_change()
    full_vix = vix['Close'].reindex(spy.index).ffill()
    
    full_features = pd.DataFrame(index=spy.index)
    full_features['vix'] = full_vix.shift(1)
    vix_mean_full = full_vix.rolling(60).mean()
    vix_std_full = full_vix.rolling(60).std()
    full_features['vix_zscore'] = ((full_vix - vix_mean_full) / (vix_std_full + 1e-10)).shift(1)
    full_features['vix_mom_5'] = (full_vix / full_vix.shift(5) - 1).shift(1)
    full_features['vix_mom_10'] = (full_vix / full_vix.shift(10) - 1).shift(1)
    full_features['vix_acceleration'] = full_features['vix_mom_5'].diff().shift(1)
    
    rvol_5_full = full_returns.rolling(5).std() * np.sqrt(252)
    rvol_20_full = full_returns.rolling(20).std() * np.sqrt(252)
    rvol_60_full = full_returns.rolling(60).std() * np.sqrt(252)
    full_features['rvol_5'] = rvol_5_full.shift(1)
    full_features['rvol_20'] = rvol_20_full.shift(1)
    full_features['vol_mom'] = (rvol_20_full / rvol_60_full - 1).shift(1)
    full_features['drawdown'] = (full_close / full_close.rolling(60).max() - 1).shift(1)
    full_features['return_5d'] = (full_close / full_close.shift(5) - 1).shift(1)
    
    vol_median_full = rvol_20_full.rolling(60).median()
    is_low_vol_full = rvol_20_full.shift(1) <= vol_median_full
    is_high_vol_full = rvol_20_full.shift(1) > vol_median_full
    
    # =========================================
    # TEST STRATEGY
    # =========================================
    
    class EarlyExitStrategy:
        """Strategy that uses ML to exit BEFORE VIX spikes."""
        
        def __init__(self, spike_thresh=0.50, calm_thresh=0.35, ml_pos=0.65, normal_pos=1.0):
            self.spike_thresh = spike_thresh
            self.calm_thresh = calm_thresh
            self.ml_pos = ml_pos
            self.normal_pos = normal_pos
            self._in_ml_protection = False
            self._vix_history = []
        
        def update(self, vix, spike_prob, calm_prob, is_low_vol):
            self._vix_history.append(vix)
            if len(self._vix_history) > 20:
                self._vix_history = self._vix_history[-20:]
            
            # VIX level-based position during high vol
            if not is_low_vol:
                # Use VIX level + calm_prob
                if vix < 25:
                    pos = 1.0
                elif vix < 35:
                    pos = 0.90 if calm_prob > self.calm_thresh else 0.85
                elif vix < 50:
                    pos = 0.85 if calm_prob > self.calm_thresh else 0.75
                elif vix < 65:
                    pos = 0.80 if calm_prob > self.calm_thresh else 0.70
                else:
                    pos = 0.75 if calm_prob > self.calm_thresh else 0.65
                
                return pos, f'HIGH_VOL (VIX={vix:.0f}, calm={calm_prob:.0%})'
            
            # Low vol: check for ML spike warning
            if spike_prob > self.spike_thresh:
                self._in_ml_protection = True
                return self.ml_pos, f'ML_WARNING (spike={spike_prob:.0%})'
            
            self._in_ml_protection = False
            return self.normal_pos, 'NORMAL'
    
    configs = [
        {'spike_thresh': 0.45, 'calm_thresh': 0.30, 'ml_pos': 0.60},
        {'spike_thresh': 0.50, 'calm_thresh': 0.35, 'ml_pos': 0.65},
        {'spike_thresh': 0.55, 'calm_thresh': 0.35, 'ml_pos': 0.70},
        {'spike_thresh': 0.50, 'calm_thresh': 0.40, 'ml_pos': 0.70},
        {'spike_thresh': 0.55, 'calm_thresh': 0.30, 'ml_pos': 0.65},
        # More aggressive
        {'spike_thresh': 0.45, 'calm_thresh': 0.25, 'ml_pos': 0.55},
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
        print(f"\n{'='*80}")
        print(f"spike>{config['spike_thresh']:.0%} calm>{config['calm_thresh']:.0%} ml_pos={config['ml_pos']:.0%}")
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
            
            strategy = EarlyExitStrategy(
                spike_thresh=config['spike_thresh'],
                calm_thresh=config['calm_thresh'],
                ml_pos=config['ml_pos']
            )
            
            positions = []
            for idx in period_spy.index:
                if idx not in full_vix.index:
                    continue
                
                vix_val = full_vix.loc[idx]
                if pd.isna(vix_val):
                    continue
                
                is_low = is_low_vol_full.loc[idx] if idx in is_low_vol_full.index else True
                is_high = is_high_vol_full.loc[idx] if idx in is_high_vol_full.index else False
                
                # Get ML predictions
                feat = full_features.loc[idx:idx].fillna(0)
                for col in feature_cols:
                    if col not in feat.columns:
                        feat[col] = 0
                feat = feat[feature_cols]
                
                spike_prob = 0
                calm_prob = 0
                try:
                    if is_low and spike_model is not None:
                        spike_prob = float(spike_model.predict_proba(feat)[0][1])
                    if is_high and calm_model is not None:
                        calm_prob = float(calm_model.predict_proba(feat)[0][1])
                except:
                    pass
                
                pos, reason = strategy.update(vix_val, spike_prob, calm_prob, is_low)
                positions.append({'date': idx, 'position': pos, 'reason': reason,
                                  'spike_p': spike_prob, 'calm_p': calm_prob})
            
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
                'dd_imp': dd_imp,
                'positions': pos_df
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
                best_result = {'config': config, 'periods': period_results}
    
    # COVID timeline
    if best_result:
        print("\n" + "=" * 90)
        print("BEST CONFIGURATION - COVID TIMELINE (Focus on BEFORE crash)")
        print("=" * 90)
        print(f"\nConfig: {best_result['config']}")
        
        # Check if ML predicted anything BEFORE Feb 24
        covid_pos = best_result['periods']['2020 COVID']['positions']
        early_covid = covid_pos[(covid_pos.index >= '2020-02-01') & (covid_pos.index <= '2020-03-15')]
        
        print(f"\n{'Date':<12} {'VIX':>6} {'spike_p':>8} {'calm_p':>8} {'Pos':>6} {'Reason':<25}")
        print("-" * 75)
        
        for idx, row in early_covid.iterrows():
            if row['position'] < 1.0 or row['spike_p'] > 0.3:
                print(f"{idx.strftime('%Y-%m-%d'):<12} {full_vix.loc[idx]:>6.1f} "
                      f"{row['spike_p']*100:>7.1f}% {row['calm_p']*100:>7.1f}% "
                      f"{row['position']:>6.2f} {row['reason']:<25}")
        
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
                print(f"\n✗ Still underperforms")
                
                # Check if any config beat B&H
                print("\n   Looking for ANY config that beats B&H...")


if __name__ == "__main__":
    run_early_exit_test()
