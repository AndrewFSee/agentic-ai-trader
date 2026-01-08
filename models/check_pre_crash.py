"""
Check ML predictions BEFORE COVID crash to see if any early warning was possible.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def check_pre_crash_predictions():
    """Check what ML predicted in the 2 weeks before COVID crash."""
    
    print("=" * 80)
    print("PRE-CRASH ML PREDICTION ANALYSIS")
    print("=" * 80)
    
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    
    spy = yf.download("SPY", start="2015-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # Train on pre-2020 data
    train_spy = spy[spy.index <= '2019-12-31']
    train_vix = vix[vix.index <= '2019-12-31']
    
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
    features['vix_acceleration'] = features['vix_mom_5'].diff().shift(1)
    
    rvol_5 = returns.rolling(5).std() * np.sqrt(252)
    rvol_20 = returns.rolling(20).std() * np.sqrt(252)
    rvol_60 = returns.rolling(60).std() * np.sqrt(252)
    features['rvol_5'] = rvol_5.shift(1)
    features['rvol_20'] = rvol_20.shift(1)
    features['vol_mom'] = (rvol_20 / rvol_60 - 1).shift(1)
    features['drawdown'] = (close / close.rolling(60).max() - 1).shift(1)
    features['return_5d'] = (close / close.shift(5) - 1).shift(1)
    
    # Target: Vol spike
    vol_median = rvol_20.rolling(60).median()
    current_is_low = rvol_20.shift(1) <= vol_median
    future_vol = returns.shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
    future_is_high = future_vol > vol_median * 1.5
    spike_target = (current_is_low & future_is_high).astype(int)
    
    # Train
    common_idx = features.dropna().index.intersection(spike_target.dropna().index)
    X = features.loc[common_idx]
    low_mask = current_is_low.loc[common_idx].fillna(False)
    X_low = X[low_mask]
    y_spike = spike_target.loc[common_idx][low_mask]
    
    spike_model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    spike_model.fit(X_low, y_spike)
    
    feature_cols = X.columns.tolist()
    
    # Now check predictions for Feb 2020 (before crash)
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
    
    # Check Feb 3 - Feb 28, 2020
    print("\nPRE-CRASH PREDICTIONS (Feb 3 - Feb 28, 2020):")
    print("=" * 80)
    print(f"{'Date':<12} {'SPY':>8} {'Return':>8} {'VIX':>6} {'VIX_z':>7} {'VIX_mom':>8} {'spike_p':>8} {'is_low':>6}")
    print("-" * 80)
    
    pre_crash = spy[(spy.index >= '2020-02-03') & (spy.index <= '2020-02-28')]
    
    for idx in pre_crash.index:
        if idx not in full_vix.index:
            continue
        
        vix_val = full_vix.loc[idx]
        is_low = is_low_vol_full.loc[idx] if idx in is_low_vol_full.index else True
        
        feat = full_features.loc[idx:idx].fillna(0)
        for col in feature_cols:
            if col not in feat.columns:
                feat[col] = 0
        feat = feat[feature_cols]
        
        spike_prob = 0
        if is_low:
            try:
                spike_prob = float(spike_model.predict_proba(feat)[0][1])
            except:
                pass
        
        spy_close = pre_crash.loc[idx]['Close']
        spy_ret = full_returns.loc[idx] if idx in full_returns.index else 0
        vix_zscore = full_features.loc[idx]['vix_zscore'] if idx in full_features.index else 0
        vix_mom = full_features.loc[idx]['vix_mom_5'] if idx in full_features.index else 0
        
        print(f"{idx.strftime('%Y-%m-%d'):<12} {spy_close:>8.2f} {spy_ret*100:>+7.2f}% {vix_val:>6.1f} "
              f"{vix_zscore:>+6.2f} {vix_mom*100:>+7.1f}% {spike_prob*100:>7.1f}% {'Y' if is_low else 'N':>6}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print("\nKey observations:")
    print("1. VIX was ~14-15 on Feb 3-19 (normal)")
    print("2. VIX jumped to 17.1 on Feb 21 (Friday)")
    print("3. VIX spiked to 25.0 on Feb 24 (Monday) - crash day")
    print("\nThe question: Could the model have predicted the spike on Feb 20-21?")
    
    # Check feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (what predicts spikes):")
    print("=" * 80)
    
    importances = pd.Series(spike_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  {feat:<20}: {imp:.3f}")


if __name__ == "__main__":
    check_pre_crash_predictions()
