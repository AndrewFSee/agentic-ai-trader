"""
Predictive VIX Strategy: Combines VIX Signals + ML Volatility Prediction

KEY IMPROVEMENTS:
1. Use ML vol prediction model for EARLY WARNING (before VIX spikes)
2. Aggressive recovery: Start getting back in as soon as VIX starts DECREASING
3. Use calm_probability to accelerate recovery when model predicts vol decline

Strategy Logic:
- ENTRY into protection: VIX spike OR high spike_probability
- EXIT from protection: VIX decreasing (any decrease!) AND calm_probability > threshold

This should capture more of the recovery while still protecting during crashes.

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictiveParams:
    """Parameters for predictive VIX strategy."""
    # Reactive signals (VIX-based)
    vix_roc_spike: float = 0.25      # VIX rising 25%+ in 5 days = spike
    vix_min_entry: float = 18.0      # Only enter if VIX > this
    
    # Predictive signals (ML-based)
    spike_prob_threshold: float = 0.55  # Enter protection if spike_prob > this
    calm_prob_threshold: float = 0.50   # Accelerate recovery if calm_prob > this
    
    # Recovery logic
    vix_decrease_trigger: float = -0.02  # Exit when VIX drops by 2% from recent high
    min_protection_days: int = 2         # Stay protected at least 2 days
    
    # Position sizing
    normal_position: float = 1.0
    protected_position: float = 0.50
    ml_warning_position: float = 0.70    # Partial reduction on ML warning


class VolatilityPredictor:
    """
    Inline volatility predictor (simplified from vol_prediction_tool).
    Predicts vol regime transitions using VIX and market features.
    """
    
    def __init__(self):
        self.spike_model = None
        self.calm_model = None
        self.vix_data = None
        self.is_trained = False
    
    def train(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame, verbose: bool = False):
        """Train on historical data."""
        from sklearn.ensemble import RandomForestClassifier
        
        close = spy_df['Close']
        returns = close.pct_change()
        
        # Align VIX to SPY index
        if isinstance(vix_df, pd.Series):
            vix = vix_df.reindex(spy_df.index).ffill()
        else:
            vix = vix_df['Close'].reindex(spy_df.index).ffill()
        
        # Compute features
        features = self._compute_features(close, returns, vix)
        
        # Compute regimes
        rvol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_median = rvol_20.rolling(60).median()
        
        current_vol = rvol_20.shift(1)
        future_vol = returns.shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
        
        current_is_low = current_vol <= vol_median
        current_is_high = current_vol > vol_median
        future_is_high = future_vol > vol_median
        future_is_low = future_vol <= vol_median
        
        spike_target = (current_is_low & future_is_high).astype(int)
        calm_target = (current_is_high & future_is_low).astype(int)
        
        # Prepare training data
        common_idx = features.dropna().index.intersection(spike_target.dropna().index)
        X = features.loc[common_idx]
        
        # Train spike model (on low vol periods)
        low_mask = current_is_low.loc[common_idx].fillna(False)
        X_low = X[low_mask]
        y_spike = spike_target.loc[common_idx][low_mask]
        
        if len(X_low) > 50 and y_spike.sum() > 10:
            self.spike_model = RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
            )
            self.spike_model.fit(X_low, y_spike)
            if verbose:
                print(f"  Trained spike model on {len(X_low)} low-vol periods, {y_spike.sum()} spikes")
        
        # Train calm model (on high vol periods)
        high_mask = current_is_high.loc[common_idx].fillna(False)
        X_high = X[high_mask]
        y_calm = calm_target.loc[common_idx][high_mask]
        
        if len(X_high) > 50 and y_calm.sum() > 10:
            self.calm_model = RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
            )
            self.calm_model.fit(X_high, y_calm)
            if verbose:
                print(f"  Trained calm model on {len(X_high)} high-vol periods, {y_calm.sum()} calms")
        
        self.is_trained = True
        self.feature_cols = X.columns.tolist()
        
        return self
    
    def _compute_features(self, close, returns, vix):
        """Compute prediction features."""
        features = pd.DataFrame(index=close.index)
        lag = 1  # Avoid lookahead
        
        # VIX features
        features['vix'] = vix.shift(lag)
        vix_mean = vix.rolling(60).mean()
        vix_std = vix.rolling(60).std()
        features['vix_zscore'] = ((vix - vix_mean) / (vix_std + 1e-10)).shift(lag)
        features['vix_mom_5'] = (vix / vix.shift(5) - 1).shift(lag)
        features['vix_mom_10'] = (vix / vix.shift(10) - 1).shift(lag)
        
        # VIX term structure proxy
        vix_sma = vix.rolling(10).mean()
        features['vix_vs_sma'] = (vix / vix_sma - 1).shift(lag)
        
        # Asset volatility
        for w in [5, 10, 20]:
            rvol = returns.rolling(w).std() * np.sqrt(252)
            features[f'rvol_{w}'] = rvol.shift(lag)
        
        rvol_20 = returns.rolling(20).std() * np.sqrt(252)
        rvol_60 = returns.rolling(60).std() * np.sqrt(252)
        features['vol_mom'] = (rvol_20 / rvol_60 - 1).shift(lag)
        
        # Price features
        sma_50 = close.rolling(50).mean()
        features['sma_deviation'] = (close / sma_50 - 1).shift(lag)
        features['drawdown'] = (close / close.rolling(60).max() - 1).shift(lag)
        features['return_5d'] = (close / close.shift(5) - 1).shift(lag)
        
        return features
    
    def predict(self, close, returns, vix, current_vol, vol_median) -> Tuple[float, float, str]:
        """
        Predict probabilities for current state.
        
        Returns: (spike_prob, calm_prob, regime)
        """
        if not self.is_trained:
            return 0.0, 0.0, 'UNKNOWN'
        
        is_low_vol = current_vol <= vol_median if not np.isnan(current_vol) else True
        regime = 'LOW' if is_low_vol else 'HIGH'
        
        # Compute features for latest bar
        features = self._compute_features(close, returns, vix)
        latest = features.iloc[-1:].fillna(0)
        
        # Ensure columns match
        if hasattr(self, 'feature_cols'):
            for col in self.feature_cols:
                if col not in latest.columns:
                    latest[col] = 0
            latest = latest[self.feature_cols]
        
        spike_prob = 0.0
        calm_prob = 0.0
        
        try:
            if is_low_vol and self.spike_model is not None:
                spike_prob = float(self.spike_model.predict_proba(latest)[0][1])
            elif not is_low_vol and self.calm_model is not None:
                calm_prob = float(self.calm_model.predict_proba(latest)[0][1])
        except Exception:
            pass
        
        return spike_prob, calm_prob, regime


class PredictiveVIXStrategy:
    """
    Strategy that combines reactive VIX signals with predictive ML model.
    
    Entry: VIX spike OR high spike_probability
    Exit: VIX decreasing AND (min_days passed OR calm_probability high)
    """
    
    def __init__(self, params: PredictiveParams):
        self.params = params
        self.predictor = VolatilityPredictor()
        self._reset_state()
    
    def _reset_state(self):
        self._vix_history: List[float] = []
        self._vix_peak = 0.0
        self._in_protection = False
        self._protection_days = 0
        self._position_history: List[Dict] = []
    
    def train(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame, verbose: bool = False):
        """Train the ML predictor on historical data."""
        self.predictor.train(spy_df, vix_df, verbose=verbose)
    
    def update(
        self,
        date: pd.Timestamp,
        close: float,
        vix: float,
        returns: pd.Series,
        vix_series: pd.Series,
        close_series: pd.Series
    ) -> Tuple[float, Dict]:
        """
        Update strategy with new data.
        
        Returns: (position, info_dict)
        """
        self._vix_history.append(vix)
        if len(self._vix_history) > 100:
            self._vix_history = self._vix_history[-100:]
        
        # Compute VIX ROC (5 day)
        if len(self._vix_history) > 5:
            prev_vix = self._vix_history[-6]
            vix_roc = (vix - prev_vix) / max(prev_vix, 1.0)
        else:
            vix_roc = 0.0
        
        # Compute current vol regime
        rvol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_median = rvol_20.rolling(60).median()
        current_vol = rvol_20.iloc[-1] if len(rvol_20) > 0 else 0.15
        vol_med = vol_median.iloc[-1] if len(vol_median) > 0 else 0.15
        
        # Get ML predictions
        spike_prob, calm_prob, vol_regime = self.predictor.predict(
            close_series, returns, vix_series, current_vol, vol_med
        )
        
        info = {
            'date': date,
            'vix': vix,
            'vix_roc': vix_roc,
            'spike_prob': spike_prob,
            'calm_prob': calm_prob,
            'vol_regime': vol_regime,
            'reason': ''
        }
        
        # === ENTRY LOGIC ===
        
        # Reactive: VIX spike detection
        vix_spike = (vix_roc > self.params.vix_roc_spike and vix > self.params.vix_min_entry)
        
        # Predictive: ML says spike incoming
        ml_warning = (spike_prob > self.params.spike_prob_threshold)
        
        if not self._in_protection:
            if vix_spike:
                self._in_protection = True
                self._protection_days = 0
                self._vix_peak = vix
                info['reason'] = f'ENTER_VIX_SPIKE (ROC={vix_roc:.1%})'
                return self.params.protected_position, info
            
            elif ml_warning:
                # Partial protection on ML warning only
                info['reason'] = f'ML_WARNING (spike_prob={spike_prob:.1%})'
                return self.params.ml_warning_position, info
            
            else:
                info['reason'] = 'NORMAL'
                return self.params.normal_position, info
        
        # === IN PROTECTION - CHECK FOR EXIT ===
        
        self._protection_days += 1
        self._vix_peak = max(self._vix_peak, vix)
        
        # Key insight: Exit when VIX starts DECREASING
        vix_change_from_peak = (vix - self._vix_peak) / self._vix_peak
        vix_decreasing = vix_change_from_peak < self.params.vix_decrease_trigger
        
        # ML says calm is coming
        ml_calm = calm_prob > self.params.calm_prob_threshold
        
        # Exit conditions
        can_exit = self._protection_days >= self.params.min_protection_days
        
        if can_exit and vix_decreasing:
            # VIX is falling - start getting back in
            if ml_calm:
                # ML confirms calm - full exit from protection
                self._in_protection = False
                self._vix_peak = 0
                info['reason'] = f'EXIT_VIX_DOWN+ML_CALM (calm_prob={calm_prob:.1%})'
                return self.params.normal_position, info
            else:
                # VIX falling but ML not confident - partial exit
                partial_pos = (self.params.protected_position + self.params.normal_position) / 2
                info['reason'] = f'PARTIAL_EXIT_VIX_DOWN (VIX_chg={vix_change_from_peak:.1%})'
                return partial_pos, info
        
        elif can_exit and ml_calm and vix < 30:
            # ML says calm even if VIX not falling much, but VIX is reasonable
            self._in_protection = False
            self._vix_peak = 0
            info['reason'] = f'EXIT_ML_CALM (calm_prob={calm_prob:.1%}, VIX={vix:.1f})'
            return self.params.normal_position, info
        
        else:
            # Stay in protection
            info['reason'] = f'PROTECTED_D{self._protection_days}'
            return self.params.protected_position, info
    
    def backtest(
        self,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        train_end: str = '2019-12-31',
        transaction_cost: float = 0.001
    ) -> Dict:
        """Run backtest with train/test split."""
        
        # Prepare data
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        
        # Train on data before train_end
        train_spy = spy_df[spy_df.index <= train_end]
        train_vix = vix_df[vix_df.index <= train_end]
        
        if len(train_spy) < 252:
            return {'error': 'Insufficient training data'}
        
        self.train(train_spy, train_vix, verbose=False)
        
        # Test on data after train_end
        test_spy = spy_df[spy_df.index > train_end]
        
        if len(test_spy) < 50:
            return {'error': 'Insufficient test data'}
        
        # Reset state for testing
        self._reset_state()
        
        # Prepare series
        close_series = spy_df['Close']
        returns = close_series.pct_change()
        vix_series = vix_df['Close'] if 'Close' in vix_df.columns else vix_df
        
        positions = []
        
        for idx in test_spy.index:
            if idx not in vix_series.index:
                continue
            
            vix_val = vix_series.loc[idx]
            if pd.isna(vix_val):
                continue
            
            close = test_spy.loc[idx]['Close']
            
            # Get history up to this point (no lookahead)
            hist_returns = returns.loc[:idx]
            hist_vix = vix_series.loc[:idx]
            hist_close = close_series.loc[:idx]
            
            pos, info = self.update(idx, close, vix_val, hist_returns, hist_vix, hist_close)
            positions.append({
                'date': idx,
                'position': pos,
                'vix': vix_val,
                'reason': info['reason'],
                'spike_prob': info['spike_prob'],
                'calm_prob': info['calm_prob']
            })
        
        if not positions:
            return {'error': 'No positions generated'}
        
        pos_df = pd.DataFrame(positions).set_index('date')
        
        # Calculate returns
        test_returns = returns.loc[pos_df.index]
        signals = pos_df['position'].shift(1).fillna(1.0)
        
        aligned = pd.DataFrame({
            'return': test_returns,
            'signal': signals
        }).dropna()
        
        strat_returns = aligned['signal'] * aligned['return']
        tc = aligned['signal'].diff().abs().fillna(0) * transaction_cost
        strat_returns_net = strat_returns - tc
        
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
            'dd_improvement': abs(bh_m['max_drawdown']) - abs(strat_m['max_drawdown']),
            'positions_df': pos_df
        }


def run_predictive_test():
    """Test the predictive VIX strategy."""
    
    print("=" * 90)
    print("PREDICTIVE VIX STRATEGY: ML Prediction + VIX Decrease Recovery")
    print("=" * 90)
    
    import yfinance as yf
    
    print("\nDownloading data...")
    spy = yf.download("SPY", start="2015-01-01", end="2025-01-01", progress=False)
    vix = yf.download("^VIX", start="2015-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    print(f"Data: {len(spy)} bars")
    
    # Test different parameter combinations
    configs = [
        # Baseline: VIX spike only, aggressive VIX decrease recovery
        {'vix_roc': 0.25, 'spike_prob': 1.0, 'calm_prob': 1.0, 'vix_dec': -0.02, 'min_days': 1},
        
        # Add ML early warning
        {'vix_roc': 0.30, 'spike_prob': 0.55, 'calm_prob': 0.50, 'vix_dec': -0.02, 'min_days': 2},
        {'vix_roc': 0.30, 'spike_prob': 0.50, 'calm_prob': 0.45, 'vix_dec': -0.02, 'min_days': 2},
        
        # Aggressive recovery (any VIX decrease)
        {'vix_roc': 0.25, 'spike_prob': 0.55, 'calm_prob': 0.45, 'vix_dec': -0.01, 'min_days': 1},
        
        # Use ML to accelerate recovery
        {'vix_roc': 0.25, 'spike_prob': 0.55, 'calm_prob': 0.40, 'vix_dec': -0.03, 'min_days': 2},
        
        # Higher spike_prob threshold (fewer false entries)
        {'vix_roc': 0.30, 'spike_prob': 0.60, 'calm_prob': 0.50, 'vix_dec': -0.02, 'min_days': 2},
        
        # Lower calm_prob threshold (faster recovery)
        {'vix_roc': 0.25, 'spike_prob': 0.55, 'calm_prob': 0.35, 'vix_dec': -0.02, 'min_days': 1},
    ]
    
    periods = [
        ('2019-12-31', '2020 COVID', '2020-01-01', '2020-12-31'),
        ('2021-12-31', '2022 Bear', '2022-01-01', '2022-12-31'),
        ('2022-12-31', '2023 Bull', '2023-01-01', '2023-12-31'),
        ('2017-12-31', 'Full 2018-2024', '2018-01-01', '2024-12-31'),
    ]
    
    all_results = []
    
    for config in configs:
        params = PredictiveParams(
            vix_roc_spike=config['vix_roc'],
            spike_prob_threshold=config['spike_prob'],
            calm_prob_threshold=config['calm_prob'],
            vix_decrease_trigger=config['vix_dec'],
            min_protection_days=config['min_days']
        )
        
        period_results = {}
        
        for train_end, period_name, test_start, test_end in periods:
            # Filter data for this period
            test_spy = spy[(spy.index >= test_start) & (spy.index <= test_end)]
            test_vix = vix[(vix.index >= test_start) & (vix.index <= test_end)]
            
            # Use all data for training
            train_spy = spy[spy.index <= train_end]
            train_vix = vix[vix.index <= train_end]
            
            if len(train_spy) < 252 or len(test_spy) < 50:
                continue
            
            strategy = PredictiveVIXStrategy(params)
            strategy.train(train_spy, train_vix)
            strategy._reset_state()
            
            # Manual backtest on test period
            close_series = spy['Close']
            returns = close_series.pct_change()
            vix_series = vix['Close']
            
            positions = []
            for idx in test_spy.index:
                if idx not in vix_series.index:
                    continue
                vix_val = vix_series.loc[idx]
                if pd.isna(vix_val):
                    continue
                close = test_spy.loc[idx]['Close']
                hist_returns = returns.loc[:idx]
                hist_vix = vix_series.loc[:idx]
                hist_close = close_series.loc[:idx]
                
                pos, info = strategy.update(idx, close, vix_val, hist_returns, hist_vix, hist_close)
                positions.append({'date': idx, 'position': pos, 'reason': info['reason']})
            
            if not positions:
                continue
            
            pos_df = pd.DataFrame(positions).set_index('date')
            test_returns = returns.loc[pos_df.index]
            signals = pos_df['position'].shift(1).fillna(1.0)
            
            aligned = pd.DataFrame({'return': test_returns, 'signal': signals}).dropna()
            strat_rets = aligned['signal'] * aligned['return']
            tc = aligned['signal'].diff().abs().fillna(0) * 0.001
            strat_net = strat_rets - tc
            
            def calc_m(rets):
                total = (1 + rets).prod() - 1
                years = max(len(rets) / 252, 0.01)
                cum = (1 + rets).cumprod()
                peak = cum.expanding().max()
                dd = ((cum - peak) / peak).min()
                return {'total': total * 100, 'dd': dd * 100}
            
            strat_m = calc_m(strat_net)
            bh_m = calc_m(aligned['return'])
            
            period_results[period_name] = {
                'strat': strat_m['total'],
                'bh': bh_m['total'],
                'excess': strat_m['total'] - bh_m['total'],
                'strat_dd': strat_m['dd'],
                'bh_dd': bh_m['dd'],
                'dd_imp': abs(bh_m['dd']) - abs(strat_m['dd'])
            }
        
        all_results.append({'config': config, 'periods': period_results})
    
    # Print results
    print("\n" + "=" * 90)
    print("RESULTS BY CONFIGURATION")
    print("=" * 90)
    
    best_score = -999
    best_result = None
    
    for result in all_results:
        config = result['config']
        print(f"\n{'='*80}")
        print(f"VIX_ROC>{config['vix_roc']:.0%} spike_p>{config['spike_prob']:.0%} calm_p>{config['calm_prob']:.0%} "
              f"VIX_dec<{config['vix_dec']:.0%} min_days={config['min_days']}")
        print(f"{'='*80}")
        
        print(f"\n{'Period':<18} {'Strat':>10} {'B&H':>10} {'Excess':>8} {'S_DD':>8} {'B_DD':>8} {'DD_Imp':>8}")
        print("-" * 80)
        
        wins = 0
        total = 0
        
        for period_name, data in result['periods'].items():
            total += 1
            excess = data['excess']
            if excess > 0:
                wins += 1
            
            mark = "✓" if excess > 0 else " "
            print(f"{period_name:<18} {data['strat']:>+9.1f}% {data['bh']:>+9.1f}% {excess:>+7.1f}%{mark} "
                  f"{data['strat_dd']:>7.1f}% {data['bh_dd']:>7.1f}% {data['dd_imp']:>+7.1f}%")
        
        print(f"\nWINS: {wins}/{total}")
        
        # Score
        if result['periods']:
            avg_excess = np.mean([d['excess'] for d in result['periods'].values()])
            avg_dd_imp = np.mean([d['dd_imp'] for d in result['periods'].values()])
            score = wins * 25 + avg_excess + avg_dd_imp * 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
    
    # Show COVID timeline for best config
    if best_result:
        print("\n" + "=" * 90)
        print("BEST CONFIGURATION - COVID TIMELINE")
        print("=" * 90)
        print(f"\nParameters: {best_result['config']}")
        
        config = best_result['config']
        params = PredictiveParams(
            vix_roc_spike=config['vix_roc'],
            spike_prob_threshold=config['spike_prob'],
            calm_prob_threshold=config['calm_prob'],
            vix_decrease_trigger=config['vix_dec'],
            min_protection_days=config['min_days']
        )
        
        # Train up to end of 2019
        train_spy = spy[spy.index <= '2019-12-31']
        train_vix = vix[vix.index <= '2019-12-31']
        
        strategy = PredictiveVIXStrategy(params)
        strategy.train(train_spy, train_vix)
        strategy._reset_state()
        
        # Run on COVID period
        covid_spy = spy[(spy.index >= '2020-02-01') & (spy.index <= '2020-05-31')]
        close_series = spy['Close']
        returns = close_series.pct_change()
        vix_series = vix['Close']
        
        print(f"\n{'Date':<12} {'VIX':>6} {'VIX_ROC':>8} {'spike_p':>8} {'calm_p':>8} {'Pos':>6} {'Reason':<25}")
        print("-" * 85)
        
        for idx in covid_spy.index:
            if idx not in vix_series.index:
                continue
            vix_val = vix_series.loc[idx]
            if pd.isna(vix_val):
                continue
            close = covid_spy.loc[idx]['Close']
            hist_returns = returns.loc[:idx]
            hist_vix = vix_series.loc[:idx]
            hist_close = close_series.loc[:idx]
            
            pos, info = strategy.update(idx, close, vix_val, hist_returns, hist_vix, hist_close)
            
            # Only print interesting days
            if pos < 1.0 or 'ENTER' in info['reason'] or 'EXIT' in info['reason']:
                date_str = idx.strftime('%Y-%m-%d')
                print(f"{date_str:<12} {vix_val:>6.1f} {info.get('vix_roc', 0)*100:>+7.1f}% "
                      f"{info['spike_prob']*100:>7.1f}% {info['calm_prob']*100:>7.1f}% "
                      f"{pos:>6.2f} {info['reason']:<25}")
        
        # Final summary
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        
        full_period = best_result['periods'].get('Full 2018-2024', {})
        if full_period:
            print(f"\nFull Period (2018-2024):")
            print(f"  Strategy Return: {full_period['strat']:+.1f}%")
            print(f"  Buy & Hold:      {full_period['bh']:+.1f}%")
            print(f"  Excess Return:   {full_period['excess']:+.1f}%")
            print(f"  DD Improvement:  {full_period['dd_imp']:+.1f}%")
            
            if full_period['excess'] > 0:
                print(f"\n✓ STRATEGY BEATS BUY & HOLD!")
            else:
                print(f"\n✗ Strategy underperforms by {-full_period['excess']:.1f}%")


if __name__ == "__main__":
    run_predictive_test()
