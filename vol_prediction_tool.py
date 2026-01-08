# vol_prediction_tool.py
"""
Volatility Prediction Tool for Agentic AI Trader

This tool predicts volatility regime transitions for position sizing and risk management.

KEY FINDINGS FROM RESEARCH:
1. VIX z-score is the #1 predictor (works across all assets)
2. HIGH→LOW transitions are more predictable (65-71% precision)
3. LOW→HIGH (vol spikes) are harder (45-62% precision at 0.6 threshold)
4. ~34% of predictions are transitions (not just "stay same")

APPROACH: Ticker-agnostic using VIX and market-derived features
- VIX z-score and momentum predict vol for ALL equities
- Asset-specific features (recent vol, drawdown) provide fine-tuning
- Model trains on SPY but transfers reasonably to other assets
"""

import os
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Lazy imports
_VOL_PREDICTOR_LOADED = False
_vol_predictor = None


def _ensure_vol_predictor_loaded():
    """Lazy load the volatility predictor."""
    global _VOL_PREDICTOR_LOADED, _vol_predictor
    
    if _VOL_PREDICTOR_LOADED:
        return True
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from sklearn.ensemble import RandomForestClassifier
        
        _VOL_PREDICTOR_LOADED = True
        return True
    except ImportError as e:
        print(f"Warning: Could not load vol predictor dependencies: {e}")
        return False


class VolatilityPredictor:
    """
    Volatility regime predictor using VIX-based universal features.
    
    Trained on SPY but works across equities because VIX reflects
    broad market fear/greed that affects all stocks.
    """
    
    def __init__(self):
        self.spike_model = None
        self.calm_model = None
        self.training_date = None
        self.is_trained = False
        self.vix_data = None
    
    def _download_vix(self, years=3):
        """Download VIX data."""
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=years*365)
        
        vix = yf.download('^VIX', start=start, end=end, progress=False)
        if hasattr(vix.columns, 'droplevel'):
            if isinstance(vix.columns, type(vix.columns)) and len(vix.columns.names) > 1:
                vix = vix.droplevel(1, axis=1)
        
        self.vix_data = vix['Close'].rename('VIX')
        return self.vix_data
    
    def _download_asset(self, symbol, years=3):
        """Download asset data."""
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=years*365)
        
        df = yf.download(symbol, start=start, end=end, progress=False)
        if hasattr(df.columns, 'droplevel'):
            if isinstance(df.columns, type(df.columns)) and len(df.columns.names) > 1:
                df = df.droplevel(1, axis=1)
        
        return df
    
    def compute_features(self, df, lag=1):
        """Compute VIX-based universal features + asset-specific features."""
        import pandas as pd
        import numpy as np
        
        features = pd.DataFrame(index=df.index)
        close = df['Close']
        returns = close.pct_change()
        
        # Ensure VIX is loaded
        if self.vix_data is None:
            self._download_vix()
        
        vix_aligned = self.vix_data.reindex(df.index).ffill()
        
        # ==================================================
        # VIX FEATURES (Universal - work across all equities)
        # ==================================================
        features['vix'] = vix_aligned.shift(lag)
        
        vix_mean = vix_aligned.rolling(60).mean()
        vix_std = vix_aligned.rolling(60).std()
        features['vix_zscore'] = ((vix_aligned - vix_mean) / (vix_std + 1e-10)).shift(lag)
        
        features['vix_mom_5'] = (vix_aligned / vix_aligned.shift(5) - 1).shift(lag)
        features['vix_mom_10'] = (vix_aligned / vix_aligned.shift(10) - 1).shift(lag)
        
        # VIX term structure proxy
        vix_sma = vix_aligned.rolling(10).mean()
        features['vix_vs_sma'] = (vix_aligned / vix_sma - 1).shift(lag)
        
        # VIX acceleration
        vix_velocity = vix_aligned.diff(5)
        features['vix_acceleration'] = vix_velocity.diff(5).shift(lag)
        
        # ==================================================
        # ASSET-SPECIFIC VOLATILITY FEATURES
        # ==================================================
        for w in [5, 10, 20, 60]:
            rvol = returns.rolling(w).std() * np.sqrt(252)
            features[f'rvol_{w}'] = rvol.shift(lag)
        
        rvol_20 = returns.rolling(20).std() * np.sqrt(252)
        rvol_60 = returns.rolling(60).std() * np.sqrt(252)
        features['vol_mom'] = (rvol_20 / rvol_60 - 1).shift(lag)
        
        # Vol compression (narrow vol often precedes expansion)
        vol_min_60 = rvol_20.rolling(60).min()
        features['vol_compression'] = (rvol_20 / (vol_min_60 + 1e-10) - 1).shift(lag)
        
        # ==================================================
        # PRICE ACTION FEATURES
        # ==================================================
        sma_50 = close.rolling(50).mean()
        features['sma_deviation'] = (close / sma_50 - 1).shift(lag)
        features['drawdown'] = (close / close.rolling(60).max() - 1).shift(lag)
        features['return_5d'] = (close / close.shift(5) - 1).shift(lag)
        features['return_20d'] = (close / close.shift(20) - 1).shift(lag)
        
        # Extreme moves
        abs_returns = returns.abs()
        daily_threshold = abs_returns.rolling(60).quantile(0.95)
        features['extreme_days_5'] = (abs_returns > daily_threshold).rolling(5).sum().shift(lag)
        
        return features
    
    def get_vol_regimes(self, returns, lookback=60, horizon=5, lag=1):
        """Define volatility regimes."""
        import numpy as np
        
        rvol_20 = returns.rolling(20).std() * np.sqrt(252)
        
        # Current vol (lagged to avoid look-ahead)
        current_vol = rvol_20.shift(lag)
        
        # Future vol (what we're predicting)
        future_vol = returns.shift(-1).rolling(horizon).std().shift(-horizon+1) * np.sqrt(252)
        
        # Threshold
        vol_median = current_vol.rolling(lookback).median()
        
        current_is_low = (current_vol <= vol_median)
        current_is_high = (current_vol > vol_median)
        future_is_high = (future_vol > vol_median)
        future_is_low = (future_vol <= vol_median)
        
        return {
            'current_low': current_is_low,
            'current_high': current_is_high,
            'spike_target': (current_is_low & future_is_high).astype(int),
            'calm_target': (current_is_high & future_is_low).astype(int),
            'current_vol': current_vol,
            'vol_median': vol_median
        }
    
    def train(self, symbol='SPY', years=3, verbose=False):
        """Train the volatility predictor."""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        
        df = self._download_asset(symbol, years)
        if self.vix_data is None:
            self._download_vix(years)
        
        returns = df['Close'].pct_change()
        features = self.compute_features(df)
        regimes = self.get_vol_regimes(returns)
        
        # Prepare data
        common_idx = features.dropna().index.intersection(regimes['spike_target'].dropna().index)
        X = features.loc[common_idx]
        
        # Train spike model (on low vol periods)
        low_mask = regimes['current_low'].loc[common_idx].fillna(False)
        X_low = X[low_mask]
        y_spike = regimes['spike_target'].loc[common_idx][low_mask]
        
        if len(X_low) > 50 and y_spike.sum() > 10:
            self.spike_model = RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
            )
            self.spike_model.fit(X_low, y_spike)
        
        # Train calm model (on high vol periods)
        high_mask = regimes['current_high'].loc[common_idx].fillna(False)
        X_high = X[high_mask]
        y_calm = regimes['calm_target'].loc[common_idx][high_mask]
        
        if len(X_high) > 50 and y_calm.sum() > 10:
            self.calm_model = RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
            )
            self.calm_model.fit(X_high, y_calm)
        
        self.training_date = common_idx[-1] if len(common_idx) > 0 else None
        self.is_trained = True
        
        if verbose:
            print(f"  Trained spike model on {len(X_low)} low-vol periods")
            print(f"  Trained calm model on {len(X_high)} high-vol periods")
        
        return self
    
    def predict(self, symbol: str) -> Dict[str, Any]:
        """
        Predict volatility regime for a symbol.
        
        Returns dict with:
        - current_regime: 'LOW' or 'HIGH'
        - spike_probability: P(transition to HIGH) if currently LOW
        - calm_probability: P(transition to LOW) if currently HIGH
        - risk_level: 'LOW', 'WATCH', 'ELEVATED', 'CALMING', or 'HIGH'
        - suggested_action: Position sizing recommendation
        - vix_zscore: Current VIX standardized score
        """
        import numpy as np
        
        if not self.is_trained:
            self.train()
        
        # Download recent data for this symbol
        df = self._download_asset(symbol, years=1)
        if len(df) < 100:
            return {
                'symbol': symbol,
                'error': f"Insufficient data for {symbol}",
                'current_regime': 'UNKNOWN',
                'risk_level': 'UNKNOWN'
            }
        
        # Ensure VIX is current
        if self.vix_data is None or len(self.vix_data) == 0:
            self._download_vix(years=1)
        
        returns = df['Close'].pct_change()
        features = self.compute_features(df)
        regimes = self.get_vol_regimes(returns)
        
        # Current regime
        current_vol = regimes['current_vol'].iloc[-1]
        vol_median = regimes['vol_median'].iloc[-1]
        is_low_vol = current_vol <= vol_median if not np.isnan(current_vol) else True
        
        # Get latest features (may have NaNs, fill with 0)
        latest = features.iloc[-1:].fillna(0)
        
        # Make predictions
        spike_prob = 0.0
        calm_prob = 0.0
        
        try:
            if is_low_vol and self.spike_model is not None:
                spike_prob = float(self.spike_model.predict_proba(latest)[0][1])
            elif not is_low_vol and self.calm_model is not None:
                calm_prob = float(self.calm_model.predict_proba(latest)[0][1])
        except Exception as e:
            pass  # Keep default 0.0
        
        # Determine risk level and action
        current_regime = 'LOW' if is_low_vol else 'HIGH'
        
        if is_low_vol:
            if spike_prob >= 0.6:
                risk_level = 'ELEVATED'
                action = 'Reduce position size 30-50%, tighten stops'
            elif spike_prob >= 0.4:
                risk_level = 'WATCH'
                action = 'Tighten stops, reduce new positions'
            else:
                risk_level = 'LOW'
                action = 'Normal position sizing'
        else:
            if calm_prob >= 0.6:
                risk_level = 'CALMING'
                action = 'Consider adding to positions, vol likely to subside'
            elif calm_prob >= 0.4:
                risk_level = 'HIGH_MAY_CALM'
                action = 'Wait for confirmation before adding'
            else:
                risk_level = 'HIGH'
                action = 'Reduce exposure, defensive positioning'
        
        vix_zscore = float(features['vix_zscore'].iloc[-1]) if not np.isnan(features['vix_zscore'].iloc[-1]) else 0.0
        
        return {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_regime': current_regime,
            'current_vol_annualized': float(current_vol) if not np.isnan(current_vol) else 0.0,
            'vol_median': float(vol_median) if not np.isnan(vol_median) else 0.0,
            'spike_probability': spike_prob,
            'calm_probability': calm_prob,
            'risk_level': risk_level,
            'suggested_action': action,
            'vix': float(features['vix'].iloc[-1]) if not np.isnan(features['vix'].iloc[-1]) else 0.0,
            'vix_zscore': vix_zscore,
            'interpretation': {
                'spike_warning': spike_prob >= 0.5,
                'calm_signal': calm_prob >= 0.5,
                'precision_note': f"Spike predictions at 0.6+ threshold have ~62% precision (vs 23% base rate). "
                                  f"Calm predictions at 0.6+ threshold have ~62% precision (vs ~42% base rate)."
            }
        }


# Global predictor instance (lazy loaded)
_predictor_instance: Optional[VolatilityPredictor] = None


def get_predictor() -> VolatilityPredictor:
    """Get or create the global predictor instance."""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = VolatilityPredictor()
    
    return _predictor_instance


def vol_prediction_tool_fn(state: dict, args: dict) -> dict:
    """
    Volatility prediction tool function for the agent.
    
    Predicts volatility regime and transition probability for position sizing
    and risk management decisions.
    
    args:
        symbol: str (required) - Stock ticker
        
    Returns state with tool_results["vol_prediction"] containing:
        - current_regime: 'LOW' or 'HIGH' volatility
        - spike_probability: P(transitioning to HIGH) if currently LOW
        - calm_probability: P(transitioning to LOW) if currently HIGH  
        - risk_level: 'LOW', 'WATCH', 'ELEVATED', 'CALMING', 'HIGH'
        - suggested_action: Position sizing recommendation
        - vix_zscore: Current VIX relative to 60-day mean
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    symbol = args.get("symbol")
    if not symbol:
        state["tool_results"]["vol_prediction"] = {
            "error": "symbol is required"
        }
        return state
    
    # Ensure dependencies are loaded
    if not _ensure_vol_predictor_loaded():
        state["tool_results"]["vol_prediction"] = {
            "symbol": symbol,
            "error": "Vol prediction dependencies not available (pandas, sklearn, yfinance)"
        }
        return state
    
    try:
        predictor = get_predictor()
        result = predictor.predict(symbol)
        state["tool_results"]["vol_prediction"] = result
    except Exception as e:
        state["tool_results"]["vol_prediction"] = {
            "symbol": symbol,
            "error": f"Vol prediction failed: {str(e)}"
        }
    
    return state


def format_vol_prediction_summary(result: Dict[str, Any]) -> str:
    """Format vol prediction result for agent prompt."""
    if result.get("error"):
        return f"Vol Prediction: Error - {result['error']}"
    
    lines = [
        f"**Volatility Prediction for {result['symbol']}** (as of {result.get('date', 'N/A')}):",
        f"  Current Regime: {result['current_regime']} vol",
        f"  Risk Level: {result['risk_level']}",
    ]
    
    if result['current_regime'] == 'LOW':
        lines.append(f"  Spike Probability: {result['spike_probability']:.1%}")
        if result['spike_probability'] >= 0.5:
            lines.append("  ⚠️ WARNING: Elevated probability of volatility spike")
    else:
        lines.append(f"  Calm Probability: {result['calm_probability']:.1%}")
        if result['calm_probability'] >= 0.5:
            lines.append("  ✓ SIGNAL: Volatility likely to calm")
    
    lines.extend([
        f"  VIX Z-score: {result['vix_zscore']:+.2f}",
        f"  Suggested Action: {result['suggested_action']}",
    ])
    
    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("=" * 80)
    print("VOLATILITY PREDICTION TOOL TEST")
    print("=" * 80)
    
    state = {}
    
    for symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'XLE']:
        print(f"\nTesting {symbol}...")
        state = vol_prediction_tool_fn(state, {"symbol": symbol})
        result = state["tool_results"]["vol_prediction"]
        print(format_vol_prediction_summary(result))
