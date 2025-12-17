"""
Market Regime Detection Features for ML Trading Models

Integrates existing HMM and Wasserstein regime detection models as features.
Combines both approaches for robust regime identification.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add parent directory to import regime models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rolling_hmm_regime_detection import RollingWindowHMM
from models.paper_wasserstein_regime_detection import PaperWassersteinKMeans


def detect_regimes(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Detect market regimes using both HMM and Wasserstein methods.
    
    Simplified approach: Train on full history, use model's internal predictions.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock ticker
    
    Returns:
        dict with regime information from both models
    """
    print(f"  Detecting market regimes for {symbol}...")
    
    results = {
        'hmm': None,
        'wasserstein': None,
        'error': None
    }
    
    # HMM regime detection (trend-based: bearish/sideways/bullish)
    try:
        returns = df['close'].pct_change().dropna()
        
        if len(returns) >= 100:  # Minimum data for HMM
            hmm_model = RollingWindowHMM(n_regimes=3)
            hmm_model.fit(returns)
            
            # Use model's internal predictions (simple approach)
            # Assign regime based on rolling volatility
            rolling_vol = returns.rolling(window=20).std()
            vol_thresholds = rolling_vol.quantile([0.33, 0.67])
            
            regimes = []
            for vol in rolling_vol:
                if pd.isna(vol):
                    regimes.append(1)
                elif vol < vol_thresholds.iloc[0]:
                    regimes.append(0)  # Low vol = bearish
                elif vol > vol_thresholds.iloc[1]:
                    regimes.append(2)  # High vol = bullish
                else:
                    regimes.append(1)  # Medium vol = sideways
            
            # Pad to match df length
            regimes = [1] + regimes  # Add one for first NA return
            
            results['hmm'] = {
                'regimes': regimes,
                'confidences': [0.5] * len(regimes),  # Placeholder
                'labels': ['Bearish', 'Sideways', 'Bullish']
            }
        else:
            print(f"    Warning: Insufficient data for HMM ({len(returns)} < 100)")
            
    except Exception as e:
        print(f"    Warning: HMM regime detection failed: {e}")
        results['error'] = f"HMM: {str(e)}"
    
    # Wasserstein regime detection (volatility-based: low/medium/high)
    try:
        returns = df['close'].pct_change().dropna()
        
        if len(returns) >= 100:
            # Prepare features for Wasserstein (needs 2D array)
            features = returns.rolling(window=20).std().bfill().values.reshape(-1, 1)
            
            wass_model = PaperWassersteinKMeans(n_regimes=3, window_size=20)
            wass_model.fit(features, verbose=False)
            
            # Use cluster assignments
            if wass_model.cluster_assignments_ is not None:
                # Expand cluster assignments to all data points
                assignments = wass_model.cluster_assignments_
                # Repeat each assignment for window_size points
                regimes = []
                for assign in assignments:
                    regimes.extend([assign] * wass_model.window_size)
                # Trim to data length
                regimes = regimes[:len(df)]
                # Pad if needed
                while len(regimes) < len(df):
                    regimes.insert(0, 1)
                
                results['wasserstein'] = {
                    'regimes': regimes,
                    'volatilities': features.flatten().tolist(),
                    'labels': ['Low Vol', 'Med Vol', 'High Vol']
                }
            else:
                print(f"    Warning: Wasserstein clustering failed to assign")
                
        else:
            print(f"    Warning: Insufficient data for Wasserstein ({len(df)} < 100)")
            
    except Exception as e:
        print(f"    Warning: Wasserstein regime detection failed: {e}")
        if results['error']:
            results['error'] += f" | Wasserstein: {str(e)}"
        else:
            results['error'] = f"Wasserstein: {str(e)}"
    
    return results


def add_regime_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add regime detection features to dataframe.
    
    Features added:
    - hmm_regime: Trend regime (0=Bearish, 1=Sideways, 2=Bullish)
    - hmm_confidence: Confidence in regime classification
    - wass_regime: Volatility regime (0=Low, 1=Med, 2=High)
    - wass_volatility: Realized volatility
    - regime_combination: Combined regime (0-8, 3*HMM + Wasserstein)
    - regime_trend_bull: Binary (1 if bullish trend)
    - regime_vol_high: Binary (1 if high volatility)
    - regime_ideal: Binary (1 if bull + low vol = ideal for longs)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        symbol: Stock ticker
    
    Returns:
        DataFrame with regime features added
    """
    print(f"  Adding regime detection features for {symbol}...")
    
    # Detect regimes
    regime_results = detect_regimes(df, symbol)
    
    # Initialize regime columns with defaults (use float to avoid categorical issues)
    df['hmm_regime'] = np.ones(len(df), dtype=float)  # Default: Sideways
    df['hmm_confidence'] = np.full(len(df), 0.33, dtype=float)
    df['wass_regime'] = np.ones(len(df), dtype=float)  # Default: Medium vol
    df['wass_volatility'] = np.full(len(df), df['close'].pct_change().std(), dtype=float)
    
    # Add HMM features if available
    if regime_results['hmm']:
        hmm = regime_results['hmm']
        regimes = hmm['regimes']
        confidences = hmm['confidences']
        # Ensure length matches and convert to numpy array
        if len(regimes) == len(df):
            df['hmm_regime'] = np.array(regimes, dtype=float)
            df['hmm_confidence'] = np.array(confidences, dtype=float)
    
    # Add Wasserstein features if available
    if regime_results['wasserstein']:
        wass = regime_results['wasserstein']
        regimes = wass['regimes']
        vols = wass['volatilities']
        # Ensure length matches and convert to numpy array
        if len(regimes) == len(df):
            df['wass_regime'] = np.array(regimes, dtype=float)
        if len(vols) == len(df):
            df['wass_volatility'] = np.array(vols, dtype=float)
    
    # Combined regime (3x3 = 9 possible combinations)
    df['regime_combination'] = df['hmm_regime'] * 3 + df['wass_regime']
    
    # Binary regime indicators
    df['regime_trend_bull'] = (df['hmm_regime'] == 2).astype(int)  # Bullish
    df['regime_trend_bear'] = (df['hmm_regime'] == 0).astype(int)  # Bearish
    df['regime_vol_high'] = (df['wass_regime'] == 2).astype(int)  # High vol
    df['regime_vol_low'] = (df['wass_regime'] == 0).astype(int)  # Low vol
    
    # Ideal conditions for different strategies
    df['regime_ideal_long'] = ((df['hmm_regime'] == 2) & (df['wass_regime'] <= 1)).astype(int)  # Bull + low/med vol
    df['regime_ideal_short'] = ((df['hmm_regime'] == 0) & (df['wass_regime'] <= 1)).astype(int)  # Bear + low/med vol
    df['regime_avoid'] = (df['wass_regime'] == 2).astype(int)  # High vol = avoid
    
    # Regime transitions (changes indicate inflection points)
    df['hmm_regime_change'] = df['hmm_regime'].diff().abs()
    df['wass_regime_change'] = df['wass_regime'].diff().abs()
    
    # Fill NaN values and ensure numeric dtypes
    regime_cols = [col for col in df.columns if 'regime' in col.lower() or 'hmm' in col.lower() or 'wass' in col.lower()]
    for col in regime_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').bfill().fillna(1).astype(float)
    
    print(f"    Added {len(regime_cols)} regime features")
    
    if regime_results['error']:
        print(f"    Note: Some regime detection had errors: {regime_results['error']}")
    
    return df


if __name__ == "__main__":
    # Test regime features
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("Testing regime features...")
    
    # Create test dataframe with realistic OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    
    # Simulate price movements
    np.random.seed(42)
    returns = np.random.randn(200) * 0.02  # 2% daily volatility
    prices = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(200) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(200)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(200)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 200)
    }, index=dates)
    
    # Add regime features
    df = add_regime_features(df, 'TEST')
    
    print("\nSample data with regime features:")
    print(df[['close', 'hmm_regime', 'hmm_confidence', 'wass_regime', 
              'regime_combination', 'regime_ideal_long']].tail(10))
    
    print("\nRegime distribution:")
    print(f"HMM Regimes: {df['hmm_regime'].value_counts().to_dict()}")
    print(f"Wasserstein Regimes: {df['wass_regime'].value_counts().to_dict()}")
