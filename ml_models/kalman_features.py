"""
Kalman Filter Features for ML Trading Models

Applies Kalman filtering to denoise price series and create
smoothed alpha factors with adaptive estimation.

Based on: Jansen, "Machine Learning for Algorithmic Trading" Chapter 4
"""

import pandas as pd
import numpy as np
from typing import Optional

# Lazy import pykalman (not in base environment)
_PYKALMAN_AVAILABLE = None
_KalmanFilter = None

def _ensure_pykalman_loaded():
    """Lazy load pykalman library."""
    global _PYKALMAN_AVAILABLE, _KalmanFilter
    
    if _PYKALMAN_AVAILABLE is not None:
        return _PYKALMAN_AVAILABLE
    
    try:
        from pykalman import KalmanFilter
        _KalmanFilter = KalmanFilter
        _PYKALMAN_AVAILABLE = True
        return True
    except ImportError:
        print("Warning: pykalman not installed. Run: pip install pykalman")
        _PYKALMAN_AVAILABLE = False
        return False


def kalman_smooth_price(prices: pd.Series, 
                         transition_covariance: float = 0.01,
                         observation_covariance: float = 1.0) -> pd.Series:
    """
    Apply Kalman filter to smooth price series.
    
    The Kalman filter is a recursive Bayesian estimator that:
    1. Predicts the next state based on the current state
    2. Updates the estimate based on new observations
    3. Adapts more sensitively to changes than fixed moving averages
    
    Args:
        prices: Price series to smooth
        transition_covariance: Process noise (lower = smoother, default 0.01)
        observation_covariance: Measurement noise (higher = smoother, default 1.0)
    
    Returns:
        Smoothed price series
    """
    if not _ensure_pykalman_loaded():
        return prices  # Return original if pykalman unavailable
    
    # Initialize Kalman filter
    # Assumes random walk model: x_t = x_{t-1} + noise
    kf = _KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=prices.iloc[0],
        initial_state_covariance=1,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Run forward filter to get state estimates
    state_means, _ = kf.filter(prices.values)
    
    return pd.Series(state_means.flatten(), index=prices.index)


def kalman_smooth_returns(returns: pd.Series,
                           transition_covariance: float = 0.001,
                           observation_covariance: float = 1.0) -> pd.Series:
    """
    Apply Kalman filter to smooth return series.
    
    Useful for denoising alpha factors (return predictions).
    Lower transition_covariance = smoother but less responsive.
    
    Args:
        returns: Return series to smooth
        transition_covariance: Process noise (default 0.001 for returns)
        observation_covariance: Measurement noise (default 1.0)
    
    Returns:
        Smoothed return series
    """
    if not _ensure_pykalman_loaded():
        return returns
    
    # Initialize Kalman filter for returns (mean-reverting around 0)
    kf = _KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Run filter
    state_means, _ = kf.filter(returns.values)
    
    return pd.Series(state_means.flatten(), index=returns.index)


def add_kalman_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add Kalman filter features to dataframe.
    
    Features added:
    - kalman_price: Smoothed price (adaptive moving average)
    - kalman_returns: Smoothed returns (denoised alpha signal)
    - price_kalman_residual: Price - Kalman estimate (mean reversion signal)
    - kalman_price_ratio: Price / Kalman price (oversold/overbought)
    - kalman_momentum: Kalman price change over 5 days
    - kalman_trend: Binary (1 if Kalman price > 20-day MA)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        verbose: Print progress
    
    Returns:
        DataFrame with Kalman features added
    """
    if verbose:
        print(f"  Adding Kalman filter features...")
    
    if not _ensure_pykalman_loaded():
        if verbose:
            print(f"    Warning: pykalman not available, using moving averages as substitute")
        # Fallback to moving averages
        df['kalman_price'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['kalman_returns'] = df['close'].pct_change().rolling(window=5, min_periods=1).mean()
        df['price_kalman_residual'] = df['close'] - df['kalman_price']
        df['kalman_price_ratio'] = df['close'] / df['kalman_price']
        df['kalman_momentum'] = df['kalman_price'].diff(5)
        df['kalman_trend'] = (df['kalman_price'] > df['kalman_price'].rolling(20, min_periods=1).mean()).astype(int)
        
        if verbose:
            print(f"    Added 6 Kalman features (using MA fallback)")
        return df
    
    # Smooth price with Kalman filter
    df['kalman_price'] = kalman_smooth_price(
        df['close'],
        transition_covariance=0.01,
        observation_covariance=1.0
    )
    
    # Smooth returns with Kalman filter
    returns = df['close'].pct_change()
    df['kalman_returns'] = kalman_smooth_returns(
        returns.fillna(0),
        transition_covariance=0.001,
        observation_covariance=1.0
    )
    
    # Derived features
    df['price_kalman_residual'] = df['close'] - df['kalman_price']
    df['kalman_price_ratio'] = df['close'] / df['kalman_price']
    df['kalman_momentum'] = df['kalman_price'].diff(5)
    
    # Kalman trend (above/below 20-day MA of Kalman price)
    kalman_ma_20 = df['kalman_price'].rolling(window=20, min_periods=1).mean()
    df['kalman_trend'] = (df['kalman_price'] > kalman_ma_20).astype(int)
    
    # Fill NaN values
    df['kalman_price'] = df['kalman_price'].fillna(method='ffill').fillna(df['close'])
    df['kalman_returns'] = df['kalman_returns'].fillna(0)
    df['price_kalman_residual'] = df['price_kalman_residual'].fillna(0)
    df['kalman_price_ratio'] = df['kalman_price_ratio'].fillna(1.0)
    df['kalman_momentum'] = df['kalman_momentum'].fillna(0)
    df['kalman_trend'] = df['kalman_trend'].fillna(0)
    
    if verbose:
        print(f"    Added 6 Kalman filter features")
        avg_residual = abs(df['price_kalman_residual']).mean()
        print(f"    Average price-Kalman residual: ${avg_residual:.2f}")
    
    return df


if __name__ == "__main__":
    # Test Kalman filter features
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_collection import fetch_daily_bars
    
    print("Testing Kalman filter features...")
    
    # Fetch AAPL data (2008-2009 financial crisis for dramatic movements)
    df = fetch_daily_bars("AAPL", "2024-01-01", "2025-12-15")
    print(f"Fetched {len(df)} rows for AAPL")
    
    # Add Kalman features
    df = add_kalman_features(df, verbose=True)
    
    print("\nSample Kalman features:")
    kalman_cols = ['close', 'kalman_price', 'price_kalman_residual', 'kalman_price_ratio', 'kalman_trend']
    print(df[kalman_cols].tail(10))
    
    print("\nKalman vs Price statistics:")
    print(f"Price std: {df['close'].std():.2f}")
    print(f"Kalman price std: {df['kalman_price'].std():.2f}")
    print(f"Noise reduction: {(1 - df['kalman_price'].std() / df['close'].std()) * 100:.1f}%")
