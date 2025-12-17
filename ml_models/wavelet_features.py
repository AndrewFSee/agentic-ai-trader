"""
Wavelet Features for ML Trading Models

Applies wavelet transforms to denoise price and return series.
Uses Daubechies wavelets for multi-scale decomposition.

Based on: Jansen, "Machine Learning for Algorithmic Trading" Chapter 4
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

# Lazy import pywavelets (may not be in base environment)
_PYWT_AVAILABLE = None
_pywt = None

def _ensure_pywt_loaded():
    """Lazy load pywavelets library."""
    global _PYWT_AVAILABLE, _pywt
    
    if _PYWT_AVAILABLE is not None:
        return _PYWT_AVAILABLE
    
    try:
        import pywt as pw
        _pywt = pw
        _PYWT_AVAILABLE = True
        return True
    except ImportError:
        print("Warning: PyWavelets not installed. Run: pip install PyWavelets")
        _PYWT_AVAILABLE = False
        return False


def wavelet_denoise(signal: pd.Series, 
                     wavelet: str = 'db6',
                     threshold_scale: float = 0.5,
                     mode: str = 'soft') -> pd.Series:
    """
    Denoise signal using wavelet transform and thresholding.
    
    Process:
    1. Decompose signal into wavelet coefficients (multi-scale)
    2. Threshold coefficients (remove small/noisy components)
    3. Reconstruct signal from remaining coefficients
    
    Args:
        signal: Signal to denoise (price or returns)
        wavelet: Wavelet family (default 'db6' = Daubechies 6)
        threshold_scale: Threshold = scale * max(signal) (default 0.5)
        mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised signal
    """
    if not _ensure_pywt_loaded():
        return signal  # Return original if pywt unavailable
    
    # Decompose signal using wavelet transform
    coefficients = _pywt.wavedec(signal.values, wavelet, mode='per')
    
    # Calculate threshold
    threshold = threshold_scale * abs(signal).max()
    
    # Threshold all detail coefficients (keep approximation coefficients[0] unchanged)
    coefficients[1:] = [_pywt.threshold(coeff, value=threshold, mode=mode) 
                         for coeff in coefficients[1:]]
    
    # Reconstruct signal from thresholded coefficients
    reconstructed = _pywt.waverec(coefficients, wavelet, mode='per')
    
    # Handle length mismatch (wavelet transform may add/remove samples)
    if len(reconstructed) > len(signal):
        reconstructed = reconstructed[:len(signal)]
    elif len(reconstructed) < len(signal):
        # Pad with last value
        reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)), 
                                mode='edge')
    
    return pd.Series(reconstructed, index=signal.index)


def wavelet_decompose_levels(signal: pd.Series,
                               wavelet: str = 'db6',
                               levels: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose signal into approximation (trend) and detail (noise) components.
    
    Args:
        signal: Signal to decompose
        wavelet: Wavelet family
        levels: Decomposition levels (default 3 = 3 detail scales)
    
    Returns:
        Tuple of (approximation, detail_1, detail_2, detail_3)
    """
    if not _ensure_pywt_loaded():
        # Fallback: use moving averages for approximation
        approx = signal.rolling(window=10, min_periods=1).mean()
        detail1 = signal - approx
        detail2 = pd.Series(0, index=signal.index)
        return approx, detail1, detail2
    
    # Multi-level decomposition
    coefficients = _pywt.wavedec(signal.values, wavelet, level=levels, mode='per')
    
    # Approximation (lowest frequency / trend)
    approx_coeffs = coefficients[0]
    # Pad to original length
    approx = np.repeat(approx_coeffs, len(signal) // len(approx_coeffs) + 1)[:len(signal)]
    
    # Details (high frequency / noise)
    details = []
    for i in range(1, min(levels + 1, len(coefficients))):
        detail_coeffs = coefficients[i]
        # Pad to original length
        detail = np.repeat(detail_coeffs, len(signal) // len(detail_coeffs) + 1)[:len(signal)]
        details.append(pd.Series(detail, index=signal.index))
    
    # Pad details list if needed
    while len(details) < 3:
        details.append(pd.Series(0, index=signal.index))
    
    return (pd.Series(approx, index=signal.index), details[0], details[1], details[2])


def add_wavelet_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add wavelet-based features to dataframe.
    
    Features added:
    - wavelet_price_smooth: Denoised price (threshold 0.5)
    - wavelet_price_aggressive: Heavily denoised price (threshold 0.1)
    - wavelet_returns_smooth: Denoised returns (threshold 0.5)
    - wavelet_returns_aggressive: Heavily denoised returns (threshold 0.1)
    - price_wavelet_residual: Price - denoised price (noise component)
    - wavelet_trend: Price approximation (low frequency trend)
    - wavelet_detail_1: High frequency details (day-to-day noise)
    - wavelet_detail_2: Medium frequency details (weekly patterns)
    - wavelet_noise_ratio: |residual| / price (signal-to-noise)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        verbose: Print progress
    
    Returns:
        DataFrame with wavelet features added
    """
    if verbose:
        print(f"  Adding wavelet features...")
    
    if not _ensure_pywt_loaded():
        if verbose:
            print(f"    Warning: PyWavelets not available, using moving averages as substitute")
        # Fallback to moving averages
        df['wavelet_price_smooth'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['wavelet_price_aggressive'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['wavelet_returns_smooth'] = df['close'].pct_change().rolling(window=5, min_periods=1).mean()
        df['wavelet_returns_aggressive'] = df['close'].pct_change().rolling(window=10, min_periods=1).mean()
        df['price_wavelet_residual'] = df['close'] - df['wavelet_price_smooth']
        df['wavelet_trend'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['wavelet_detail_1'] = df['close'].diff()
        df['wavelet_detail_2'] = df['close'].diff(5)
        df['wavelet_noise_ratio'] = abs(df['price_wavelet_residual']) / df['close']
        
        if verbose:
            print(f"    Added 9 wavelet features (using MA fallback)")
        return df
    
    # Denoise price with different thresholds
    df['wavelet_price_smooth'] = wavelet_denoise(
        df['close'],
        wavelet='db6',
        threshold_scale=0.5,  # Moderate denoising
        mode='soft'
    )
    
    df['wavelet_price_aggressive'] = wavelet_denoise(
        df['close'],
        wavelet='db6',
        threshold_scale=0.1,  # Heavy denoising
        mode='soft'
    )
    
    # Denoise returns
    returns = df['close'].pct_change().fillna(0)
    
    df['wavelet_returns_smooth'] = wavelet_denoise(
        returns,
        wavelet='db6',
        threshold_scale=0.5,
        mode='soft'
    )
    
    df['wavelet_returns_aggressive'] = wavelet_denoise(
        returns,
        wavelet='db6',
        threshold_scale=0.1,
        mode='soft'
    )
    
    # Residual (noise component)
    df['price_wavelet_residual'] = df['close'] - df['wavelet_price_smooth']
    
    # Multi-scale decomposition
    trend, detail1, detail2, _ = wavelet_decompose_levels(df['close'], wavelet='db6', levels=3)
    df['wavelet_trend'] = trend
    df['wavelet_detail_1'] = detail1  # High frequency (daily noise)
    df['wavelet_detail_2'] = detail2  # Medium frequency (weekly patterns)
    
    # Signal-to-noise ratio
    df['wavelet_noise_ratio'] = abs(df['price_wavelet_residual']) / (df['close'] + 1e-8)
    
    # Fill NaN values
    for col in ['wavelet_price_smooth', 'wavelet_price_aggressive', 'wavelet_returns_smooth',
                'wavelet_returns_aggressive', 'price_wavelet_residual', 'wavelet_trend',
                'wavelet_detail_1', 'wavelet_detail_2', 'wavelet_noise_ratio']:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    if verbose:
        print(f"    Added 9 wavelet features")
        avg_noise_ratio = df['wavelet_noise_ratio'].mean()
        print(f"    Average noise ratio: {avg_noise_ratio:.4f}")
    
    return df


if __name__ == "__main__":
    # Test wavelet features
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_collection import fetch_daily_bars
    
    print("Testing wavelet features...")
    
    # Fetch AAPL data
    df = fetch_daily_bars("AAPL", "2024-01-01", "2025-12-15")
    print(f"Fetched {len(df)} rows for AAPL")
    
    # Add wavelet features
    df = add_wavelet_features(df, verbose=True)
    
    print("\nSample wavelet features:")
    wavelet_cols = ['close', 'wavelet_price_smooth', 'wavelet_price_aggressive', 
                    'price_wavelet_residual', 'wavelet_noise_ratio']
    print(df[wavelet_cols].tail(10))
    
    print("\nNoise statistics:")
    print(f"Average noise ratio: {df['wavelet_noise_ratio'].mean():.4f}")
    print(f"Price std: {df['close'].std():.2f}")
    print(f"Smooth price std: {df['wavelet_price_smooth'].std():.2f}")
    print(f"Aggressive smooth std: {df['wavelet_price_aggressive'].std():.2f}")
