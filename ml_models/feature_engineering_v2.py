"""
Improved feature engineering with regime detection and feature interactions.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rolling_hmm_regime_detection import RollingWindowHMM
from models.paper_wasserstein_regime_detection import PaperWassersteinKMeans


def add_regime_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add regime detection features from HMM and Wasserstein models.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        
    Returns:
        DataFrame with added regime features
    """
    print(f"  Adding regime features for {symbol}...")
    
    # Prepare data for regime detection (need returns and volatility)
    regime_data = df[['close']].copy()
    regime_data['returns'] = regime_data['close'].pct_change()
    regime_data['volatility'] = regime_data['returns'].rolling(window=20).std()
    
    # Drop NaN rows for regime detection
    regime_data = regime_data.dropna()
    
    if len(regime_data) < 100:
        print(f"    Warning: Not enough data for regime detection ({len(regime_data)} rows)")
        df['regime_hmm'] = 1  # Sideways default
        df['regime_hmm_prob'] = 0.33
        df['regime_vol'] = 1  # Medium default
        df['regime_vol_confidence'] = 0.33
        return df
    
    try:
        # HMM regime detection (trend-based)
        hmm = RollingWindowHMM(n_regimes=3, window_size=252*3)  # 3-year window
        hmm_df = pd.DataFrame({
            'returns': regime_data['returns'].values
        }, index=regime_data.index)
        hmm.fit(hmm_df)
        
        # Get regime labels and confidence
        hmm_regimes = hmm.labels_
        hmm_probs = hmm.regime_probs_
        hmm_confidence = np.max(hmm_probs, axis=1)
        
        # Map regimes by mean return: 0=bearish, 1=sideways, 2=bullish
        regime_means = []
        for i in range(3):
            regime_returns = regime_data['returns'].values[hmm_regimes == i]
            regime_means.append(np.mean(regime_returns) if len(regime_returns) > 0 else 0)
        
        regime_order = np.argsort(regime_means)
        regime_map = {old: new for new, old in enumerate(regime_order)}
        hmm_regimes_ordered = np.array([regime_map[r] for r in hmm_regimes])
        
        # Align with original dataframe
        df.loc[regime_data.index, 'regime_hmm'] = hmm_regimes_ordered
        df.loc[regime_data.index, 'regime_hmm_prob'] = hmm_confidence
        
        # Forward fill for NaN rows
        df['regime_hmm'] = df['regime_hmm'].ffill().fillna(1)
        df['regime_hmm_prob'] = df['regime_hmm_prob'].ffill().fillna(0.33)
        
    except Exception as e:
        print(f"    Warning: HMM regime detection failed: {e}")
        df['regime_hmm'] = 1
        df['regime_hmm_prob'] = 0.33
    
    try:
        # Wasserstein regime detection (volatility-based)
        # Use rolling windows of volatility
        vol_values = regime_data['volatility'].values.reshape(-1, 1)
        
        wass = PaperWassersteinKMeans(n_clusters=3, max_iter=50, window_size=20)
        wass_regimes = wass.fit_predict(vol_values)
        wass_centers = wass.centers_
        
        # Sort by volatility level: 0=low, 1=medium, 2=high
        vol_order = np.argsort(wass_centers.flatten())
        vol_map = {old: new for new, old in enumerate(vol_order)}
        wass_regimes_ordered = np.array([vol_map[r] for r in wass_regimes])
        
        # Calculate confidence as inverse distance to center
        distances = np.abs(vol_values - wass_centers[wass_regimes].reshape(-1, 1))
        wass_confidence = 1 / (1 + distances.flatten() * 100)  # Scale to [0, 1]
        
        # Align with original dataframe
        df.loc[regime_data.index, 'regime_vol'] = wass_regimes_ordered
        df.loc[regime_data.index, 'regime_vol_confidence'] = wass_confidence
        
        # Forward fill
        df['regime_vol'] = df['regime_vol'].ffill().fillna(1)
        df['regime_vol_confidence'] = df['regime_vol_confidence'].ffill().fillna(0.33)
        
    except Exception as e:
        print(f"    Warning: Wasserstein regime detection failed: {e}")
        df['regime_vol'] = 1
        df['regime_vol_confidence'] = 0.33
    
    # Add derived regime features
    df['regime_days'] = df.groupby('regime_hmm').cumcount() + 1  # Days in current regime
    df['regime_change'] = (df['regime_hmm'].diff() != 0).astype(int)  # Regime transition
    
    print(f"    Added 6 regime features")
    return df


def add_feature_interactions(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Add important feature interactions (products and ratios).
    
    Args:
        df: DataFrame with features
        feature_list: List of feature names to create interactions from
        
    Returns:
        DataFrame with added interaction features
    """
    print("  Adding feature interactions...")
    
    # Define important interactions based on trading logic
    interactions = [
        # Momentum * Volume (strong move with volume = more reliable)
        ('momentum_5d', 'volume_sma_ratio'),
        ('momentum_10d', 'volume_sma_ratio'),
        
        # RSI * Volume (oversold/overbought with volume)
        ('rsi', 'volume_sma_ratio'),
        
        # Return * Volatility (risk-adjusted returns)
        ('return_5d', 'volatility_5d'),
        ('return_10d', 'volatility_10d'),
        
        # MACD * RSI (momentum confirmation)
        ('macd_histogram', 'rsi'),
        
        # Bollinger position * Volume
        ('bb_position', 'volume_sma_ratio'),
        
        # Regime * Return (regime-specific momentum)
        ('regime_hmm', 'return_5d'),
        ('regime_vol', 'volatility_5d'),
    ]
    
    count = 0
    for feat1, feat2 in interactions:
        if feat1 in df.columns and feat2 in df.columns:
            # Product interaction
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            count += 1
            
            # Ratio interaction (avoid division by zero)
            if (df[feat2] != 0).all():
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
                count += 1
    
    print(f"    Added {count} interaction features")
    return df


def add_polynomial_features(df: pd.DataFrame, top_features: list, degree: int = 2) -> pd.DataFrame:
    """
    Add polynomial features for top important features.
    
    Args:
        df: DataFrame with features
        top_features: List of most important features
        degree: Polynomial degree (2 for squared terms)
        
    Returns:
        DataFrame with polynomial features
    """
    print(f"  Adding polynomial features (degree={degree})...")
    
    count = 0
    for feat in top_features:
        if feat in df.columns:
            # Squared term
            df[f'{feat}_squared'] = df[feat] ** 2
            count += 1
            
            # Cubic term (if degree >= 3)
            if degree >= 3:
                df[f'{feat}_cubed'] = df[feat] ** 3
                count += 1
    
    print(f"    Added {count} polynomial features")
    return df


def engineer_features_v2(df: pd.DataFrame, symbol: str, config: dict) -> pd.DataFrame:
    """
    Enhanced feature engineering with regime and interactions.
    
    This is version 2 that adds:
    1. Regime features from HMM and Wasserstein
    2. Feature interactions
    3. Polynomial terms
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        config: Configuration dict
        
    Returns:
        DataFrame with all features
    """
    print(f"\nEngineering features v2 for {symbol}...")
    
    # Import original feature engineering
    from feature_engineering import (
        calculate_returns,
        calculate_technical_indicators,
        calculate_volume_features,
        calculate_volatility_features,
        calculate_momentum_features,
        calculate_market_relative_features,
        calculate_seasonality_features,
        create_target_variables
    )
    
    # Step 1: Original features
    print("  Calculating original features...")
    df = calculate_returns(df)
    df = calculate_technical_indicators(df)
    df = calculate_volume_features(df)
    df = calculate_volatility_features(df)
    df = calculate_momentum_features(df)
    
    # Note: Market relative features require SPY data (skip for now)
    # df = calculate_market_relative_features(df, spy_df)
    
    df = calculate_seasonality_features(df)
    
    # Create target variables manually since function uses global config
    for horizon in config['prediction_horizons']:
        # Forward returns
        df[f'target_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
        # Binary classification (up vs down) - must match train_models.py expectations
        df[f'target_direction_{horizon}d'] = (df[f'target_{horizon}d'] > 0).astype(int)
    
    # Step 2: Add regime features
    df = add_regime_features(df, symbol)
    
    # Step 3: Add feature interactions
    feature_list = [col for col in df.columns if col not in 
                   ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    df = add_feature_interactions(df, feature_list)
    
    # Step 4: Add polynomial features for top features
    # Use top technical indicators
    top_features = ['rsi', 'macd_histogram', 'bb_position', 'momentum_5d', 
                   'volatility_10d', 'volume_sma_ratio', 'return_5d']
    df = add_polynomial_features(df, top_features, degree=2)
    
    print(f"  Total features: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])}")
    
    return df


if __name__ == "__main__":
    # Test on single stock
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_collection import fetch_daily_bars
    from config import CONFIG
    
    print("Testing enhanced feature engineering on AAPL...")
    
    # Fetch data
    df = fetch_daily_bars("AAPL", "2020-01-01", "2024-12-15")
    
    # Engineer features
    df_features = engineer_features_v2(df, "AAPL", CONFIG)
    
    # Print feature summary
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"\nRegime features:")
    regime_features = [f for f in feature_cols if 'regime' in f]
    print(f"  {regime_features}")
    
    print(f"\nInteraction features:")
    interaction_features = [f for f in feature_cols if '_x_' in f or '_div_' in f]
    print(f"  Count: {len(interaction_features)}")
    print(f"  Examples: {interaction_features[:5]}")
    
    print(f"\nPolynomial features:")
    poly_features = [f for f in feature_cols if 'squared' in f or 'cubed' in f]
    print(f"  {poly_features}")
    
    print(f"\nMissing values:")
    print(df_features[feature_cols].isnull().sum().sum())
    
    print("\n" + "="*80)
    print("Test complete!")
