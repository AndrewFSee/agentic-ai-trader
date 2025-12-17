"""
Feature engineering for ML models.
Creates technical, volume, volatility, momentum, and regime features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from config import TECHNICAL_PARAMS, FEATURE_GROUPS, PREDICTION_HORIZONS, VERBOSE


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various return metrics."""
    df = df.copy()
    
    # Simple returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)
    
    # Log returns
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price momentum (scaled returns)
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # NEW: Gap statistics (predictive of volatility and intraday moves)
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)
    df['gap_up'] = (df['gap'] > 0).astype(int)
    df['gap_down'] = (df['gap'] < 0).astype(int)
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, Bollinger Bands, SMAs, EMAs."""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=TECHNICAL_PARAMS['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=TECHNICAL_PARAMS['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
    
    # NEW: RSI divergence (price vs momentum indicator)
    # Normalize price change to compare with RSI
    price_change_20d = (df['close'] / df['close'].shift(20) - 1) * 100  # Scale to RSI range
    df['rsi_divergence'] = price_change_20d - df['rsi']
    df['rsi_divergence_normalized'] = df['rsi_divergence'] / 100
    
    # MACD
    ema_fast = df['close'].ewm(span=TECHNICAL_PARAMS['macd_fast'], adjust=False).mean()
    ema_slow = df['close'].ewm(span=TECHNICAL_PARAMS['macd_slow'], adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=TECHNICAL_PARAMS['macd_signal'], adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']  # Normalize by price
    
    # Bollinger Bands
    sma = df['close'].rolling(window=TECHNICAL_PARAMS['bb_period']).mean()
    std = df['close'].rolling(window=TECHNICAL_PARAMS['bb_period']).std()
    df['bb_upper'] = sma + (std * TECHNICAL_PARAMS['bb_std'])
    df['bb_lower'] = sma - (std * TECHNICAL_PARAMS['bb_std'])
    df['bb_middle'] = sma
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Normalized width
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])  # %B
    
    # SMAs
    df['sma_20'] = df['close'].rolling(window=TECHNICAL_PARAMS['sma_short']).mean()
    df['sma_50'] = df['close'].rolling(window=TECHNICAL_PARAMS['sma_medium']).mean()
    df['sma_200'] = df['close'].rolling(window=TECHNICAL_PARAMS['sma_long']).mean()
    
    # Price relative to SMAs
    df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_to_sma200'] = df['close'] / df['sma_200'] - 1
    
    # SMA crossovers
    df['sma20_cross_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['sma50_cross_sma200'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # EMAs
    for period in TECHNICAL_PARAMS['ema_periods']:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema{period}'] = df['close'] / df[f'ema_{period}'] - 1
    
    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features."""
    df = df.copy()
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(window=TECHNICAL_PARAMS['volume_sma']).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volume momentum
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_normalized'] = df['obv'] / df['obv'].rolling(window=50).mean()
    
    # Volume-price correlation
    df['volume_price_corr_10d'] = df['close'].rolling(10).corr(df['volume'])
    
    return df


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility-based features."""
    df = df.copy()
    
    # Realized volatility (multiple windows)
    df['realized_vol_5d'] = df['log_return_1d'].rolling(window=5).std() * np.sqrt(252)
    df['realized_vol_10d'] = df['log_return_1d'].rolling(window=10).std() * np.sqrt(252)
    df['realized_vol_20d'] = df['log_return_1d'].rolling(window=20).std() * np.sqrt(252)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=TECHNICAL_PARAMS['atr_period']).mean()
    df['atr_normalized'] = df['atr'] / df['close']  # Normalize by price
    
    # Volatility regime
    df['vol_regime'] = pd.cut(df['realized_vol_20d'], bins=3, labels=['low', 'medium', 'high'])
    df['vol_regime_low'] = (df['vol_regime'] == 'low').astype(int)
    df['vol_regime_medium'] = (df['vol_regime'] == 'medium').astype(int)
    df['vol_regime_high'] = (df['vol_regime'] == 'high').astype(int)
    
    # Parkinson volatility (high-low estimator)
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2)
    df['parkinson_vol_20d'] = df['parkinson_vol'].rolling(window=20).mean() * np.sqrt(252)
    
    return df


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum features."""
    df = df.copy()
    
    # Rate of change (ROC)
    for period in [5, 10, 20, 50]:
        df[f'roc_{period}d'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
    
    # Trend strength (consecutive up/down days)
    df['up_days'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_ups'] = df['up_days'].groupby((df['up_days'] != df['up_days'].shift()).cumsum()).cumsum()
    df['consecutive_downs'] = (~df['up_days'].astype(bool)).astype(int).groupby((df['up_days'] != df['up_days'].shift()).cumsum()).cumsum()
    
    # Drawdown
    rolling_max = df['close'].rolling(window=252, min_periods=1).max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max
    df['drawdown_duration'] = (df['close'] < rolling_max).astype(int).groupby((df['close'] >= rolling_max).cumsum()).cumsum()
    
    # NEW: Day-of-week patterns (optimal buy day indicator)
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 4=Friday
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    return df


def calculate_market_relative_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features relative to market (SPY)."""
    df = df.copy()
    
    # Align dates
    spy_returns = spy_df['close'].pct_change()
    
    # Beta (20-day rolling)
    stock_returns = df['return_1d']
    df['beta_20d'] = stock_returns.rolling(20).cov(spy_returns) / spy_returns.rolling(20).var()
    
    # Alpha (excess return vs SPY)
    df['alpha_5d'] = df['return_5d'] - spy_returns.rolling(5).sum()
    df['alpha_10d'] = df['return_10d'] - spy_returns.rolling(10).sum()
    
    # Relative strength
    df['relative_strength'] = df['close'] / spy_df['close']
    df['relative_strength_change'] = df['relative_strength'].pct_change(20)
    
    return df


def calculate_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate seasonality features."""
    df = df.copy()
    
    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Month
    df['month'] = df.index.month
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    
    # Quarter
    df['quarter'] = df.index.quarter
    
    return df


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for supervised learning."""
    df = df.copy()
    
    for horizon in PREDICTION_HORIZONS:
        # Forward returns
        df[f'target_return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Binary classification (up vs down)
        df[f'target_direction_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)
        
        # Multi-class classification (strong down, down, flat, up, strong up)
        df[f'target_class_{horizon}d'] = pd.cut(
            df[f'target_return_{horizon}d'],
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=['strong_down', 'down', 'flat', 'up', 'strong_up']
        ).cat.codes.astype(float)  # Convert to numeric codes (0, 1, 2, 3, 4)
    
    return df


def engineer_features(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        symbol: Stock ticker
        df: Raw OHLCV data
        spy_df: SPY data for market-relative features
    
    Returns:
        DataFrame with engineered features
    """
    if VERBOSE:
        print(f"Engineering features for {symbol}...")
    
    df = df.copy()
    
    # Basic return features
    if FEATURE_GROUPS['price_features']:
        df = calculate_returns(df)
    
    # Technical indicators
    if FEATURE_GROUPS['technical']:
        df = calculate_technical_indicators(df)
    
    # Volume features
    if FEATURE_GROUPS['volume']:
        df = calculate_volume_features(df)
    
    # Volatility features
    if FEATURE_GROUPS['volatility']:
        df = calculate_volatility_features(df)
    
    # Momentum features
    if FEATURE_GROUPS['momentum']:
        df = calculate_momentum_features(df)
    
    # Market-relative features
    if FEATURE_GROUPS['market_relative'] and spy_df is not None:
        df = calculate_market_relative_features(df, spy_df)
    
    # Seasonality
    if FEATURE_GROUPS['seasonality']:
        df = calculate_seasonality_features(df)
    
    # NEW: Sentiment features (GDELT disabled for real-time to avoid hangs)
    try:
        from sentiment_features import add_sentiment_features
        df = add_sentiment_features(df, symbol, use_gdelt=False)  # Disable GDELT for real-time (too slow/unreliable)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add sentiment features: {e}")
    
    # NEW: Regime detection features
    try:
        from regime_features import add_regime_features
        df = add_regime_features(df, symbol)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add regime features: {e}")
    
    # NEW: Fundamental features
    try:
        from fundamental_features import add_fundamental_features
        df = add_fundamental_features(df, symbol)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add fundamental features: {e}")
    
    # NEW: Options features
    try:
        from options_features import add_options_features
        df = add_options_features(df, symbol)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add options features: {e}")
    
    # NEW: VIX features (market fear gauge)
    try:
        from vix_features import add_vix_features
        df = add_vix_features(df, symbol, verbose=VERBOSE)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add VIX features: {e}")
    
    # NEW: Kalman filter features (adaptive smoothing)
    try:
        from kalman_features import add_kalman_features
        df = add_kalman_features(df, verbose=VERBOSE)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add Kalman features: {e}")
    
    # NEW: Wavelet features (multi-scale denoising)
    try:
        from wavelet_features import add_wavelet_features
        df = add_wavelet_features(df, verbose=VERBOSE)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add wavelet features: {e}")
    
    # NEW: Macro correlation features (USD, oil, gold)
    try:
        from macro_features import add_macro_features
        df = add_macro_features(df, symbol, verbose=VERBOSE)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add macro features: {e}")
    
    # NEW: Market breadth features (advance-decline)
    try:
        from breadth_features import add_breadth_features
        df = add_breadth_features(df, symbol, verbose=VERBOSE)
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not add breadth features: {e}")
    
    # Target variables (must be last)
    df = create_target_variables(df)
    
    # DROP low-importance features (based on feature importance analysis Dec 16, 2024)
    # These 30 features had 0.000 importance across all XGBoost models and horizons
    # Dropping them reduces noise and should improve model performance
    FEATURES_TO_DROP = [
        # Macro features (16 total, all had 0.000 importance)
        'usd_corr_10d', 'usd_corr_20d', 'usd_corr_50d', 'usd_momentum_divergence',
        'oil_corr_10d', 'oil_corr_20d', 'oil_corr_50d', 'oil_momentum_divergence',
        'gold_corr_10d', 'gold_corr_20d', 'gold_corr_50d', 'gold_momentum_divergence',
        'macro_risk_off', 'macro_risk_on',
        # Breadth features (8 total, all had 0.000 importance)
        'ad_line', 'ad_ratio', 'ad_line_ma20', 'ad_ratio_ma20',
        'breadth_momentum_10d', 'breadth_divergence', 'breadth_strong', 'breadth_weak',
        # Fundamental features (8 total, no data or 0.000 importance)
        'debt_to_equity', 'payout_ratio', 'roa', 'roe', 'current_ratio',
        'pe_ratio', 'pb_ratio', 'peg_ratio', 'pe_to_growth', 'forward_pe',
        'gross_margin', 'operating_margin', 'profit_margin', 'earnings_growth',
        'revenue_growth', 'financial_health', 'quality_score', 'valuation_score',
        'dividend_yield', 'dividend_attractive',
        # Other low-importance features
        'sentiment_extreme', 'wass_regime_change', 'wass_volatility', 'regime_trend_bull'
    ]
    
    # Drop features that exist in the dataframe
    features_dropped = []
    for feature in FEATURES_TO_DROP:
        if feature in df.columns:
            df = df.drop(columns=[feature])
            features_dropped.append(feature)
    
    if VERBOSE and features_dropped:
        print(f"  Dropped {len(features_dropped)} low-importance features for noise reduction")
    
    # Add metadata
    df['symbol'] = symbol
    
    if VERBOSE:
        print(f"  Created {len(df.columns)} features (after dropping {len(features_dropped)})")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding targets and metadata)."""
    exclude_patterns = ['target_', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                       'vwap', 'transactions', 'timestamp', 'vol_regime']
    
    # Only include numeric columns
    feature_cols = [col for col in df.columns 
                   if not any(pattern in col for pattern in exclude_patterns)
                   and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from data_collection import fetch_daily_bars, fetch_spy_data
    from config import START_DATE_STR, END_DATE_STR
    
    print("Testing feature engineering...")
    
    # Fetch sample data
    symbol = "AAPL"
    df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
    spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
    
    # Engineer features
    df_features = engineer_features(symbol, df, spy_df)
    
    # Print summary
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Engineered columns: {len(df_features.columns)}")
    print(f"\nFeature columns:")
    feature_cols = get_feature_columns(df_features)
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"  {i}. {col}")
    print(f"  ... and {len(feature_cols) - 20} more")
    
    print(f"\nTarget columns:")
    target_cols = [col for col in df_features.columns if 'target_' in col]
    for col in target_cols:
        print(f"  - {col}")
    
    print(f"\nData shape: {df_features.shape}")
    print(f"Missing values per column:")
    print(df_features.isnull().sum().sort_values(ascending=False).head(10))
