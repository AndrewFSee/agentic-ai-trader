"""
Macro correlation features - currency and commodity correlations.
Tracks rolling correlations with USD, oil, and gold to capture macro sensitivity.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Cache for macro data to avoid repeated downloads
_macro_cache = {}


def _fetch_macro_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch macro asset data with caching."""
    cache_key = f"{symbol}_{start_date}_{end_date}"
    
    if cache_key in _macro_cache:
        return _macro_cache[cache_key].copy()
    
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
            _macro_cache[cache_key] = data.copy()
        return data
    except Exception:
        return pd.DataFrame()


def add_macro_features(df: pd.DataFrame, symbol: str, verbose: bool = False) -> pd.DataFrame:
    """
    Add macro correlation features.
    
    Features:
    - Rolling correlations with USD (DX-Y.NYB), Oil (CL=F), Gold (GC=F)
    - Cross-asset momentum indicators
    - Macro regime indicators
    
    Args:
        df: DataFrame with OHLCV data (must have DatetimeIndex)
        symbol: Stock symbol (for logging)
        verbose: Print progress messages
        
    Returns:
        DataFrame with added macro features
    """
    if not YFINANCE_AVAILABLE:
        if verbose:
            print(f"  Warning: yfinance not available, skipping macro features")
        return df
    
    df = df.copy()
    
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    if verbose:
        print(f"  Adding macro correlation features for {symbol}...")
    
    try:
        # Get date range with buffer for lookback periods
        start_date = (df.index.min() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        # Fetch macro assets
        # USD Index (DX-Y.NYB), Oil (CL=F), Gold (GC=F)
        usd_data = _fetch_macro_data('DX-Y.NYB', start_date, end_date)
        oil_data = _fetch_macro_data('CL=F', start_date, end_date)
        gold_data = _fetch_macro_data('GC=F', start_date, end_date)
        
        # Calculate stock returns for correlation
        stock_returns = df['close'].pct_change()
        
        # USD correlations
        if not usd_data.empty:
            usd_returns = usd_data['close'].pct_change()
            # Align to stock dates
            usd_returns = usd_returns.reindex(df.index, method='ffill')
            
            df['usd_corr_10d'] = stock_returns.rolling(10).corr(usd_returns)
            df['usd_corr_20d'] = stock_returns.rolling(20).corr(usd_returns)
            df['usd_corr_50d'] = stock_returns.rolling(50).corr(usd_returns)
            
            # USD momentum (relative to stock)
            usd_mom = usd_data['close'].pct_change(20).reindex(df.index, method='ffill')
            stock_mom = df['close'].pct_change(20)
            df['usd_momentum_divergence'] = stock_mom - usd_mom
            
        else:
            df['usd_corr_10d'] = 0
            df['usd_corr_20d'] = 0
            df['usd_corr_50d'] = 0
            df['usd_momentum_divergence'] = 0
        
        # Oil correlations
        if not oil_data.empty:
            oil_returns = oil_data['close'].pct_change()
            oil_returns = oil_returns.reindex(df.index, method='ffill')
            
            df['oil_corr_10d'] = stock_returns.rolling(10).corr(oil_returns)
            df['oil_corr_20d'] = stock_returns.rolling(20).corr(oil_returns)
            df['oil_corr_50d'] = stock_returns.rolling(50).corr(oil_returns)
            
            # Oil momentum
            oil_mom = oil_data['close'].pct_change(20).reindex(df.index, method='ffill')
            df['oil_momentum_divergence'] = stock_mom - oil_mom
            
        else:
            df['oil_corr_10d'] = 0
            df['oil_corr_20d'] = 0
            df['oil_corr_50d'] = 0
            df['oil_momentum_divergence'] = 0
        
        # Gold correlations
        if not gold_data.empty:
            gold_returns = gold_data['close'].pct_change()
            gold_returns = gold_returns.reindex(df.index, method='ffill')
            
            df['gold_corr_10d'] = stock_returns.rolling(10).corr(gold_returns)
            df['gold_corr_20d'] = stock_returns.rolling(20).corr(gold_returns)
            df['gold_corr_50d'] = stock_returns.rolling(50).corr(gold_returns)
            
            # Gold momentum (safe haven indicator)
            gold_mom = gold_data['close'].pct_change(20).reindex(df.index, method='ffill')
            df['gold_momentum_divergence'] = stock_mom - gold_mom
            
        else:
            df['gold_corr_10d'] = 0
            df['gold_corr_20d'] = 0
            df['gold_corr_50d'] = 0
            df['gold_momentum_divergence'] = 0
        
        # Macro regime indicators
        # High gold correlation + negative oil correlation = risk-off
        df['macro_risk_off'] = ((df['gold_corr_20d'] > 0.3) & (df['oil_corr_20d'] < -0.2)).astype(int)
        
        # High oil correlation + negative gold correlation = risk-on
        df['macro_risk_on'] = ((df['oil_corr_20d'] > 0.3) & (df['gold_corr_20d'] < -0.2)).astype(int)
        
        if verbose:
            print(f"    Added 16 macro correlation features")
        
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not add macro features: {e}")
        # Add dummy features to maintain consistent feature count
        for feature in ['usd_corr_10d', 'usd_corr_20d', 'usd_corr_50d', 'usd_momentum_divergence',
                       'oil_corr_10d', 'oil_corr_20d', 'oil_corr_50d', 'oil_momentum_divergence',
                       'gold_corr_10d', 'gold_corr_20d', 'gold_corr_50d', 'gold_momentum_divergence',
                       'macro_risk_off', 'macro_risk_on']:
            df[feature] = 0
    
    return df


if __name__ == "__main__":
    # Test macro features
    print("Testing macro correlation features...")
    
    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed")
        exit(1)
    
    # Fetch sample data
    symbol = "AAPL"
    df = yf.download(symbol, start='2024-01-01', end='2024-12-16', progress=False)
    df.columns = [col.lower() for col in df.columns]
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Add macro features
    df_with_macro = add_macro_features(df, symbol, verbose=True)
    
    print(f"\nNew shape: {df_with_macro.shape}")
    print(f"Added {df_with_macro.shape[1] - df.shape[1]} features")
    
    # Show sample features
    macro_cols = [col for col in df_with_macro.columns if any(x in col for x in ['usd', 'oil', 'gold', 'macro'])]
    print(f"\nMacro features ({len(macro_cols)}):")
    for col in macro_cols:
        print(f"  - {col}")
    
    print(f"\nSample data (last 5 rows):")
    print(df_with_macro[macro_cols].tail())
    
    print(f"\nNaN counts:")
    print(df_with_macro[macro_cols].isnull().sum())
