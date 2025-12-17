"""
Market breadth features - advance-decline indicators.
Provides macro context by tracking market-wide momentum.
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

# Cache for breadth data
_breadth_cache = {}


def _fetch_breadth_proxy(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch market breadth proxy using sector ETFs.
    True advance-decline data requires premium data, so we approximate using sector performance.
    """
    cache_key = f"breadth_{start_date}_{end_date}"
    
    if cache_key in _breadth_cache:
        return _breadth_cache[cache_key].copy()
    
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        # Use major sector ETFs as proxy for breadth
        sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
        
        sector_data = {}
        for sector in sectors:
            data = yf.download(sector, start=start_date, end=end_date, progress=False)
            if not data.empty:
                sector_data[sector] = data['Close']
        
        if not sector_data:
            return pd.DataFrame()
        
        # Combine into single DataFrame
        breadth_df = pd.DataFrame(sector_data)
        
        # Calculate daily returns for each sector
        sector_returns = breadth_df.pct_change()
        
        # Advance-Decline proxy: count of sectors with positive returns
        breadth_df['advancing'] = (sector_returns > 0).sum(axis=1)
        breadth_df['declining'] = (sector_returns < 0).sum(axis=1)
        breadth_df['unchanged'] = (sector_returns == 0).sum(axis=1)
        
        # Advance-Decline Line (cumulative)
        breadth_df['ad_line'] = (breadth_df['advancing'] - breadth_df['declining']).cumsum()
        
        # Advance-Decline Ratio
        breadth_df['ad_ratio'] = breadth_df['advancing'] / (breadth_df['declining'] + 1)  # +1 to avoid div by zero
        
        _breadth_cache[cache_key] = breadth_df.copy()
        return breadth_df
        
    except Exception:
        return pd.DataFrame()


def add_breadth_features(df: pd.DataFrame, symbol: str, verbose: bool = False) -> pd.DataFrame:
    """
    Add market breadth features.
    
    Features:
    - Advance-Decline Line (cumulative breadth)
    - Advance-Decline Ratio
    - Breadth momentum indicators
    - Divergence between price and breadth
    
    Args:
        df: DataFrame with OHLCV data (must have DatetimeIndex)
        symbol: Stock symbol (for logging)
        verbose: Print progress messages
        
    Returns:
        DataFrame with added breadth features
    """
    if not YFINANCE_AVAILABLE:
        if verbose:
            print(f"  Warning: yfinance not available, skipping breadth features")
        return df
    
    df = df.copy()
    
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    if verbose:
        print(f"  Adding market breadth features for {symbol}...")
    
    try:
        # Get date range with buffer
        start_date = (df.index.min() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        # Fetch breadth data
        breadth_df = _fetch_breadth_proxy(start_date, end_date)
        
        if breadth_df.empty:
            if verbose:
                print(f"    Warning: Could not fetch breadth data")
            # Add dummy features
            for feature in ['ad_line', 'ad_ratio', 'ad_line_ma20', 'ad_ratio_ma20',
                           'breadth_momentum_10d', 'breadth_divergence', 'breadth_strong', 'breadth_weak']:
                df[feature] = 0
            return df
        
        # Align to stock dates
        breadth_aligned = breadth_df.reindex(df.index, method='ffill')
        
        # Raw breadth indicators
        df['ad_line'] = breadth_aligned['ad_line']
        df['ad_ratio'] = breadth_aligned['ad_ratio']
        
        # Smoothed breadth (moving averages)
        df['ad_line_ma20'] = breadth_aligned['ad_line'].rolling(20).mean()
        df['ad_ratio_ma20'] = breadth_aligned['ad_ratio'].rolling(20).mean()
        
        # Breadth momentum (change in advance-decline line)
        df['breadth_momentum_10d'] = breadth_aligned['ad_line'].diff(10)
        
        # Divergence between price and breadth
        # Normalize both to compare
        price_norm = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        breadth_norm = (breadth_aligned['ad_line'] - breadth_aligned['ad_line'].rolling(50).mean()) / breadth_aligned['ad_line'].rolling(50).std()
        df['breadth_divergence'] = price_norm - breadth_norm
        
        # Breadth regime indicators
        df['breadth_strong'] = (breadth_aligned['ad_ratio'] > 1.5).astype(int)  # More advancing than declining
        df['breadth_weak'] = (breadth_aligned['ad_ratio'] < 0.67).astype(int)  # More declining than advancing
        
        if verbose:
            print(f"    Added 8 market breadth features")
        
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not add breadth features: {e}")
        # Add dummy features
        for feature in ['ad_line', 'ad_ratio', 'ad_line_ma20', 'ad_ratio_ma20',
                       'breadth_momentum_10d', 'breadth_divergence', 'breadth_strong', 'breadth_weak']:
            df[feature] = 0
    
    return df


if __name__ == "__main__":
    # Test breadth features
    print("Testing market breadth features...")
    
    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed")
        exit(1)
    
    # Fetch sample data
    symbol = "AAPL"
    df = yf.download(symbol, start='2024-01-01', end='2024-12-16', progress=False)
    df.columns = [col.lower() for col in df.columns]
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Add breadth features
    df_with_breadth = add_breadth_features(df, symbol, verbose=True)
    
    print(f"\nNew shape: {df_with_breadth.shape}")
    print(f"Added {df_with_breadth.shape[1] - df.shape[1]} features")
    
    # Show sample features
    breadth_cols = [col for col in df_with_breadth.columns if any(x in col for x in ['ad_', 'breadth_'])]
    print(f"\nBreadth features ({len(breadth_cols)}):")
    for col in breadth_cols:
        print(f"  - {col}")
    
    print(f"\nSample data (last 5 rows):")
    print(df_with_breadth[breadth_cols].tail())
    
    print(f"\nNaN counts:")
    print(df_with_breadth[breadth_cols].isnull().sum())
