"""
VIX Features for ML Trading Models

Fetches VIX (volatility index) data and calculates derived features
for risk-adjusted alpha factors. VIX measures market fear/uncertainty.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import requests
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
DATA_DIR = "data"

def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch VIX data with fallback: Polygon -> yfinance.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with VIX OHLC data
    """
    # Check cache first
    cache_file = os.path.join(DATA_DIR, f"VIX_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    # Try Polygon first
    if POLYGON_API_KEY:
        url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{start_date}/{end_date}"
        params = {"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "OK" and "results" in data:
                    # Parse results
                    bars = data["results"]
                    df = pd.DataFrame(bars)
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                    df = df.set_index('timestamp')
                    
                    # Rename columns
                    df = df.rename(columns={
                        'o': 'vix_open',
                        'h': 'vix_high', 
                        'l': 'vix_low',
                        'c': 'vix_close',
                        'v': 'vix_volume'
                    })
                    
                    # Keep only OHLCV
                    df = df[['vix_open', 'vix_high', 'vix_low', 'vix_close', 'vix_volume']]
                    
                    # Cache
                    os.makedirs(DATA_DIR, exist_ok=True)
                    df.to_parquet(cache_file)
                    
                    return df
        except Exception as e:
            pass  # Fall through to yfinance
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        if vix.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns from yfinance (happens with single ticker)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # Rename columns to match expected format
        vix.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in vix.columns]
        vix = vix.rename(columns={
            'open': 'vix_open',
            'high': 'vix_high',
            'low': 'vix_low',
            'close': 'vix_close',
            'volume': 'vix_volume',
            'adj close': 'vix_close'  # Use adj close if available
        })
        
        # Ensure UTC timezone
        if vix.index.tz is None:
            vix.index = vix.index.tz_localize('UTC')
        else:
            vix.index = vix.index.tz_convert('UTC')
        
        # Keep only OHLCV
        available_cols = [col for col in ['vix_open', 'vix_high', 'vix_low', 'vix_close', 'vix_volume'] 
                          if col in vix.columns]
        vix = vix[available_cols]
        
        # Cache
        os.makedirs(DATA_DIR, exist_ok=True)
        vix.to_parquet(cache_file)
        
        return vix
        
    except Exception as e:
        print(f"VIX fetch failed (Polygon + yfinance): {e}")
        return pd.DataFrame()


def add_vix_features(df: pd.DataFrame, symbol: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Add VIX-based features to dataframe.
    
    Features added:
    - vix_level: Current VIX level (market fear gauge)
    - vix_change: Daily change in VIX
    - vix_pct_change: Percentage change in VIX
    - vix_ma_20: 20-day moving average of VIX
    - vix_ma_50: 50-day moving average of VIX
    - vix_std_20: 20-day rolling std of VIX (volatility of volatility)
    - vix_spike: Binary indicator for VIX > mean + 2*std
    - vix_regime: VIX regime (0=low <15, 1=normal 15-25, 2=high >25)
    - vix_trend: VIX above/below 20-day MA
    - stock_vix_correlation: 20-day rolling correlation with VIX (inverse relationship)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        symbol: Stock ticker (for correlation calc)
        verbose: Print progress
    
    Returns:
        DataFrame with VIX features added
    """
    if verbose:
        print(f"  Adding VIX features...")
    
    # Get date range
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    # Fetch VIX data
    vix_df = fetch_vix_data(start_date, end_date)
    
    if vix_df.empty:
        if verbose:
            print(f"    Warning: No VIX data available, using zeros")
        # Create dummy features
        for col in ['vix_level', 'vix_change', 'vix_pct_change', 'vix_ma_20', 'vix_ma_50',
                    'vix_std_20', 'vix_spike', 'vix_regime', 'vix_trend', 'stock_vix_correlation']:
            df[col] = 0.0
        return df
    
    # Ensure timezone compatibility before reindexing
    if df.index.tz is not None and vix_df.index.tz is None:
        vix_df.index = vix_df.index.tz_localize('UTC')
    elif df.index.tz is None and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    
    # Ensure timezone compatibility before reindexing
    if df.index.tz is not None and vix_df.index.tz is None:
        vix_df.index = vix_df.index.tz_localize('UTC')
    elif df.index.tz is None and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    
    # Align VIX data with stock data (forward fill for any missing dates)
    vix_aligned = vix_df.reindex(df.index, method='ffill')
    
    # Basic VIX features
    df['vix_level'] = vix_aligned['vix_close']
    df['vix_change'] = vix_aligned['vix_close'].diff()
    df['vix_pct_change'] = vix_aligned['vix_close'].pct_change()
    
    # Moving averages
    df['vix_ma_20'] = vix_aligned['vix_close'].rolling(window=20, min_periods=1).mean()
    df['vix_ma_50'] = vix_aligned['vix_close'].rolling(window=50, min_periods=1).mean()
    
    # Volatility of volatility
    df['vix_std_20'] = vix_aligned['vix_close'].rolling(window=20, min_periods=1).std()
    
    # VIX spike indicator (VIX > mean + 2*std over 60-day window)
    vix_mean_60 = vix_aligned['vix_close'].rolling(window=60, min_periods=20).mean()
    vix_std_60 = vix_aligned['vix_close'].rolling(window=60, min_periods=20).std()
    df['vix_spike'] = (vix_aligned['vix_close'] > vix_mean_60 + 2 * vix_std_60).astype(int)
    
    # VIX regime
    df['vix_regime'] = 1  # Default: normal
    df.loc[df['vix_level'] < 15, 'vix_regime'] = 0  # Low vol
    df.loc[df['vix_level'] > 25, 'vix_regime'] = 2  # High vol
    
    # VIX trend (above/below 20-day MA)
    df['vix_trend'] = (df['vix_level'] > df['vix_ma_20']).astype(int)
    
    # Stock-VIX correlation (typically negative - stocks fall when VIX rises)
    if 'close' in df.columns:
        stock_returns = df['close'].pct_change()
        vix_returns = vix_aligned['vix_close'].pct_change()
        df['stock_vix_correlation'] = stock_returns.rolling(window=20, min_periods=10).corr(vix_returns)
    else:
        df['stock_vix_correlation'] = 0.0
    
    # Fill NaN values
    df['vix_level'] = df['vix_level'].fillna(method='ffill').fillna(0)
    df['vix_change'] = df['vix_change'].fillna(0)
    df['vix_pct_change'] = df['vix_pct_change'].fillna(0)
    df['vix_ma_20'] = df['vix_ma_20'].fillna(method='ffill').fillna(0)
    df['vix_ma_50'] = df['vix_ma_50'].fillna(method='ffill').fillna(0)
    df['vix_std_20'] = df['vix_std_20'].fillna(0)
    df['vix_spike'] = df['vix_spike'].fillna(0)
    df['vix_regime'] = df['vix_regime'].fillna(1)
    df['vix_trend'] = df['vix_trend'].fillna(0)
    df['stock_vix_correlation'] = df['stock_vix_correlation'].fillna(0)
    
    if verbose:
        print(f"    Added 10 VIX features")
        print(f"    VIX range: {df['vix_level'].min():.2f} - {df['vix_level'].max():.2f}")
        print(f"    VIX spikes: {df['vix_spike'].sum()} days")
    
    return df


if __name__ == "__main__":
    # Test VIX features
    from data_collection import fetch_daily_bars
    
    print("Testing VIX features...")
    
    # Fetch AAPL data
    df = fetch_daily_bars("AAPL", "2020-12-16", "2025-12-15")
    print(f"Fetched {len(df)} rows for AAPL")
    
    # Add VIX features
    df = add_vix_features(df, symbol="AAPL", verbose=True)
    
    print("\nSample VIX features:")
    vix_cols = [col for col in df.columns if 'vix' in col.lower()]
    print(df[vix_cols].tail(10))
    
    print("\nVIX regime distribution:")
    print(df['vix_regime'].value_counts().sort_index())
