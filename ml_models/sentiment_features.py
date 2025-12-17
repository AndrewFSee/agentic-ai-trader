"""
News Sentiment Features for ML Trading Models

Uses existing news_sentiment_finviz_finbert tool with caching for historical data.
Caches sentiment scores in SQLite DB to avoid re-computation during retraining.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests

# Add parent directory to import sentiment tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cache database
CACHE_DB = "data/sentiment_cache.db"


def _init_cache_db():
    """Initialize sentiment cache database."""
    os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            symbol TEXT,
            date TEXT,
            sentiment_score REAL,
            sentiment_count INTEGER,
            scraped_at TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    
    # Index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_date 
        ON sentiment_cache(symbol, date)
    """)
    
    conn.commit()
    conn.close()


def _get_cached_sentiment(symbol: str, date: str) -> Optional[Dict]:
    """Get sentiment from cache if available."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sentiment_score, sentiment_count, scraped_at
        FROM sentiment_cache
        WHERE symbol = ? AND date = ?
    """, (symbol, date))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'sentiment_score': row[0],
            'sentiment_count': row[1],
            'scraped_at': row[2]
        }
    return None


def _cache_sentiment(symbol: str, date: str, sentiment_score: float, sentiment_count: int):
    """Save sentiment to cache."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO sentiment_cache 
        (symbol, date, sentiment_score, sentiment_count, scraped_at)
        VALUES (?, ?, ?, ?, ?)
    """, (symbol, date, sentiment_score, sentiment_count, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_gdelt_sentiment(symbol: str, date: str) -> Dict:
    """
    Fetch historical news sentiment from GDELT Project (2015+).
    
    GDELT provides free historical news archives with pre-computed sentiment.
    Uses 'tonechart' mode which aggregates sentiment across all articles for the date.
    
    Args:
        symbol: Stock ticker
        date: Date string (YYYY-MM-DD)
    
    Returns:
        dict with sentiment_score (-1 to +1) and sentiment_count
    """
    try:
        # GDELT DOC 2.0 API - use tonechart for aggregated sentiment
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Note: OR queries must be wrapped in parentheses
        params = {
            'query': f'({symbol} OR "{symbol} stock")',
            'mode': 'tonechart',
            'format': 'json',
            'startdatetime': f'{date.replace("-", "")}000000',
            'enddatetime': f'{date.replace("-", "")}235959',
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            # Silently return 0 (don't spam logs)
            return {'sentiment_score': 0.0, 'sentiment_count': 0}
        
        data = response.json()
        tonechart = data.get('tonechart', [])
        
        if not tonechart:
            return {'sentiment_score': 0.0, 'sentiment_count': 0}
        
        # Calculate weighted average tone from tone bins
        # GDELT tone bins are -10 to +10, we normalize to -1 to +1
        total_articles = sum(bin['count'] for bin in tonechart)
        weighted_sum = sum(bin['bin'] * bin['count'] for bin in tonechart)
        
        if total_articles == 0:
            return {'sentiment_score': 0.0, 'sentiment_count': 0}
        
        avg_tone = weighted_sum / total_articles
        normalized_sentiment = avg_tone / 10.0  # Scale from -10..+10 to -1..+1
        
        return {
            'sentiment_score': normalized_sentiment,
            'sentiment_count': total_articles
        }
        
    except Exception as e:
        # Silently return 0 (don't spam logs for every date)
        return {'sentiment_score': 0.0, 'sentiment_count': 0}


def get_news_sentiment_for_date(symbol: str, date: str, use_gdelt: bool = False) -> Dict:
    """
    Get news sentiment for a specific date.
    Uses cache if available, otherwise fetches and caches.
    
    Args:
        symbol: Stock ticker
        date: Date string (YYYY-MM-DD)
        use_gdelt: If True, uses GDELT for historical data (slower but has archives)
    
    Returns:
        dict with sentiment_score (-1 to +1) and sentiment_count
    """
    # Check cache first
    cached = _get_cached_sentiment(symbol, date)
    if cached:
        return {
            'sentiment_score': cached['sentiment_score'],
            'sentiment_count': cached['sentiment_count'],
            'from_cache': True
        }
    
    # If use_gdelt flag is enabled, fetch from GDELT
    if use_gdelt:
        result = get_gdelt_sentiment(symbol, date)
        # Cache the result
        _cache_sentiment(symbol, date, result['sentiment_score'], result['sentiment_count'])
        return {**result, 'from_cache': False, 'source': 'gdelt'}
    
    # Otherwise, return 0.0 (Finviz only has recent news, not historical)
    # NOTE: For backtesting on historical dates, set use_gdelt=True
    #       or pre-build a sentiment database using GDELT
    _cache_sentiment(symbol, date, 0.0, 0)
    return {'sentiment_score': 0.0, 'sentiment_count': 0, 'from_cache': False, 'source': 'none'}


def add_sentiment_features(df: pd.DataFrame, symbol: str, use_gdelt: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Add news sentiment features to dataframe.
    
    Features added:
    - sentiment_score: Average sentiment (-1 to +1)
    - sentiment_count: Number of news articles
    - sentiment_momentum_3d: Change in sentiment over 3 days
    - sentiment_momentum_7d: Change in sentiment over 7 days
    - sentiment_volatility: Rolling std of sentiment
    - sentiment_positive: Binary (1 if positive sentiment)
    - sentiment_extreme: Binary (1 if |sentiment| > 0.5)
    - sentiment_strength: Product of score and count (confidence-weighted)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        symbol: Stock ticker
        use_gdelt: If True, fetches historical sentiment from GDELT (slower but accurate)
        verbose: If True, prints progress every 100 dates
    
    Returns:
        DataFrame with sentiment features added
    """
    if use_gdelt:
        print(f"  Adding news sentiment features for {symbol} (using GDELT - may take 1-2 minutes)...")
    else:
        print(f"  Adding news sentiment features for {symbol} (WARNING: returns 0.0 for historical dates)...")
    
    # Initialize cache DB
    _init_cache_db()
    
    sentiment_scores = []
    sentiment_counts = []
    
    # Get sentiment for each date in dataframe
    for i, date in enumerate(df.index):
        date_str = date.strftime('%Y-%m-%d')
        sentiment = get_news_sentiment_for_date(symbol, date_str, use_gdelt=use_gdelt)
        sentiment_scores.append(sentiment['sentiment_score'])
        sentiment_counts.append(sentiment['sentiment_count'])
        
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(df.index)} dates...")
    
    # Add raw sentiment features
    df['sentiment_score'] = sentiment_scores
    df['sentiment_count'] = sentiment_counts
    
    # Sentiment momentum (change over time)
    df['sentiment_momentum_3d'] = df['sentiment_score'].diff(3)
    df['sentiment_momentum_7d'] = df['sentiment_score'].diff(7)
    
    # Sentiment volatility (how erratic is the news?)
    df['sentiment_volatility'] = df['sentiment_score'].rolling(window=14).std()
    
    # Binary indicators
    df['sentiment_positive'] = (df['sentiment_score'] > 0).astype(int)
    df['sentiment_extreme'] = (df['sentiment_score'].abs() > 0.5).astype(int)
    
    # Interaction with news count (high count + extreme sentiment = strong signal)
    df['sentiment_strength'] = df['sentiment_score'].abs() * np.log1p(df['sentiment_count'])
    
    # Fill NaN values with 0 (neutral sentiment)
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)
    
    print(f"    Added {len(sentiment_cols)} sentiment features")
    
    return df


def clear_sentiment_cache(symbol: Optional[str] = None):
    """
    Clear sentiment cache for debugging or refresh.
    
    Args:
        symbol: If provided, only clear cache for this symbol. Otherwise clear all.
    """
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    if symbol:
        cursor.execute("DELETE FROM sentiment_cache WHERE symbol = ?", (symbol,))
        print(f"Cleared sentiment cache for {symbol}")
    else:
        cursor.execute("DELETE FROM sentiment_cache")
        print("Cleared entire sentiment cache")
    
    conn.commit()
    conn.close()


def get_cache_stats() -> Dict:
    """Get statistics about sentiment cache."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM sentiment_cache")
    total_entries = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM sentiment_cache")
    unique_symbols = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(date), MAX(date) FROM sentiment_cache")
    date_range = cursor.fetchone()
    
    conn.close()
    
    return {
        'total_entries': total_entries,
        'unique_symbols': unique_symbols,
        'date_range': date_range
    }


if __name__ == "__main__":
    # Test sentiment features
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("Testing sentiment features...")
    
    # Create test dataframe
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    df = pd.DataFrame({
        'close': np.random.randn(30).cumsum() + 100
    }, index=dates)
    
    # Add sentiment features
    df = add_sentiment_features(df, 'AAPL')
    
    print("\nSample data with sentiment features:")
    print(df[['close', 'sentiment_score', 'sentiment_count', 
              'sentiment_momentum_3d', 'sentiment_strength']].tail(10))
    
    print("\nCache stats:")
    print(get_cache_stats())
