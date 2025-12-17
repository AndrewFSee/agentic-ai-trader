"""
Test sentiment features with GDELT on a small subset.
Check timing to estimate full dataset cost.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import time
from data_collection import fetch_daily_bars
from sentiment_features import add_sentiment_features, clear_sentiment_cache

def test_timing():
    """Test GDELT on 50 dates to estimate full cost."""
    
    print("="*80)
    print("GDELT SENTIMENT TIMING TEST")
    print("="*80)
    
    # Clear cache to force fresh API calls
    clear_sentiment_cache()
    
    # Fetch AAPL data (90 days ~ 3 months of trading days)
    print("\nFetching AAPL price data...")
    from datetime import datetime, timedelta
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    df = fetch_daily_bars("AAPL", start, end)
    print(f"  Got {len(df)} rows")
    
    # Take only first 50 for timing test
    df_small = df.head(50).copy()
    print(f"  Testing on {len(df_small)} dates")
    
    # Time the sentiment fetch
    print("\nFetching GDELT sentiment with use_gdelt=True...")
    start = time.time()
    
    df_result = add_sentiment_features(df_small, "AAPL", use_gdelt=True, verbose=True)
    
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average: {elapsed / len(df_small):.2f} seconds per date")
    
    # Check results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    sentiment = df_result['sentiment_score']
    count = df_result['sentiment_count']
    
    print(f"\nSentiment Score:")
    print(f"  Mean: {sentiment.mean():.4f}")
    print(f"  Std: {sentiment.std():.4f}")
    print(f"  Min: {sentiment.min():.4f}")
    print(f"  Max: {sentiment.max():.4f}")
    print(f"  Non-zero: {(sentiment != 0).sum()} / {len(sentiment)}")
    
    print(f"\nArticle Count:")
    print(f"  Mean: {count.mean():.1f}")
    print(f"  Min: {count.min()}")
    print(f"  Max: {count.max()}")
    
    # Estimate full dataset time
    print("\n" + "="*80)
    print("FULL DATASET ESTIMATES")
    print("="*80)
    
    avg_per_date = elapsed / len(df_small)
    
    scenarios = [
        ("1 stock, 1 year (252 days)", 252),
        ("1 stock, 5 years (1260 days)", 1260),
        ("25 stocks, 5 years (31,500 dates)", 31500),
    ]
    
    for desc, num_dates in scenarios:
        est_time = avg_per_date * num_dates
        hours = est_time / 3600
        print(f"\n{desc}:")
        print(f"  Estimated time: {hours:.1f} hours ({est_time/60:.0f} minutes)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nFor training on multiple stocks, options:")
    print("1. Use caching (2nd run will be instant)")
    print("2. Pre-build sentiment database once (run overnight)")
    print("3. Parallelize with ThreadPoolExecutor (5-10x speedup)")
    

if __name__ == "__main__":
    test_timing()
