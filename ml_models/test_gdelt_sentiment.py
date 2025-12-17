"""
Quick test to verify GDELT sentiment fetching works.
Tests on a single date before running full pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_features import get_gdelt_sentiment, _init_cache_db, clear_sentiment_cache
from datetime import datetime, timedelta

def test_gdelt():
    """Test GDELT sentiment on various dates."""
    
    print("="*80)
    print("TESTING GDELT SENTIMENT API")
    print("="*80)
    
    # Test on a few historical dates
    symbol = "AAPL"
    test_dates = [
        "2024-01-15",  # ~1 year ago
        "2023-06-20",  # ~1.5 years ago
        "2022-11-10",  # ~2 years ago
    ]
    
    print(f"\nTesting sentiment for {symbol} on historical dates:")
    print("-" * 80)
    
    for date_str in test_dates:
        print(f"\nDate: {date_str}")
        result = get_gdelt_sentiment(symbol, date_str)
        
        print(f"  Sentiment Score: {result['sentiment_score']:.4f}")
        print(f"  Article Count: {result['sentiment_count']}")
        
        if result['sentiment_count'] > 0:
            print(f"  ✓ SUCCESS - Found {result['sentiment_count']} articles with sentiment")
        else:
            print(f"  ⚠ WARNING - No articles found for this date")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    # Check if any date had sentiment
    results = [get_gdelt_sentiment(symbol, d) for d in test_dates]
    success_count = sum(1 for r in results if r['sentiment_count'] > 0)
    
    print(f"\nSummary: {success_count}/{len(test_dates)} dates had sentiment data")
    
    if success_count > 0:
        print("✓ GDELT API is working - ready to use with use_gdelt=True")
    else:
        print("⚠ No sentiment data found - check GDELT API status or ticker symbol")
    
    return success_count > 0


if __name__ == "__main__":
    # Clear cache to test fresh fetches
    clear_sentiment_cache()
    
    # Run test
    success = test_gdelt()
    
    if success:
        print("\nNext steps:")
        print("1. Update feature_engineering.py to use: add_sentiment_features(df, symbol, use_gdelt=True)")
        print("2. Or build pre-computed sentiment DB for faster training")
    else:
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify GDELT API is accessible: https://api.gdeltproject.org/")
        print("3. Try different ticker symbols")
