"""
Pre-build sentiment database for all stocks in the universe.
This script fetches historical sentiment from GDELT for all training dates
and caches them in SQLite. Future training runs will use the cache instantly.

Run this once overnight (~12 hours for 25 stocks x 5 years).
After completion, all feature engineering will be instant.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import List
import traceback

# Add parent for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import STOCK_UNIVERSE, START_DATE_STR, END_DATE_STR
from data_collection import fetch_daily_bars
from sentiment_features import add_sentiment_features, get_cache_stats, clear_sentiment_cache

def build_sentiment_for_stock(symbol: str, start_date: str, end_date: str, verbose: bool = True) -> dict:
    """
    Build sentiment cache for a single stock.
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Show progress
    
    Returns:
        dict with stats (dates_processed, time_taken, errors)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Building sentiment for {symbol}")
        print(f"{'='*80}")
    
    try:
        # Fetch price data to get all trading dates
        if verbose:
            print(f"  Fetching trading dates from {start_date} to {end_date}...")
        
        df = fetch_daily_bars(symbol, start_date, end_date)
        
        if df.empty:
            return {
                'symbol': symbol,
                'status': 'error',
                'dates_processed': 0,
                'time_taken': 0,
                'error': 'No price data available'
            }
        
        num_dates = len(df)
        if verbose:
            print(f"  Found {num_dates} trading dates")
            print(f"  Fetching GDELT sentiment (estimated time: {num_dates * 1.3 / 60:.1f} minutes)...")
        
        # Fetch sentiment with GDELT
        start_time = time.time()
        
        df_with_sentiment = add_sentiment_features(
            df, 
            symbol, 
            use_gdelt=True,
            verbose=verbose
        )
        
        elapsed = time.time() - start_time
        
        # Check results
        sentiment_scores = df_with_sentiment['sentiment_score']
        non_zero = (sentiment_scores != 0).sum()
        
        if verbose:
            print(f"\n  ✓ Completed in {elapsed/60:.1f} minutes ({elapsed/num_dates:.2f} sec/date)")
            print(f"  Sentiment stats:")
            print(f"    Non-zero scores: {non_zero}/{num_dates} ({non_zero/num_dates*100:.1f}%)")
            print(f"    Mean: {sentiment_scores.mean():.4f}")
            print(f"    Std: {sentiment_scores.std():.4f}")
            print(f"    Range: [{sentiment_scores.min():.4f}, {sentiment_scores.max():.4f}]")
        
        return {
            'symbol': symbol,
            'status': 'success',
            'dates_processed': num_dates,
            'time_taken': elapsed,
            'non_zero_count': non_zero,
            'avg_sentiment': sentiment_scores.mean(),
            'error': None
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"\n  ✗ ERROR: {error_msg}")
            print(f"\n  Stack trace:")
            traceback.print_exc()
        
        return {
            'symbol': symbol,
            'status': 'error',
            'dates_processed': 0,
            'time_taken': 0,
            'error': error_msg
        }


def build_full_database(
    stock_universe: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    resume: bool = True,
    clear_cache: bool = False
) -> pd.DataFrame:
    """
    Build sentiment database for all stocks.
    
    Args:
        stock_universe: List of stock tickers (default: from config)
        start_date: Start date (default: from config)
        end_date: End date (default: from config)
        resume: If True, skip stocks already in cache
        clear_cache: If True, clear all cached sentiment before starting
    
    Returns:
        DataFrame with build statistics per stock
    """
    # Use defaults from config
    if stock_universe is None:
        # Flatten STOCK_UNIVERSE dict into list
        if isinstance(STOCK_UNIVERSE, dict):
            stock_universe = [s for stocks in STOCK_UNIVERSE.values() for s in stocks]
        else:
            stock_universe = STOCK_UNIVERSE
    if start_date is None:
        start_date = START_DATE_STR
    if end_date is None:
        end_date = END_DATE_STR
    
    print("="*80)
    print("SENTIMENT DATABASE BUILD")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Stocks: {len(stock_universe)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Resume: {resume}")
    print(f"  Clear cache: {clear_cache}")
    
    # Clear cache if requested
    if clear_cache:
        print(f"\nClearing sentiment cache...")
        clear_sentiment_cache()
        print(f"  Cache cleared")
    
    # Check existing cache
    if resume:
        print(f"\nChecking existing cache...")
        stats = get_cache_stats()
        print(f"  Current cache: {stats['total_entries']} entries")
        print(f"  Unique symbols: {stats['unique_symbols']}")
    
    # Estimate total time
    total_dates = len(stock_universe) * 252 * 5  # Approx 252 trading days/year * 5 years
    est_hours = (total_dates * 1.3) / 3600
    print(f"\nEstimated time: {est_hours:.1f} hours")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process each stock
    results = []
    start_time = time.time()
    
    for i, symbol in enumerate(stock_universe, 1):
        print(f"\n{'='*80}")
        print(f"Stock {i}/{len(stock_universe)}: {symbol}")
        print(f"{'='*80}")
        
        # Build sentiment for this stock
        result = build_sentiment_for_stock(symbol, start_date, end_date, verbose=True)
        results.append(result)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time_per_stock = elapsed / i
        remaining = (len(stock_universe) - i) * avg_time_per_stock
        
        print(f"\nProgress: {i}/{len(stock_universe)} stocks completed")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        print(f"Estimated remaining: {remaining/3600:.2f} hours")
        print(f"ETA: {datetime.fromtimestamp(time.time() + remaining).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("BUILD COMPLETE")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    success = df_results[df_results['status'] == 'success']
    errors = df_results[df_results['status'] == 'error']
    
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"Successful: {len(success)}/{len(stock_universe)}")
    print(f"Errors: {len(errors)}/{len(stock_universe)}")
    
    if len(success) > 0:
        print(f"\nSentiment statistics:")
        print(f"  Total dates processed: {success['dates_processed'].sum():,}")
        print(f"  Average dates per stock: {success['dates_processed'].mean():.0f}")
        print(f"  Total time: {success['time_taken'].sum()/3600:.2f} hours")
        print(f"  Average time per stock: {success['time_taken'].mean()/60:.1f} minutes")
    
    if len(errors) > 0:
        print(f"\nErrors occurred for:")
        for _, row in errors.iterrows():
            print(f"  {row['symbol']}: {row['error']}")
    
    # Check final cache stats
    print(f"\nFinal cache statistics:")
    final_stats = get_cache_stats()
    print(f"  Total entries: {final_stats['total_entries']:,}")
    print(f"  Unique symbols: {final_stats['unique_symbols']}")
    print(f"  Cache size: ~{final_stats['total_entries'] * 100 / 1024 / 1024:.1f} MB")
    
    print(f"\n{'='*80}")
    print("DATABASE READY FOR TRAINING")
    print("{'='*80}")
    print(f"\nAll future training runs will use cached sentiment (instant).")
    print(f"Cache location: data/sentiment_cache.db")
    
    # Save results
    results_file = f"results/sentiment_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    df_results.to_csv(results_file, index=False)
    print(f"\nBuild statistics saved to: {results_file}")
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build sentiment database for ML training')
    parser.add_argument('--clear', action='store_true', help='Clear existing cache before building')
    parser.add_argument('--no-resume', action='store_true', help='Rebuild all stocks (ignore cache)')
    parser.add_argument('--test', action='store_true', help='Test on 3 stocks only')
    
    args = parser.parse_args()
    
    if args.test:
        print("\n*** TEST MODE: Building for 3 stocks only ***\n")
        stock_universe = ['AAPL', 'NVDA', 'JPM']
    else:
        stock_universe = None  # Use full universe from config
    
    # Build the database
    results_df = build_full_database(
        stock_universe=stock_universe,
        clear_cache=args.clear,
        resume=not args.no_resume
    )
    
    print("\n✓ Build complete!")
