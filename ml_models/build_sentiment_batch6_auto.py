"""
Build sentiment database for the SIXTH batch of 25 S&P 500 stocks.
AUTO-RUN VERSION - No confirmation required (for overnight automation)
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add parent for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import START_DATE_STR, END_DATE_STR
from build_sentiment_database import build_sentiment_for_stock, get_cache_stats

# Sixth batch of 25 S&P 500 stocks (diversified across sectors)
# Verified NO REPEATS from batches 1-5
BATCH_6_STOCKS = [
    # Insurance & Financial Services (5)
    "MET", "ALL", "TRV", "PGR", "AIG",
    # Energy & Materials (5)
    "EOG", "PSX", "VLO", "MPC", "OXY",
    # Consumer & Retail (5)
    "EBAY", "ETSY", "YUM", "CMG", "ULTA",
    # Technology & Cloud (5)
    "FTNT", "CRWD", "ZS", "NET", "OKTA",
    # Healthcare Services & Equipment (5)
    "ELV", "ZTS", "IDXX", "SYK", "BSX"
]

def main():
    """Build sentiment cache for batch 6 stocks - AUTO RUN."""
    
    print("=" * 80)
    print("SENTIMENT DATABASE BUILDER - BATCH 6 (AUTO-RUN)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Stocks: {len(BATCH_6_STOCKS)}")
    print(f"  Date range: {START_DATE_STR} to {END_DATE_STR}")
    print(f"\nStocks to process:")
    for i, symbol in enumerate(BATCH_6_STOCKS, 1):
        print(f"  {i:2d}. {symbol}")
    
    # Check existing cache
    print(f"\nChecking existing cache...")
    stats = get_cache_stats()
    print(f"  Current cache: {stats['total_entries']} entries")
    print(f"  Unique symbols: {stats['unique_symbols']}")
    
    # Estimate total time
    total_dates = len(BATCH_6_STOCKS) * 252 * 5  # Approx 252 trading days/year * 5 years
    est_hours = (total_dates * 1.3) / 3600
    print(f"\nEstimated time: {est_hours:.1f} hours")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSTARTING AUTOMATICALLY IN 5 SECONDS...")
    time.sleep(5)
    
    # Process each stock
    results = []
    start_time = time.time()
    
    for i, symbol in enumerate(BATCH_6_STOCKS, 1):
        print(f"\n{'='*80}")
        print(f"Stock {i}/{len(BATCH_6_STOCKS)}: {symbol}")
        print(f"{'='*80}")
        
        # Build sentiment for this stock
        result = build_sentiment_for_stock(symbol, START_DATE_STR, END_DATE_STR, verbose=True)
        results.append(result)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time_per_stock = elapsed / i
        remaining = (len(BATCH_6_STOCKS) - i) * avg_time_per_stock
        
        print(f"\nProgress: {i}/{len(BATCH_6_STOCKS)} stocks completed")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        print(f"Estimated remaining: {remaining/3600:.2f} hours")
        print(f"ETA: {datetime.fromtimestamp(time.time() + remaining).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save intermediate results after each stock (in case of crash)
        df_temp = pd.DataFrame(results)
        temp_file = f"sentiment_batch6_progress.csv"
        df_temp.to_csv(temp_file, index=False)
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("BATCH 6 COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\nResults:")
    print(f"  Successful: {len(successful)}/{len(BATCH_6_STOCKS)}")
    print(f"  Failed: {len(failed)}/{len(BATCH_6_STOCKS)}")
    
    if successful:
        total_dates = sum(r['dates_processed'] for r in successful)
        total_non_zero = sum(r['non_zero_count'] for r in successful)
        avg_sentiment = sum(r['avg_sentiment'] for r in successful) / len(successful)
        
        print(f"\nSentiment stats:")
        print(f"  Total dates processed: {total_dates:,}")
        print(f"  Non-zero scores: {total_non_zero:,} ({total_non_zero/total_dates*100:.1f}%)")
        print(f"  Average sentiment: {avg_sentiment:.4f}")
    
    if failed:
        print(f"\nFailed stocks:")
        for r in failed:
            print(f"  {r['symbol']}: {r['error']}")
    
    # Check final cache stats
    print(f"\nFinal cache stats:")
    final_stats = get_cache_stats()
    print(f"  Total entries: {final_stats['total_entries']:,}")
    print(f"  Unique symbols: {final_stats['unique_symbols']}")
    
    # Save final results to CSV
    df_results = pd.DataFrame(results)
    output_file = f"sentiment_build_batch6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ALL DONE! You can now close this window.")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nINTERRUPTED BY USER")
        print("Partial results saved in sentiment_batch6_progress.csv")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
