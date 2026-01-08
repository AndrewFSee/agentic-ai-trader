"""
Sentiment Database Builder - Batch 8 (Auto-Run)

This script collects 5 years of daily news sentiment for 25 S&P 500 stocks.
It will start automatically in 5 seconds.

Stocks in Batch 8: INTU, ADSK, WDAY, TEAM, ZM, PLD, EQIX, DLR, AMT, CCI, 
                   BDX, A, RMD, BAX, EW, TT, ROK, PH, ITW, ETN, SHW, 
                   PPG, NUE, STLD, VMC

NO REPEATS - Verified against batches 1-7 (175 stocks)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Eighth batch of 25 S&P 500 stocks (diversified across sectors)
# Verified NO REPEATS from batches 1-7
BATCH_8_STOCKS = [
    # Software & SaaS (5)
    "INTU", "ADSK", "WDAY", "TEAM", "ZM",
    
    # Real Estate & Infrastructure (5)
    "PLD", "EQIX", "DLR", "AMT", "CCI",
    
    # Medical Devices & Equipment (5)
    "BDX", "A", "RMD", "BAX", "EW",
    
    # Industrial Equipment & Automation (5)
    "TT", "ROK", "PH", "ITW", "ETN",
    
    # Materials & Chemicals (5)
    "SHW", "PPG", "NUE", "STLD", "VMC"
]

from config import START_DATE_STR, END_DATE_STR
from build_sentiment_database import build_sentiment_for_stock, get_cache_stats
import pandas as pd
from datetime import datetime
import time

def main():
    """Build sentiment cache for batch 8 stocks - AUTO RUN."""
    
    print("=" * 80)
    print("SENTIMENT DATABASE BUILDER - BATCH 8 (AUTO-RUN)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Stocks: {len(BATCH_8_STOCKS)}")
    print(f"  Date range: {START_DATE_STR} to {END_DATE_STR}")
    print(f"\nStocks to process:")
    for i, symbol in enumerate(BATCH_8_STOCKS, 1):
        print(f"  {i:2d}. {symbol}")
    
    # Check existing cache
    print(f"\nChecking existing cache...")
    stats = get_cache_stats()
    print(f"  Current cache: {stats['total_entries']} entries")
    print(f"  Unique symbols: {stats['unique_symbols']}")
    
    # Estimate total time
    total_dates = len(BATCH_8_STOCKS) * 252 * 5  # Approx 252 trading days/year * 5 years
    est_hours = (total_dates * 1.3) / 3600
    print(f"\nEstimated time: {est_hours:.1f} hours")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSTARTING AUTOMATICALLY IN 5 SECONDS...")
    time.sleep(5)
    
    # Process each stock
    results = []
    total_start = time.time()
    
    for i, symbol in enumerate(BATCH_8_STOCKS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(BATCH_8_STOCKS)}] Processing {symbol}")
        print(f"{'='*80}")
        
        stock_start = time.time()
        result = build_sentiment_for_stock(symbol, START_DATE_STR, END_DATE_STR, verbose=True)
        stock_elapsed = time.time() - stock_start
        
        results.append({
            'symbol': symbol,
            'status': result['status'],
            'dates_processed': result['dates_processed'],
            'time_seconds': stock_elapsed
        })
        
        print(f"\n✓ {symbol} complete:")
        print(f"  Status: {result['status']}")
        print(f"  Dates processed: {result['dates_processed']}")
        print(f"  Time: {stock_elapsed/60:.1f} minutes")
        print(f"  Remaining stocks: {len(BATCH_8_STOCKS) - i}")
        
        # Show updated cache stats
        if i % 5 == 0:  # Every 5 stocks
            stats = get_cache_stats()
            print(f"\n  Cache update: {stats['total_entries']} total entries, {stats['unique_symbols']} symbols")
    
    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"BATCH 8 COMPLETE!")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    print(f"\nResults:")
    print(results_df.to_string(index=False))
    
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"Average per stock: {total_elapsed/len(BATCH_8_STOCKS)/60:.1f} minutes")
    
    # Final cache stats
    final_stats = get_cache_stats()
    print(f"\nFinal cache: {final_stats['total_entries']} entries, {final_stats['unique_symbols']} symbols")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
