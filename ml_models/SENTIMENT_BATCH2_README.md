# Sentiment Database Builder - Batch 2

## Overview

This builds GDELT news sentiment cache for the **second batch of 25 S&P 500 stocks**.

**Batch 1 (Already Completed - 25 stocks):**
- Growth: NVDA, TSLA, GOOGL, META, AMZN
- Value: JPM, BAC, WFC, XOM, CVX
- Momentum: MSFT, AAPL, V, MA, COST
- Defensive: JNJ, PG, KO, PEP, WMT
- Volatility: GME, AMC, COIN, MSTR, PLTR

**Batch 2 (This Run - 25 NEW stocks):**
- Large Cap Tech: ORCL, ADBE, CRM, NFLX, INTC
- Financials: GS, MS, C, BLK, SCHW
- Healthcare: UNH, LLY, ABBV, TMO, DHR
- Consumer: HD, MCD, NKE, DIS, SBUX
- Industrial: CAT, UNP, HON, BA, RTX

## Estimated Time

**~10-12 hours** for 25 stocks × 5 years of data (~1,260 trading days per stock)

## How to Run

### Option 1: Auto-Run (Recommended for Overnight)

```powershell
cd ml_models
python build_sentiment_batch2_auto.py
```

- Starts automatically after 5 seconds
- Saves progress after each stock
- Logs everything to console
- Creates `sentiment_batch2_progress.csv` with intermediate results
- Creates final timestamped CSV when complete

### Option 2: Interactive (Requires Confirmation)

```powershell
cd ml_models
python build_sentiment_batch2.py
```

- Asks "Start processing? (y/n)" before beginning
- Otherwise same as auto-run

### Option 3: Windows Batch Script

```cmd
cd ml_models
run_sentiment_batch2.bat
```

- Runs auto version
- Logs output to `sentiment_batch2_log.txt`
- Shows "COMPLETE" message when done

## What It Does

For each stock:
1. Fetches ~1,260 trading dates (5 years)
2. Calls GDELT API for news sentiment on each date (~1.3 sec/date)
3. Caches results in SQLite database
4. Shows progress and stats

## Output Files

- **`sentiment_batch2_progress.csv`**: Updated after each stock (in case of crash)
- **`sentiment_build_batch2_YYYYMMDD_HHMMSS.csv`**: Final results with stats
- **`sentiment_batch2_log.txt`**: Full console output (if using .bat file)

## Monitoring Progress

The script prints:
```
Stock 5/25: INTC
Elapsed: 2.15 hours
Estimated remaining: 8.60 hours
ETA: 2025-12-17 06:30:00
```

You can check `sentiment_batch2_progress.csv` at any time to see which stocks are done.

## If Something Goes Wrong

**Interrupted?**
- Partial results are saved in `sentiment_batch2_progress.csv`
- GDELT cache is already saved for completed stocks
- Just restart the script - it will re-process only the remaining stocks

**Rate Limited?**
- GDELT API has rate limits
- Script sleeps 1.3 seconds between calls (conservative)
- If you hit limits, wait 1 hour and restart

**Errors?**
- Script continues even if individual stocks fail
- Failed stocks listed in final summary
- You can manually retry failed stocks later

## After Completion

**Check cache stats:**
```python
from sentiment_features import get_cache_stats
stats = get_cache_stats()
print(f"Total entries: {stats['total_entries']:,}")
print(f"Unique symbols: {stats['unique_symbols']}")
```

Should show:
- **Batch 1**: ~31,500 entries (25 stocks × ~1,260 dates)
- **Batch 2**: ~31,500 entries (25 stocks × ~1,260 dates)
- **Total**: ~63,000 entries for 50 stocks

**Future batches:**
- Copy `build_sentiment_batch2_auto.py` → `build_sentiment_batch3_auto.py`
- Update `BATCH_3_STOCKS` list with next 25 stocks
- Run overnight
- Repeat until all S&P 500 stocks are cached

## Speed Tips

**Faster processing:**
- Use a faster internet connection
- Run on a server with better GDELT access
- Run multiple batches in parallel (different stocks)

**Current speed:**
- ~1.3 sec per date (conservative, includes sleeps)
- ~27 minutes per stock (1,260 dates × 1.3 sec)
- ~11 hours for 25 stocks

## Example Output

```
================================================================================
Stock 1/25: ORCL
================================================================================
  Fetching trading dates from 2020-12-16 to 2025-12-16...
  Found 1,258 trading dates
  Fetching GDELT sentiment (estimated time: 27.3 minutes)...

  ✓ Completed in 26.8 minutes (1.28 sec/date)
  Sentiment stats:
    Non-zero scores: 1,015/1,258 (80.7%)
    Mean: 0.0234
    Std: 0.1456
    Range: [-0.4567, 0.6789]

Progress: 1/25 stocks completed
Elapsed: 0.45 hours
Estimated remaining: 10.80 hours
ETA: 2025-12-17 07:15:00
```

## Questions?

- **Where is the cache stored?** `data/sentiment_cache.db` (SQLite)
- **Can I run this on a server?** Yes, works on any machine with Python
- **Do I need API keys?** No, GDELT is public
- **Can I interrupt and resume?** Yes, cache is saved after each stock
- **How do I use the cache?** Automatically used by `add_sentiment_features()`
