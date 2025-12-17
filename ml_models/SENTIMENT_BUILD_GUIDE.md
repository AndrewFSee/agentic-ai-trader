# Sentiment Database Build Guide

## What This Does

The sentiment database builder fetches historical news sentiment from GDELT for all stocks and dates in your training dataset. After building once, all future training runs will use the cached sentiment (instant).

## Test Run Results (3 stocks)

✓ Successfully completed in 1.23 hours
- AAPL: 1,255 dates, 96.6% coverage
- NVDA: 1,255 dates, 95.4% coverage  
- JPM: 1,255 dates, 95.9% coverage

## Full Build (25 stocks)

### Command
```bash
cd C:\Users\Andrew\projects\agentic_ai_trader\ml_models
python build_sentiment_database.py
```

### Expected Duration
- **Estimated: 10-11 hours** (based on test run: 24.5 min/stock × 25 stocks)
- Best to run overnight or during work day

### What It Does
1. Fetches all trading dates for each stock (2020-12-16 to 2025-12-15)
2. Queries GDELT API for news sentiment (~1.2 seconds per date)
3. Caches results in SQLite database (`data/sentiment_cache.db`)
4. Shows progress every 100 dates with ETA
5. Saves build statistics to `results/sentiment_build_TIMESTAMP.csv`

### Options
```bash
# Full build (default)
python build_sentiment_database.py

# Clear existing cache first (rebuild everything)
python build_sentiment_database.py --clear

# Don't resume (ignore cache, rebuild all)
python build_sentiment_database.py --no-resume

# Test on 3 stocks only (already done)
python build_sentiment_database.py --test
```

## After Build Completes

### 1. Update Feature Engineering

Edit `ml_models/feature_engineering.py` around line 265:

**Change from:**
```python
try:
    from sentiment_features import add_sentiment_features
    df = add_sentiment_features(df, symbol)
except Exception as e:
```

**Change to:**
```python
try:
    from sentiment_features import add_sentiment_features
    df = add_sentiment_features(df, symbol, use_gdelt=True)  # ← Add use_gdelt=True
except Exception as e:
```

### 2. Verify Cache

```bash
cd ml_models
python -c "from sentiment_features import get_cache_stats; print(get_cache_stats())"
```

Expected output after full build:
```
{
  'total_entries': ~31,375,  # 25 stocks × ~1255 dates
  'unique_symbols': 25,
  'date_range': ('2020-12-16', '2025-12-15')
}
```

### 3. Test Enhanced Features

```bash
cd ml_models
python test_enhanced_features.py
```

Should show:
- Sentiment scores varying (not all 0.0)
- Processing time: < 10 seconds (vs 23 minutes without cache)

## Resume Capability

If the build is interrupted:
- Cache is saved after each stock
- Rerun same command - it will resume from where it stopped
- To force rebuild: add `--clear` flag

## Monitoring Progress

The script shows:
- Current stock being processed (e.g., "Stock 5/25: MSFT")
- Dates processed every 100 entries
- Time elapsed and estimated remaining
- ETA for completion

## Expected Final Output

```
================================================================================
BUILD COMPLETE
================================================================================

Total time: 10.2 hours
Successful: 25/25
Errors: 0/25

Sentiment statistics:
  Total dates processed: 31,375
  Average dates per stock: 1,255
  Total time: 10.2 hours
  Average time per stock: 24.5 minutes

Final cache statistics:
  Total entries: 31,375
  Unique symbols: 25
  Cache size: ~3.0 MB
```

## Troubleshooting

### If a stock fails
- Error will be logged
- Build continues with next stock
- Failed stocks will have `status: 'error'` in results CSV

### If GDELT API has issues
- Script retries failed dates
- Falls back to 0.0 sentiment if API unavailable
- Check results CSV for details

### If you need to stop
- Ctrl+C to stop
- Progress is saved after each stock
- Resume with same command

## Next Steps After Build

1. ✓ Cache built (~3 MB SQLite file)
2. Update `feature_engineering.py` with `use_gdelt=True`
3. Run `test_enhanced_features.py` (should be fast)
4. Fix regime detection models (next priority)
5. Debug options data (Polygon API)
6. Train models with all enhanced features

## Cache Details

- **Location**: `data/sentiment_cache.db`
- **Size**: ~100 bytes per entry = ~3 MB total
- **Schema**: `(symbol, date, sentiment_score, sentiment_count, scraped_at)`
- **Query time**: Instant (SQLite indexed)
- **Rebuild**: Only needed if adding new stocks or extending date range
