# Feature Implementation Issues & Solutions

## 1. News Sentiment - NEEDS FIX

### Current Status: ❌ NOT WORKING
**Problem**: Sentiment scores all return 0.0 because:
- Finviz only shows recent news (2-3 days)
- Cannot fetch historical news for backtesting dates (2020-2025)
- Our tool calls Finviz with historical dates → gets no results

### Solutions (in priority order):

#### Option A: GDELT Project (RECOMMENDED for backtesting)
**What**: Global Database of Events, Language, and Tone - massive free historical news archive

**Pros**:
- ✅ FREE with massive historical data (back to 2015+)
- ✅ Already has sentiment scores (TONE metric -100 to +100)
- ✅ Can query by ticker, date range
- ✅ Covers millions of news sources globally

**Cons**:
- Requires API integration (different from current Finviz approach)
- Sentiment may be noisier than FinBERT (but has more data)

**Implementation**:
```python
import requests

def get_gdelt_sentiment(symbol: str, date: str) -> Dict:
    """Fetch historical sentiment from GDELT."""
    # GDELT DOC 2.0 API
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    params = {
        'query': f'{symbol} OR company_name',
        'mode': 'timelinemodesubjectlanguage',
        'format': 'json',
        'startdatetime': f'{date}000000',
        'enddatetime': f'{date}235959'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract sentiment (TONE field)
    articles = data.get('articles', [])
    if articles:
        tones = [article.get('tone', 0) for article in articles]
        avg_tone = sum(tones) / len(tones)
        # Convert GDELT tone (-100 to +100) to our scale (-1 to +1)
        sentiment_score = avg_tone / 100
        return {
            'sentiment_score': sentiment_score,
            'sentiment_count': len(articles)
        }
    
    return {'sentiment_score': 0.0, 'sentiment_count': 0}
```

#### Option B: NewsAPI.org (LIMITED free tier)
**What**: News aggregator with historical search

**Pros**:
- ✅ 100 requests/day on free tier
- ✅ 1 month of historical data
- ✅ Clean API with good docs

**Cons**:
- ❌ Only 1 month history (not enough for 5-year backtest)
- ❌ 100 calls/day = only 4 stocks per day with caching
- ❌ Still need FinBERT to compute sentiment (extra cost)

**Best Use**: Live trading only, not backtesting

#### Option C: Pre-compute Sentiment Database (HYBRID APPROACH)
**What**: Build historical sentiment database once, use for all backtests

**Steps**:
1. For each training date (2020-2025), fetch news ONCE:
   - Use GDELT for historical dates
   - Use Finviz + FinBERT for recent dates (last 30 days)
2. Store in SQLite: `(symbol, date, sentiment_score, sentiment_count, source)`
3. During training: Load from cache only (no API calls)
4. During live trading: Fetch real-time from Finviz + FinBERT

**Benefits**:
- ✅ Fast training (no API calls)
- ✅ Reproducible results
- ✅ Best quality sentiment (FinBERT on recent, GDELT on historical)

**Code**:
```python
def build_sentiment_database(symbols: List[str], start_date: str, end_date: str):
    """One-time build of historical sentiment database."""
    for symbol in symbols:
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # Check if already cached
            if _get_cached_sentiment(symbol, date_str):
                continue
            
            # Fetch sentiment
            if date < datetime.now() - timedelta(days=30):
                # Use GDELT for historical
                sentiment = get_gdelt_sentiment(symbol, date_str)
            else:
                # Use Finviz + FinBERT for recent
                sentiment = get_finviz_sentiment(symbol)
            
            # Cache it
            _cache_sentiment(symbol, date_str, 
                           sentiment['sentiment_score'], 
                           sentiment['sentiment_count'])
```

### RECOMMENDED ACTION:
1. **Immediate**: Implement GDELT integration for historical sentiment
2. **Long-term**: Build pre-computed sentiment database
3. **Live trading**: Keep Finviz + FinBERT for real-time

---

## 2. Fundamentals Downsampling - ✅ WORKING CORRECTLY

### Current Status: ✅ CORRECT IMPLEMENTATION

**What it does**:
```python
# Fetch current fundamentals (P/E, ROE, etc.)
fundamentals = get_fundamentals(symbol)  # Gets latest values

# Apply to ALL rows (broadcast)
df['pe_ratio'] = fundamentals.get('pe_ratio')  # Same value for every date
df['roe'] = fundamentals.get('roe')  # Same value for every date
```

**Why this is correct**:
- Fundamentals update **quarterly** (earnings reports)
- Between reports, values stay constant
- You can't predict future fundamentals (look-ahead bias!)
- Industry standard: Use latest known values

**Example**:
```
Date         Close    P/E Ratio    ROE
2025-12-01   $100     25.0         15%
2025-12-02   $101     25.0         15%  ← Same fundamental
2025-12-03   $99      25.0         15%  ← Same fundamental
...
2025-12-15   $105     25.0         15%  ← Same fundamental
```

**What models learn**:
- How price CHANGES relative to static fundamentals
- E.g., "When P/E is high and price drops 5%, likely mean reversion"
- Not trying to predict fundamentals themselves

**Advanced Improvement** (future):
To capture quarterly updates, you'd need to:
1. Fetch historical earnings dates
2. Get fundamentals AS OF each earnings date
3. Forward-fill between earnings

But this requires:
- Historical fundamental data (expensive: FactSet, Bloomberg)
- yfinance only gives CURRENT values (free but limited)

**Bottom Line**: Current implementation is correct for free data. Models will learn to use constant fundamental ratios in combination with changing price action.

---

## 3. Options Data - ⚠️ NEEDS POLYGON API KEY

### Current Status: ⚠️ PARTIAL

Test showed:
```python
put_call_ratio: 0
avg_iv: NaN (100% missing)
```

**Issue**: Options module checks `POLYGON_API_KEY` but may not be fetching data correctly.

**Debug Steps**:
1. Verify Polygon API key is set: `echo $env:POLYGON_API_KEY`
2. Check Polygon tier includes options (Starter = $29/month)
3. Test options endpoint manually:
   ```python
   import requests
   url = "https://api.polygon.io/v3/reference/options/contracts"
   params = {'underlying_ticker': 'AAPL', 'apiKey': 'YOUR_KEY'}
   response = requests.get(url, params=params)
   print(response.json())
   ```

**If Polygon doesn't work**: Options features will be zeros/NaN but won't break training.

---

## 4. Regime Detection - ⚠️ PARTIAL FAILURE

### Current Status: ⚠️ MIXED RESULTS

Test showed:
```
Warning: HMM regime detection failed: 'RollingWindowHMM' object has no attribute 'fit'
Warning: Wasserstein regime detection failed: (slice(None, None, None), 0)
```

**Issues**:
1. **HMM**: Model class needs `.fit()` method added
2. **Wasserstein**: Index error in regime prediction

**Impact**: Regime features created but with defaults (all "neutral")

**Fix Required**: Update regime model classes to have proper scikit-learn-like API:
```python
class RollingWindowHMM:
    def fit(self, returns):
        """Fit HMM on returns data."""
        self.model = hmm.GaussianHMM(n_components=3)
        self.model.fit(returns.reshape(-1, 1))
        return self
    
    def predict(self, window_data):
        """Predict regime for window."""
        return self.model.predict(window_data.reshape(-1, 1))
```

---

## Summary of Action Items

### High Priority (Blocking Model Performance):
1. ✅ **Fundamentals**: Working correctly, no action needed
2. ❌ **Sentiment**: Implement GDELT integration ASAP (biggest gap)
3. ⚠️ **Regimes**: Fix HMM and Wasserstein `.fit()` methods

### Medium Priority (Incremental Improvements):
4. ⚠️ **Options**: Debug Polygon API access, verify tier

### Low Priority (Future Enhancements):
5. Build pre-computed sentiment database for faster training
6. Fetch quarterly historical fundamentals (requires paid data)

---

## Testing Strategy

After fixes, verify with:
```bash
python ml_models/test_enhanced_features.py
```

Look for:
- ✅ Sentiment scores NOT all 0.0
- ✅ No "Warning: HMM regime detection failed"
- ✅ Fundamentals populated (not NaN)
- ✅ Options data if Polygon tier correct
