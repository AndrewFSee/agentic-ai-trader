# Feature Enhancement Implementation Plan
## Phase 1: Research & Discovery (IN PROGRESS)

### Current Research Script Running
**File**: `ml_models/research_predictive_features.py`
**Status**: Running comprehensive GPT-Researcher queries
**Expected Duration**: 10-15 minutes
**Output**: `results/ml_models/feature_research_[timestamp].json` and `.md`

### Research Topics:
1. **ML Trading Features** - What actually works in academic literature?
2. **Historical News Sentiment** - Data sources and backfilling techniques
3. **Options Data** - Polygon.io and alternatives, key metrics (IV, put/call)
4. **Alternative Data** - Social sentiment, insider trading, 13F filings, etc.
5. **Feature Engineering** - Lopez de Prado techniques, best practices

---

## Phase 2: Quick Wins (Implement These First)

### 2.1 News Sentiment Integration ⭐ HIGHEST PRIORITY
**Why First**: We already have the tool (`news_sentiment_finviz_finbert`), just need to integrate into features

**Implementation**:
```python
# ml_models/sentiment_features.py (NEW FILE)
def get_historical_news_sentiment(symbol: str, date: str) -> dict:
    """
    Scrape Finviz news for given date and run FinBERT sentiment.
    For historical data: Scrape with date filter, cache results.
    """
    # TODO: Add date range parameter to existing tool
    # TODO: Cache sentiment scores in local DB for retraining
    pass

def add_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add news sentiment features to dataframe."""
    # sentiment_score: -1 (bearish) to +1 (bullish)
    # sentiment_count: number of news articles
    # sentiment_momentum_3d: change in sentiment over 3 days
    # sentiment_momentum_7d: change over 7 days
    pass
```

**Challenges**:
- Historical news not available from Finviz (recent only)
- **Solution**: Use NewsAPI.org (historical archives) or GDELT dataset
- Cache all sentiment scores to avoid re-computation

**Estimated Impact**: Medium (sentiment is noisy but can help on earnings/events)

---

### 2.2 Regime Detection Features ⭐ HIGH PRIORITY
**Why**: We already have HMM and Wasserstein models trained and working

**Implementation**:
```python
# ml_models/regime_features.py (NEW FILE)
def add_regime_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add regime detection features."""
    from models.rolling_hmm_regime_detection import RollingWindowHMM
    from models.paper_wasserstein_regime_detection import PaperWassersteinKMeans
    
    # HMM regime (bearish/sideways/bullish)
    hmm = RollingWindowHMM()
    hmm_regime = hmm.detect_regime(symbol)
    df['hmm_regime'] = hmm_regime['regime']  # 0, 1, 2
    df['hmm_confidence'] = hmm_regime['confidence']
    
    # Wasserstein regime (low/med/high volatility)
    wass = PaperWassersteinKMeans()
    wass_regime = wass.detect_regime(symbol)
    df['wass_regime'] = wass_regime['regime']  # 0, 1, 2
    df['wass_volatility'] = wass_regime['volatility']
    
    # Regime interaction (combine both)
    df['regime_combination'] = df['hmm_regime'] * 3 + df['wass_regime']  # 0-8
    
    return df
```

**Challenges**:
- Need to run regime detection for ALL historical dates (slow)
- **Solution**: Pre-compute regimes for all stocks, save to CSV
- Update during live trading only

**Estimated Impact**: High (regimes capture market structure changes)

---

### 2.3 Fundamental Ratios 📊 MEDIUM PRIORITY
**Why**: Fundamentals provide forward-looking information (P/E, growth)

**Data Sources** (research will confirm best option):
1. **Polygon.io** - `polygon_ticker_details` gives some fundamentals (market cap)
2. **Alpha Vantage** - OVERVIEW endpoint (free tier, limited calls)
3. **yfinance** - Free but unofficial API
4. **Financial Modeling Prep** - $15/month for fundamentals

**Features to Add**:
```python
def add_fundamental_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add fundamental ratio features."""
    # Valuation ratios
    df['pe_ratio'] = get_pe_ratio(symbol)
    df['pb_ratio'] = get_pb_ratio(symbol)
    df['ps_ratio'] = get_ps_ratio(symbol)
    
    # Growth metrics
    df['revenue_growth_yoy'] = get_revenue_growth(symbol)
    df['earnings_growth_yoy'] = get_earnings_growth(symbol)
    
    # Profitability
    df['profit_margin'] = get_profit_margin(symbol)
    df['roe'] = get_return_on_equity(symbol)
    
    # Relative valuation (vs sector/market)
    df['pe_vs_sector'] = df['pe_ratio'] / get_sector_avg_pe(symbol)
    
    return df
```

**Challenges**:
- Fundamentals update quarterly (not daily)
- **Solution**: Forward-fill values between earnings reports
- Need to handle restatements and revisions

**Estimated Impact**: High (fundamentals drive long-term returns)

---

## Phase 3: Advanced Features (After Research)

### 3.1 Options Data (If Available via Polygon)
**Polygon Starter Tier** ($29/month) includes options data:
- Implied Volatility (IV)
- Put/Call ratios
- Open Interest
- Options volume

**Features**:
```python
def add_options_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add options-derived features."""
    df['iv_percentile'] = get_iv_percentile(symbol)  # IV rank 0-100
    df['put_call_ratio'] = get_put_call_ratio(symbol)  # Bearish if > 1
    df['iv_skew'] = get_iv_skew(symbol)  # OTM put premium
    df['max_pain'] = get_max_pain_strike(symbol)  # Options pinning level
    return df
```

**Estimated Impact**: High (options predict future volatility/direction)

---

### 3.2 Alternative Data
**Based on Research Findings**:

**A. Social Sentiment** (Reddit, StockTwits, Twitter)
- Reddit WallStreetBets mentions and sentiment
- StockTwits message volume and bullish %
- Twitter/X trending tickers

**B. Insider Trading** (SEC Form 4)
- Insider buys vs sells (net insider activity)
- Cluster of insider buys (strong signal)
- C-suite buying (CEO, CFO)

**C. Institutional Holdings** (SEC 13F filings)
- Top institutional ownership %
- Changes quarter-over-quarter
- Smart money flow (hedge fund buys)

**D. Web Scraping**
- Glassdoor employee sentiment
- Google Trends search volume
- Product reviews (Amazon, App Store)

**E. Alternative Data APIs**
- Quiver Quant (insider trading, lobbying, gov contracts)
- Unusual Whales (options flow, dark pool)
- Alternative.me (crypto sentiment can affect tech stocks)

---

## Phase 4: Feature Engineering Best Practices

### 4.1 Interaction Terms
```python
# Combine features for non-linear relationships
df['rsi_x_volume'] = df['rsi'] * df['volume_ratio']
df['sentiment_x_momentum'] = df['news_sentiment'] * df['return_5d']
df['regime_x_volatility'] = df['hmm_regime'] * df['atr']
```

### 4.2 Temporal Features
```python
# Time-based patterns
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_earnings_week'] = get_earnings_calendar(symbol)
df['days_to_ex_dividend'] = get_dividend_calendar(symbol)
```

### 4.3 Market-Relative Features
```python
# Stock performance vs market
df['beta_vs_spy'] = calculate_beta(df, spy_df)
df['alpha_vs_spy'] = df['return_10d'] - (df['beta_vs_spy'] * spy_df['return_10d'])
df['correlation_vs_spy_30d'] = df['close'].rolling(30).corr(spy_df['close'])
```

### 4.4 Regime-Conditional Features
```python
# Different features for different regimes
if current_regime == 'high_vol':
    # Use momentum features in high vol
    weight_momentum = 2.0
    weight_fundamentals = 0.5
elif current_regime == 'low_vol':
    # Use mean reversion in low vol
    weight_mean_reversion = 2.0
```

---

## Implementation Sequence

### Week 1: Foundation
- [x] Run comprehensive research (GPT-Researcher)
- [ ] Analyze research findings
- [ ] Create `sentiment_features.py` with historical news capability
- [ ] Create `regime_features.py` with pre-computation
- [ ] Test on 3 stocks (AAPL, NVDA, JPM)

### Week 2: Core Features
- [ ] Implement fundamental data fetching (choose best source)
- [ ] Create `fundamental_features.py`
- [ ] Add options data if Polygon tier allows
- [ ] Create `options_features.py`
- [ ] Test combined features on 10 stocks

### Week 3: Alternative Data
- [ ] Based on research, implement highest-impact alt data sources
- [ ] Insider trading features (SEC EDGAR API)
- [ ] Social sentiment (Reddit API, StockTwits)
- [ ] Test on full 25-stock universe

### Week 4: Optimization
- [ ] Feature selection (RandomForest feature importance)
- [ ] Hyperparameter tuning with new features
- [ ] Regime-conditional model selection
- [ ] Final backtest and comparison vs baseline

---

## Success Metrics

### Current Performance (Technical Indicators Only):
- Mean Sharpe: 0.64 across all horizons
- Mean Return: 10-12%
- Win Rate vs Buy-Hold: 11-21%
- Train Accuracy: 49% (random chance)

### Target Performance (With New Features):
- Mean Sharpe: > 1.0 (at least 50% improvement)
- Mean Return: > 20%
- Win Rate vs Buy-Hold: > 60%
- Train Accuracy: > 55% (learning actual patterns)

---

## Next Immediate Actions

1. **Wait for research to complete** (running now, ~10 min remaining)
2. **Read and analyze** all 5 research reports
3. **Prioritize features** based on:
   - Ease of implementation
   - Data availability (free/cheap)
   - Academic evidence of efficacy
4. **Start with sentiment** (easiest, tool already exists)
5. **Then regimes** (models already trained)
6. **Then fundamentals** (choose data source based on research)

---

## Open Questions for Research to Answer

1. **Historical news sentiment**: Best approach? NewsAPI, GDELT, or scraping archive.org?
2. **Options data**: Is Polygon Starter tier worth $29/month? Any free alternatives?
3. **Most impactful alt data**: What do quant funds actually use?
4. **Feature engineering**: Which Lopez de Prado techniques are most practical?
5. **Data leakage**: How to avoid look-ahead bias with forward-looking features?
6. **Non-stationarity**: Should we retrain monthly? Quarterly? Use online learning?
7. **Regime-specific models**: Train separate models per regime or use regime as feature?

---

## Risk Mitigation

### Data Quality
- Validate all external data sources
- Check for missing values and outliers
- Cross-reference with multiple sources

### Overfitting Prevention
- Walk-forward validation (not just train/test split)
- Out-of-sample testing on recent data only
- Feature selection before hyperparameter tuning
- Regularization (L1/L2) to reduce feature count

### Implementation Bugs
- Unit tests for all feature functions
- Verify forward-fill logic (no look-ahead bias)
- Compare feature values manually for 1-2 stocks
- Version control for reproducibility

---

**CURRENT STATUS**: Research phase active. Waiting for GPT-Researcher results to guide implementation priorities.
