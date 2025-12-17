# ML Pipeline Diagnosis - December 15, 2024

## Summary: The Problem is **UNDERFITTING**, Not Overfitting

**Key Finding**: Models are barely learning from the training data at all.

### Evidence

1. **Train Accuracy**: 49% (essentially random chance)
   - Decision Tree: 49.2%
   - Random Forest: 48.0%
   - XGBoost: 47.9%
   - Logistic: 51.5%

2. **Buy-and-Hold Comparison**: Models lose on average
   - XGBoost beats B&H on only 40% of stocks (return) and 54% (Sharpe)
   - Mean return deficit vs B&H: -2.4% to -9.7%
   - Only defensive and volatility categories show some wins

3. **ROC-AUC**: 0.52-0.56 (barely above 0.5 random baseline)

## What This Means

The models are **not overfitting** (learning training noise). They're **underfitting** (failing to learn real patterns).

Possible reasons:
1. **No predictive signal at 5-10 day horizon** - Markets may be too efficient
2. **Wrong features** - Technical indicators don't contain forward-looking information
3. **Label noise** - Binary up/down classification is too simplistic
4. **Data quality** - Missing important variables (options flow, order book, news, etc.)
5. **Regime mixing** - Combining bull/bear/sideways regimes confuses models

## Incorrect Recommendations from First Pass

❌ **DON'T increase regularization** - This will make underfitting worse
❌ **DON'T reduce model complexity** - Models need MORE capacity, not less
❌ **DON'T prune features** - We need MORE informative features

## Correct Path Forward

### 1. Feature Engineering (HIGHEST PRIORITY)

**Add Forward-Looking Features:**
```python
# Option-implied features
- IV percentile (if available from data provider)
- Put/call ratio
- Dark pool prints (if available)

# Sentiment features  
- FinBERT news sentiment (we have this!)
- Social media sentiment
- Insider buying/selling

# Microstructure features
- Bid-ask spread
- Order flow imbalance
- Volume profile

# Inter-asset features
- Correlation with sector ETF
- Relative strength vs peers
- Beta to VIX
```

**Add Feature Interactions:**
```python
# Non-linear combinations
- RSI * volume_ratio
- return_5d * volatility
- macd * rsi
- Polynomial features (degree=2 for top 10 features)
```

**Add Regime as Feature:**
```python
# Use our HMM/Wasserstein regime detection
- regime_hmm (0=bearish, 1=sideways, 2=bullish)
- regime_vol (0=low, 1=medium, 2=high)
- regime_confidence
- days_in_regime
```

### 2. Alternative Target Formulations

Instead of binary classification, try:

**A. Regression on actual returns:**
```python
# Predict continuous return, then apply threshold
target = df['return_5d_forward']
# Use RandomForestRegressor, XGBRegressor
# Only trade if |predicted_return| > 2%
```

**B. Multi-class classification:**
```python
# 5 classes based on quintiles
# Only trade top quintile (strong buy) or bottom quintile (strong sell)
target = pd.qcut(df['return_5d_forward'], q=5, labels=[0,1,2,3,4])
```

**C. Triple-barrier method (Lopez de Prado):**
```python
# Label: 1 if hits profit target first, -1 if hits stop-loss first
# More sophisticated than simple forward return
# Accounts for path, not just endpoint
```

### 3. Regime-Specific Models

Train separate models for each market regime:

```python
# Split data by HMM regime
bull_data = df[df['regime_hmm'] == 2]
bear_data = df[df['regime_hmm'] == 0]
sideways_data = df[df['regime_hmm'] == 1]

# Train 3 separate XGBoost models
model_bull = train_xgboost(bull_data, ...)
model_bear = train_xgboost(bear_data, ...)  
model_sideways = train_xgboost(sideways_data, ...)

# At inference: Use current regime to select model
```

### 4. Model Complexity

**INCREASE capacity, don't decrease:**
```python
MODELS = {
    "xgboost": {
        "n_estimators": 500,  # More trees (was 100)
        "max_depth": 8,  # Deeper trees (was 6)
        "learning_rate": 0.1,  # Keep same
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # REMOVE regularization initially
        # "reg_alpha": 0,
        # "reg_lambda": 1
    },
    "random_forest": {
        "n_estimators": 300,  # More trees (was 100)
        "max_depth": 15,  # Much deeper (was 10)
        "min_samples_split": 20,  # LOWER (was 50)
        "min_samples_leaf": 10,  # LOWER (was 20)
    }
}
```

### 5. Walk-Forward Validation

Current 60/20/20 split might have regime bias:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(df):
    # Train on expanding window
    # Test on next period
    # Average performance across folds
```

### 6. Ensemble Methods

Even if individual models are weak, ensemble might help:
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',  # Average probabilities
    weights=[2, 1, 1]  # XGBoost gets more weight
)
```

### 7. Use News Sentiment

We already have `news_sentiment_finviz_finbert` tool. Add to features:
```python
# Before training:
sentiment = get_news_sentiment(symbol, date)
df['news_sentiment_score'] = sentiment['score']
df['news_count'] = sentiment['count']
df['sentiment_change_1d'] = df['news_sentiment_score'].diff()
```

### 8. Feature Importance Analysis

Check which features XGBoost/RF actually use:
```python
# After training
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 30 features only
top_features = importance_df.head(30)['feature'].tolist()
```

## Immediate Next Steps

1. **Add regime as feature** (easiest, 1 hour)
2. **Add feature interactions** (polynomial, 2 hours)
3. **Try regression instead of classification** (1 hour)  
4. **Increase model capacity** (30 min)
5. **Run smaller test** (AAPL, NVDA, SPY only, 1 hour)
6. **If still poor**: Accept that 5-10 day horizon has no edge with technical features alone

## Harsh Reality Check

If after these improvements we still can't beat buy-and-hold:

**The signal might not exist** at 5-10 day horizon with technical features.

Markets are highly efficient at this time scale. You might need:
- Much shorter horizon (intraday with tick data - requires paid API)
- Much longer horizon (20+ days)
- Fundamental data (earnings, revenue growth, etc. - also requires paid data)
- Alternative data (satellite imagery, credit card transactions, etc.)

The fact that defensive and volatility stocks show some edge suggests:
- Mean reversion works better than momentum
- Regime detection helps (these stocks have clearer regimes)
- Lower trading volume = less efficiency = more opportunity

## Category-Specific Insights

From the results:

- **Growth stocks (NVDA, AAPL, META)**: Models fail completely
  - Reason: Strong trends, high efficiency, hard to time
  - Conclusion: Just buy and hold growth

- **Defensive stocks (PG, JNJ, COST)**: 80% win rate with Logistic Regression
  - Reason: Mean-reverting, low volatility, predictable
  - Conclusion: Focus ML efforts here

- **Volatility stocks (MSTR, GME, AMC)**: Mixed results
  - MSTR: Huge wins (+200%)
  - AMC/GME: Huge losses
  - Reason: Meme stock behavior is unpredictable
  - Conclusion: Either skip or use regime filter

## Recommendation: Pivot Strategy

Instead of trying to beat buy-and-hold on all stocks:

**Use ML for position sizing, not timing:**
```python
# Always be in the market (buy-and-hold)
# Use ML to predict confidence
# Adjust position size by confidence

position_size = base_size * model_probability * kelly_fraction
```

This way you capture buy-and-hold returns while reducing drawdowns during predicted down periods.
