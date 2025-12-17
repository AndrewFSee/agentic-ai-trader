# Feature Combination Testing - Overview

## Test Configuration

**Objective**: Identify which minimal 3-feature combinations work best for HMM regime detection across different sectors and stock types.

### Test Parameters
- **Stocks**: 12 total (2 per sector)
  - **Tech**: AAPL, MSFT
  - **Finance**: JPM, BAC
  - **Energy**: XOM, CVX
  - **Consumer**: WMT, HD
  - **Healthcare**: JNJ, UNH
  - **Industrial**: CAT, BA

- **Feature Combinations**: 120 total (all possible 3-feature combinations from 10 features)
- **HMM Parameters**:
  - Training window: 500 days (~2 years)
  - Retrain frequency: 126 days (quarterly)
  - State persistence: 0.80
  - States: 3 (Low Vol/Bearish, Med Vol/Sideways, High Vol/Bullish)

- **Total Tests**: 1,440 (120 combinations × 12 stocks)

## Feature Library (10 Total Features)

### Current Features (Baseline)
1. **vol_norm_return**: Volatility-normalized returns (returns / rolling vol)
2. **realized_vol**: Rolling realized volatility (20-day)
3. **trend_strength**: Price vs 50-day MA, normalized

### New Features Added
4. **momentum_10d**: 10-day price momentum (rate of change)
5. **volume_momentum**: Volume vs 20-day average (normalized)
6. **hl_range**: High-Low range normalized by close
7. **price_zscore**: Mean reversion signal (z-score of price)
8. **vol_momentum**: Change in volatility (volatility momentum)
9. **rsi_momentum**: RSI-like momentum indicator (centered at 0)
10. **vol_regime**: Current vol vs long-term average vol

## Evaluation Metrics

For each combination, we measure:

1. **Sharpe Improvement**: Strategy Sharpe - Buy & Hold Sharpe
   - Primary ranking metric
   - Measures risk-adjusted return improvement
   
2. **Drawdown Improvement**: Strategy DD - Buy & Hold DD
   - Risk reduction metric
   - Negative values = better (less drawdown)
   
3. **Return Difference**: Strategy Return - Buy & Hold Return
   - Absolute performance
   - May be negative (trade-off for better risk metrics)
   
4. **Success Rate**: Number of stocks where combo worked / total stocks

## Analysis Goals

### 1. Find Universal Best Combinations
- Which 3-feature combos work well across ALL stocks?
- Top 15 combinations by average Sharpe improvement

### 2. Identify Sector-Specific Patterns
- Do certain features work better for specific sectors?
  - **Tech**: High volatility, growth-oriented
  - **Finance**: Cyclical, rate-sensitive
  - **Energy**: Commodity-driven, volatile
  - **Consumer**: Defensive, stable
  - **Healthcare**: Defensive, regulatory-driven
  - **Industrial**: Economic cycle-dependent

### 3. Feature Effectiveness Analysis
- Which features appear most in top combinations?
- Which features are most versatile?
- Which features are sector-specific?

### 4. Robustness Assessment
- Do top combinations work on majority of stocks?
- Or only on specific types?
- Trade-offs: universality vs specialization

## Expected Outcomes

### Hypothesis 1: Volatility Features Dominate
- Expect `realized_vol` and `vol_regime` to appear frequently
- HMMs naturally good at volatility regime detection

### Hypothesis 2: Sector-Specific Patterns
- **High vol sectors** (Tech, Energy): Momentum features important
- **Low vol sectors** (Consumer, Healthcare): Mean reversion features
- **Cyclical sectors** (Finance, Industrial): Trend features

### Hypothesis 3: Trade-offs
- Best Sharpe improvement ≠ best absolute returns
- Most combinations will reduce returns but improve risk metrics
- Goal: Find combinations that minimize return sacrifice while maximizing risk improvement

## Results Format

The test will produce:

1. **Ranked Combinations**: Top 15 by avg Sharpe improvement
2. **Sector Analysis**: Best combo for each sector
3. **Feature Frequency**: Which features appear most in top combos
4. **Stock-Specific Results**: Detailed metrics for each stock × combo
5. **JSON Output**: Complete results for further analysis

## Next Steps After Testing

1. **Validate Top Combinations**: Deep dive on top 3-5 combinations
2. **Test on Additional Stocks**: Expand beyond 2 per sector
3. **Ensemble Approach**: Combine predictions from multiple feature sets
4. **Adaptive Selection**: Dynamically choose features based on market regime
5. **Production Implementation**: Deploy best combo in live trading agent

---

**Test Started**: December 15, 2025
**Expected Duration**: 30-60 minutes
**Output File**: `feature_combination_results_YYYYMMDD_HHMMSS.json`
