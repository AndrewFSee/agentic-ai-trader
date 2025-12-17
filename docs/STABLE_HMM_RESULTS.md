# Stable HMM Methods - Stability Test Results

## Executive Summary

**Tested two advanced HMM stabilization techniques:**
1. **Constrained Transition Matrix** - Prevents direct jumps between extreme states
2. **Bayesian HMM with Dirichlet Priors** - Regularizes transition probabilities

**CRITICAL FINDING: Both methods FAILED to achieve production stability (≥90%)**

---

## Test Configuration

- **Stocks**: AAPL, MSFT, NVDA, TSLA, XOM, JPM (6 stocks)
- **Method**: Walk-forward stability test
- **Initial Window**: 250 days
- **Step Size**: 50 days (3 incremental steps)
- **Total Days**: 400 days
- **Bayesian Prior Strength**: 10.0

---

## Results Summary

### Method 1: Constrained Transition Matrix

**Overall Average Stability: 58.0%**

| Stock | Avg Stability | Min Stability | Max Stability | Status |
|-------|--------------|---------------|---------------|---------|
| AAPL  | **89.0%** 🟡 | 80.4% | 96.6% | Best performer (just below 90%) |
| MSFT  | 30.9% 🔴 | 19.6% | 44.0% | Massive label switching |
| NVDA  | 47.0% 🔴 | 22.7% | 83.2% | High volatility |
| TSLA  | 67.6% 🟡 | 52.4% | 92.9% | Inconsistent |
| XOM   | 60.1% 🟡 | 30.8% | 87.3% | Moderate instability |
| JPM   | 53.2% 🔴 | 20.9% | 97.0% | Extreme swings |

**Stable Stocks (≥90% avg): 0/6 (0%)**

---

### Method 2: Bayesian HMM with Dirichlet Priors

**Overall Average Stability: 53.0%**

| Stock | Avg Stability | Min Stability | Max Stability | Status |
|-------|--------------|---------------|---------------|---------|
| AAPL  | 68.4% 🟡 | 64.0% | 77.1% | Most consistent |
| MSFT  | 57.0% 🔴 | 54.4% | 60.9% | Low stability |
| NVDA  | 37.0% 🔴 | **11.7%** | 68.0% | CRITICAL - 88% label flip! |
| TSLA  | 62.3% 🟡 | 41.6% | 76.9% | Volatile |
| XOM   | 58.2% 🔴 | 41.7% | 68.0% | Unstable |
| JPM   | 35.1% 🔴 | 23.7% | 50.4% | Severe instability |

**Stable Stocks (≥90% avg): 0/6 (0%)**

---

## Comparative Analysis

### All Methods Tested

| Method | Avg Stability | Stable Stocks | Best Stock | Worst Stock |
|--------|--------------|---------------|------------|-------------|
| **Original HMM** | 55.0% | 0/12 (0%) | JPM 66.6% | PG 26.4% |
| **Wasserstein K-medoids** | 63.5% | 0/6 (0%) | MSFT 69.0% | XOM 58.1% |
| **Constrained HMM** | **58.0%** | 0/6 (0%) | AAPL 89.0% | MSFT 30.9% |
| **Bayesian HMM** | **53.0%** | 0/6 (0%) | AAPL 68.4% | JPM 35.1% |

### Key Findings

1. **Constrained Transition Matrix**: 
   - Slight improvement over original HMM (58.0% vs 55.0%)
   - AAPL nearly reached 90% threshold (89.0% avg)
   - Still suffers from label switching on most stocks
   - Constraints help but not enough
   
2. **Bayesian HMM with Priors**:
   - **WORSE than original HMM** (53.0% vs 55.0%)
   - More consistent within stocks but lower overall
   - NVDA showed catastrophic 11.7% agreement (88% label flip!)
   - Priors didn't prevent retroactive label changes
   
3. **Best Overall Method**: 
   - Wasserstein K-medoids (63.5%) remains the winner
   - But still 26.5 percentage points short of production threshold

---

## Detailed Stock-by-Stock Analysis

### AAPL - Most Stable Stock

**Constrained HMM**: 89.0% avg (BEST, but still <90%)
- Step 2: 80.4% (MODERATE)
- Step 3: 90.0% (HIGH) ✅
- Step 4: 96.6% (HIGH) ✅

**Bayesian HMM**: 68.4% avg
- Step 2: 64.0% (LOW)
- Step 3: 64.0% (LOW)
- Step 4: 77.1% (MODERATE)

**Conclusion**: Constrained method works best for AAPL, nearly production-ready

---

### MSFT - Most Unstable (Constrained)

**Constrained HMM**: 30.9% avg (WORST)
- Step 2: **19.6%** (80% label flip!)
- Step 3: 29.0% (CRITICAL)
- Step 4: 44.0% (CRITICAL)

**Bayesian HMM**: 57.0% avg (better but still unstable)
- Step 2: 54.4% (CRITICAL)
- Step 3: 55.7% (CRITICAL)
- Step 4: 60.9% (LOW)

**Conclusion**: Bayesian method significantly better for MSFT, but still fails

---

### NVDA - Catastrophic Bayesian Failure

**Constrained HMM**: 47.0% avg
- Step 2: 83.2% (MODERATE)
- Step 3: **22.7%** (CRITICAL)
- Step 4: 35.1% (CRITICAL)

**Bayesian HMM**: 37.0% avg (WORST)
- Step 2: 68.0% (LOW)
- Step 3: **11.7%** (CRITICAL - 88% label flip!)
- Step 4: 31.4% (CRITICAL)

**Conclusion**: Neither method works for high-volatility tech stocks

---

### TSLA - High Variance

**Constrained HMM**: 67.6% avg
- Step 2: 52.4% (CRITICAL)
- Step 3: 57.7% (CRITICAL)
- Step 4: **92.9%** (HIGH) ✅

**Bayesian HMM**: 62.3% avg
- Step 2: 41.6% (CRITICAL)
- Step 3: 68.3% (LOW)
- Step 4: 76.9% (MODERATE)

**Conclusion**: Constrained method shows promise in final step, but overall unstable

---

### XOM - Energy Sector

**Constrained HMM**: 60.1% avg
- Step 2: **30.8%** (CRITICAL)
- Step 3: 87.3% (MODERATE)
- Step 4: 62.3% (LOW)

**Bayesian HMM**: 58.2% avg
- Step 2: 64.8% (LOW)
- Step 3: 68.0% (LOW)
- Step 4: 41.7% (CRITICAL)

**Conclusion**: Both methods struggle with energy sector volatility

---

### JPM - Financial Sector

**Constrained HMM**: 53.2% avg
- Step 2: 41.6% (CRITICAL)
- Step 3: **97.0%** (HIGH) ✅
- Step 4: **20.9%** (CRITICAL - 79% label flip!)

**Bayesian HMM**: 35.1% avg (WORST)
- Step 2: 50.4% (CRITICAL)
- Step 3: 31.3% (CRITICAL)
- Step 4: **23.7%** (CRITICAL)

**Conclusion**: Extreme instability, wild swings between steps

---

## Root Cause Analysis

### Why These Methods Failed

1. **Constrained Transitions**:
   - ✅ Prevents some illogical transitions (bearish → bullish directly)
   - ❌ Doesn't prevent regimes from shifting boundaries
   - ❌ New data still causes retroactive relabeling
   - ❌ Constraints applied to model, not to regime assignment logic

2. **Bayesian Priors**:
   - ✅ Regularizes transition probabilities
   - ✅ More consistent regime persistence within a window
   - ❌ Priors don't stabilize regime **definitions**
   - ❌ When new data shifts cluster centers, labels still change
   - ❌ Actually performed WORSE than vanilla HMM

### Fundamental Problem

**All clustering/regime detection methods suffer from the same flaw:**
- Regime boundaries are **data-dependent**
- Adding new observations shifts cluster centers
- Historical regime assignments change retroactively
- No amount of constraints or priors can fix this

**The problem isn't the algorithm—it's the approach.**

---

## Recommendations

### ❌ What Doesn't Work

1. Standard HMM (55% stability)
2. Bayesian HMM with priors (53% stability) - WORSE
3. Constrained transition matrices (58% stability)
4. Wasserstein clustering (63.5% stability) - best but still fails
5. Any unsupervised clustering on incremental data

### ✅ Potential Solutions

**Option 1: Fixed Regime Definitions (Rule-Based)**
- Define regimes using **fixed thresholds** (not clustering)
- Example: VIX >20 = high vol, VIX <15 = low vol
- Regimes are **deterministic** and don't change with new data
- Sacrifice sophistication for stability

**Option 2: Supervised Learning**
- Label historical data with known market conditions
- Train classifier (Random Forest, XGBoost) to predict regimes
- Model learns patterns, not clustering boundaries
- More robust to incremental data

**Option 3: Online/Incremental Clustering**
- Use algorithms designed for streaming data (Online k-means, StreamKM++)
- Explicitly designed to handle new data without full retraining
- May reduce but not eliminate label drift

**Option 4: Abandon Regime Detection**
- Focus on direct signal generation (RSI, MACD, moving averages)
- Use ensemble methods without regime overlay
- Simpler and more transparent

**Option 5: Hybrid Approach**
- Use fixed rules for major regimes (bull/bear/sideways)
- Apply ML for sub-regimes within each major regime
- Reduces degrees of freedom, increases stability

### 🎯 Recommended Next Step

**Try Option 1 (Rule-Based Fixed Regimes) first:**
1. Define regimes using deterministic rules:
   - **Low Vol**: 20-day volatility < 15th percentile
   - **Medium Vol**: Between 15th and 85th percentile
   - **High Vol**: > 85th percentile
   OR
   - **Bearish**: Price below 200-day MA AND RSI < 45
   - **Sideways**: Price near 200-day MA OR RSI 45-55
   - **Bullish**: Price above 200-day MA AND RSI > 55

2. Test stability - regimes should be 100% stable by definition

3. If performance is good, you have a production-ready system

4. If performance is poor, at least you know stability isn't the issue

---

## Conclusion

**Both constrained and Bayesian HMM methods FAILED to achieve production stability.**

- Constrained: 58.0% avg (3pp improvement over vanilla HMM)
- Bayesian: 53.0% avg (2pp WORSE than vanilla HMM)
- Neither achieved 90% threshold
- 0/6 stocks stable with either method

**The fundamental problem is not solvable by algorithmic tweaks to HMMs or clustering.**

Regime detection on incremental data inherently causes retroactive label changes. The solution requires a fundamentally different approach:
- Fixed rule-based regimes
- Supervised learning
- Online clustering algorithms
- Or abandoning regime detection entirely

**Recommendation**: Try rule-based fixed regime definitions next. They sacrifice sophistication for guaranteed stability.
