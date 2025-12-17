# Paper-Faithful Wasserstein vs HMM: Final Verdict

## Executive Summary

After implementing the paper-faithful Wasserstein distance regime detection method from Horvath et al. (2021) and comparing it against rolling HMM across multiple stocks, the results show **mixed performance** that does **NOT** support the paper's claim of "vastly outperforming" HMM.

## Implementation Details

### Paper-Faithful Techniques Implemented ✓

1. **Median Barycenters** (Proposition 2.6, Equation 22):
   - Used median (not mean) for computing regime centroids
   - `aⱼ = Median(α¹ⱼ,...,αᴹⱼ)` across all distributions in cluster

2. **Fast 1D Wasserstein Distance** (Proposition 2.5, Equation 21):
   - O(N log N) computation via sorted atoms
   - `W₁(μ,ν) = ∫|F⁻¹_μ(t) - F⁻¹_ν(t)|dt`

3. **MMD Cluster Quality** (Equation 53):
   - Gaussian kernel with σ=0.1
   - Within-cluster vs between-cluster evaluation
   - `MMD²(X,Y) = 𝔼[k(x,x')] + 𝔼[k(y,y')] - 2𝔼[k(x,y)]`

4. **Empirical Distributions** (Definition 1.3):
   - Sliding windows treated as probability distributions
   - Equal-weighted atoms: `μ = (1/T)∑δ_xᵢ`

5. **Algorithm 1** - Full k-means:
   - Proper convergence checking
   - Centroid updates until no label changes
   - Deterministic median-based cluster centers

### Architecture

- **Training window**: 500 trading days (~2 years)
- **Retraining frequency**: 63 days (quarterly)
- **Feature set**: `realized_vol`, `trend_strength`, `volume_momentum` (universal winner combo)
- **Number of regimes**: k=3 (Low/Medium/High volatility)
- **Window size**: 20 days for empirical distributions

## Head-to-Head Comparison Results

### Stock-by-Stock Breakdown

| Stock | Wass Sharpe | HMM Sharpe | Winner | Score (W/H) | Notes |
|-------|-------------|------------|---------|-------------|-------|
| **AAPL** | 1.180 | **1.187** | HMM | 1/4 | Narrow HMM win, both ~identical to B&H |
| **MSFT** | **1.922** | 1.691 | Wass | 5/2 | Strong Wasserstein win, +5.56% better DD |
| **JPM** | **2.140** | 1.631 | Tie | 2/2 | Tie in score, but Wass Sharpe much better |
| **JNJ** | **0.238** | -0.012 | Wass | 5/3 | Decisive Wass win, only method with positive return |

### Key Findings

1. **Not "Vastly Superior"**: Results are **mixed**
   - Wasserstein wins decisively: 2/4 stocks (MSFT, JNJ)
   - HMM wins: 1/4 stocks (AAPL)
   - Tie: 1/4 stocks (JPM, though Wass has better Sharpe)

2. **Sharpe Ratio Comparison**:
   - Mean Wasserstein Sharpe: **1.370**
   - Mean HMM Sharpe: **1.124**
   - Advantage: +0.246 for Wasserstein (22% improvement)
   - **Conclusion**: Wasserstein shows meaningful edge, but not "vast"

3. **Return Comparison**:
   - Wasserstein average return: **16.13%**
   - HMM average return: **13.35%**
   - Difference: +2.78% for Wasserstein

4. **Stock-Specific Performance**:
   - **Tech stocks** (MSFT): Wasserstein dominated
   - **Financial stocks** (AAPL, JPM): Mixed results
   - **Healthcare** (JNJ): Wasserstein much better in down markets

## Multi-Stock Backtest Results (Wasserstein Only)

Comprehensive backtest on 12 stocks across 6 sectors (2020-01-01 to 2024-12-01):

### Summary Statistics
- **Mean Sharpe improvement**: +0.013 (modest)
- **Median Sharpe improvement**: +0.029
- **Win rate**: 6/12 stocks positive

### Top Performers
1. **JNJ**: +0.246 Sharpe (best)
2. **MSFT**: +0.233 Sharpe
3. **BAC**: +0.127 Sharpe
4. **CVX**: +0.081 Sharpe
5. **BA**: +0.079 Sharpe

### Bottom Performers
1. **HD**: -0.401 Sharpe (worst)
2. **CAT**: -0.157 Sharpe
3. **PFE**: -0.071 Sharpe

## Technical Analysis

### What Matters Most

1. **Median vs Mean Barycenters**:
   - Critical for robustness to outliers
   - Prevents regime centers from being pulled by extreme values
   - Paper implementation superior to mean approximation

2. **Fast 1D Wasserstein**:
   - O(N log N) makes it computationally viable
   - Proper probability distance metric (unlike Euclidean)
   - Captures distributional differences effectively

3. **MMD Evaluation**:
   - Provides quantitative cluster quality metric
   - Helps diagnose when clustering is effective vs poor
   - Not used for trading decisions, but valuable for analysis

### What Doesn't Matter Much

1. **Convergence Tolerance**:
   - Most cases converge in 3-6 iterations
   - Final convergence (loss ≈ 0) vs early stopping makes little difference

2. **Number of Regimes (k)**:
   - k=3 works well across all stocks tested
   - Paper used k=6, but simpler structure seems adequate

3. **Exact MMD Kernel Width**:
   - σ=0.1 from paper works fine
   - Results not highly sensitive to this parameter

## Comparison with Earlier Simplified Implementation

### Simplified Wasserstein (Previous)
- Used **mean** for barycenters (not paper-faithful)
- Lacked MMD evaluation
- Simpler Algorithm 1 implementation
- **Result**: Lost to HMM significantly (-0.490 vs -0.167 Sharpe on test case)

### Paper-Faithful Wasserstein (Current)
- Uses **median** for barycenters (Proposition 2.6)
- Includes MMD cluster quality
- Full Algorithm 1 with proper convergence
- **Result**: Mixed vs HMM, wins 2/4 stocks decisively, loses 1/4 narrowly

**Conclusion**: Paper-faithful implementation makes a **significant difference**, but still doesn't achieve "vast" superiority.

## Final Verdict

### Does Paper-Faithful Wasserstein Beat HMM?

**Answer: SOMETIMES** ✓

- **Not universally superior**: 50% win rate in head-to-head (2 wins, 1 loss, 1 tie)
- **Better average performance**: +22% Sharpe improvement over HMM
- **Stock-dependent**: Excels on tech/healthcare, mixed on financials
- **Risk-adjusted edge**: Better max drawdown control on average

### Paper's Claim of "Vastly Outperforms"

**Verdict: NOT VALIDATED** ✗

The paper's claim that Wasserstein "vastly outperforms" other methods is **overstated**:

1. **Modest aggregate gains**: +0.013 mean Sharpe in backtest (not "vast")
2. **High variance**: Some stocks +0.246, others -0.401
3. **Mixed head-to-head**: 50% win rate vs HMM, not dominant
4. **Similar to B&H**: Often comparable to buy-and-hold baseline

### When to Use Each Method

**Use Wasserstein when**:
- Stock has distinct volatility regimes (tech, healthcare)
- Strong trending or momentum behavior
- Need interpretable regime clusters
- Computational cost not a concern

**Use HMM when**:
- Stock has smooth regime transitions
- Need probabilistic predictions
- Want forward-filtering without look-ahead bias
- Faster computation required (no k-means iterations)

### Recommendations

1. **Neither is universally better**: Use ensemble of both methods
2. **Paper techniques matter**: Median barycenters are crucial
3. **Feature engineering dominates**: Universal winner combo (`realized_vol`, `trend_strength`, `volume_momentum`) works for both
4. **Validation essential**: Backtest on specific stock before deployment
5. **Modest expectations**: Neither method consistently beats buy-and-hold by large margins

## Code Repository

- **Paper-faithful implementation**: `paper_wasserstein_regime_detection.py` (926 lines)
- **Rolling HMM**: `rolling_hmm_regime_detection.py` (562 lines)
- **Head-to-head comparison**: `final_wasserstein_vs_hmm.py` (273 lines)
- **Multi-stock backtest**: `test_paper_wasserstein_backtest.py` (477 lines)
- **Results**: `final_comparison_*.json`

## References

Horvath, B., Teichmann, J., & Žurič, Ž. (2021). "Clustering Market Regimes using the Wasserstein Distance." arXiv:2110.13087.
