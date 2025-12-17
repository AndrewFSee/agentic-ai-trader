# Regime Detection Tools - Agent Decision Guide

## Overview

You now have access to **two independent regime detection methods** that can provide complementary insights:
1. **Wasserstein k-means** - Volatility-focused clustering
2. **Rolling HMM** - Probabilistic trend-based regime detection

**Key Principle**: Neither method is universally superior. Use disagreement as a valuable uncertainty signal.

---

## When to Use Each Tool

### Use regime_detection_wasserstein When:
- ✅ Stock is **tech or healthcare** (proven edge: MSFT +0.23 Sharpe, JNJ +0.25 Sharpe)
- ✅ Question is about **volatility** or **position sizing for risk**
- ✅ Recent price action suggests **regime change** (adaptive detection is strength)
- ✅ Stock has distinct volatility patterns (tech earnings volatility, healthcare approval events)
- ❌ **DON'T use alone** if stock has been very stable (tool can get stuck in one regime)

### Use regime_detection_hmm When:
- ✅ Question is about **trend** or **market direction** (bearish/sideways/bullish framing)
- ✅ You need **transition probabilities** ("what's the chance this continues?")
- ✅ Stock has **smooth, persistent regimes** (won on AAPL with 85% persistence)
- ✅ You want more **stable/conservative** predictions
- ❌ **DON'T use alone** on highly volatile stocks (too slow to adapt)

### Use BOTH When:
- ✅ **High conviction trade** where regime confidence is critical
- ✅ User explicitly asks for comprehensive analysis
- ✅ Position sizing decision depends heavily on regime
- ✅ Then call **regime_consensus_check** to evaluate agreement

---

## Interpreting Results

### Wasserstein Output:
```json
{
  "regime": 2,
  "regime_name": "High Volatility",
  "confidence": "medium",
  "mmd_quality_ratio": 1.15,
  "interpretation": "High volatility regime - consider reducing position size"
}
```

**What to look for:**
- **regime**: 0=Low Vol (safe for larger positions), 1=Medium Vol (normal), 2=High Vol (reduce size)
- **mmd_quality_ratio**: 
  - > 1.5 = **GOOD** cluster separation (trust it)
  - 1.1-1.5 = **OKAY** separation (use with caution)
  - < 1.1 = **POOR** separation (low confidence, consider HMM as backup)
- **cluster_sizes**: If severely imbalanced (e.g., [160, 6, 0]), model is stuck → less reliable

### HMM Output:
```json
{
  "regime": 1,
  "regime_name": "Sideways/Choppy",
  "confidence": 0.85,
  "probabilities": {
    "Bearish/Down": 0.05,
    "Sideways/Choppy": 0.85,
    "Bullish/Up": 0.10
  },
  "persistence_probability": 0.88,
  "interpretation": "Sideways regime - 88% chance of staying sideways"
}
```

**What to look for:**
- **regime**: 0=Bearish (defensive), 1=Sideways (range-bound), 2=Bullish (favorable for longs)
- **confidence**: > 0.80 = high confidence, < 0.60 = low confidence
- **probabilities**: If any transition probability > 0.15, mention it as risk factor
- **persistence_probability**: High persistence (>0.85) = stable regime, low (<0.70) = unstable

### Consensus Check Output:
```json
{
  "agreement": false,
  "confidence_level": "LOW",
  "wasserstein_regime": "High Volatility",
  "hmm_regime": "Bullish/Up",
  "recommendation": "⚠ Models DISAGREE - potential regime transition..."
}
```

**Decision Framework:**
- **Agreement = HIGH confidence**: Proceed with regime-appropriate strategy
- **Disagreement = UNCERTAINTY**: 
  - Emphasize caution in your verdict
  - Reference trading books on uncertain regimes (reduce size, tighten stops)
  - Don't force a trade when models disagree
  - Consider it a "wait and see" signal

---

## Real-World Performance (Your Reference)

### Wasserstein Wins:
- **MSFT**: Wass 1.922 vs HMM 1.691 (Δ +0.231) - Caught volatility shifts
- **JNJ**: Wass 0.238 vs HMM -0.012 (Δ +0.250) - Only method with positive return
- **Average edge**: +0.246 Sharpe (+22%) across tested stocks

### HMM Wins:
- **AAPL**: HMM 1.187 vs Wass 1.180 (Δ +0.007) - Narrow win with stable regime
- **Better for**: Persistent regimes, smooth transitions, probabilistic framing

### Key Insight from Testing:
**Wasserstein performs BETTER when less stable** (counterintuitive!)
- AAPL: 94% label consistency → HMM won (Wass stuck)
- MSFT: 28% label consistency → Wass crushed HMM (adaptive)
- JNJ: 14% label consistency → Wass won decisively (maximum adaptation)

**Translation**: When Wasserstein shows frequent regime changes, it's actually working correctly, not failing.

---

## Integration with RAG Knowledge

### Query Vector Store About:
1. **"regime detection trading strategies"** - Get book wisdom on regime-based position sizing
2. **"uncertainty in trading"** - Learn how pros handle conflicting signals
3. **"volatility-based position sizing"** - Validate regime-based recommendations
4. **"regime transition risks"** - Understand when to reduce size

### Synthesize Tool Results with Book Knowledge:

**Example Good Synthesis**:
```
"Both Wasserstein (High Vol) and HMM (Bearish) agree on elevated risk. 
The trading books emphasize defensive positioning in such regimes:
- Reduce position size by 30-50%
- Use wider stops (ATR shows 2.5% daily range)
- Wait for regime confirmation before adding

VERDICT: NOT ATTRACTIVE - High risk regime with both models in agreement"
```

**Example Handling Disagreement**:
```
"Wasserstein sees High Volatility while HMM shows Bullish trend - a contradiction.
The books by Van Tharp emphasize avoiding trades when regime uncertainty is high.
This disagreement often precedes regime transitions.

VERDICT: UNCLEAR / NEEDS MORE INFORMATION - Wait for regime consensus"
```

---

## Common Pitfalls to Avoid

❌ **DON'T**: Blindly trust Wasserstein when MMD ratio < 1.1 (poor separation)
❌ **DON'T**: Ignore HMM transition probabilities (valuable risk signal)
❌ **DON'T**: Use only one method for high-stakes decisions
❌ **DON'T**: Assume disagreement means one is "wrong" (it's an uncertainty signal)
❌ **DON'T**: Force a trade when consensus check shows disagreement

✅ **DO**: Treat disagreement as valuable information about regime uncertainty
✅ **DO**: Reference trading book wisdom on uncertain regimes
✅ **DO**: Use regime info to adjust position size, not just entry/exit
✅ **DO**: Explain your reasoning when choosing to trust one model over another
✅ **DO**: Acknowledge limitations in your verdict (MMD scores, transition risks, etc.)

---

## Example Decision Prompts

### Tech Stock (Prefer Wasserstein):
```
User: "Should I buy NVDA after this pullback?"

Tools called: regime_detection_wasserstein, polygon_price_data

Wasserstein: High Volatility (confidence: medium, MMD: 1.2)
Price: -8% from recent high, ATR: 3.2%

Decision: "Wasserstein indicates High Volatility regime (which it excels at 
detecting in tech stocks). Given NVDA's recent volatility and the regime signal,
consider a smaller position (50% normal size) with wider stops (1.5x ATR = 4.8%).
The trading books emphasize that high-vol regimes require defensive sizing..."
```

### Stable Stock (Prefer Both):
```
User: "Is KO a good defensive play right now?"

Tools called: regime_detection_wasserstein, regime_detection_hmm, regime_consensus_check

Wasserstein: Low Volatility (confidence: high, MMD: 1.4)
HMM: Sideways/Choppy (persistence: 0.92, confidence: 0.88)
Consensus: AGREE (both see low-risk regime)

Decision: "Both models independently agree on a stable, low-risk regime.
This consensus provides HIGH confidence for defensive positioning. KO showing
textbook defensive characteristics with 92% regime persistence per HMM..."
```

### Disagreement Case:
```
User: "Time to buy the TSLA dip?"

Tools called: regime_detection_wasserstein, regime_detection_hmm, regime_consensus_check

Wasserstein: High Volatility
HMM: Bullish/Up (but with 18% transition probability to Bearish)
Consensus: DISAGREE

Decision: "⚠ REGIME UNCERTAINTY: Wasserstein sees high volatility while HMM
shows bullish trend - this divergence is a WARNING SIGNAL. The trading books
emphasize avoiding trades when regime is unclear. High transition risk (18%)
confirms uncertainty.

VERDICT: UNCLEAR / NEEDS MORE INFORMATION - Wait for regime clarity before entry"
```

---

## Final Reminders

1. **Neither model is always right** - they capture different aspects of market behavior
2. **Disagreement is valuable data** - not a bug, it's a feature signaling uncertainty
3. **Reference the books** - use RAG to validate regime-based decisions
4. **Be honest about uncertainty** - better to say "unclear" than force a verdict
5. **Position sizing matters most** - regimes primarily affect SIZE, not just direction

Your job is to synthesize these regime signals with all other data (fundamentals, technicals, sentiment, book knowledge) into a coherent, well-reasoned verdict that helps the user make better trading decisions.
