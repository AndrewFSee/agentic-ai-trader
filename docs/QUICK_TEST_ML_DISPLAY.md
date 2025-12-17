# Quick Test Guide - ML Display Enhancement

## What Changed?

The ML MODEL PREDICTIONS section in the agent's output is now **much more prominent and detailed**.

## How to Test

```bash
python analyze_trade_agent.py
```

**Input:**
- Trading idea: `buy or sell`
- Symbol: `AAPL`
- Then: `q` (to quit after analysis)

## What You Should See

### In the Data Section (fed to LLM):

You'll see formatted ML data like this:

```
================================================================================
🤖 MACHINE LEARNING MODEL PREDICTIONS
Symbol: AAPL | Horizon: 5d | Models Trained on 25 Stocks (2020-2025)
================================================================================

📊 INDIVIDUAL MODEL FORECASTS:
--------------------------------------------------------------------------------

📉 **Random Forest**:
   Prediction: DOWN/FLAT
   Confidence: 60.4%
   Probability Breakdown: UP 39.6% | DOWN 60.4%
   Historical Performance: Sharpe Ratio 1.52, Win Rate 17.9%

📉 **XGBoost**:
   Prediction: DOWN/FLAT
   Confidence: 97.5%
   Probability Breakdown: UP 2.5% | DOWN 97.5%
   Historical Performance: Sharpe Ratio 1.34, Win Rate 15.5%

[... 2 more models ...]

================================================================================
🎯 CONSENSUS VERDICT:
================================================================================

🔴 Consensus Direction: STRONG DOWN
   Agreement Level: 100% (STRONG ⚡)
   Vote Distribution: 0 models UP, 4 models DOWN (Total: 4)

💡 Interpretation:
   - STRONG consensus = High confidence. Models are aligned.
   - Suggested sizing: NORMAL position size (e.g., 2-3% of account)
```

### In the LLM's Analysis Output:

You should see a **dedicated section 6** (or whatever number after regime/sentiment) titled:

```
6. ML MODEL PREDICTIONS

[The LLM will provide comprehensive analysis covering:]

a) INDIVIDUAL MODEL REVIEW
   - Discussion of each model (RF, XGBoost, Logistic, Decision Tree)
   - Their confidence levels and probability breakdowns
   - Any disagreements noted

b) CONSENSUS INTERPRETATION
   - Clear statement of consensus direction (e.g., "STRONG DOWN")
   - Agreement level (e.g., "100% agreement - all 4 models agree")
   - Strength classification (STRONG/MODERATE/WEAK)

c) CROSS-VALIDATION WITH OTHER SIGNALS
   - How ML compares with RSI, MACD, Bollinger Bands
   - Alignment with regime analysis (volatility and trend)
   - Agreement/disagreement with FinBERT sentiment

d) POSITION SIZING GUIDANCE
   - Based on consensus strength:
     * STRONG (≥75%) → Normal size (2-3% of account)
     * MODERATE (50-74%) → Reduced by 30-50%
     * WEAK (<50%) → Skip or minimal (<1%)

e) ML LIMITATIONS & CONTEXT
   - Feature-driven, not news-reactive
   - May miss recent catalysts
   - Trained on 2020-2025 data
   - Short-term tactical (5-day) signals only

f) FINAL ML VERDICT
   - Does ML support or oppose the trade idea?
   - How does it affect overall confidence?
```

## Key Differences from Before

**BEFORE:**
```
6. ML MODEL PREDICTIONS (5-DAY HORIZON)
   - All individual models: RF, XGBoost, LR, DT = DOWN
   - Probabilities/confidence = 0% (metadata inconsistent)
   - Consensus: "STRONG DOWN" but 0% agreement, 0 votes
   [brief mention, maybe 3-4 sentences total]
```

**AFTER:**
```
6. ML MODEL PREDICTIONS (MANDATORY DEDICATED SECTION)

a) INDIVIDUAL MODEL REVIEW:
   [Detailed discussion of each model - 5-10 sentences]

b) CONSENSUS INTERPRETATION:
   [Clear explanation of consensus strength - 4-6 sentences]

c) CROSS-VALIDATION WITH OTHER SIGNALS:
   [Comparison with technicals, regime, sentiment - 6-10 sentences]

d) POSITION SIZING GUIDANCE:
   [Sizing recommendations based on consensus - 3-5 sentences]

e) ML LIMITATIONS & CONTEXT:
   [Caveats and context - 4-6 sentences]

f) FINAL ML VERDICT:
   [Summary and impact on confidence - 3-5 sentences]

[Total: 25-40+ sentences, comprehensive analysis]
```

## Success Criteria

✅ Section 6 (or appropriate number) is titled "ML MODEL PREDICTIONS"  
✅ All 4 models are discussed individually  
✅ Consensus strength is clearly interpreted (STRONG/MODERATE/WEAK)  
✅ Position sizing guidance is provided  
✅ ML is cross-validated against technicals, regime, and sentiment  
✅ ML limitations are acknowledged  
✅ Section length is substantial (not just 3-4 sentences)

## If You Don't See This

If the ML section is still brief or missing:

1. **Check debug output:**
   ```
   [DEBUG] Planner selected tools: ..., ml_prediction
   [DEBUG] ML prediction SUCCESS for AAPL (5dd)
   ```
   If you see this, the tool is working correctly.

2. **Check for errors:**
   Look for any error messages about model loading or feature mismatches.

3. **Try another symbol:**
   Test with NVDA, TSLA, MSFT, or JPM (all have trained models).

4. **Check model files exist:**
   ```bash
   ls ml_models/saved_models/AAPL_5d/
   ```
   Should show: `random_forest.pkl`, `xgboost.pkl`, `logistic.pkl`, `decision_tree.pkl`

## Troubleshooting

**"No trained models found" error:**
- Models only exist for 25 stocks (AAPL, NVDA, TSLA, MSFT, etc.)
- For other symbols, ML section will show "not available"

**"Feature mismatch" error:**
- All models were retrained on Dec 16, 2025 with 125 features
- If you see this, check that you're using the latest code

**Emoji not displaying:**
- Windows console may not show emoji properly
- This is cosmetic only - the LLM still sees the formatted text
- For best display, use Windows Terminal or run from WSL

## Expected Runtime

- Tool selection (planner): ~2-5 seconds
- Tool execution (8-10 tools including ML): ~15-30 seconds
- LLM analysis generation: ~10-20 seconds
- **Total: ~30-60 seconds**

ML prediction adds minimal overhead (~2-3 seconds) since models load quickly.

## Next Steps After Testing

Once you confirm the ML section is prominent and detailed:

1. **Test with multiple symbols** to verify robustness
2. **Compare ML predictions across different market conditions**
3. **Consider adding more horizons** (currently 5d, can add 3d and 10d)
4. **Retrain models periodically** as new data accumulates

## Need Help?

If ML section still doesn't appear prominently after testing:
1. Share the full agent output
2. Check if [DEBUG] messages show ML tool was called
3. Verify which model predictions were returned
4. Look for any error messages in the output

The enhanced formatter and strengthened prompt should ensure ML gets equal prominence to price, technicals, fundamentals, regime, and sentiment! 🚀
