# Quick Start - Regime Detection in Agent

## 🚀 Test It Now

```bash
# 1. Run integration test (validates all 3 tools work)
python tests/test_agent_regime_integration.py

# 2. Start the agent
python analyze_trade_agent.py

# 3. Try these queries:
```

### Example Queries

**Volatility Regime (Should call Wasserstein)**:
```
"What's the current volatility regime for MSFT?"
"Is NVDA in a high volatility period right now?"
```

**Trend Regime (Should call HMM)**:
```
"Is AAPL in a bullish regime?"
"What's the market regime for JPM - bearish or bullish?"
```

**High Conviction (Should call BOTH + consensus)**:
```
"Should I buy JNJ? Need high confidence on regime."
"Large position on MSFT - what's the regime consensus?"
```

---

## 🔍 What to Watch For

### Planner Tool Selection

**Good**: Planner chooses appropriate tool(s) based on:
- Stock type (MSFT → Wasserstein, AAPL → HMM or both)
- Question type ("volatility" → Wass, "trend/bullish" → HMM)
- High stakes → Both + consensus check

**Bad**: Always picks same tool regardless of context

### Agent Reasoning

**Good**: Agent explains:
- Why it trusts/distrusts regime based on confidence metrics
- How regime affects position sizing (High Vol → reduce 30-50%)
- References trading book wisdom on uncertain regimes
- Mentions when models disagree as uncertainty signal

**Bad**: Just reports regime without context or position sizing implications

### Consensus Handling

**Agreement**: Should increase confidence in recommendation
```
"Both Wasserstein (High Vol) and HMM (Bearish) agree → 
Defensive positioning required. Reduce size 50%..."
```

**Disagreement**: Should flag as uncertainty
```
"⚠ REGIME UNCERTAINTY: Models diverge (Wass=High Vol, HMM=Bullish).
Van Tharp emphasizes avoiding uncertain regimes.
VERDICT: UNCLEAR - Wait for consensus"
```

---

## 📊 Tool Output Quick Reference

### Wasserstein
```json
{
  "regime": 2,                           // 0=Low, 1=Med, 2=High Vol
  "regime_name": "High Volatility",
  "confidence": "medium",                // based on MMD quality
  "mmd_quality_ratio": 1.15,            // <1.1 poor, >1.5 good
  "interpretation": "Reduce size 30%"
}
```

**Key Metric**: `mmd_quality_ratio`
- < 1.1 = Don't trust (poor separation)
- 1.1-1.5 = Use with caution
- \> 1.5 = Good confidence

### HMM
```json
{
  "regime": 1,                          // 0=Bearish, 1=Sideways, 2=Bullish
  "regime_name": "Sideways/Choppy",
  "confidence": 0.85,                   // Forward filter probability
  "persistence_probability": 0.88,      // Stay in regime
  "interpretation": "88% stays sideways"
}
```

**Key Metric**: `persistence_probability`
- \> 0.85 = Stable regime
- < 0.70 = Unstable (likely transition)

### Consensus
```json
{
  "agreement": false,                    // Do regimes align?
  "confidence_level": "LOW",             // HIGH or LOW
  "recommendation": "⚠ Uncertainty..."
}
```

**Agreement = HIGH confidence**  
**Disagreement = Reduce size or wait**

---

## 🎯 Expected Planner Behavior

### Stock Type Recognition

| Stock | Type | Preferred Tool | Reasoning |
|-------|------|----------------|-----------|
| MSFT, NVDA, AAPL | Tech | Wasserstein first | Won +0.23 on MSFT |
| JNJ, PFE | Healthcare | Wasserstein first | Won +0.25 on JNJ |
| JPM, BAC | Financial | Both or HMM | More stable patterns |
| KO, PG | Stable/Consumer | HMM first | Won on AAPL (similar) |

### Question Type Recognition

| Question Contains | Should Call | Why |
|------------------|-------------|-----|
| "volatility", "vol", "risky" | Wasserstein | Vol-focused regimes |
| "trend", "bullish", "bearish" | HMM | Trend-focused regimes |
| "regime transition", "probability" | HMM | Has transition probs |
| "high conviction", "large position" | BOTH + consensus | Need confidence |

---

## ⚠️ Common Issues

### Issue: Tools not loading
**Symptom**: Error about missing models  
**Fix**: Check `models/` folder has both `.py` files  
**Workaround**: Tools return error dict gracefully (agent should handle)

### Issue: Planner always calls both tools
**Symptom**: Every query uses both + consensus  
**Fix**: Refine planner prompt with more specific guidance  
**Note**: Not necessarily bad (just slower)

### Issue: Agent ignores regime results
**Symptom**: Verdict doesn't mention regime or position sizing  
**Fix**: Check if tool results being passed to decision LLM  
**Workaround**: Add regime mentions to RAG query

### Issue: MMD always ~1.0 (poor quality)
**Symptom**: Wasserstein shows "low confidence" every time  
**Note**: This is expected (universal in testing)  
**Response**: Agent should mention as caveat, rely more on HMM for consensus

---

## 📈 Performance Reference

Quick lookup for when agent should favor each method:

### Wasserstein Wins
- **MSFT**: +0.231 Sharpe edge (1.922 vs 1.691)
- **JNJ**: +0.250 Sharpe edge (0.238 vs -0.012)
- **Avg edge**: +22% across tested stocks

### HMM Wins  
- **AAPL**: +0.007 Sharpe edge (1.187 vs 1.180) - narrow
- **Better when**: Stable regimes, smooth transitions

### Tie
- **JPM**: 2-2 score, but Wass higher Sharpe (2.140 vs 1.631)

**Rule of Thumb**: Use Wasserstein for tech/healthcare, HMM for stable stocks or when need transition probs

---

## 📚 Documentation

**Full Guides**:
1. [`docs/REGIME_DETECTION_AGENT_GUIDE.md`](REGIME_DETECTION_AGENT_GUIDE.md) - Complete decision framework (210 lines)
2. [`docs/INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md) - What was built and why (220 lines)
3. [`docs/WASSERSTEIN_VS_HMM_VERDICT.md`](WASSERSTEIN_VS_HMM_VERDICT.md) - Detailed test results

**Code**:
- [`tools.py`](../tools.py) - Tools implementation (~1430 lines, see lines 1030-1430)
- [`planner.py`](../planner.py) - Planner guidance (lines 178-210)
- [`tests/test_agent_regime_integration.py`](../tests/test_agent_regime_integration.py) - Integration test

---

## ✅ Success Checklist

Before considering integration "working", verify:

- [ ] Integration test passes: `python tests/test_agent_regime_integration.py`
- [ ] Agent calls Wasserstein for tech stock + volatility query
- [ ] Agent calls HMM for stable stock + trend query  
- [ ] Agent calls both + consensus for "high conviction" query
- [ ] Agent mentions position sizing when regime is high risk
- [ ] Agent flags uncertainty when models disagree
- [ ] Agent references MMD quality in Wasserstein confidence
- [ ] Agent uses HMM persistence probability in interpretation

---

## 🐛 Debug Commands

If something isn't working:

```bash
# Check tools registered
python -c "from tools import get_all_tools; print([t['name'] for t in get_all_tools()])"

# Test Wasserstein directly
python -c "from tools import regime_detection_wasserstein_tool_fn; print(regime_detection_wasserstein_tool_fn({'tool_results': {}}, {'symbol': 'MSFT', 'description': 'test'}))"

# Check planner can see tools
python -c "from planner import plan_tools; print(plan_tools('What is MSFT volatility regime?', 'MSFT'))"

# Run agent with verbose logging
python analyze_trade_agent.py  # Then paste query and check planner output
```

---

**Ready to test? Run**: `python tests/test_agent_regime_integration.py`
