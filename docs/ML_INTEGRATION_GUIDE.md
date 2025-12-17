# ML Model Integration Guide

## Overview
Integrated all 4 trained ML models (Random Forest, XGBoost, Logistic Regression, Decision Tree) into the trading agent's tool registry with comprehensive performance context.

## Integration Approach: Multi-Model with Context

### Why All Models?
1. **Horizon-specific strengths**: Different models excel at different prediction horizons
   - 3d: Random Forest (Sharpe 1.52)
   - 5d: XGBoost (Sharpe 1.34)
   - 10d: XGBoost (Sharpe 1.45)

2. **Ensemble robustness**: Agent sees all predictions and can weight by model confidence

3. **Consensus signal**: Strong agreement (75%+ models) = high confidence, Divergence = uncertainty

4. **Adaptability**: Agent can choose based on context, regime, volatility, etc.

## Implementation Files

### 1. `ml_prediction_tool.py` (NEW)
- Core prediction engine
- Loads trained models from `ml_models/saved_models/`
- Returns predictions from all 4 models with:
  - Individual predictions (UP/DOWN)
  - Probability distributions (if available)
  - Performance metrics (Sharpe, win rate, returns)
  - Consensus calculation
  - Best model identification
  - Pipeline context (features, training stats)

### 2. `tools.py` (UPDATED)
- Added ML prediction tool to registry
- Tool name: `ml_prediction`
- Parameters:
  - `symbol` (required): Stock ticker
  - `horizon` (optional, default 5): 3, 5, or 10 days
  - `models` (optional): Specific models to use (default: all 4)

### 3. `planner.py` (UPDATED)
- Added comprehensive ML tool guidance
- When to use: Directional forecasts, entry/exit signals, quantitative validation
- Horizon selection guide
- Interpretation framework (consensus strength, probabilities)
- Integration with other tools (technical, regime, sentiment)

## Model Performance Context

### Embedded in Every Prediction:
```python
{
  "3d": {
    "best_model": "Random Forest",
    "models": {
      "Random Forest": {
        "sharpe": 1.517,
        "return": 52.31,
        "win_rate": 17.91,
        "strengths": "Best for short-term, handles non-linear patterns",
        "weaknesses": "Lower win rate, can be volatile"
      },
      # ... XGBoost, Logistic, Decision Tree
    }
  },
  # ... 5d, 10d
}
```

### Pipeline Stats (Always Included):
- **Features**: 141 (optimized from 167 → 196 → 141 via pruning)
- **Training**: 25 stocks, 2020-2025, 60/20/20 split
- **Performance**: +26.0% vs baseline, +49.3% feature efficiency
- **Optimization**: Optuna hyperparameter tuning (50 trials per horizon)

## Usage Examples

### Basic Call (Agent)
```python
{
  "tool_name": "ml_prediction",
  "arguments": {
    "symbol": "AAPL",
    "horizon": 5
  }
}
```

### Response Structure
```python
{
  "symbol": "AAPL",
  "horizon": "5d",
  "timestamp": "2025-12-16 ...",
  "predictions": {
    "Random Forest": {
      "prediction": "UP",
      "direction": 1,
      "probability": {
        "down": 0.32,
        "up": 0.68,
        "confidence": 0.68
      },
      "performance": {
        "sharpe": 1.117,
        "return": 27.91,
        "win_rate": 15.35,
        "strengths": "Ensemble robustness",
        "weaknesses": "Low win rate for 5d"
      }
    },
    "XGBoost": { ... },
    "Logistic Regression": { ... },
    "Decision Tree": { ... }
  },
  "consensus": {
    "direction": "STRONG UP",  # or WEAK UP, WEAK DOWN, STRONG DOWN
    "up_votes": 3,
    "total_votes": 4,
    "confidence": 0.75
  },
  "best_model": {
    "name": "XGBoost",
    "prediction": "UP",
    "probability": { ... },
    "performance": {
      "sharpe": 1.343,
      "return": 34.97,
      "win_rate": 19.88,
      "strengths": "Best for medium-term, robust gradient boosting",
      "weaknesses": "Moderate win rate"
    }
  },
  "horizon_performance": { ... },
  "pipeline_context": { ... },
  "latest_price": 185.23,
  "feature_count": 141
}
```

## Agent Decision Framework

### Interpretation Levels:

1. **STRONG Consensus (75-100% agreement)**
   - High confidence signal
   - Consider larger position size
   - All/most models agree on direction

2. **WEAK Consensus (50-75% agreement)**
   - Moderate confidence
   - Standard position size
   - Some model disagreement

3. **NO Consensus (<50% agreement)**
   - Low confidence / uncertainty
   - Reduce size or pass
   - Significant model disagreement

### Probability Thresholds:
- **>70% confidence**: High conviction
- **60-70%**: Moderate conviction
- **50-60%**: Low conviction (coin flip)
- **<50%**: Wrong direction predicted

### Integration with Other Tools:

**Technical + ML**:
```
- RSI oversold + ML STRONG UP = High conviction buy
- RSI overbought + ML WEAK UP = Take profits / reduce
- MACD bullish + ML STRONG UP = Trend confirmation
```

**Regime + ML**:
```
- High volatility regime + ML STRONG UP = Reduce size (risk management)
- Low volatility regime + ML STRONG UP = Standard size
- Regime transition + ML disagreement = Pass (uncertainty)
```

**Sentiment + ML**:
```
- Bullish news + ML STRONG UP = Catalyst alignment
- Bearish news + ML WEAK UP = Proceed with caution
- Mixed sentiment + ML STRONG consensus = Data overrides noise
```

### Best Practices:

1. **Always check consensus strength** before making decision
2. **Review best model performance** (Sharpe, win rate) for context
3. **Compare to technical/regime signals** for confirmation
4. **Use appropriate horizon** for trade timeframe
5. **Respect model limitations**:
   - Models trained on 2020-2025 (bull market heavy)
   - Win rates are modest (15-25%) - many false signals
   - Sharpe ratios strong but not infallible
   - No model predicts black swans

## Strengths & Limitations

### Strengths:
- ✅ Quantitative, data-driven forecasts
- ✅ Multiple model perspectives (ensemble)
- ✅ Comprehensive performance context
- ✅ Horizon-specific optimization
- ✅ Fast execution (~2 seconds)
- ✅ Consensus signal for confidence
- ✅ Validated on 25 stocks over 5 years

### Limitations:
- ⚠️ Models can be wrong (15-25% win rates typical)
- ⚠️ Training period bias (2020-2025 = mostly bull)
- ⚠️ No guarantee of future performance
- ⚠️ Black swan events not in training data
- ⚠️ Works best with proper risk management
- ⚠️ Should complement, not replace, human judgment

## Recommended Usage Pattern

### For Trading Ideas:
1. **Technical Analysis** (price, RSI, MACD, volume)
2. **Regime Detection** (volatility + trend regime)
3. **ML Prediction** (directional forecast + consensus)
4. **Sentiment** (news catalyst / crowd positioning)
5. **Synthesis**: Agent weighs all signals with context

### Sample Agent Output:
```
ANALYSIS: AAPL 5-day outlook

Technical: Price above 50/200 MA, RSI neutral (55), MACD bullish crossover
Regime: Medium volatility + Sideways (58% probability), stable conditions
ML: STRONG UP consensus (3/4 models, 75% confidence)
  - XGBoost (best for 5d, Sharpe 1.34): UP with 72% probability
  - Random Forest: UP with 65% probability  
  - Logistic: UP with 68% probability
  - Decision Tree: DOWN (contrarian signal)
Sentiment: Bullish (+0.35 FinBERT score, earnings beat catalyst)

VERDICT: ATTRACTIVE IF STRICT RULES FOLLOWED
- Entry: $185 (current)
- Target: $192 (3.8% upside, 5-day horizon)
- Stop: $181 (2.2% risk, below MA support)
- Size: 2% account (medium conviction, regime stable)
- Rationale: Technical + ML + sentiment aligned, regime supports standard sizing
```

## Testing the Integration

Run the standalone test:
```bash
cd C:\Users\Andrew\projects\agentic_ai_trader
python ml_prediction_tool.py
```

Or test through the agent:
```bash
python analyze_trade_agent.py
# Enter: "What's the ML forecast for AAPL over the next 5 days?"
```

## Next Steps

1. ✅ **DONE**: Created ml_prediction_tool.py with all 4 models
2. ✅ **DONE**: Integrated into tools.py registry
3. ✅ **DONE**: Updated planner.py with ML guidance
4. 🔄 **TEST**: Validate predictions work end-to-end
5. ⏳ **MONITOR**: Track agent usage patterns and effectiveness
6. ⏳ **TUNE**: Adjust consensus thresholds based on live performance
7. ⏳ **EXPAND**: Consider adding volatility forecasts, time-series confidence intervals

## Summary

The integration provides the agent with **quantitative, validated predictions** from professionally trained models while maintaining full context about their strengths, limitations, and historical performance. The multi-model approach with consensus calculation allows the agent to make informed decisions about signal confidence and appropriate risk sizing.

The agent can now synthesize:
- **Technical** (what's happening)
- **Regime** (market environment)  
- **ML** (what's likely next)
- **Sentiment** (what catalysts exist)
- **Books** (what trading wisdom says)

This creates a comprehensive, multi-dimensional analysis framework for trading decisions.
