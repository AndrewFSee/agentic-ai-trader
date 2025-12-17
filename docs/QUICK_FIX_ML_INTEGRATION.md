# Quick Fix: ML Predictions Not Showing

## Problem 1: Missing pyarrow in WSL conda environment

The error "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'" means pyarrow is not installed in your WSL conda environment (`agentic_ai_trader_env`).

### Solution:
```bash
# In WSL, activate your conda environment
conda activate agentic_ai_trader_env

# Install pyarrow
pip install pyarrow

# Or use conda
conda install -c conda-forge pyarrow
```

## Problem 2: torch/transformers warning

The warning "torch and/or transformers not installed; FinBERT sentiment will return 'unknown' labels" appears because:
- The sentiment tool tries to load FinBERT for ML-based sentiment analysis
- Without torch/transformers, it falls back to basic sentiment (which still works)

### Solutions:

**Option A: Install torch/transformers (for full FinBERT sentiment)**
```bash
# In WSL conda environment
pip install torch transformers

# Note: This adds ~2GB of dependencies
```

**Option B: Suppress the warning (if you don't need ML sentiment)**
Edit `tools.py` and change the warning to only show once or suppress it entirely.

## Problem 3: Planner not selecting ml_prediction

I added debug logging to show:
1. Which tools the planner selects
2. Which tools return results  
3. If ml_prediction fails, what the error is

Run the agent again and you'll see output like:
```
[DEBUG] Planner selected tools: polygon_price_data, ml_prediction, regime_hmm, ...
[DEBUG] Tool results available: polygon_price_data, regime_hmm, ...
[DEBUG] ML prediction error: Unable to find a usable engine...
```

This will tell us exactly what's happening.

## Quick Test After Installing pyarrow:

```bash
# In WSL
conda activate agentic_ai_trader_env
pip install pyarrow
python analyze_trade_agent.py
```

Then enter:
- Trading idea: `buy or sell recommendation`
- Symbol: `MSFT`

You should now see:
1. `[DEBUG] Planner selected tools: ...` (should include `ml_prediction`)
2. `[DEBUG] ML prediction SUCCESS for MSFT (5d)` 
3. An "ML MODEL PREDICTIONS" section in the analysis with:
   - Individual model predictions (XGBoost, Random Forest, etc.)
   - Consensus direction and strength
   - Performance metrics

## Expected Output Structure:

```
=== ANALYSIS ===

1. PRICE & TREND
2. TECHNICALS
3. FUNDAMENTALS
4. REGIME ANALYSIS
5. NEWS & SENTIMENT
6. ML MODEL PREDICTIONS          ← NEW SECTION
   Individual Model Predictions:
   - XGBoost: UP (65% prob, HIGH conf)
   - Random Forest: UP (58% prob, MEDIUM conf)
   - ...
   
   Consensus:
   - Direction: STRONG UP
   - Agreement: 75%
   - Recommendation: ...
7. EDGE ASSESSMENT
8. RISK MANAGEMENT
9. VERDICT
...
```

## If Still Not Working:

Share the debug output and I'll help diagnose further. The debug logs will show:
- Did planner select ml_prediction? (if no, planner prompt needs adjustment)
- Did ml_prediction return results? (if no, data fetching issue)
- Did ml_prediction return an error? (shows specific error message)
