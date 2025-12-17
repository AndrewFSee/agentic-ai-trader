# Quick Reference - Feature Pruning & Hyperparameter Tuning

## Current Status
- ✅ Feature pruning implemented (171 → 141 features)
- ✅ 3-stock test validated (AAPL, NVDA, JPM showing +40-80% Sharpe improvements)
- ⏳ Full 25-stock training in progress
- ⏳ Analysis pending (after training completes)
- ⏳ Hyperparameter tuning ready to run

## Quick Commands

### Monitor Training Progress
```bash
# Get latest results file info
cd ml_models
ls -ltr results/ml_models/*.json | tail -1

# Count completed stocks
python -c "import json; f=sorted(__import__('pathlib').Path('results/ml_models').glob('ml_pipeline_results_*.json'))[-1]; d=json.load(open(f)); print(f'Completed: {len(d[\"individual_results\"])} / 75')"
```

### After Training Completes

#### 1. Analyze Results
```bash
cd ml_models
python analyze_pruned_results.py
```
**Output:** Three-way comparison (baseline vs enhanced vs pruned)

#### 2. If Pruning Successful (Mean Sharpe > 1.15)
```bash
# Run hyperparameter tuning on 3 stocks (2-3 hours)
python tune_hyperparameters.py

# Review and apply best parameters
python apply_tuned_params.py

# Full training with tuned parameters (~60 min)
python run_pipeline.py --test

# Analyze final results
python analyze_pruned_results.py
```

#### 3. If Pruning Insufficient (Mean Sharpe < 1.15)
```bash
# More aggressive pruning - edit feature_engineering.py
# Add to FEATURES_TO_DROP:
#   'day_of_week', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday',
#   'gap', 'gap_pct', 'gap_up', 'gap_down'

# Re-run training
python run_pipeline.py --test

# Analyze
python analyze_pruned_results.py
```

## Files Overview

### Analysis Scripts
- **[analyze_feature_importance.py](analyze_feature_importance.py)** - Identifies low-importance features
- **[analyze_pruned_results.py](analyze_pruned_results.py)** - Three-way comparison (baseline/enhanced/pruned)
- **[analyze_enhanced_results.py](analyze_enhanced_results.py)** - Two-way comparison (baseline/enhanced)

### Tuning Scripts
- **[tune_hyperparameters.py](tune_hyperparameters.py)** - Optuna hyperparameter optimization
- **[apply_tuned_params.py](apply_tuned_params.py)** - Apply tuned params to config

### Configuration
- **[feature_engineering.py](feature_engineering.py)** - Contains FEATURES_TO_DROP list (lines 360-395)
- **[features_to_drop.txt](features_to_drop.txt)** - List of 30 dropped features

### Documentation
- **[FEATURE_PRUNING_SUMMARY.md](FEATURE_PRUNING_SUMMARY.md)** - Comprehensive summary
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - This file

## Performance Targets

### Current Results
- **Baseline (167 features):** Sharpe 1.141
- **Enhanced (196 features):** Sharpe 1.090 (-4.1%)
- **Pruned (141 features):** TESTING (target: >1.15)

### Expected Outcomes

**Pruned Features (no tuning):**
- Conservative: Sharpe 1.15-1.20 (+5-8% vs enhanced)
- Optimistic: Sharpe 1.25-1.30 (+15-19% vs enhanced)

**Pruned + Hyperparameter Tuning:**
- Conservative: Sharpe 1.30-1.35 (+19-24% vs enhanced)
- Optimistic: Sharpe 1.40-1.50 (+28-37% vs enhanced)

**Buy-and-Hold Baseline:** Sharpe 0.428

## Key Decisions

### Decision Point 1: Pruning Results
**After `analyze_pruned_results.py`**

✅ **If Mean Sharpe > 1.15:**
- Pruning successful, proceed to hyperparameter tuning
- Keep gap/RSI/day-of-week features (marginal signal acceptable)

❌ **If Mean Sharpe < 1.15:**
- More aggressive pruning needed
- Drop gap (4) + day-of-week (6) features
- Focus on proven high-signal features only

### Decision Point 2: Hyperparameter Tuning
**After `tune_hyperparameters.py`**

✅ **If Test Sharpe > Val Sharpe (within 10%):**
- Parameters generalize well, apply to full training

❌ **If Test Sharpe << Val Sharpe (>20% drop):**
- Overfitting detected, need regularization
- Increase reg_alpha/reg_lambda
- Reduce max_depth or n_estimators

## Rollback Plan

If results get worse at any stage:

```bash
# Revert feature_engineering.py changes
cd ml_models
git checkout feature_engineering.py

# Or manually comment out FEATURES_TO_DROP section (lines 360-395)

# Re-run with baseline features
python run_pipeline.py --test
```

## Hyperparameter Tuning Details

### Parameters to Tune
```python
learning_rate:     0.01 - 0.3   (log scale)
max_depth:         3 - 10
n_estimators:      100 - 500    (step 50)
subsample:         0.6 - 1.0
colsample_bytree:  0.6 - 1.0
gamma:             0.0 - 1.0
min_child_weight:  1 - 7
reg_alpha:         0.0 - 1.0
reg_lambda:        0.0 - 1.0
```

### Tuning Strategy
1. **Test on 3 stocks first** (AAPL, NVDA, JPM)
2. **Focus on 5d and 10d horizons** (best performing)
3. **50 trials per symbol/horizon** (~2-3 hours total)
4. **Use best params for full 25-stock training**

### Expected Improvements
- Baseline XGBoost: Sharpe 1.38 (with default params)
- Tuned XGBoost: Sharpe 1.50-1.60 (+8-15% improvement)

## Monitoring Output

### During Training
```
Engineering features for NVDA...
  Dropped 46 low-importance features for noise reduction
  Created 154 features (after dropping 46)
  
Features: 125
```
✅ Confirms pruning is active

### After Backtest
```
BEST MODEL: XGBoost
   Sharpe Ratio: 2.84
   Total Return: 122.17%
   vs Buy-Hold:  +248.40%
```
🎯 Look for Sharpe > 1.5 consistently

### Analysis Output
```
PRUNED vs ENHANCED: +12.3% (1.230 vs 1.090)
PRUNED vs BASELINE: +7.8% (1.230 vs 1.141)

HYPOTHESIS VALIDATION
  ✓ CONFIRMED: Noise reduction successful!
  ✓ EXCEEDS BASELINE
```
✅ Validates approach

## Tips

1. **Wait for training to complete** before running analysis (75 results needed)
2. **Check terminal output** for any errors or warnings
3. **Save all results JSONs** for comparison and rollback
4. **Monitor RAM usage** during hyperparameter tuning (can be intensive)
5. **Use background training** for long runs (--test flag)

## Contact Points

If you need to stop/restart:
- Training: `Ctrl+C` in terminal
- Resume: `python run_pipeline.py --test` (cached data)
- Kill background: Find process and terminate

## Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Feature pruning | 20 min | ✅ DONE |
| 3-stock test | 10 min | ✅ DONE |
| Full 25-stock training | 40-60 min | ⏳ IN PROGRESS |
| Results analysis | 10 min | ⏳ PENDING |
| Hyperparameter tuning | 2-3 hours | ⏳ PENDING |
| Final tuned training | 60 min | ⏳ PENDING |
| **Total** | **4-5 hours** | - |

---

**Next Action:** Wait for 25-stock training to complete, then run `python analyze_pruned_results.py`
