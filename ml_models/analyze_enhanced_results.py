"""
Analysis of enhanced features results (196 features vs 167 baseline).
Compares performance and prepares for feature importance + hyperparameter tuning.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load results
baseline_file = Path('results/ml_models/ml_pipeline_results_20251216_101007.json')  # 167 features
enhanced_file = Path('results/ml_models/ml_pipeline_results_20251216_121916.json')  # 196 features

print("="*80)
print("ENHANCED FEATURES ANALYSIS")
print("="*80)
print("\nBaseline: 167 features (VIX, Kalman, Wavelets)")
print("Enhanced: 196 features (+29: gap, RSI div, day-of-week, macro, breadth)")
print("="*80)

# Load data
with open(baseline_file) as f:
    baseline_data = json.load(f)

with open(enhanced_file) as f:
    enhanced_data = json.load(f)

# Extract aggregated results
def get_aggregate_stats(data):
    """Extract aggregated performance by model and horizon."""
    stats = {}
    
    for result in data['individual_results']:
        horizon = result['horizon']
        if horizon not in stats:
            stats[horizon] = {
                'XGBoost': [], 'Random Forest': [], 
                'Logistic Regression': [], 'Decision Tree': [],
                'Buy-Hold': []
            }
        
        # Collect Sharpe ratios - backtest_metrics is a list of dicts
        for backtest in result['backtest_metrics']:
            model_name = backtest['model']
            if model_name in stats[horizon]:
                sharpe = backtest['sharpe_ratio']
                stats[horizon][model_name].append(sharpe)
        
        stats[horizon]['Buy-Hold'].append(result['buy_hold']['sharpe_ratio'])
    
    # Calculate means
    aggregated = {}
    for horizon in stats:
        aggregated[horizon] = {
            model: np.mean(sharpes) for model, sharpes in stats[horizon].items()
        }
    
    return aggregated

baseline_stats = get_aggregate_stats(baseline_data)
enhanced_stats = get_aggregate_stats(enhanced_data)

print("\n" + "="*80)
print("SHARPE RATIO COMPARISON")
print("="*80)

for horizon in [3, 5, 10]:
    print(f"\n{horizon}-Day Prediction Horizon:")
    print("-" * 80)
    print(f"{'Model':<25} {'Baseline':<12} {'Enhanced':<12} {'Change':<12} {'% Change'}")
    print("-" * 80)
    
    for model in ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree']:
        base_sharpe = baseline_stats[horizon][model]
        enh_sharpe = enhanced_stats[horizon][model]
        change = enh_sharpe - base_sharpe
        pct_change = (change / base_sharpe * 100) if base_sharpe != 0 else 0
        
        symbol = "+" if change > 0 else ""
        print(f"{model:<25} {base_sharpe:>11.3f} {enh_sharpe:>11.3f} {symbol}{change:>10.3f} {symbol}{pct_change:>9.1f}%")
    
    # Buy-and-hold
    base_bh = baseline_stats[horizon]['Buy-Hold']
    enh_bh = enhanced_stats[horizon]['Buy-Hold']
    print("-" * 80)
    print(f"{'Buy-and-Hold':<25} {base_bh:>11.3f} {enh_bh:>11.3f} {'-':>11} {'-':>10}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Calculate overall improvements
improvements = []
for horizon in [3, 5, 10]:
    for model in ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree']:
        base = baseline_stats[horizon][model]
        enh = enhanced_stats[horizon][model]
        pct_change = ((enh - base) / base * 100) if base != 0 else 0
        improvements.append({
            'horizon': horizon,
            'model': model,
            'baseline': base,
            'enhanced': enh,
            'change': enh - base,
            'pct_change': pct_change
        })

df_improvements = pd.DataFrame(improvements)

print(f"\n1. OVERALL PERFORMANCE:")
print(f"   - Mean Sharpe (baseline): {df_improvements['baseline'].mean():.3f}")
print(f"   - Mean Sharpe (enhanced): {df_improvements['enhanced'].mean():.3f}")
print(f"   - Average improvement: {df_improvements['change'].mean():.3f} ({df_improvements['pct_change'].mean():.1f}%)")

print(f"\n2. BEST IMPROVEMENTS:")
top_5 = df_improvements.nlargest(5, 'pct_change')
for idx, row in top_5.iterrows():
    print(f"   - {row['model']:<20} {row['horizon']:>2}d: {row['baseline']:.3f} -> {row['enhanced']:.3f} (+{row['pct_change']:.1f}%)")

print(f"\n3. WORST CHANGES:")
bottom_5 = df_improvements.nsmallest(5, 'pct_change')
for idx, row in bottom_5.iterrows():
    print(f"   - {row['model']:<20} {row['horizon']:>2}d: {row['baseline']:.3f} -> {row['enhanced']:.3f} ({row['pct_change']:.1f}%)")

print(f"\n4. CONSISTENCY:")
positive_changes = (df_improvements['change'] > 0).sum()
negative_changes = (df_improvements['change'] < 0).sum()
print(f"   - Improvements: {positive_changes}/{len(df_improvements)} ({positive_changes/len(df_improvements)*100:.1f}%)")
print(f"   - Degradations: {negative_changes}/{len(df_improvements)} ({negative_changes/len(df_improvements)*100:.1f}%)")

# Best overall model
print(f"\n5. BEST MODELS (ENHANCED):")
for horizon in [3, 5, 10]:
    best_model = max(enhanced_stats[horizon].items(), 
                     key=lambda x: x[1] if x[0] != 'Buy-Hold' else -np.inf)
    bh_sharpe = enhanced_stats[horizon]['Buy-Hold']
    improvement_vs_bh = (best_model[1] / bh_sharpe - 1) * 100
    print(f"   - {horizon:>2}d: {best_model[0]:<20} Sharpe {best_model[1]:.3f} (+{improvement_vs_bh:.0f}% vs buy-hold)")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR NEXT STEPS")
print("="*80)

print("\n1. FEATURE IMPORTANCE ANALYSIS:")
print("   - Run SHAP or permutation importance on best models")
print("   - Identify top 20 most predictive features")
print("   - Identify bottom 20 least useful features")
print("   - Target: Drop 20-30 low-value features to reduce noise")

print("\n2. HYPERPARAMETER TUNING:")
print("   - Focus on XGBoost (best performer)")
print("   - Use Optuna with time-series cross-validation")
print("   - Tune: learning_rate, max_depth, n_estimators, subsample")
print("   - Expected gain: +5-10% Sharpe improvement")

print("\n3. FEATURE CATEGORIES TO ANALYZE:")
baseline_features = [
    "Technical (26)", "Sentiment (8)", "Regime (13)", "Fundamental (21)",
    "Volatility (13)", "VIX (10)", "Kalman (6)", "Wavelet (9)", "Other (61)"
]
new_features = [
    "Gap statistics (4)", "RSI divergence (2)", "Day-of-week (6)",
    "Macro correlations (16)", "Market breadth (8)"
]
print("   Baseline categories:")
for cat in baseline_features:
    print(f"     - {cat}")
print("   New categories:")
for cat in new_features:
    print(f"     - {cat}")

print("\n4. EXPECTED OUTCOMES:")
print("   - Remove 20-30 noisy features -> +2-5% Sharpe")
print("   - Hyperparameter tuning -> +5-10% Sharpe")
print("   - Combined target: Sharpe 1.50-1.70 (from current 1.28-1.54)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
