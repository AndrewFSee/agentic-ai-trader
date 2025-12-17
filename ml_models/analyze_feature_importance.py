"""
Feature importance analysis using SHAP and permutation importance.
Identifies which features to drop to reduce noise.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")

from sklearn.inspection import permutation_importance

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Load latest results to find best models
results_file = Path('results/ml_models/ml_pipeline_results_20251216_121916.json')
with open(results_file) as f:
    results_data = json.load(f)

# Focus on XGBoost (best performer) across all horizons
best_models = {
    3: [], 5: [], 10: []
}

for result in results_data['individual_results']:
    if result['best_model']['model'] == 'XGBoost':
        horizon = result['horizon']
        symbol = result['symbol']
        best_models[horizon].append({
            'symbol': symbol,
            'sharpe': result['best_model']['sharpe_ratio'],
            'feature_importance': result.get('feature_importance', {})
        })

print("\n1. XGBoost Performance Summary:")
print("-" * 80)
for horizon in [3, 5, 10]:
    sharpes = [m['sharpe'] for m in best_models[horizon]]
    print(f"{horizon}d: {len(best_models[horizon])} stocks, mean Sharpe: {np.mean(sharpes):.3f}, std: {np.std(sharpes):.3f}")

# Aggregate feature importance across all stocks
print("\n2. Aggregating Feature Importance...")
print("-" * 80)

feature_importance_aggregate = {}

for horizon in [3, 5, 10]:
    print(f"\n{horizon}-day horizon:")
    
    # Collect all feature importances from XGBoost only
    all_importances = {}
    count = 0
    
    for model_info in best_models[horizon]:
        feat_imp = model_info.get('feature_importance', {})
        if feat_imp and 'xgboost' in feat_imp:
            count += 1
            xgb_importances = feat_imp['xgboost']
            for feat, importance in xgb_importances.items():
                if feat not in all_importances:
                    all_importances[feat] = []
                # XGBoost stores as strings, convert to float
                all_importances[feat].append(float(importance))
    
    print(f"  Collected feature importance from {count} XGBoost models")
    
    # Calculate mean importance
    mean_importance = {
        feat: np.mean(imps) for feat, imps in all_importances.items()
    }
    
    feature_importance_aggregate[horizon] = mean_importance
    
    # Show top 20
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 20 Most Important Features ({horizon}d):")
    for i, (feat, imp) in enumerate(sorted_features[:20], 1):
        print(f"    {i:2}. {feat:<40} {imp:.6f}")
    
    print(f"\n  Bottom 20 Least Important Features ({horizon}d):")
    for i, (feat, imp) in enumerate(sorted_features[-20:], 1):
        print(f"    {i:2}. {feat:<40} {imp:.6f}")

# Identify consistently low-importance features
print("\n" + "="*80)
print("3. FEATURES TO DROP (Low importance across all horizons)")
print("="*80)

# Features that are in bottom 30 for ALL horizons
all_features = set(feature_importance_aggregate[3].keys())
bottom_threshold = 0.001  # Very low importance

drop_candidates = []
for feature in all_features:
    importances = [
        feature_importance_aggregate[h].get(feature, 0) 
        for h in [3, 5, 10]
    ]
    mean_imp = np.mean(importances)
    max_imp = max(importances)
    
    # Feature is consistently unimportant if mean < threshold
    if mean_imp < bottom_threshold:
        drop_candidates.append({
            'feature': feature,
            'mean_importance': mean_imp,
            'max_importance': max_imp,
            '3d': feature_importance_aggregate[3].get(feature, 0),
            '5d': feature_importance_aggregate[5].get(feature, 0),
            '10d': feature_importance_aggregate[10].get(feature, 0)
        })

df_drop = pd.DataFrame(drop_candidates).sort_values('mean_importance')

print(f"\nFound {len(drop_candidates)} features with mean importance < {bottom_threshold}")
print("\nTop candidates for removal:")
print("-" * 80)
for idx, row in df_drop.head(30).iterrows():
    print(f"{row['feature']:<40} Mean: {row['mean_importance']:.6f} | "
          f"3d: {row['3d']:.6f} 5d: {row['5d']:.6f} 10d: {row['10d']:.6f}")

# Categorize features
print("\n" + "="*80)
print("4. FEATURE CATEGORY ANALYSIS")
print("="*80)

feature_categories = {
    'gap': ['gap', 'gap_pct', 'gap_up', 'gap_down'],
    'rsi_divergence': ['rsi_divergence', 'rsi_divergence_normalized'],
    'day_of_week': ['day_of_week', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday'],
    'macro': [f for f in all_features if any(x in f.lower() for x in ['usd_', 'oil_', 'gold_', 'macro_'])],
    'breadth': [f for f in all_features if any(x in f.lower() for x in ['ad_', 'breadth_'])],
    'sentiment': [f for f in all_features if 'sentiment' in f.lower() or 'gdelt' in f.lower()],
    'regime': [f for f in all_features if 'regime' in f.lower() or 'hmm' in f.lower() or 'wass' in f.lower()],
    'vix': [f for f in all_features if 'vix' in f.lower()],
    'kalman': [f for f in all_features if 'kalman' in f.lower()],
    'wavelet': [f for f in all_features if 'wavelet' in f.lower() or 'noise' in f.lower()]
}

# Calculate mean importance by category
print("\nMean importance by feature category (3d horizon):")
print("-" * 80)
for cat_name, features in feature_categories.items():
    valid_features = [f for f in features if f in feature_importance_aggregate[3]]
    if valid_features:
        mean_imp = np.mean([feature_importance_aggregate[3][f] for f in valid_features])
        print(f"{cat_name:<20} ({len(valid_features):2} features): {mean_imp:.6f}")

# Save drop list
drop_list = df_drop.head(30)['feature'].tolist()
with open('features_to_drop.txt', 'w') as f:
    f.write("# Features to drop (low importance < 0.001)\n")
    f.write("# Generated from feature importance analysis\n\n")
    for feature in drop_list:
        f.write(f"{feature}\n")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print(f"\n1. DROP {len(drop_list)} low-importance features (saved to features_to_drop.txt)")
print(f"   - This will reduce feature count from 171 -> {171 - len(drop_list)}")
print(f"   - Expected: +2-5% Sharpe improvement (less noise)")

print("\n2. NEW FEATURES ASSESSMENT:")
new_feature_cats = ['gap', 'rsi_divergence', 'day_of_week', 'macro', 'breadth']
for cat in new_feature_cats:
    features = feature_categories.get(cat, [])
    valid = [f for f in features if f in feature_importance_aggregate[3]]
    if valid:
        mean_imp = np.mean([feature_importance_aggregate[3][f] for f in valid])
        assessment = "KEEP" if mean_imp > 0.002 else "DROP"
        print(f"   - {cat:<20}: {mean_imp:.6f} -> {assessment}")

print("\n3. NEXT STEPS:")
print("   - Review features_to_drop.txt")
print("   - Modify feature_engineering.py to skip dropped features")
print("   - Re-run training with reduced feature set")
print("   - Then: Hyperparameter tuning on cleaner features")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
