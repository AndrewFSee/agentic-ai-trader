"""
Analyze pruned features results vs enhanced and baseline.
Validates noise reduction hypothesis.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_results(json_path):
    """Load results JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_aggregate_stats(data):
    """Extract aggregate Sharpe ratios by model and horizon."""
    stats = {3: {}, 5: {}, 10: {}}
    
    for result in data['individual_results']:
        horizon = result['horizon']
        
        # Iterate through backtest_metrics list
        for backtest in result['backtest_metrics']:
            model_name = backtest['model']
            sharpe = backtest['sharpe_ratio']
            
            if model_name not in stats[horizon]:
                stats[horizon][model_name] = []
            stats[horizon][model_name].append(sharpe)
    
    # Calculate means
    aggregated = {
        horizon: {
            model: np.mean(sharpes) 
            for model, sharpes in stats[horizon].items()
        }
        for horizon in stats
    }
    
    return aggregated

def print_comparison(baseline_stats, enhanced_stats, pruned_stats):
    """Print three-way comparison: baseline vs enhanced vs pruned."""
    print("\n" + "="*80)
    print("THREE-WAY SHARPE RATIO COMPARISON")
    print("="*80)
    print("Format: Model | Baseline (167 feat) | Enhanced (196 feat) | Pruned (141 feat)")
    print("-"*80)
    
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
    horizons = [3, 5, 10]
    
    for horizon in horizons:
        print(f"\n{horizon}-Day Horizon:")
        print("-"*80)
        
        for model in models:
            baseline_sharpe = baseline_stats[horizon].get(model, 0)
            enhanced_sharpe = enhanced_stats[horizon].get(model, 0)
            pruned_sharpe = pruned_stats[horizon].get(model, 0)
            
            # Calculate changes
            enhanced_change = ((enhanced_sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
            pruned_vs_baseline = ((pruned_sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
            pruned_vs_enhanced = ((pruned_sharpe - enhanced_sharpe) / enhanced_sharpe * 100) if enhanced_sharpe > 0 else 0
            
            print(f"{model:20s} | {baseline_sharpe:5.3f} | {enhanced_sharpe:5.3f} ({enhanced_change:+5.1f}%) | "
                  f"{pruned_sharpe:5.3f} ({pruned_vs_baseline:+5.1f}% vs base, {pruned_vs_enhanced:+5.1f}% vs enh)")

def analyze_feature_impact():
    """Main analysis comparing all three versions."""
    print("="*80)
    print("FEATURE PRUNING IMPACT ANALYSIS")
    print("="*80)
    print("\nComparing three versions:")
    print("  1. BASELINE (167 features) - VIX, Kalman, Wavelets")
    print("  2. ENHANCED (196 features) - Added gap, RSI div, day-of-week, macro, breadth")
    print("  3. PRUNED   (141 features) - Dropped 30 zero-importance features (all macro/breadth)")
    print()
    
    # Find latest results files
    results_dir = Path('results/ml_models')
    
    # Load baseline (from earlier training)
    baseline_files = sorted(results_dir.glob('ml_pipeline_results_*_baseline.json'))
    if not baseline_files:
        # Try to find by date pattern (assuming Dec 15 or earlier)
        baseline_files = [f for f in sorted(results_dir.glob('ml_pipeline_results_*.json')) 
                         if '20251215' in f.name or '20251214' in f.name]
    
    # Load enhanced (from yesterday's training with 196 features)
    enhanced_files = [f for f in sorted(results_dir.glob('ml_pipeline_results_*.json')) 
                     if '20251216_121916' in f.name]  # Known enhanced results
    
    # Load pruned (today's training)
    all_files = sorted(results_dir.glob('ml_pipeline_results_*.json'))
    pruned_files = [f for f in all_files if f not in baseline_files and f not in enhanced_files]
    pruned_files = [f for f in pruned_files if datetime.strptime(f.stem.split('_')[-2] + f.stem.split('_')[-1], 
                                                                  '%Y%m%d%H%M%S') > datetime(2025, 12, 16, 12, 30)]
    
    if not baseline_files:
        print("ERROR: Could not find baseline results")
        return
    if not enhanced_files:
        print("ERROR: Could not find enhanced results (20251216_121916)")
        return
    if not pruned_files:
        print("ERROR: Could not find pruned results (training may still be running)")
        return
    
    baseline_path = baseline_files[0]
    enhanced_path = enhanced_files[0]
    pruned_path = pruned_files[-1]  # Latest
    
    print(f"Loading results...")
    print(f"  Baseline: {baseline_path.name}")
    print(f"  Enhanced: {enhanced_path.name}")
    print(f"  Pruned:   {pruned_path.name}")
    
    baseline_data = load_results(baseline_path)
    enhanced_data = load_results(enhanced_path)
    pruned_data = load_results(pruned_path)
    
    # Get aggregate stats
    baseline_stats = get_aggregate_stats(baseline_data)
    enhanced_stats = get_aggregate_stats(enhanced_data)
    pruned_stats = get_aggregate_stats(pruned_data)
    
    # Print comparison
    print_comparison(baseline_stats, enhanced_stats, pruned_stats)
    
    # Calculate overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    def calc_mean_sharpe(stats):
        all_sharpes = []
        for horizon_stats in stats.values():
            all_sharpes.extend(horizon_stats.values())
        return np.mean(all_sharpes)
    
    baseline_mean = calc_mean_sharpe(baseline_stats)
    enhanced_mean = calc_mean_sharpe(enhanced_stats)
    pruned_mean = calc_mean_sharpe(pruned_stats)
    
    enhanced_change = ((enhanced_mean - baseline_mean) / baseline_mean * 100)
    pruned_vs_baseline = ((pruned_mean - baseline_mean) / baseline_mean * 100)
    pruned_vs_enhanced = ((pruned_mean - enhanced_mean) / enhanced_mean * 100)
    
    print(f"\nMean Sharpe across all models/horizons:")
    print(f"  Baseline (167 feat):  {baseline_mean:.4f}")
    print(f"  Enhanced (196 feat):  {enhanced_mean:.4f} ({enhanced_change:+.1f}% vs baseline)")
    print(f"  Pruned   (141 feat):  {pruned_mean:.4f} ({pruned_vs_baseline:+.1f}% vs baseline, {pruned_vs_enhanced:+.1f}% vs enhanced)")
    print()
    
    # Hypothesis validation
    print("="*80)
    print("HYPOTHESIS VALIDATION")
    print("="*80)
    print("\nH0: Adding macro/breadth features degraded performance due to noise")
    print(f"  Enhanced vs Baseline: {enhanced_change:+.1f}% {'✓ CONFIRMED DEGRADATION' if enhanced_change < 0 else '✗ No degradation'}")
    print()
    print("H1: Removing zero-importance features improves performance")
    if pruned_vs_enhanced > 0:
        print(f"  Pruned vs Enhanced: {pruned_vs_enhanced:+.1f}% ✓ CONFIRMED IMPROVEMENT")
        if pruned_mean > baseline_mean:
            print(f"  Pruned vs Baseline: {pruned_vs_baseline:+.1f}% ✓ EXCEEDS BASELINE")
            print("\n  CONCLUSION: Noise reduction successful! Pruned features beat both versions.")
        else:
            print(f"  Pruned vs Baseline: {pruned_vs_baseline:+.1f}% ✓ Improvement but below baseline")
            print("\n  CONCLUSION: Noise reduction worked, but new features (gap, RSI div) need refinement.")
    else:
        print(f"  Pruned vs Enhanced: {pruned_vs_enhanced:+.1f}% ✗ No improvement")
        print("\n  CONCLUSION: Pruning alone insufficient. May need hyperparameter tuning.")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if pruned_mean > enhanced_mean:
        print("✓ Pruning successful")
        print("→ Proceed with hyperparameter tuning on pruned features")
        print("→ Focus on XGBoost: learning_rate, max_depth, n_estimators")
        print("→ Target: Sharpe 1.35-1.45 with optimized hyperparameters")
    else:
        print("✗ Pruning insufficient")
        print("→ Investigate why gap/RSI features underperforming")
        print("→ Consider dropping day-of-week features too")
        print("→ May need more aggressive feature selection")
    
    print("="*80)

if __name__ == "__main__":
    analyze_feature_impact()
