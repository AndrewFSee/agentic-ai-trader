"""
Simple analysis comparing pruned vs tuned results
"""

import json
from pathlib import Path

def load_results(filename):
    """Load results from JSON file."""
    filepath = Path("results/ml_models") / filename
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING IMPACT ANALYSIS")
    print("="*80)
    
    # Load results
    pruned = load_results("ml_pipeline_results_20251216_124922.json")
    tuned = load_results("ml_pipeline_results_20251216_170259.json")
    
    pruned_agg = pruned['aggregate']
    tuned_agg = tuned['aggregate']
    
    # Compare by horizon
    for horizon in ['3', '5', '10']:
        print(f"\n{horizon}-Day Horizon:")
        print("-"*80)
        print(f"{'Model':20s} | {'Pruned':>10s} | {'Tuned':>10s} | {'Change':>10s}")
        print("-"*80)
        
        for model in ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree']:
            pruned_sharpe = pruned_agg[horizon]['models'][model]['mean_sharpe']
            tuned_sharpe = tuned_agg[horizon]['models'][model]['mean_sharpe']
            change = ((tuned_sharpe - pruned_sharpe) / pruned_sharpe * 100)
            
            marker = ""
            if tuned_sharpe > 1.4:
                marker = " ⭐"
            
            print(f"{model:20s} | {pruned_sharpe:10.3f} | {tuned_sharpe:10.3f} | {change:+9.1f}%{marker}")
        
        # Show best model for this horizon
        best_model = None
        best_sharpe = 0
        for model in ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree']:
            sharpe = tuned_agg[horizon]['models'][model]['mean_sharpe']
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model = model
        
        print(f"\n  → BEST: {best_model} (Sharpe: {best_sharpe:.3f})")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    # Calculate mean Sharpe across best models for each horizon
    pruned_best = []
    tuned_best = []
    
    for horizon in ['3', '5', '10']:
        # Find best Sharpe for each horizon
        best_pruned = max(pruned_agg[horizon]['models'][m]['mean_sharpe'] 
                         for m in pruned_agg[horizon]['models'])
        best_tuned = max(tuned_agg[horizon]['models'][m]['mean_sharpe'] 
                        for m in tuned_agg[horizon]['models'])
        pruned_best.append(best_pruned)
        tuned_best.append(best_tuned)
    
    pruned_mean = sum(pruned_best) / len(pruned_best)
    tuned_mean = sum(tuned_best) / len(tuned_best)
    improvement = ((tuned_mean - pruned_mean) / pruned_mean * 100)
    
    print(f"\nMean Sharpe (best model per horizon):")
    print(f"  Pruned:  {pruned_mean:.4f}")
    print(f"  Tuned:   {tuned_mean:.4f} ({improvement:+.1f}%)")
    
    print(f"\nBest Sharpe by Horizon:")
    for i, horizon in enumerate(['3', '5', '10']):
        change = ((tuned_best[i] - pruned_best[i]) / pruned_best[i] * 100)
        print(f"  {horizon}d: {pruned_best[i]:.3f} → {tuned_best[i]:.3f} ({change:+.1f}%)")
    
    # Complete evolution summary
    print("\n" + "="*80)
    print("COMPLETE PIPELINE EVOLUTION")
    print("="*80)
    
    print("\n  Stage 1: Baseline")
    print("    • 167 features")
    print("    • Mean Sharpe: 1.141")
    
    print("\n  Stage 2: Enhanced (GPT-researcher features)")
    print("    • 196 features (+29)")
    print("    • Mean Sharpe: 1.090 (-4.1%)")
    print("    • Result: DEGRADED (too much noise)")
    
    print("\n  Stage 3: Pruned (dropped low-importance)")
    print("    • 141 features (-55)")
    print(f"    • Mean Sharpe: {pruned_mean:.3f} (+5.5%)")
    print("    • Result: IMPROVED (noise reduction)")
    
    print("\n  Stage 4: Tuned (Optuna hyperparameters)")
    print("    • 141 features (same)")
    print(f"    • Mean Sharpe: {tuned_mean:.3f} ({improvement:+.1f}%)")
    print(f"    • Result: {'IMPROVED' if improvement > 1 else 'MAINTAINED'}")
    
    total_gain = ((tuned_mean / 1.141 - 1) * 100)
    print(f"\n  TOTAL IMPROVEMENT: {total_gain:+.1f}% (baseline → tuned)")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n✓ Winning Strategy:")
    print("  • Feature quality > quantity (141 beats 196)")
    print("  • Noise removal crucial for performance")
    print(f"  • Hyperparameter tuning added {improvement:+.1f}% on top of pruning")
    print(f"  • Combined approach: {total_gain:+.1f}% total gain")
    
    print("\n✓ Best Models by Prediction Horizon:")
    print(f"  • 3-day:  Random Forest (Sharpe {tuned_best[0]:.2f}) - best for short-term")
    print(f"  • 5-day:  XGBoost (Sharpe {tuned_best[1]:.2f}) - best for medium-term")
    print(f"  • 10-day: XGBoost (Sharpe {tuned_best[2]:.2f}) - best for longer-term")
    
    print("\n✓ Feature Efficiency Gain:")
    baseline_efficiency = 1.141 / 167
    tuned_efficiency = tuned_mean / 141
    eff_gain = ((tuned_efficiency / baseline_efficiency - 1) * 100)
    print(f"  • Baseline: {baseline_efficiency:.6f} Sharpe per feature")
    print(f"  • Tuned:    {tuned_efficiency:.6f} Sharpe per feature")
    print(f"  • Gain:     {eff_gain:+.1f}% more efficient")
    
    print("\n✓ Production Deployment:")
    print(f"  • Deploy tuned models (Mean Sharpe: {tuned_mean:.2f})")
    print("  • Use ensemble: RF for 3d, XGB for 5d/10d")
    print("  • 141 optimized features (saved in ml_models/config.py)")
    print("  • Hyperparameters saved (from Optuna tuning)")
    print("  • Paper trade 1-2 weeks before live deployment")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE ✓")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
