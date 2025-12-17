"""
Analyze ML results for overfitting and model diagnostics.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_latest_results():
    """Load the most recent results file."""
    results_dir = Path("../results/ml_models")
    result_files = list(results_dir.glob("ml_pipeline_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError("No results files found")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file) as f:
        return json.load(f)


def analyze_overfitting(results):
    """Analyze train vs validation performance to detect overfitting."""
    
    print("="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)
    
    overfitting_data = []
    
    for result in results['individual_results']:
        symbol = result['symbol']
        horizon = result['horizon']
        
        for model_name, train_metrics in result['training_metrics'].items():
            # Find corresponding backtest metrics
            backtest = next(
                (b for b in result['backtest_metrics'] 
                 if b['model'] == train_metrics['model']),
                None
            )
            
            if backtest and backtest['sharpe_ratio'] != 0:
                # Compare train accuracy to test Sharpe (proxy for performance)
                train_acc = train_metrics['accuracy']
                train_roc = train_metrics['roc_auc']
                test_sharpe = backtest['sharpe_ratio']
                test_return = backtest['total_return']
                
                # Overfitting indicators:
                # 1. High train accuracy but poor test performance
                # 2. Large gap between train ROC-AUC and test Sharpe
                overfit_score = train_acc - max(0, test_sharpe)  # Rough proxy
                
                overfitting_data.append({
                    'symbol': symbol,
                    'horizon': horizon,
                    'model': train_metrics['model'],
                    'train_accuracy': train_acc,
                    'train_roc_auc': train_roc,
                    'test_sharpe': test_sharpe,
                    'test_return': test_return,
                    'overfit_score': overfit_score
                })
    
    df = pd.DataFrame(overfitting_data)
    
    # Group by model
    print("\nTRAIN VS TEST PERFORMANCE BY MODEL:")
    print("-"*80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        print(f"\n{model}:")
        print(f"  Mean Train Accuracy:  {model_df['train_accuracy'].mean():.3f}")
        print(f"  Mean Train ROC-AUC:   {model_df['train_roc_auc'].mean():.3f}")
        print(f"  Mean Test Sharpe:     {model_df['test_sharpe'].mean():.3f}")
        print(f"  Mean Test Return:     {model_df['test_return'].mean():.3%}")
        print(f"  Overfit Score:        {model_df['overfit_score'].mean():.3f}")
        
        # Identify worst overfitters
        worst = model_df.nlargest(3, 'overfit_score')
        if not worst.empty:
            print(f"  Worst Overfitters:")
            for _, row in worst.iterrows():
                print(f"    {row['symbol']} ({row['horizon']}d): "
                      f"Train={row['train_accuracy']:.2f}, "
                      f"Test Sharpe={row['test_sharpe']:.2f}")
    
    # Overall overfitting check
    print("\n" + "="*80)
    print("OVERFITTING INDICATORS:")
    print("="*80)
    
    mean_train_acc = df['train_accuracy'].mean()
    mean_test_sharpe = df['test_sharpe'].mean()
    
    print(f"\nMean Train Accuracy: {mean_train_acc:.3f}")
    print(f"Mean Test Sharpe:    {mean_test_sharpe:.3f}")
    
    if mean_train_acc > 0.6 and mean_test_sharpe < 0.5:
        print("\n⚠️  HIGH OVERFITTING DETECTED")
        print("   - Models learn training data well but fail to generalize")
        print("   - Recommendations:")
        print("     1. Increase regularization (L1/L2)")
        print("     2. Reduce feature count (feature selection)")
        print("     3. Increase min_samples_split/leaf in tree models")
        print("     4. Add dropout or early stopping")
    elif mean_train_acc < 0.55:
        print("\n⚠️  UNDERFITTING DETECTED")
        print("   - Models can't learn training data patterns")
        print("   - Recommendations:")
        print("     1. Add more features or interactions")
        print("     2. Increase model complexity")
        print("     3. Try non-linear transformations")
    else:
        print("\n✓ Models appear reasonably balanced")
    
    return df


def analyze_buy_hold_comparison(results):
    """Compare ML models to buy-and-hold."""
    
    print("\n" + "="*80)
    print("BUY-AND-HOLD COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for result in results['individual_results']:
        symbol = result['symbol']
        horizon = result['horizon']
        category = result.get('category', 'unknown')
        
        # Buy-hold is a separate field in the result
        buy_hold = result.get('buy_hold')
        
        if buy_hold:
            bh_return = buy_hold.get('total_return', 0)
            bh_sharpe = buy_hold.get('sharpe_ratio', 0)
            
            for backtest in result['backtest_metrics']:
                if backtest['model'] != 'Buy-and-Hold':
                    comparison_data.append({
                        'symbol': symbol,
                        'category': category,
                        'horizon': horizon,
                        'model': backtest['model'],
                        'ml_return': backtest['total_return'],
                        'bh_return': bh_return,
                        'ml_sharpe': backtest['sharpe_ratio'],
                        'bh_sharpe': bh_sharpe,
                        'return_diff': backtest['total_return'] - bh_return,
                        'sharpe_diff': backtest['sharpe_ratio'] - bh_sharpe,
                        'beats_bh_return': backtest['total_return'] > bh_return,
                        'beats_bh_sharpe': backtest['sharpe_ratio'] > bh_sharpe
                    })
    
    df = pd.DataFrame(comparison_data)
    
    if df.empty:
        print("\n⚠️  No comparison data available")
        return df
    
    print("\nWIN RATE VS BUY-AND-HOLD:")
    print("-"*80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        win_rate_return = (model_df['beats_bh_return'].sum() / len(model_df)) * 100
        win_rate_sharpe = (model_df['beats_bh_sharpe'].sum() / len(model_df)) * 100
        
        print(f"\n{model}:")
        print(f"  Beats B&H (Return): {win_rate_return:.1f}% ({model_df['beats_bh_return'].sum()}/{len(model_df)} stocks)")
        print(f"  Beats B&H (Sharpe): {win_rate_sharpe:.1f}% ({model_df['beats_bh_sharpe'].sum()}/{len(model_df)} stocks)")
        print(f"  Mean Return Diff:   {model_df['return_diff'].mean():.2%}")
        print(f"  Mean Sharpe Diff:   {model_df['sharpe_diff'].mean():.2f}")
    
    # By category
    print("\n\nPERFORMANCE BY STOCK CATEGORY:")
    print("-"*80)
    
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        best_model = cat_df.groupby('model')['sharpe_diff'].mean().idxmax()
        
        print(f"\n{category.upper()}:")
        print(f"  Best Model: {best_model}")
        print(f"  Models that beat B&H:")
        
        for model in cat_df['model'].unique():
            model_cat = cat_df[cat_df['model'] == model]
            wins = model_cat['beats_bh_sharpe'].sum()
            total = len(model_cat)
            if wins > total / 2:
                print(f"    {model}: {wins}/{total} stocks")
    
    return df


def recommend_improvements():
    """Recommend specific improvements based on analysis."""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)
    
    print("""
1. INCREASE REGULARIZATION:
   - XGBoost: Increase reg_alpha (L1) and reg_lambda (L2)
   - Random Forest: Increase min_samples_leaf from 20 to 50
   - Logistic: Try stronger C penalty (reduce C from 1.0 to 0.1)

2. FEATURE SELECTION:
   - Use feature importance to keep only top 20-30 features
   - Remove correlated features (correlation > 0.9)
   - Try different feature combinations (technical only, volume only, etc.)

3. HYPERPARAMETER TUNING:
   - XGBoost: Reduce learning_rate to 0.05, increase n_estimators
   - Random Forest: Try max_features='sqrt' instead of 'auto'
   - Add early stopping with more patience

4. CROSS-VALIDATION:
   - Use TimeSeriesSplit with 5 folds
   - Validate on multiple time periods
   - Check consistency across CV folds

5. ENSEMBLE METHODS:
   - Stack XGBoost + Random Forest predictions
   - Use voting classifier with probability averaging
   - Try blending with buy-and-hold signal

6. ALTERNATIVE TARGETS:
   - Instead of binary up/down, try:
     * Top/bottom quintile classification (0/1/2/3/4)
     * Regression on actual returns (then threshold)
     * Predict probability and use Kelly criterion for sizing

7. REGIME-SPECIFIC MODELS:
   - Train separate models for high/low volatility regimes
   - Use HMM/Wasserstein regime as feature or filter
   - Only trade when regime confidence is high
""")


if __name__ == "__main__":
    print("Loading latest results...")
    results = load_latest_results()
    
    print(f"\nAnalyzing {len(results['individual_results'])} experiments...")
    print(f"Stocks: {len(results['config']['symbols'])}")
    print(f"Horizons: {results['config']['horizons']}")
    
    # Analyze overfitting
    overfit_df = analyze_overfitting(results)
    
    # Compare to buy-and-hold
    comparison_df = analyze_buy_hold_comparison(results)
    
    # Recommend improvements
    recommend_improvements()
    
    # Save analysis
    overfit_df.to_csv("../results/ml_models/overfitting_analysis.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv("../results/ml_models/buyhold_comparison.csv", index=False)
    
    print("\n" + "="*80)
    print("Analysis saved to results/ml_models/")
    print("  - overfitting_analysis.csv")
    if not comparison_df.empty:
        print("  - buyhold_comparison.csv")
    print("="*80)
