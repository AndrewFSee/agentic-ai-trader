#!/usr/bin/env python3
"""
Analyze Robust HMM Feature Test Results
========================================

Loads saved JSON results and produces detailed analysis:
- Feature performance by category
- Universal winners across categories
- Statistical significance testing
- Feature stability metrics
- Recommendations for production use

Usage:
    python analyze_robust_results.py hmm_robust_test_20251214_175443.json
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import argparse
from collections import defaultdict


def load_results(filename: str) -> Dict:
    """Load test results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_consistency_score(outperformances: List[float]) -> float:
    """
    Calculate consistency score (0-100) based on:
    - % positive results
    - Low standard deviation (stability)
    - High mean performance
    """
    if len(outperformances) < 2:
        return 0
    
    pct_positive = sum(1 for x in outperformances if x > 0) / len(outperformances)
    mean_perf = np.mean(outperformances)
    std_perf = np.std(outperformances)
    
    # Normalize components (0-1 scale)
    consistency_component = pct_positive
    stability_component = max(0, 1 - (std_perf / 100))  # Penalize high std
    performance_component = max(0, min(1, mean_perf / 50))  # Cap at 50% outperformance
    
    # Weighted average
    score = (0.4 * consistency_component + 
             0.3 * stability_component + 
             0.3 * performance_component) * 100
    
    return score


def analyze_by_category(data: Dict) -> pd.DataFrame:
    """Analyze feature performance within each category."""
    
    category_agg = data['category_aggregates']
    all_analyses = []
    
    for category, features_dict in category_agg.items():
        category_features = []
        
        for feature_str, results in features_dict.items():
            features = feature_str.split('|')
            outperfs = [r['outperformance'] for r in results]
            sharpes = [r['sharpe'] for r in results]
            returns = [r['return'] for r in results]
            
            if len(outperfs) < 2:
                continue
            
            # Statistical tests
            t_stat, p_value = stats.ttest_1samp(outperfs, 0)  # Test if mean > 0
            
            category_features.append({
                'category': category,
                'features': features,
                'feature_count': len(features),
                'n_stocks': len(results),
                'mean_outperformance': np.mean(outperfs),
                'median_outperformance': np.median(outperfs),
                'std_outperformance': np.std(outperfs),
                'min_outperformance': np.min(outperfs),
                'max_outperformance': np.max(outperfs),
                'pct_positive': sum(1 for x in outperfs if x > 0) / len(outperfs) * 100,
                'mean_sharpe': np.mean(sharpes),
                'mean_return': np.mean(returns),
                'consistency_score': calculate_consistency_score(outperfs),
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        all_analyses.extend(category_features)
    
    return pd.DataFrame(all_analyses)


def find_universal_features(data: Dict, min_categories: int = 3, 
                            min_consistency: float = 60) -> pd.DataFrame:
    """Find features that work well across multiple categories."""
    
    feature_performance = defaultdict(lambda: {
        'categories': [],
        'outperformances': [],
        'consistency_scores': [],
        'stocks': []
    })
    
    category_agg = data['category_aggregates']
    
    for category, features_dict in category_agg.items():
        for feature_str, results in features_dict.items():
            features = tuple(sorted(feature_str.split('|')))
            outperfs = [r['outperformance'] for r in results]
            mean_outperf = np.mean(outperfs)
            consistency = calculate_consistency_score(outperfs)
            
            # Only count if mean outperformance > 0 AND reasonable consistency
            if mean_outperf > 0 and consistency >= min_consistency:
                feature_performance[features]['categories'].append(category)
                feature_performance[features]['outperformances'].append(mean_outperf)
                feature_performance[features]['consistency_scores'].append(consistency)
                feature_performance[features]['stocks'].extend([r['stock'] for r in results])
    
    # Build results
    universal = []
    for features, perf in feature_performance.items():
        if len(perf['categories']) >= min_categories:
            universal.append({
                'features': list(features),
                'feature_count': len(features),
                'n_categories': len(perf['categories']),
                'categories': ', '.join(sorted(perf['categories'])),
                'n_stocks': len(perf['stocks']),
                'avg_outperformance': np.mean(perf['outperformances']),
                'min_outperformance': np.min(perf['outperformances']),
                'max_outperformance': np.max(perf['outperformances']),
                'avg_consistency': np.mean(perf['consistency_scores']),
                'overall_score': np.mean(perf['outperformances']) * np.mean(perf['consistency_scores']) / 100
            })
    
    df = pd.DataFrame(universal)
    if len(df) > 0:
        df = df.sort_values('overall_score', ascending=False)
    return df


def find_best_by_feature_count(data: Dict) -> pd.DataFrame:
    """Find best performing feature sets by number of features."""
    
    df = analyze_by_category(data)
    
    results = []
    for n_features in sorted(df['feature_count'].unique()):
        subset = df[df['feature_count'] == n_features]
        if len(subset) == 0:
            continue
        
        # Get top performer
        best = subset.nlargest(1, 'consistency_score').iloc[0]
        
        results.append({
            'n_features': n_features,
            'features': best['features'],
            'category': best['category'],
            'mean_outperformance': best['mean_outperformance'],
            'consistency_score': best['consistency_score'],
            'pct_positive': best['pct_positive'],
            'n_stocks': best['n_stocks']
        })
    
    return pd.DataFrame(results)


def generate_recommendations(data: Dict) -> Dict[str, List[str]]:
    """Generate feature recommendations for each category."""
    
    df = analyze_by_category(data)
    recommendations = {}
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        
        # Filter: positive mean, good consistency, significant
        good_features = cat_data[
            (cat_data['mean_outperformance'] > 0) &
            (cat_data['consistency_score'] > 60) &
            (cat_data['pct_positive'] >= 50)
        ]
        
        if len(good_features) > 0:
            # Sort by overall performance
            good_features = good_features.copy()
            good_features['score'] = (good_features['mean_outperformance'] * 
                                     good_features['consistency_score'] / 100)
            good_features = good_features.sort_values('score', ascending=False)
            
            top_3 = good_features.head(3)
            recommendations[category] = [
                {
                    'features': row['features'],
                    'mean_outperformance': row['mean_outperformance'],
                    'consistency_score': row['consistency_score'],
                    'pct_positive': row['pct_positive']
                }
                for _, row in top_3.iterrows()
            ]
    
    return recommendations


def print_summary(data: Dict):
    """Print comprehensive summary of results."""
    
    print("\n" + "="*80)
    print("ROBUST HMM FEATURE TESTING - ANALYSIS SUMMARY")
    print("="*80)
    
    # Basic stats
    total_tests = len(data['results'])
    categories = set(r['category'] for r in data['results'])
    stocks = set(r['stock'] for r in data['results'])
    
    print(f"\nTest Overview:")
    print(f"  Total tests: {total_tests}")
    print(f"  Categories: {len(categories)}")
    print(f"  Unique stocks: {len(stocks)}")
    print(f"  Lookback period: {data['lookback_days']} days")
    
    # Category analysis
    print(f"\n{'='*80}")
    print("BEST FEATURES BY CATEGORY")
    print(f"{'='*80}\n")
    
    df = analyze_by_category(data)
    for category in sorted(df['category'].unique()):
        cat_data = df[df['category'] == category]
        top_3 = cat_data.nlargest(3, 'consistency_score')
        
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"{'Rank':<6}{'Features':<35}{'Mean%':>8}{'Cons':>8}{'%Win':>8}{'Sharpe':>8}")
        print("-" * 73)
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            feat_str = ', '.join(row['features'][:3])
            if len(row['features']) > 3:
                feat_str += f" +{len(row['features'])-3}"
            
            print(f"{i:<6}{feat_str:<35}"
                  f"{row['mean_outperformance']:>7.1f}%"
                  f"{row['consistency_score']:>7.0f}"
                  f"{row['pct_positive']:>7.0f}%"
                  f"{row['mean_sharpe']:>8.2f}")
    
    # Universal features
    print(f"\n{'='*80}")
    print("UNIVERSAL FEATURES (work across multiple categories)")
    print(f"{'='*80}\n")
    
    universal_df = find_universal_features(data, min_categories=3, min_consistency=60)
    if len(universal_df) > 0:
        print(f"{'Features':<35}{'#Cats':>8}{'#Stocks':>10}{'AvgOut%':>10}{'Consistency':>12}{'Score':>10}")
        print("-" * 85)
        
        for _, row in universal_df.head(10).iterrows():
            feat_str = ', '.join(row['features'][:3])
            if len(row['features']) > 3:
                feat_str += f" +{len(row['features'])-3}"
            
            print(f"{feat_str:<35}"
                  f"{row['n_categories']:>8}"
                  f"{row['n_stocks']:>10}"
                  f"{row['avg_outperformance']:>9.1f}%"
                  f"{row['avg_consistency']:>11.0f}"
                  f"{row['overall_score']:>10.1f}")
    else:
        print("No features performed consistently across 3+ categories")
    
    # Feature complexity analysis
    print(f"\n{'='*80}")
    print("BEST FEATURES BY COMPLEXITY")
    print(f"{'='*80}\n")
    
    complexity_df = find_best_by_feature_count(data)
    print(f"{'#Features':>10}{'Features':<35}{'Category':<20}{'Mean%':>10}{'Consistency':>12}")
    print("-" * 87)
    
    for _, row in complexity_df.iterrows():
        feat_str = ', '.join(row['features'][:3])
        if len(row['features']) > 3:
            feat_str += f" +{len(row['features'])-3}"
        
        print(f"{row['n_features']:>10}"
              f"{feat_str:<35}"
              f"{row['category']:<20}"
              f"{row['mean_outperformance']:>9.1f}%"
              f"{row['consistency_score']:>11.0f}")
    
    # Production recommendations
    print(f"\n{'='*80}")
    print("PRODUCTION RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    recommendations = generate_recommendations(data)
    for category, recs in sorted(recommendations.items()):
        if len(recs) > 0:
            print(f"\n{category.upper().replace('_', ' ')}:")
            best = recs[0]
            print(f"  Primary: {', '.join(best['features'])}")
            print(f"    → {best['mean_outperformance']:.1f}% outperformance, "
                  f"{best['consistency_score']:.0f} consistency, "
                  f"{best['pct_positive']:.0f}% win rate")
            
            if len(recs) > 1:
                alt = recs[1]
                print(f"  Alternate: {', '.join(alt['features'])}")


def main():
    parser = argparse.ArgumentParser(description='Analyze robust HMM test results')
    parser.add_argument('file', type=str, help='JSON results file to analyze')
    parser.add_argument('--export-csv', action='store_true', 
                       help='Export detailed results to CSV')
    args = parser.parse_args()
    
    # Load and analyze
    data = load_results(args.file)
    print_summary(data)
    
    # Optional CSV export
    if args.export_csv:
        df = analyze_by_category(data)
        csv_file = args.file.replace('.json', '_analysis.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Detailed analysis exported to {csv_file}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
