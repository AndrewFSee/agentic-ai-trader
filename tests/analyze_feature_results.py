#!/usr/bin/env python3
"""
Analyze feature combination test results.
Identify patterns, best combinations, sector-specific insights.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_results(filepath: str) -> Dict:
    """Load results JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_top_combinations(results: Dict, top_n: int = 15) -> pd.DataFrame:
    """Analyze top N feature combinations by Sharpe improvement"""
    combos = results['combination_results']
    
    data = []
    for combo in combos[:top_n]:
        data.append({
            'Rank': len(data) + 1,
            'Features': ', '.join(combo['features']),
            'Avg_Sharpe_Improvement': combo['avg_sharpe_improvement'],
            'Avg_Drawdown_Improvement': combo['avg_drawdown_improvement'],
            'Avg_Return_Diff': combo['avg_return_diff'],
            'Success_Rate': f"{combo['success_count']}/12",
            'Feature_1': combo['features'][0],
            'Feature_2': combo['features'][1],
            'Feature_3': combo['features'][2]
        })
    
    return pd.DataFrame(data)

def analyze_feature_frequency(results: Dict, top_n: int = 30) -> pd.DataFrame:
    """Count how often each feature appears in top N combinations"""
    combos = results['combination_results'][:top_n]
    
    feature_counts = Counter()
    feature_sharpe_sum = defaultdict(float)
    feature_combos = defaultdict(list)
    
    for combo in combos:
        for feature in combo['features']:
            feature_counts[feature] += 1
            feature_sharpe_sum[feature] += combo['avg_sharpe_improvement']
            feature_combos[feature].append(', '.join(combo['features']))
    
    data = []
    for feature, count in feature_counts.most_common():
        data.append({
            'Feature': feature,
            'Appearances': count,
            'Frequency_%': round(count / top_n * 100, 1),
            'Avg_Sharpe_When_Present': round(feature_sharpe_sum[feature] / count, 4),
            'Example_Combos': ' | '.join(feature_combos[feature][:3])
        })
    
    return pd.DataFrame(data)

def analyze_sector_performance(results: Dict) -> pd.DataFrame:
    """Analyze which sectors benefit most from HMM strategy"""
    stocks = results['tested_stocks']
    combos = results['combination_results']
    
    # Get top 10 combinations
    top_combos = combos[:10]
    
    sector_data = defaultdict(lambda: {
        'sharpe_improvements': [],
        'drawdown_improvements': [],
        'return_diffs': []
    })
    
    for combo in top_combos:
        for symbol, sector in stocks.items():
            stock_result = combo['stock_results'].get(symbol, {})
            if 'sharpe_improvement' in stock_result:
                sector_data[sector]['sharpe_improvements'].append(stock_result['sharpe_improvement'])
                sector_data[sector]['drawdown_improvements'].append(stock_result['drawdown_improvement'])
                sector_data[sector]['return_diffs'].append(stock_result['return_diff'])
    
    data = []
    for sector, metrics in sector_data.items():
        data.append({
            'Sector': sector.title(),
            'Avg_Sharpe_Improvement': round(np.mean(metrics['sharpe_improvements']), 4),
            'Sharpe_Std': round(np.std(metrics['sharpe_improvements']), 4),
            'Avg_Drawdown_Improvement': round(np.mean(metrics['drawdown_improvements']), 4),
            'Avg_Return_Diff': round(np.mean(metrics['return_diffs']), 4),
            'Samples': len(metrics['sharpe_improvements'])
        })
    
    df = pd.DataFrame(data)
    return df.sort_values('Avg_Sharpe_Improvement', ascending=False)

def analyze_stock_winners_losers(results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify which stocks benefit most/least from HMM"""
    stocks = results['tested_stocks']
    combos = results['combination_results']
    
    # Get top 20 combinations
    top_combos = combos[:20]
    
    stock_performance = defaultdict(lambda: {
        'sharpe_improvements': [],
        'drawdown_improvements': [],
        'return_diffs': []
    })
    
    for combo in top_combos:
        for symbol in stocks.keys():
            stock_result = combo['stock_results'].get(symbol, {})
            if 'sharpe_improvement' in stock_result:
                stock_performance[symbol]['sharpe_improvements'].append(stock_result['sharpe_improvement'])
                stock_performance[symbol]['drawdown_improvements'].append(stock_result['drawdown_improvement'])
                stock_performance[symbol]['return_diffs'].append(stock_result['return_diff'])
    
    data = []
    for symbol, metrics in stock_performance.items():
        sector = stocks[symbol]
        data.append({
            'Symbol': symbol,
            'Sector': sector.title(),
            'Avg_Sharpe_Improvement': round(np.mean(metrics['sharpe_improvements']), 4),
            'Best_Sharpe_Improvement': round(max(metrics['sharpe_improvements']), 4),
            'Worst_Sharpe_Improvement': round(min(metrics['sharpe_improvements']), 4),
            'Consistency_Score': round(np.mean(metrics['sharpe_improvements']) / (np.std(metrics['sharpe_improvements']) + 0.01), 2),
            'Avg_Drawdown_Improvement': round(np.mean(metrics['drawdown_improvements']), 4),
            'Avg_Return_Diff': round(np.mean(metrics['return_diffs']), 4)
        })
    
    df = pd.DataFrame(data)
    winners = df.nlargest(6, 'Avg_Sharpe_Improvement')
    losers = df.nsmallest(6, 'Avg_Sharpe_Improvement')
    
    return winners, losers

def find_sector_specific_combos(results: Dict) -> Dict[str, Dict]:
    """Find best feature combination for each sector"""
    stocks = results['tested_stocks']
    combos = results['combination_results']
    
    sector_best = defaultdict(lambda: {'combo': None, 'sharpe': -999, 'features': None})
    
    for combo in combos:
        # Calculate sector-specific performance
        sector_sharpes = defaultdict(list)
        
        for symbol, sector in stocks.items():
            stock_result = combo['stock_results'].get(symbol, {})
            if 'sharpe_improvement' in stock_result:
                sector_sharpes[sector].append(stock_result['sharpe_improvement'])
        
        # Check if best for any sector
        for sector, sharpes in sector_sharpes.items():
            avg_sharpe = np.mean(sharpes)
            if avg_sharpe > sector_best[sector]['sharpe']:
                sector_best[sector] = {
                    'combo': ', '.join(combo['features']),
                    'sharpe': round(avg_sharpe, 4),
                    'features': combo['features'],
                    'dd_improvement': round(np.mean([
                        combo['stock_results'][s]['drawdown_improvement']
                        for s in stocks if stocks[s] == sector
                        and 'drawdown_improvement' in combo['stock_results'].get(s, {})
                    ]), 4)
                }
    
    return dict(sector_best)

def analyze_feature_pairs(results: Dict, top_n: int = 30) -> pd.DataFrame:
    """Identify which feature pairs work well together"""
    combos = results['combination_results'][:top_n]
    
    pair_performance = defaultdict(lambda: {'sharpe_sum': 0, 'count': 0, 'combos': []})
    
    for combo in combos:
        features = sorted(combo['features'])
        sharpe = combo['avg_sharpe_improvement']
        
        # All pairs within the combo
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                pair = f"{features[i]} + {features[j]}"
                pair_performance[pair]['sharpe_sum'] += sharpe
                pair_performance[pair]['count'] += 1
                pair_performance[pair]['combos'].append(', '.join(combo['features']))
    
    data = []
    for pair, metrics in pair_performance.items():
        if metrics['count'] >= 3:  # Only pairs that appear at least 3 times
            data.append({
                'Feature_Pair': pair,
                'Appearances': metrics['count'],
                'Avg_Sharpe': round(metrics['sharpe_sum'] / metrics['count'], 4),
                'Example_Combos': ' | '.join(set(metrics['combos'][:2]))
            })
    
    df = pd.DataFrame(data)
    return df.sort_values('Avg_Sharpe', ascending=False).head(15)

def print_summary_report(results: Dict):
    """Print comprehensive summary report"""
    
    print("=" * 100)
    print("FEATURE COMBINATION TEST - COMPREHENSIVE ANALYSIS")
    print("=" * 100)
    
    config = results['test_config']
    print(f"\nTest Configuration:")
    print(f"  - Stocks Tested: {len(results['tested_stocks'])} ({config['stocks_per_sector']} per sector)")
    print(f"  - Feature Combinations: {len(results['combination_results'])}")
    print(f"  - Training Window: {config['training_window']} days")
    print(f"  - Retrain Frequency: {config['retrain_freq']} days")
    print(f"  - Persistence Prior: {config['persistence']}")
    print(f"  - Test Date: {config['test_date']}")
    
    # Top combinations
    print("\n" + "=" * 100)
    print("TOP 15 FEATURE COMBINATIONS (by Avg Sharpe Improvement)")
    print("=" * 100)
    top_df = analyze_top_combinations(results, top_n=15)
    print(top_df.to_string(index=False))
    
    # Feature frequency
    print("\n" + "=" * 100)
    print("FEATURE FREQUENCY IN TOP 30 COMBINATIONS")
    print("=" * 100)
    freq_df = analyze_feature_frequency(results, top_n=30)
    print(freq_df.to_string(index=False))
    
    # Sector performance
    print("\n" + "=" * 100)
    print("SECTOR PERFORMANCE (Top 10 Combinations)")
    print("=" * 100)
    sector_df = analyze_sector_performance(results)
    print(sector_df.to_string(index=False))
    
    # Stock winners/losers
    print("\n" + "=" * 100)
    print("TOP 6 STOCKS (Best Sharpe Improvement)")
    print("=" * 100)
    winners, losers = analyze_stock_winners_losers(results)
    print(winners.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("BOTTOM 6 STOCKS (Worst Sharpe Improvement)")
    print("=" * 100)
    print(losers.to_string(index=False))
    
    # Sector-specific best combos
    print("\n" + "=" * 100)
    print("BEST FEATURE COMBINATION PER SECTOR")
    print("=" * 100)
    sector_combos = find_sector_specific_combos(results)
    for sector, info in sorted(sector_combos.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        print(f"\n{sector.upper()}:")
        print(f"  Best Combo: {info['combo']}")
        print(f"  Avg Sharpe Improvement: {info['sharpe']}")
        print(f"  Avg Drawdown Improvement: {info['dd_improvement']:.2%}")
    
    # Feature pairs
    print("\n" + "=" * 100)
    print("TOP FEATURE PAIRS (appear together in top combos)")
    print("=" * 100)
    pairs_df = analyze_feature_pairs(results, top_n=30)
    print(pairs_df.to_string(index=False))
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 100)
    
    # Universal best
    best_combo = results['combination_results'][0]
    print(f"\n1. UNIVERSAL BEST COMBINATION:")
    print(f"   Features: {', '.join(best_combo['features'])}")
    print(f"   Avg Sharpe Improvement: +{best_combo['avg_sharpe_improvement']:.4f}")
    print(f"   Avg Drawdown Improvement: {best_combo['avg_drawdown_improvement']:.2%}")
    print(f"   Avg Return Trade-off: {best_combo['avg_return_diff']:.2%}")
    print(f"   Success Rate: {best_combo['success_count']}/12 stocks")
    
    # Most versatile features
    freq_df = analyze_feature_frequency(results, top_n=30)
    top_features = freq_df.head(3)
    print(f"\n2. MOST VERSATILE FEATURES (appear in top 30 combos):")
    for _, row in top_features.iterrows():
        print(f"   - {row['Feature']}: {row['Appearances']}/30 ({row['Frequency_%']}%)")
    
    # Sector insights
    sector_df = analyze_sector_performance(results)
    best_sector = sector_df.iloc[0]
    worst_sector = sector_df.iloc[-1]
    print(f"\n3. SECTOR EFFECTIVENESS:")
    print(f"   Best: {best_sector['Sector']} (+{best_sector['Avg_Sharpe_Improvement']:.4f} Sharpe)")
    print(f"   Worst: {worst_sector['Sector']} ({worst_sector['Avg_Sharpe_Improvement']:.4f} Sharpe)")
    
    # Return trade-off
    avg_return_diff = np.mean([c['avg_return_diff'] for c in results['combination_results'][:10]])
    avg_sharpe_gain = np.mean([c['avg_sharpe_improvement'] for c in results['combination_results'][:10]])
    print(f"\n4. RETURN vs RISK TRADE-OFF (Top 10 combos):")
    print(f"   Avg Return Sacrifice: {avg_return_diff:.2%}")
    print(f"   Avg Sharpe Gain: +{avg_sharpe_gain:.4f}")
    print(f"   Conclusion: {'Worth it!' if avg_sharpe_gain > 0.15 else 'Marginal benefit'}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Find most recent results file
        import glob
        files = glob.glob("feature_combination_results_*.json")
        if not files:
            print("No results file found!")
            sys.exit(1)
        filepath = sorted(files)[-1]
    
    print(f"Analyzing: {filepath}\n")
    
    results = load_results(filepath)
    print_summary_report(results)
    
    print("\n✓ Analysis complete!")
