#!/usr/bin/env python3
"""
Run Wasserstein vs HMM comparison on multiple stocks.
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

STOCKS = [
    # Tech
    'AAPL', 'MSFT',
    # Finance
    'JPM', 'BAC',
    # Energy
    'XOM', 'CVX',
    # Consumer
    'WMT', 'HD',
    # Healthcare
    'JNJ', 'PFE',
    # Industrial
    'BA', 'CAT'
]

def run_comparison_on_stock(symbol: str) -> dict:
    """Run comparison on one stock."""
    print(f"\n{'='*80}")
    print(f"Running comparison on {symbol}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            ['python', 'final_wasserstein_vs_hmm.py', '--symbol', symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"Error on {symbol}:")
            print(result.stderr)
            return None
        
        # Find result file
        result_files = sorted(Path('.').glob(f'final_comparison_{symbol}_*.json'))
        if result_files:
            with open(result_files[-1], 'r') as f:
                return json.load(f)
        else:
            print(f"No result file found for {symbol}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout on {symbol}")
        return None
    except Exception as e:
        print(f"Exception on {symbol}: {e}")
        return None

def main():
    results = {}
    
    for symbol in STOCKS:
        result = run_comparison_on_stock(symbol)
        if result:
            results[symbol] = result
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}\n")
    
    wass_wins = 0
    hmm_wins = 0
    ties = 0
    
    wass_sharpes = []
    hmm_sharpes = []
    wass_returns = []
    hmm_returns = []
    
    print(f"{'Stock':<8} {'Wass Sharpe':<12} {'HMM Sharpe':<12} {'Winner':<10}")
    print("-" * 50)
    
    for symbol, result in results.items():
        wass_sharpe = result['paper_wass']['sharpe']
        hmm_sharpe = result['hmm']['sharpe']
        
        wass_sharpes.append(wass_sharpe)
        hmm_sharpes.append(hmm_sharpe)
        wass_returns.append(result['paper_wass']['total_return'])
        hmm_returns.append(result['hmm']['total_return'])
        
        if wass_sharpe > hmm_sharpe:
            winner = "Wass"
            wass_wins += 1
        elif hmm_sharpe > wass_sharpe:
            winner = "HMM"
            hmm_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(f"{symbol:<8} {wass_sharpe:<12.3f} {hmm_sharpe:<12.3f} {winner:<10}")
    
    print("\n" + "=" * 50)
    print(f"Wins: Wasserstein={wass_wins}, HMM={hmm_wins}, Ties={ties}")
    print(f"\nMean Sharpe: Wasserstein={sum(wass_sharpes)/len(wass_sharpes):.3f}, HMM={sum(hmm_sharpes)/len(hmm_sharpes):.3f}")
    print(f"Mean Return: Wasserstein={sum(wass_returns)/len(wass_returns):.2%}, HMM={sum(hmm_returns)/len(hmm_returns):.2%}")
    
    # Save aggregate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_file = f"final_comparison_aggregate_{timestamp}.json"
    
    with open(aggregate_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'wass_wins': wass_wins,
                'hmm_wins': hmm_wins,
                'ties': ties,
                'mean_wass_sharpe': sum(wass_sharpes) / len(wass_sharpes),
                'mean_hmm_sharpe': sum(hmm_sharpes) / len(hmm_sharpes),
                'mean_wass_return': sum(wass_returns) / len(wass_returns),
                'mean_hmm_return': sum(hmm_returns) / len(hmm_returns)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {aggregate_file}")

if __name__ == '__main__':
    main()
