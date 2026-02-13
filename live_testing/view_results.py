"""
View Forward Paper Trading Results
Quick summary of current performance and positions
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_state():
    """Load current trading state."""
    state_file = Path(__file__).parent / "data" / "trading_state.json"
    with open(state_file, 'r') as f:
        return json.load(f)

def load_performance():
    """Load daily performance log."""
    perf_file = Path(__file__).parent / "trading_logs" / "daily_performance.csv"
    return pd.read_csv(perf_file)

def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"

def format_percent(value):
    """Format value as percentage."""
    return f"{value:+.2f}%"

def main():
    print("=" * 80)
    print("FORWARD PAPER TRADING - CURRENT STATUS")
    print("=" * 80)
    
    # Load data
    state = load_state()
    df = load_performance()
    
    # Basic info
    print(f"\nStart Date: {state['start_date']}")
    print(f"Last Run: {state['last_run_date']}")
    print(f"Trading Days: {state['trading_days']}")
    
    # Get latest performance for each strategy
    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date].copy()
    
    print(f"\nLatest Data: {latest_date}")
    print(f"\n{'Strategy':<30} {'Equity':>12} {'Return':>10} {'Positions':>10}")
    print("-" * 80)
    
    # Sort by return
    latest = latest.sort_values('total_return_pct', ascending=False)
    
    for _, row in latest.iterrows():
        equity = format_currency(row['equity'])
        ret = format_percent(row['total_return_pct'])
        pos = int(row['num_positions'])
        
        print(f"{row['strategy']:<30} {equity:>12} {ret:>10} {pos:>10}")
    
    # Show positions for each strategy
    print(f"\n{'='*80}")
    print("CURRENT POSITIONS BY STRATEGY")
    print("=" * 80)
    
    for strategy_name, portfolio in state['portfolios'].items():
        positions = portfolio['positions']
        cash = portfolio['cash']
        
        print(f"\n{strategy_name}:")
        print(f"  Cash: {format_currency(cash)}")
        
        if positions:
            print(f"  Positions ({len(positions)}):")
            for symbol, pos_info in positions.items():
                shares = pos_info['shares']
                entry = pos_info['entry_price']
                pos_type = pos_info['position_type'].upper()
                print(f"    {symbol}: {shares:.2f} shares @ {format_currency(entry)} ({pos_type})")
        else:
            print(f"  Positions: None (100% cash)")
    
    # Performance summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY (Since Start)")
    print("=" * 80)
    
    summary = latest[['strategy', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']].copy()
    summary = summary.sort_values('total_return_pct', ascending=False)
    
    print(f"\n{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        ret = format_percent(row['total_return_pct'])
        sharpe = f"{row['sharpe_ratio']:.2f}"
        dd = format_percent(row['max_drawdown_pct'])
        
        print(f"{row['strategy']:<30} {ret:>10} {sharpe:>10} {dd:>10}")
    
    # Daily returns
    print(f"\n{'='*80}")
    print("DAILY RETURNS")
    print("=" * 80)
    
    for date in sorted(df['date'].unique()):
        print(f"\n{date}:")
        day_data = df[df['date'] == date].sort_values('total_return_pct', ascending=False)
        
        for _, row in day_data.iterrows():
            ret = format_percent(row['total_return_pct'])
            pos = int(row['num_positions'])
            print(f"  {row['strategy']:<28} {ret:>10} ({pos} pos)")
    
    print(f"\n{'='*80}")
    print(f"Run 'python paper_trader.py --status' for more details")
    print("=" * 80)

if __name__ == "__main__":
    main()
