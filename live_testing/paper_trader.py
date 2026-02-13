"""
Forward Paper Trading Runner

Runs once per day to make trading decisions and track performance going forward.
This is NOT a backtest - it starts TODAY and tracks real-time paper trading.

Usage:
    python paper_trader.py              # Run today's decision
    python paper_trader.py --reset      # Start fresh (clears all state)
    python paper_trader.py --status     # Show current positions and performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import argparse
import requests
from dotenv import load_dotenv
import pandas_market_calendars as mcal

# Load environment variables from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from config import *
from portfolio_tracker import PortfolioTracker, PositionType
from strategies import create_strategy

# State file location
STATE_FILE = "data/trading_state.json"
TRADES_LOG = "trading_logs/paper_trades.csv"
PERFORMANCE_LOG = "trading_logs/daily_performance.csv"


def fetch_latest_price(symbol: str) -> Dict:
    """Fetch most recent price data from Polygon"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"error": "POLYGON_API_KEY not found"}
    
    # Try to get data from the last 10 days (handles weekends, holidays, and intraday delays)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "desc", "limit": 5, "apiKey": api_key}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Accept OK or DELAYED status (DELAYED means data exists but market is still open)
        if data.get('status') in ['OK', 'DELAYED'] and data.get('results') and len(data['results']) > 0:
            bar = data['results'][0]
            # Convert timestamp to date
            bar_date = datetime.fromtimestamp(bar['t'] / 1000).strftime("%Y-%m-%d")
            return {
                "symbol": symbol,
                "date": bar_date,
                "price": bar.get('c', 0),
                "open": bar.get('o', 0),
                "high": bar.get('h', 0),
                "low": bar.get('l', 0),
                "volume": bar.get('v', 0)
            }
        else:
            return {"error": f"No data for {symbol}: {data.get('status', 'Unknown error')}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_historical_data(symbol: str, days: int = 252) -> Dict:
    """
    Fetch historical prices AND volumes for regime detection.
    
    Returns:
        Dict with 'prices' and 'volumes' lists
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"prices": [], "volumes": []}
    
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days+10)).strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') in ['OK', 'DELAYED'] and data.get('results'):
            prices = [bar['c'] for bar in data['results']]
            volumes = [bar.get('v', 0) for bar in data['results']]
            return {"prices": prices, "volumes": volumes}
        return {"prices": [], "volumes": []}
    except:
        return {"prices": [], "volumes": []}


def fetch_historical_prices(symbol: str, days: int = 252) -> List[float]:
    """
    Fetch historical prices for regime detection.
    DEPRECATED: Use fetch_historical_data() instead to get volumes too.
    """
    result = fetch_historical_data(symbol, days)
    return result.get("prices", [])


class PaperTradingRunner:
    """Manages forward paper trading state and execution"""
    
    def __init__(self):
        self.state_file = STATE_FILE
        self.trades_log = TRADES_LOG
        self.performance_log = PERFORMANCE_LOG
        
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("trading_logs", exist_ok=True)
        
        # Load or initialize state
        self.state = self.load_state()
        
        # Initialize portfolios
        self.portfolios: Dict[str, PortfolioTracker] = {}
        for strategy_name, strategy_config in STRATEGIES.items():
            # Load existing portfolio or create new
            if strategy_name in self.state["portfolios"]:
                self.portfolios[strategy_name] = self.deserialize_portfolio(
                    self.state["portfolios"][strategy_name],
                    strategy_name
                )
            else:
                self.portfolios[strategy_name] = PortfolioTracker(
                    initial_capital=INITIAL_CAPITAL,
                    commission_per_trade=COMMISSION_PER_TRADE,
                    slippage_bps=SLIPPAGE_BPS,
                    strategy_name=strategy_name
                )
        
        # Initialize strategies
        self.strategies = {
            name: create_strategy(config) 
            for name, config in STRATEGIES.items()
        }
    
    def load_state(self) -> Dict:
        """Load trading state from disk"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize new state
            return {
                "start_date": datetime.now().strftime("%Y-%m-%d"),
                "last_run_date": None,
                "trading_days": 0,
                "portfolios": {},
                "symbols": TEST_SYMBOLS
            }
    
    def save_state(self):
        """Save trading state to disk"""
        # Serialize portfolios
        serialized_portfolios = {}
        for name, portfolio in self.portfolios.items():
            serialized_portfolios[name] = self.serialize_portfolio(portfolio)
        
        self.state["portfolios"] = serialized_portfolios
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def serialize_portfolio(self, portfolio: PortfolioTracker) -> Dict:
        """Convert portfolio to JSON-serializable dict"""
        return {
            "cash": portfolio.cash,
            "positions": {
                symbol: {
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date.isoformat(),
                    "position_type": pos.position_type.value
                }
                for symbol, pos in portfolio.positions.items()
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat(),
                    "exit_date": t.exit_date.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "position_type": t.position_type.value,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct
                }
                for t in portfolio.trades
            ],
            "equity_curve": portfolio.equity_curve
        }
    
    def deserialize_portfolio(self, data: Dict, strategy_name: str) -> PortfolioTracker:
        """Restore portfolio from saved state"""
        from portfolio_tracker import Position
        
        portfolio = PortfolioTracker(
            initial_capital=INITIAL_CAPITAL,
            commission_per_trade=COMMISSION_PER_TRADE,
            slippage_bps=SLIPPAGE_BPS,
            strategy_name=strategy_name
        )
        
        portfolio.cash = data["cash"]
        
        # Restore positions
        for symbol, pos_data in data["positions"].items():
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=pos_data["shares"],
                entry_price=pos_data["entry_price"],
                entry_date=datetime.fromisoformat(pos_data["entry_date"]),
                position_type=PositionType(pos_data["position_type"])
            )
        
        # Restore history
        portfolio.equity_curve = data["equity_curve"]
        
        return portfolio
    
    def already_ran_today(self) -> bool:
        """Check if we already processed today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.state.get("last_run_date") == today
    
    def is_trading_day(self, date_str: str = None) -> bool:
        """Check if date is a trading day on NYSE"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        try:
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(start_date=date_str, end_date=date_str)
            return len(schedule) > 0
        except:
            # If market calendar fails, assume it's a trading day (fail safe)
            return True
    
    def run_daily_trading(self, force: bool = False):
        """Execute today's trading decisions
        
        Args:
            force: If True, run even if market is closed or already ran today
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if market is open today
        if not self.is_trading_day(today):
            if force:
                print(f"⚠️  Market closed on {today} but --force specified, running anyway...")
            else:
                print(f"Market closed on {today} (weekend/holiday)")
                print("No trading decisions to make. Use --force to run anyway.")
                return
        
        # Check if already ran
        if self.already_ran_today():
            if force:
                print(f"⚠️  Already ran today but --force specified, running again...")
            else:
                print(f"Already processed trading for {today}")
                print("Run with --status to see current positions, or --force to run again")
                return
        
        print(f"\n{'='*80}")
        print(f"FORWARD PAPER TRADING - {today}")
        print(f"{'='*80}")
        print(f"Trading Day #{self.state['trading_days'] + 1} since {self.state['start_date']}")
        print(f"Symbols: {', '.join(TEST_SYMBOLS)}")
        print(f"{'='*80}\n")
        
        # Fetch latest prices
        print("Fetching latest prices...")
        current_prices = {}
        for symbol in TEST_SYMBOLS:
            data = fetch_latest_price(symbol)
            if "error" not in data:
                current_prices[symbol] = data["price"]
                print(f"  {symbol}: ${data['price']:.2f}")
            else:
                print(f"  {symbol}: Error - {data['error']}")
        
        if not current_prices:
            print("\nERROR: No price data available. Cannot make trading decisions.")
            return
        
        print(f"\n{'='*80}")
        print("GENERATING SIGNALS")
        print(f"{'='*80}\n")
        
        # Generate signals for each strategy
        for strategy_name, strategy in self.strategies.items():
            portfolio = self.portfolios[strategy_name]
            
            print(f"\n{strategy_name}:")
            print(f"  Cash: ${portfolio.cash:,.2f}")
            print(f"  Positions: {len(portfolio.positions)}")
            
            # Process each symbol
            for symbol in TEST_SYMBOLS:
                if symbol not in current_prices:
                    continue
                
                price = current_prices[symbol]
                current_position = portfolio.get_position_type(symbol)
                
                # Get historical data for regime detection (prices AND volumes)
                historical_data = fetch_historical_data(symbol, days=252)
                
                # Generate signal
                try:
                    signal = strategy.generate_signal(
                        symbol=symbol,
                        current_position=current_position,
                        date=datetime.now(),
                        price=price,
                        historical_data=historical_data
                    )
                    
                    print(f"    {symbol}: {current_position.value} → {signal}")
                    
                    # Execute signal
                    if signal == "buy" and portfolio.cash > 0:
                        target_value = min(portfolio.cash * MAX_POSITION_SIZE, portfolio.cash)
                        success = portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.LONG,
                            date=datetime.now()
                        )
                        if success:
                            print(f"      → Opened LONG ${target_value:.2f}")
                    
                    elif signal == "sell" and current_position == PositionType.LONG:
                        trade = portfolio.close_position(symbol, price, datetime.now())
                        if trade:
                            print(f"      → Closed LONG (P&L: ${trade.pnl:,.2f}, {trade.pnl_pct:.2f}%)")
                    
                    elif signal == "short" and portfolio.cash > 0:
                        target_value = min(portfolio.cash * MAX_POSITION_SIZE, portfolio.cash)
                        success = portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.SHORT,
                            date=datetime.now()
                        )
                        if success:
                            print(f"      → Opened SHORT ${target_value:.2f}")
                    
                    elif signal == "cover" and current_position == PositionType.SHORT:
                        trade = portfolio.close_position(symbol, price, datetime.now())
                        if trade:
                            print(f"      → Closed SHORT (P&L: ${trade.pnl:,.2f}, {trade.pnl_pct:.2f}%)")
                
                except Exception as e:
                    print(f"    {symbol}: Error - {e}")
            
            # Update equity curve
            portfolio.update_equity_curve(datetime.now(), current_prices)
        
        # Update state
        self.state["last_run_date"] = today
        self.state["trading_days"] += 1
        
        # Save everything
        self.save_state()
        self.save_daily_performance(today, current_prices)
        
        print(f"\n{'='*80}")
        print("DAILY SUMMARY")
        print(f"{'='*80}\n")
        
        self.show_performance()
    
    def save_daily_performance(self, date: str, prices: Dict[str, float]):
        """Log daily performance to CSV"""
        rows = []
        for strategy_name, portfolio in self.portfolios.items():
            equity = portfolio.get_equity(prices)
            metrics = portfolio.get_performance_metrics()
            
            rows.append({
                "date": date,
                "strategy": strategy_name,
                "equity": equity,
                "cash": portfolio.cash,
                "num_positions": len(portfolio.positions),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0)
            })
        
        df = pd.DataFrame(rows)
        
        # Append to existing file or create new
        if os.path.exists(self.performance_log):
            df.to_csv(self.performance_log, mode='a', header=False, index=False)
        else:
            df.to_csv(self.performance_log, index=False)
    
    def show_performance(self):
        """Display current performance summary"""
        print(f"\n{'Strategy':<30} {'Equity':>15} {'Return %':>12} {'Positions':>10}")
        print("-" * 70)
        
        for strategy_name, portfolio in self.portfolios.items():
            # Get latest prices for equity calculation
            current_prices = {}
            for symbol in TEST_SYMBOLS:
                data = fetch_latest_price(symbol)
                if "error" not in data:
                    current_prices[symbol] = data["price"]
            
            equity = portfolio.get_equity(current_prices)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            print(f"{strategy_name:<30} ${equity:>14,.2f} {return_pct:>11.2f}% {len(portfolio.positions):>10}")
        
        print()
    
    def show_status(self):
        """Show detailed current status"""
        print(f"\n{'='*80}")
        print(f"FORWARD PAPER TRADING STATUS")
        print(f"{'='*80}")
        print(f"Start Date: {self.state['start_date']}")
        print(f"Last Run: {self.state.get('last_run_date', 'Never')}")
        print(f"Trading Days: {self.state['trading_days']}")
        print(f"{'='*80}\n")
        
        self.show_performance()
        
        # Show positions
        print(f"\n{'='*80}")
        print("CURRENT POSITIONS")
        print(f"{'='*80}\n")
        
        for strategy_name, portfolio in self.portfolios.items():
            if portfolio.positions:
                print(f"\n{strategy_name}:")
                for symbol, pos in portfolio.positions.items():
                    days_held = (datetime.now() - pos.entry_date).days
                    print(f"  {symbol}: {pos.shares:.2f} shares @ ${pos.entry_price:.2f} ({pos.position_type.value}, {days_held} days)")
            else:
                print(f"\n{strategy_name}: No positions")
    
    def reset(self):
        """Reset all state and start fresh"""
        print("Resetting all trading state...")
        
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        
        self.state = self.load_state()
        
        self.portfolios = {
            name: PortfolioTracker(
                initial_capital=INITIAL_CAPITAL,
                commission_per_trade=COMMISSION_PER_TRADE,
                slippage_bps=SLIPPAGE_BPS,
                strategy_name=name
            )
            for name in STRATEGIES.keys()
        }
        
        self.save_state()
        print("Reset complete. Ready to start fresh.")


def main():
    parser = argparse.ArgumentParser(description="Forward Paper Trading System")
    parser.add_argument('--reset', action='store_true', help="Reset and start fresh")
    parser.add_argument('--status', action='store_true', help="Show current status")
    parser.add_argument('--force', action='store_true', help="Force run even if market closed or already ran today")
    args = parser.parse_args()
    
    runner = PaperTradingRunner()
    
    if args.reset:
        runner.reset()
    elif args.status:
        runner.show_status()
    else:
        runner.run_daily_trading(force=args.force)


if __name__ == "__main__":
    main()
