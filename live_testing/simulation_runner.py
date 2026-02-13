"""
Simulation Runner

Main orchestrator for running live testing simulations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import *
from portfolio_tracker import PortfolioTracker, PositionType
from strategies import create_strategy


def fetch_polygon_data(symbol: str, start_date: str, end_date: str) -> Dict:
    """
    Fetch daily bars from Polygon API
    
    Returns dict with 'bars' list containing OHLCV data
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"error": "POLYGON_API_KEY not found in environment"}
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            # Convert Polygon format to our format
            bars = []
            for bar in data['results']:
                bars.append({
                    'date': datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d'),
                    'open': bar.get('o', 0),
                    'high': bar.get('h', 0),
                    'low': bar.get('l', 0),
                    'close': bar.get('c', 0),
                    'volume': bar.get('v', 0)
                })
            return {"bars": bars}
        else:
            return {"error": f"No data returned: {data.get('status', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


class SimulationRunner:
    """
    Runs backtests for multiple strategies and compares performance
    """
    
    def __init__(self, config_override: Dict = None):
        # Load config
        self.symbols = TEST_SYMBOLS
        self.start_date = START_DATE
        # Polygon free tier has delayed data - use yesterday as end date
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.end_date = END_DATE if END_DATE != datetime.now().strftime("%Y-%m-%d") else yesterday
        self.initial_capital = INITIAL_CAPITAL
        self.strategies_config = STRATEGIES
        
        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(self, key, value)
        
        # Create portfolios for each strategy
        self.portfolios: Dict[str, PortfolioTracker] = {}
        self.strategies: Dict[str, BaseStrategy] = {}
        
        for strategy_name, strategy_config in self.strategies_config.items():
            self.portfolios[strategy_name] = PortfolioTracker(
                initial_capital=self.initial_capital,
                commission_per_trade=COMMISSION_PER_TRADE,
                slippage_bps=SLIPPAGE_BPS,
                strategy_name=strategy_name
            )
            self.strategies[strategy_name] = create_strategy(strategy_config)
        
        # Results storage
        self.results = {}
    
    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            if VERBOSE:
                print(f"  Fetching data for {symbol}...")
            
            data = fetch_polygon_data(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if "error" in data:
                print(f"  Error fetching {symbol}: {data['error']}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["bars"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df = df.set_index("date")
            
            return df
            
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def run_simulation(self):
        """Run simulation for all strategies"""
        print(f"\n{'='*80}")
        print(f"LIVE TESTING SIMULATION")
        print(f"{'='*80}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Strategies: {len(self.strategies)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*80}\n")
        
        # Fetch data for all symbols
        print("Fetching historical data...")
        all_data = {}
        for symbol in self.symbols:
            df = self.fetch_historical_data(symbol)
            if not df.empty:
                all_data[symbol] = df
            else:
                print(f"  Skipping {symbol} (no data)")
        
        if not all_data:
            print("ERROR: No data fetched. Cannot run simulation.")
            return
        
        print(f"Fetched data for {len(all_data)} symbols\n")
        
        # Get date range (intersection of all symbols)
        date_ranges = [df.index for df in all_data.values()]
        common_dates = set(date_ranges[0])
        for dates in date_ranges[1:]:
            common_dates = common_dates.intersection(set(dates))
        
        trading_dates = sorted(list(common_dates))
        
        if not trading_dates:
            print("ERROR: No common trading dates found.")
            return
        
        print(f"Running simulation over {len(trading_dates)} trading days...")
        print(f"{'='*80}\n")
        
        # Run simulation day by day
        for date in tqdm(trading_dates, desc="Simulating"):
            # Get prices for all symbols on this date
            current_prices = {}
            for symbol, df in all_data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, "close"]
            
            # Run each strategy
            for strategy_name, strategy in self.strategies.items():
                portfolio = self.portfolios[strategy_name]
                
                # Process each symbol
                for symbol in self.symbols:
                    if symbol not in current_prices:
                        continue
                    
                    price = current_prices[symbol]
                    current_position = portfolio.get_position_type(symbol)
                    
                    # Get historical data for this symbol up to current date (prices AND volumes)
                    symbol_df = all_data[symbol].loc[:date]
                    historical_data = {
                        "prices": symbol_df["close"].tolist(),
                        "volumes": symbol_df["volume"].tolist() if "volume" in symbol_df.columns else []
                    }
                    
                    # Generate signal
                    signal = strategy.generate_signal(
                        symbol=symbol,
                        current_position=current_position,
                        date=date,
                        price=price,
                        historical_data=historical_data
                    )
                    
                    # Execute signal
                    if signal == "buy":
                        # Open long position
                        target_value = portfolio.cash * MAX_POSITION_SIZE
                        portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.LONG,
                            date=date
                        )
                    
                    elif signal == "sell":
                        # Close long position
                        portfolio.close_position(symbol, price, date)
                    
                    elif signal == "short":
                        # Open short position
                        target_value = portfolio.cash * MAX_POSITION_SIZE
                        portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.SHORT,
                            date=date
                        )
                    
                    elif signal == "cover":
                        # Close short position
                        portfolio.close_position(symbol, price, date)
                
                # Update equity curve
                portfolio.update_equity_curve(date, current_prices)
        
        print(f"\n{'='*80}")
        print("Simulation complete!")
        print(f"{'='*80}\n")
        
        # Calculate performance metrics
        self.calculate_results()
        
        # Save results
        self.save_results()
        
        # Display summary
        self.display_summary()
    
    def calculate_results(self):
        """Calculate performance metrics for all strategies"""
        for strategy_name, portfolio in self.portfolios.items():
            self.results[strategy_name] = portfolio.get_performance_metrics()
    
    def save_results(self):
        """Save results to files"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary metrics
        if SAVE_SUMMARY_CSV:
            summary_df = pd.DataFrame([
                self.results[strategy_name] 
                for strategy_name in self.results
            ])
            summary_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary to: {summary_path}")
        
        # Save equity curves
        if SAVE_EQUITY_CURVE:
            for strategy_name, portfolio in self.portfolios.items():
                equity_df = pd.DataFrame(portfolio.equity_curve)
                equity_path = os.path.join(
                    RESULTS_DIR, 
                    f"equity_{strategy_name}_{timestamp}.csv"
                )
                equity_df.to_csv(equity_path, index=False)
            print(f"Saved equity curves to: {RESULTS_DIR}")
        
        # Save trade logs
        if SAVE_TRADE_LOG:
            for strategy_name, portfolio in self.portfolios.items():
                if portfolio.trades:
                    trades_df = pd.DataFrame([
                        {
                            "symbol": t.symbol,
                            "entry_date": t.entry_date,
                            "exit_date": t.exit_date,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "shares": t.shares,
                            "position_type": t.position_type.value,
                            "pnl": t.pnl,
                            "pnl_pct": t.pnl_pct,
                            "commission": t.commission,
                            "slippage": t.slippage
                        }
                        for t in portfolio.trades
                    ])
                    trades_path = os.path.join(
                        LOGS_DIR,
                        f"trades_{strategy_name}_{timestamp}.csv"
                    )
                    trades_df.to_csv(trades_path, index=False)
            print(f"Saved trade logs to: {LOGS_DIR}")
        
        # Save full results as JSON
        results_path = os.path.join(RESULTS_DIR, f"full_results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved full results to: {results_path}\n")
    
    def display_summary(self):
        """Display summary of results"""
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}\n")
        
        # Sort by Sharpe ratio
        sorted_strategies = sorted(
            self.results.items(),
            key=lambda x: x[1].get("sharpe_ratio", -999),
            reverse=True
        )
        
        for strategy_name, metrics in sorted_strategies:
            print(f"{strategy_name}:")
            print(f"  Total Return:      {metrics['total_return_pct']:>8.2f}%")
            print(f"  Annualized Return: {metrics['annualized_return_pct']:>8.2f}%")
            print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
            print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:>8.2f}%")
            print(f"  Win Rate:          {metrics['win_rate_pct']:>8.2f}%")
            print(f"  Num Trades:        {metrics['num_trades']:>8}")
            print()
        
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    runner = SimulationRunner()
    runner.run_simulation()


if __name__ == "__main__":
    main()
