"""
Forward Paper Trading Runner - VIX ROC & Reflexion Agent Edition

Tests the new validated strategies:
- VIX ROC three-tier (15/15 walk-forward wins)
- Reflexion Agent with self-critique
- Buy-and-hold benchmark for comparison

Usage:
    python paper_trader_new.py              # Run today's decision
    python paper_trader_new.py --reset      # Start fresh (clears all state)
    python paper_trader_new.py --status     # Show current positions and performance
    python paper_trader_new.py --force      # Force run even if market closed
    python paper_trader_new.py --backfill N # Backfill N days of history
"""

import sys
import os
# Add both parent (project root) and current (live_testing) to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import argparse
import requests
from dotenv import load_dotenv
import pandas_market_calendars as mcal
import time

# Load environment variables from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# Use NEW config and strategies
from config_new import (
    TEST_SYMBOLS, INITIAL_CAPITAL, COMMISSION_PER_TRADE, 
    SLIPPAGE_BPS, MAX_POSITION_SIZE, STRATEGIES, START_DATE
)
from portfolio_tracker import PortfolioTracker, PositionType
from strategies_new import create_strategy

# Re-insert live_testing directory to front of path since other imports may have modified it
if _script_dir not in sys.path or sys.path.index(_script_dir) > 0:
    sys.path.insert(0, _script_dir)

from trade_logging import (
    TradeRecord, log_trade, extract_vix_roc_context, 
    extract_vol_context, extract_agent_rationale
)

# State file location - NEW location to not interfere with old testing
STATE_FILE = "data/trading_state_vixroc.json"
TRADES_LOG = "trading_logs/paper_trades_vixroc.csv"
PERFORMANCE_LOG = "trading_logs/daily_performance_vixroc.csv"


def _retry_api_call(fn, *args, max_retries: int = 3, backoff: float = 2.0, **kwargs):
    """
    Retry an API call with exponential backoff.
    Returns the function's result, or the last exception's fallback.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"      ⚠️ API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise last_error


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
        def _do_request():
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        
        data = _retry_api_call(_do_request)
        
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


def fetch_price_on_date(symbol: str, date: datetime) -> Optional[float]:
    """Fetch closing price for a specific date"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return None
    
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{date_str}/{date_str}"
    params = {"adjusted": "true", "apiKey": api_key}
    
    try:
        def _do_request():
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        
        data = _retry_api_call(_do_request)
        
        if data.get('status') in ['OK', 'DELAYED'] and data.get('results'):
            return data['results'][0].get('c', 0)
        return None
    except Exception as e:
        print(f"      ⚠️ Error fetching price for {symbol} on {date_str}: {e}")
        return None


def fetch_historical_data(symbol: str, days: int = 252) -> Dict:
    """
    Fetch historical prices AND volumes for analysis.
    
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
        def _do_request():
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        
        data = _retry_api_call(_do_request)
        
        if data.get('status') in ['OK', 'DELAYED'] and data.get('results'):
            prices = [bar['c'] for bar in data['results']]
            volumes = [bar.get('v', 0) for bar in data['results']]
            return {"prices": prices, "volumes": volumes}
        return {"prices": [], "volumes": []}
    except Exception as e:
        print(f"      ⚠️ Error fetching historical data for {symbol}: {e}")
        return {"prices": [], "volumes": []}


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of NYSE trading days between two dates"""
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=start_date.strftime("%Y-%m-%d"), 
            end_date=end_date.strftime("%Y-%m-%d")
        )
        return [d.to_pydatetime() for d in schedule.index]
    except Exception as e:
        # Fallback: return all weekdays
        print(f"  ⚠️ Market calendar error: {e} - using weekday fallback")
        days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                days.append(current)
            current += timedelta(days=1)
        return days


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
            # Handle START_DATE as either datetime or string
            start_str = START_DATE if isinstance(START_DATE, str) else START_DATE.strftime("%Y-%m-%d")
            return {
                "start_date": start_str,
                "last_run_date": None,
                "trading_days": 0,
                "portfolios": {},
                "symbols": TEST_SYMBOLS,
                "version": "2.0-vixroc"
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
                    "position_type": pos.position_type.value,
                    "stop_loss": pos.stop_loss,
                    "target_price": pos.target_price,
                    "confidence": pos.confidence,
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
                position_type=PositionType(pos_data["position_type"]),
                stop_loss=pos_data.get("stop_loss"),
                target_price=pos_data.get("target_price"),
                confidence=pos_data.get("confidence", 5),
            )
        
        # Restore history
        portfolio.equity_curve = data["equity_curve"]
        
        return portfolio
    
    def _log_closed_trade(
        self,
        trade,
        strategy_name: str,
        strategy,
        reason: str,
        date: datetime
    ) -> None:
        """
        Log a closed trade to the trade memory system.
        
        Creates a TradeRecord with full context and logs to JSONL.
        """
        try:
            # Extract context from strategy
            vix_context = extract_vix_roc_context(strategy)
            vol_context = extract_vol_context(strategy)
            agent_rationale = extract_agent_rationale(strategy)
            
            # Use reason as rationale if no agent rationale available
            if not agent_rationale:
                agent_rationale = reason
            
            # Calculate holding days
            holding_days = (trade.exit_date - trade.entry_date).days
            
            # Create trade record
            record = TradeRecord(
                trade_id=TradeRecord.generate_id(),
                date=trade.exit_date.strftime("%Y-%m-%d"),
                symbol=trade.symbol,
                side=trade.position_type.value,
                size=trade.shares * trade.entry_price,
                shares=trade.shares,
                entry_ts=trade.entry_date.isoformat(),
                exit_ts=trade.exit_date.isoformat(),
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                max_drawdown_pct=None,  # Would need to track during holding
                holding_days=holding_days,
                vix_roc_tier=vix_context.get("vix_roc_tier"),
                vix_roc_signal=vix_context.get("vix_roc_signal"),
                vix_roc_value=vix_context.get("vix_roc_value"),
                vol_regime=vol_context.get("vol_regime"),
                vol_sizing_mult=vol_context.get("vol_sizing_mult"),
                agent_rationale=agent_rationale,
                strategy_name=strategy_name,
                tool_snapshot={},  # Could populate with tool results
                tags=[strategy_name, trade.symbol],
            )
            
            log_trade(record)
            
        except Exception as e:
            # Log but don't crash the trading loop
            print(f"      ⚠️ Failed to log trade to memory: {e}")
    
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
        except Exception as e:
            # If market calendar fails, assume it's a trading day (fail safe)
            print(f"      ⚠️ Market calendar error: {e} - assuming trading day")
            return True
    
    def run_daily_trading(self, force: bool = False, date: datetime = None):
        """Execute trading decisions for a given day
        
        Args:
            force: If True, run even if market is closed or already ran today
            date: Optional specific date to run for (for backfilling)
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Check if market is open today (only for live runs, not backfills)
        if date.date() == datetime.now().date():
            if not self.is_trading_day(date_str):
                if force:
                    print(f"⚠️  Market closed on {date_str} but --force specified, running anyway...")
                else:
                    print(f"Market closed on {date_str} (weekend/holiday)")
                    print("No trading decisions to make. Use --force to run anyway.")
                    return
            
            # Check if already ran (only for today)
            if self.already_ran_today():
                if force:
                    print(f"⚠️  Already ran today but --force specified, running again...")
                else:
                    print(f"Already processed trading for {date_str}")
                    print("Run with --status to see current positions, or --force to run again")
                    return
        
        print(f"\n{'='*80}")
        print(f"FORWARD PAPER TRADING - VIX ROC EDITION - {date_str}")
        print(f"{'='*80}")
        print(f"Trading Day #{self.state['trading_days'] + 1} since {self.state['start_date']}")
        print(f"Symbols: {', '.join(TEST_SYMBOLS)}")
        print(f"Strategies: {', '.join(STRATEGIES.keys())}")
        print(f"{'='*80}\n")
        
        # Fetch latest prices
        print("Fetching prices...")
        current_prices = {}
        for symbol in TEST_SYMBOLS:
            if date.date() == datetime.now().date():
                data = fetch_latest_price(symbol)
                if "error" not in data:
                    current_prices[symbol] = data["price"]
                    print(f"  {symbol}: ${data['price']:.2f}")
                else:
                    print(f"  {symbol}: Error - {data['error']}")
            else:
                # Backfilling - get historical price
                price = fetch_price_on_date(symbol, date)
                if price:
                    current_prices[symbol] = price
                    print(f"  {symbol}: ${price:.2f}")
                else:
                    print(f"  {symbol}: No data for {date_str}")
        
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
                
                # ----- STOP-LOSS CHECK (before strategy signal) -----
                if current_position == PositionType.LONG:
                    pos = portfolio.positions.get(symbol)
                    if pos and pos.stop_loss and price <= pos.stop_loss:
                        trade = portfolio.close_position(symbol, price, date)
                        if trade:
                            stop_reason = (
                                f"STOP-LOSS triggered: price ${price:.2f} <= "
                                f"stop ${pos.stop_loss:.2f} "
                                f"(entry ${pos.entry_price:.2f})"
                            )
                            print(f"    {symbol}: ⛔ {stop_reason}")
                            print(f"      → Closed LONG (P&L: ${trade.pnl:,.2f}, {trade.pnl_pct:.2f}%)")
                            self._log_closed_trade(trade, strategy_name, strategy, stop_reason, date)
                        continue  # Skip normal signal generation for this symbol
                
                # Get historical data
                historical_data = fetch_historical_data(symbol, days=252)
                
                # Generate signal
                try:
                    signal, reason = strategy.generate_signal(
                        symbol=symbol,
                        current_position=current_position,
                        date=date,
                        price=price,
                        historical_data=historical_data
                    )
                    
                    print(f"    {symbol}: {current_position.value} → {signal}")
                    # Display reason (truncate if too long)
                    reason_display = reason[:120] + "..." if len(reason) > 120 else reason
                    print(f"      Reason: {reason_display}")
                    
                    # Execute signal
                    if signal == "buy" and portfolio.cash > 0:
                        target_value = min(portfolio.cash * MAX_POSITION_SIZE, portfolio.cash)
                        
                        # Extract stop-loss from strategy if available
                        stop_loss_price = None
                        target_price = None
                        confidence = 5
                        if hasattr(strategy, 'get_stop_loss'):
                            stop_loss_price = strategy.get_stop_loss(symbol)
                        if hasattr(strategy, 'get_confidence'):
                            confidence = strategy.get_confidence(symbol)
                        # Also check last decision for target
                        if hasattr(strategy, '_last_decisions'):
                            cached = strategy._last_decisions.get(symbol, {})
                            dec = cached.get("decision")
                            if isinstance(dec, dict):
                                rm = dec.get("risk_management", {})
                                target_price = rm.get("target_price")
                                # If no explicit stop from strategy, use default 5% below entry
                                if not stop_loss_price:
                                    stop_pct = rm.get("stop_loss_pct")
                                    if stop_pct and stop_pct > 0:
                                        stop_loss_price = price * (1 - stop_pct / 100.0)
                        
                        # Default stop: 5% below entry if none set
                        if not stop_loss_price:
                            stop_loss_price = price * 0.95
                        
                        success = portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.LONG,
                            date=date,
                            stop_loss=stop_loss_price,
                            target_price=target_price,
                            confidence=confidence,
                        )
                        if success:
                            stop_info = f", stop=${stop_loss_price:.2f}" if stop_loss_price else ""
                            print(f"      → Opened LONG ${target_value:.2f}{stop_info} (conf {confidence}/10)")
                    
                    elif signal == "sell" and current_position == PositionType.LONG:
                        trade = portfolio.close_position(symbol, price, date)
                        if trade:
                            print(f"      → Closed LONG (P&L: ${trade.pnl:,.2f}, {trade.pnl_pct:.2f}%)")
                            # Log to trade memory
                            self._log_closed_trade(trade, strategy_name, strategy, reason, date)
                    
                    elif signal == "short" and portfolio.cash > 0:
                        target_value = min(portfolio.cash * MAX_POSITION_SIZE, portfolio.cash)
                        success = portfolio.open_position(
                            symbol=symbol,
                            price=price,
                            target_value=target_value,
                            position_type=PositionType.SHORT,
                            date=date
                        )
                        if success:
                            print(f"      → Opened SHORT ${target_value:.2f}")
                    
                    elif signal == "cover" and current_position == PositionType.SHORT:
                        trade = portfolio.close_position(symbol, price, date)
                        if trade:
                            print(f"      → Closed SHORT (P&L: ${trade.pnl:,.2f}, {trade.pnl_pct:.2f}%)")
                            # Log to trade memory
                            self._log_closed_trade(trade, strategy_name, strategy, reason, date)
                
                except Exception as e:
                    print(f"    {symbol}: Error - {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update equity curve
            portfolio.update_equity_curve(date, current_prices)
        
        # Update state
        self.state["last_run_date"] = date_str
        self.state["trading_days"] += 1
        
        # Save everything
        self.save_state()
        self.save_daily_performance(date_str, current_prices)
        
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
        # Get latest prices
        current_prices = {}
        for symbol in TEST_SYMBOLS:
            data = fetch_latest_price(symbol)
            if "error" not in data:
                current_prices[symbol] = data["price"]
        
        print(f"\n{'Strategy':<30} {'Equity':>15} {'Return %':>12} {'vs B&H':>10} {'Positions':>10}")
        print("-" * 80)
        
        # Get buy & hold return for comparison
        bh_return = 0
        if "buy_and_hold" in self.portfolios:
            bh_equity = self.portfolios["buy_and_hold"].get_equity(current_prices)
            bh_return = ((bh_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        for strategy_name, portfolio in self.portfolios.items():
            equity = portfolio.get_equity(current_prices)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            # Excess vs buy & hold
            excess = return_pct - bh_return if strategy_name != "buy_and_hold" else 0
            excess_str = f"{excess:+.2f}%" if strategy_name != "buy_and_hold" else "---"
            
            print(f"{strategy_name:<30} ${equity:>14,.2f} {return_pct:>11.2f}% {excess_str:>10} {len(portfolio.positions):>10}")
        
        print()
    
    def show_status(self):
        """Show detailed current status"""
        print(f"\n{'='*80}")
        print(f"FORWARD PAPER TRADING STATUS - VIX ROC EDITION")
        print(f"{'='*80}")
        print(f"Version: {self.state.get('version', '1.0')}")
        print(f"Start Date: {self.state['start_date']}")
        print(f"Last Run: {self.state.get('last_run_date', 'Never')}")
        print(f"Trading Days: {self.state['trading_days']}")
        print(f"Test Symbols: {', '.join(self.state.get('symbols', TEST_SYMBOLS))}")
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
        
        # Show trades
        print(f"\n{'='*80}")
        print("RECENT TRADES")
        print(f"{'='*80}\n")
        
        for strategy_name, portfolio in self.portfolios.items():
            if portfolio.trades:
                print(f"\n{strategy_name}:")
                for trade in portfolio.trades[-5:]:  # Last 5 trades
                    print(f"  {trade.symbol}: {trade.position_type.value} {trade.entry_date.strftime('%Y-%m-%d')} → {trade.exit_date.strftime('%Y-%m-%d')}, P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:.2f}%)")
    
    def backfill(self, days: int = 5):
        """Backfill trading decisions for past N days"""
        print(f"Backfilling {days} trading days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)  # Extra buffer for non-trading days
        
        trading_days = get_trading_days(start_date, end_date)
        trading_days = trading_days[-days:]  # Last N trading days
        
        for day in trading_days:
            if day.date() < datetime.now().date():  # Don't process today yet
                print(f"\n{'='*80}")
                print(f"BACKFILL: {day.strftime('%Y-%m-%d')}")
                print(f"{'='*80}")
                
                self.run_daily_trading(force=True, date=day)
                time.sleep(1)  # Rate limiting for Polygon API
    
    def reset(self):
        """Reset all state and start fresh"""
        print("Resetting all trading state...")
        
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        
        # Also clear logs
        if os.path.exists(self.trades_log):
            os.remove(self.trades_log)
        if os.path.exists(self.performance_log):
            os.remove(self.performance_log)
        
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
        print(f"Reset complete. Fresh start from {self.state['start_date']}")
        print(f"Strategies: {', '.join(STRATEGIES.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Forward Paper Trading - VIX ROC Edition")
    parser.add_argument('--reset', action='store_true', help="Reset and start fresh")
    parser.add_argument('--status', action='store_true', help="Show current status")
    parser.add_argument('--force', action='store_true', help="Force run even if market closed or already ran today")
    parser.add_argument('--backfill', type=int, metavar='N', help="Backfill N trading days of history")
    args = parser.parse_args()
    
    runner = PaperTradingRunner()
    
    if args.reset:
        runner.reset()
    elif args.status:
        runner.show_status()
    elif args.backfill:
        runner.backfill(days=args.backfill)
    else:
        runner.run_daily_trading(force=args.force)


if __name__ == "__main__":
    main()
