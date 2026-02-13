"""
Portfolio Tracker

Tracks positions, cash, equity, trades, and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


@dataclass
class Position:
    """Represents a stock position"""
    symbol: str
    shares: float  # Positive for long, negative for short
    entry_price: float
    entry_date: datetime
    position_type: PositionType
    stop_loss: Optional[float] = None     # Price level for stop-loss exit
    target_price: Optional[float] = None  # Price level for profit target exit
    confidence: int = 5                   # Agent conviction score (1-10)
    
    @property
    def market_value(self) -> float:
        """Current market value (always positive)"""
        return abs(self.shares) * self.entry_price
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.position_type == PositionType.LONG:
            return self.shares * (current_price - self.entry_price)
        elif self.position_type == PositionType.SHORT:
            return abs(self.shares) * (self.entry_price - current_price)
        return 0.0


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    position_type: PositionType
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float


class PortfolioTracker:
    """
    Tracks portfolio state over time
    """
    
    def __init__(
        self,
        initial_capital: float,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 5.0,
        strategy_name: str = "unknown"
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.strategy_name = strategy_name
        
        # Current state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # History
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (cash + positions)"""
        position_value = sum(
            pos.shares * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of position"""
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        return pos.shares * current_price
    
    def get_position_type(self, symbol: str) -> PositionType:
        """Check current position type"""
        if symbol not in self.positions:
            return PositionType.FLAT
        return self.positions[symbol].position_type
    
    def calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage cost"""
        slippage_multiplier = self.slippage_bps / 10000.0
        if side in [OrderSide.BUY, OrderSide.COVER]:
            return price * slippage_multiplier  # Pay more when buying
        else:
            return -price * slippage_multiplier  # Receive less when selling
    
    def open_position(
        self,
        symbol: str,
        price: float,
        target_value: float,
        position_type: PositionType,
        date: datetime,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        confidence: int = 5,
    ) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Stock symbol
            price: Current price
            target_value: Dollar value to invest
            position_type: LONG or SHORT
            date: Trade date
            stop_loss: Stop-loss price level (optional)
            target_price: Profit target price level (optional)
            confidence: Agent conviction score 1-10 (default 5)
            
        Returns:
            True if successful, False if insufficient capital
        """
        # Calculate costs
        slippage = self.calculate_slippage(
            price,
            OrderSide.BUY if position_type == PositionType.LONG else OrderSide.SHORT
        )
        commission = self.commission_per_trade
        effective_price = price + slippage
        
        # Calculate shares
        shares = target_value / effective_price
        total_cost = (shares * effective_price) + commission
        
        # Check if we have enough capital
        if position_type == PositionType.LONG:
            if total_cost > self.cash:
                return False
            self.cash -= total_cost
        else:  # SHORT
            # For shorting, we receive cash but need margin
            self.cash += (shares * effective_price) - commission
        
        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares if position_type == PositionType.LONG else -shares,
            entry_price=effective_price,
            entry_date=date,
            position_type=position_type,
            stop_loss=stop_loss,
            target_price=target_price,
            confidence=confidence,
        )
        
        return True
    
    def close_position(self, symbol: str, price: float, date: datetime) -> Optional[Trade]:
        """
        Close an existing position
        
        Returns:
            Trade object if successful, None if no position exists
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Calculate costs
        slippage = self.calculate_slippage(
            price,
            OrderSide.SELL if pos.position_type == PositionType.LONG else OrderSide.COVER
        )
        commission = self.commission_per_trade
        effective_price = price + slippage
        
        # Calculate P&L
        if pos.position_type == PositionType.LONG:
            pnl = pos.shares * (effective_price - pos.entry_price) - commission
            self.cash += pos.shares * effective_price - commission
        else:  # SHORT
            pnl = abs(pos.shares) * (pos.entry_price - effective_price) - commission
            self.cash -= abs(pos.shares) * effective_price + commission
        
        pnl_pct = (pnl / (abs(pos.shares) * pos.entry_price)) * 100
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=effective_price,
            shares=abs(pos.shares),
            position_type=pos.position_type,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission * 2,  # Entry + exit
            slippage=abs(slippage) * abs(pos.shares) * 2
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        return trade
    
    def update_equity_curve(self, date: datetime, prices: Dict[str, float]):
        """Record equity at this point in time"""
        equity = self.get_equity(prices)
        
        # Update max drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions_value": equity - self.cash,
            "drawdown": drawdown,
            "num_positions": len(self.positions)
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return {}
        
        final_equity = self.equity_curve[-1]["equity"]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate returns series
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]["equity"]
            curr_equity = self.equity_curve[i]["equity"]
            ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            returns.append(ret)
        
        # Annualized metrics
        days = len(self.equity_curve)
        years = days / 252.0
        annualized_return = (pow(final_equity / self.initial_capital, 1/years) - 1) if years > 0 else 0
        
        # Risk metrics
        returns_array = np.array(returns)
        volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 0 else 0
        sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free
        
        # Downside volatility (for Sortino)
        downside_returns = returns_array[returns_array < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annualized_return - 0.02) / downside_vol if downside_vol > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        # Calmar ratio
        calmar = annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        return {
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "calmar_ratio": calmar,
            "num_trades": len(self.trades),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_duration": np.mean([(t.exit_date - t.entry_date).days for t in self.trades]) if self.trades else 0,
            "total_commission": sum(t.commission for t in self.trades),
            "total_slippage": sum(t.slippage for t in self.trades),
        }
