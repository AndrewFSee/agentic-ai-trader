"""
Strategy Implementations

Defines different trading strategies for backtesting and live trading.

Fixes applied (2026-01-02):
1. Uses ACTUAL regime detection models instead of simplified volatility thresholds
2. Proper HMM initialization with correct feature format
3. Exit positions on bearish signals (long-only strategies can sell, just not short)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import Dict, Optional, Literal
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import regime detection models
from models.paper_wasserstein_regime_detection import (
    RollingPaperWassersteinDetector,
    PaperWassersteinKMeans,
    fetch_polygon_bars,
    calculate_features
)
from models.rolling_hmm_regime_detection import RollingWindowHMM

# Import ML prediction tool
from ml_prediction_tool import get_ml_prediction

# Import agent (for full agent decisions with RAG)
from analyze_trade_agent import analyze_trade_agent, load_vectorstore
import re

from portfolio_tracker import PositionType


class BaseStrategy:
    """Base class for all strategies"""
    
    def __init__(self, name: str, long_only: bool = True):
        self.name = name
        self.long_only = long_only
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> Literal["buy", "sell", "short", "cover", "hold"]:
        """
        Generate trading signal
        
        Args:
            symbol: Stock symbol
            current_position: Current position type (LONG, SHORT, FLAT)
            date: Current date
            price: Current price
            historical_data: Dictionary with price history and other data
            
        Returns:
            Signal: "buy", "sell", "short", "cover", "hold"
        """
        raise NotImplementedError


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold baseline"""
    
    def __init__(self):
        super().__init__("Buy & Hold", long_only=True)
        self.has_bought = {}
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> str:
        # Buy once and hold forever
        if current_position == PositionType.FLAT:
            if symbol not in self.has_bought:
                self.has_bought[symbol] = True
                return "buy"
        return "hold"


class RegimeStrategy(BaseStrategy):
    """Strategy based on ACTUAL regime detection models (not simplified thresholds)"""
    
    def __init__(
        self,
        model_type: Literal["wasserstein", "hmm"],
        long_only: bool = True,
        lookback_days: int = 252,
        window_size: int = 20  # For Wasserstein distributions
    ):
        name = f"{model_type.upper()} {'Long' if long_only else 'Long/Short'}"
        super().__init__(name, long_only)
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.window_size = window_size
        self.detector = None
        self.last_regime = {}  # Track regime changes per symbol
        self.scaler = None
        self._trained = False
    
    def _prepare_features(self, prices: list, volumes: list = None) -> pd.DataFrame:
        """Prepare features for regime detection from price history"""
        df = pd.DataFrame({'close': prices})
        if volumes:
            df['volume'] = volumes
        else:
            df['volume'] = 1000000  # Default volume if not provided
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Realized volatility (20-day rolling std)
        df['realized_vol'] = df['returns'].rolling(20).std()
        
        # Trend strength
        df['trend_strength'] = df['returns'].rolling(20).mean().abs()
        
        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_momentum'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1e-10)
        
        # Momentum
        df['momentum_10d'] = df['close'].pct_change(10)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # High-low range proxy (use returns as proxy)
        df['hl_range'] = df['returns'].abs()
        
        # Vol-adjusted returns
        df['vol_adj_returns'] = df['returns'] / (df['realized_vol'] + 1e-10)
        
        df = df.dropna()
        return df
    
    def get_regime(self, symbol: str, date: datetime, historical_data: Dict) -> Optional[str]:
        """Get current regime classification using ACTUAL models"""
        try:
            prices = historical_data.get("prices", [])
            volumes = historical_data.get("volumes", [])
            
            if len(prices) < 100:  # Need enough history
                return None
            
            # Prepare features
            df = self._prepare_features(prices, volumes if volumes else None)
            
            if len(df) < self.window_size * 3:  # Need at least 3 windows
                return None
            
            if self.model_type == "wasserstein":
                return self._get_wasserstein_regime(df, symbol)
            elif self.model_type == "hmm":
                return self._get_hmm_regime(df, symbol)
            
        except Exception as e:
            print(f"Error getting regime for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return None
    
    def _get_wasserstein_regime(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """Get regime using actual Wasserstein k-means clustering"""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Feature columns for Wasserstein
            feature_cols = ['realized_vol', 'trend_strength', 'momentum_20d']
            features = df[feature_cols].values
            
            # Scale features
            if self.scaler is None:
                self.scaler = StandardScaler()
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)
            
            # Initialize or use existing model
            if self.detector is None or not self._trained:
                self.detector = PaperWassersteinKMeans(
                    n_regimes=3,
                    window_size=self.window_size,
                    max_iter=50,
                    random_state=42
                )
                # Fit on all available data
                self.detector.fit(features_scaled, verbose=False)
                self._trained = True
            
            # Get latest window and predict regime
            if len(features_scaled) >= self.window_size:
                latest_window = features_scaled[-self.window_size:]
                regime_idx = self.detector.predict_distribution(latest_window)
                
                # Map to volatility-based regime names
                # 0 = low_vol, 1 = medium_vol, 2 = high_vol
                regime_map = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}
                return regime_map.get(regime_idx, "medium_vol")
            
        except Exception as e:
            print(f"Wasserstein regime error for {symbol}: {e}")
            return None
        
        return None
    
    def _get_hmm_regime(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """Get regime using actual HMM model"""
        try:
            # HMM feature columns
            feature_cols = ['vol_adj_returns', 'realized_vol', 'trend_strength']
            
            # Check if features exist
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < 2:
                available_cols = ['realized_vol', 'returns']
            
            features = df[available_cols].values
            
            # Initialize or use existing HMM
            if self.detector is None or not self._trained:
                self.detector = RollingWindowHMM(
                    n_regimes=3,
                    training_window_days=min(252, len(features) - 1),
                    retrain_frequency_days=126,
                    persistence_strength=0.90
                )
                # Train on available data
                self.detector.train_on_window(features_array=features)
                self._trained = True
            
            # Predict using forward filter (no backward pass = no relabeling)
            result = self.detector.predict_forward_filter(features_array=features[-50:])
            state_idx = result.get('most_likely_state', 1)
            
            # Map state to regime (0=bearish, 1=sideways, 2=bullish)
            state_mapping = {0: "bearish", 1: "sideways", 2: "bullish"}
            return state_mapping.get(state_idx, "sideways")
            
        except Exception as e:
            print(f"HMM regime error for {symbol}: {e}")
            return None
        
        return None
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> str:
        regime = self.get_regime(symbol, date, historical_data)
        
        if regime is None:
            return "hold"
        
        # Track regime changes for logging
        old_regime = self.last_regime.get(symbol)
        if old_regime != regime:
            print(f"      Regime change for {symbol}: {old_regime} -> {regime}")
            self.last_regime[symbol] = regime
        
        # Wasserstein regimes (volatility-based)
        if self.model_type == "wasserstein":
            if regime == "low_vol":
                # Low volatility = good for long positions
                if current_position == PositionType.FLAT:
                    return "buy"
                elif current_position == PositionType.SHORT and not self.long_only:
                    return "cover"
                return "hold"
                
            elif regime == "high_vol":
                # High volatility = exit longs or go short
                if current_position == PositionType.LONG:
                    return "sell"  # Long-only CAN sell to exit
                elif current_position == PositionType.FLAT and not self.long_only:
                    return "short"
                return "hold"
                
            else:  # medium_vol
                # Neutral regime - hold current position
                return "hold"
        
        # HMM regimes (trend-based)
        elif self.model_type == "hmm":
            if regime == "bullish":
                # Bullish trend = go long
                if current_position == PositionType.FLAT:
                    return "buy"
                elif current_position == PositionType.SHORT and not self.long_only:
                    return "cover"
                return "hold"
                
            elif regime == "bearish":
                # Bearish trend = exit longs or go short
                if current_position == PositionType.LONG:
                    return "sell"  # Long-only CAN sell to exit
                elif current_position == PositionType.FLAT and not self.long_only:
                    return "short"
                return "hold"
                
            else:  # sideways
                # Neutral regime - hold current position
                return "hold"
        
        return "hold"


class MLStrategy(BaseStrategy):
    """Strategy based on ML consensus predictions"""
    
    def __init__(
        self,
        long_only: bool = True,
        confidence_threshold: float = 0.6
    ):
        name = f"ML Consensus {'Long' if long_only else 'Long/Short'}"
        super().__init__(name, long_only)
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> str:
        try:
            # Get ML prediction (5-day horizon matches training)
            prediction = get_ml_prediction(symbol, horizon=5)
            
            if not prediction or "error" in prediction:
                error_msg = prediction.get("error", "Unknown error") if prediction else "No prediction returned"
                print(f"      ML Error: {error_msg}")
                # Log to file for debugging
                import os
                log_dir = os.path.join(os.path.dirname(__file__), "trading_logs")
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "ml_predictions.log"), "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {symbol} - ERROR: {error_msg}\n")
                return "hold"
            
            consensus = prediction.get("consensus", {})
            consensus_direction = consensus.get("direction", "").upper()  # "STRONG UP", "WEAK DOWN", etc.
            confidence = consensus.get("confidence", 0)  # 0 to 1.0
            up_votes = consensus.get("up_votes", 0)
            total_votes = consensus.get("total_votes", 0)
            
            # Log to file for debugging (avoid encoding issues in console)
            import os
            log_dir = os.path.join(os.path.dirname(__file__), "trading_logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "ml_predictions.log"), "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: {symbol} - {consensus_direction} | Votes: {up_votes}/{total_votes} | "
                       f"Confidence: {confidence:.3f} | Threshold: {self.confidence_threshold} | "
                       f"Position: {current_position.value} | Long-only: {self.long_only}\n")
            
            # Debug output (safe characters only)
            print(f"      ML: {consensus_direction} (conf: {confidence:.2f}, thresh: {self.confidence_threshold}, votes: {up_votes}/{total_votes})")
            
            # Only act if confidence is high enough
            if confidence < self.confidence_threshold:
                print(f"      -> Confidence too low, holding")
                return "hold"
            
            # Parse direction from consensus string
            is_up = "UP" in consensus_direction
            is_strong = "STRONG" in consensus_direction
            
            # Bullish signal (any UP consensus)
            if is_up:
                if current_position == PositionType.FLAT:
                    return "buy"
                elif current_position == PositionType.SHORT and not self.long_only:
                    return "cover"
                # If already long, hold
                return "hold"
            
            # Bearish signal (any DOWN consensus)
            elif "DOWN" in consensus_direction:
                if current_position == PositionType.LONG:
                    return "sell"
                elif current_position == PositionType.FLAT and not self.long_only:
                    return "short"
                # If already short, hold
                return "hold"
            
        except Exception as e:
            print(f"      ML Exception: {e}")
        
        return "hold"


class MLSingleModelStrategy(BaseStrategy):
    """Strategy based on a single ML model prediction"""
    
    def __init__(
        self,
        model_name: str,
        long_only: bool = True,
        confidence_threshold: float = 0.6
    ):
        name = f"{model_name} {'Long' if long_only else 'Long/Short'}"
        super().__init__(name, long_only)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> str:
        try:
            # Get ML prediction (5-day horizon, single model)
            prediction = get_ml_prediction(symbol, horizon=5, models=[self.model_name])
            
            if not prediction or "error" in prediction:
                error_msg = prediction.get("error", "Unknown error") if prediction else "No prediction returned"
                print(f"      {self.model_name} Error: {error_msg}")
                return "hold"
            
            predictions = prediction.get("predictions", {})
            model_pred = predictions.get(self.model_name, {})
            
            if "error" in model_pred:
                print(f"      {self.model_name} Error: {model_pred['error']}")
                return "hold"
            
            direction = model_pred.get("direction", 0)  # 0 or 1
            probability = model_pred.get("probability", {})
            
            if not probability:
                print(f"      {self.model_name}: No probability available")
                return "hold"
            
            confidence = probability.get("confidence", 0)
            
            # Debug output
            pred_label = "UP" if direction == 1 else "DOWN"
            print(f"      {self.model_name}: {pred_label} (conf: {confidence:.2f}, thresh: {self.confidence_threshold})")
            
            # Only act if confidence is high enough
            if confidence < self.confidence_threshold:
                print(f"      -> Confidence too low, holding")
                return "hold"
            
            # Bullish signal (direction = 1)
            if direction == 1:
                if current_position == PositionType.FLAT:
                    return "buy"
                elif current_position == PositionType.SHORT and not self.long_only:
                    return "cover"
                return "hold"
            
            # Bearish signal (direction = 0)
            else:
                if current_position == PositionType.LONG:
                    return "sell"
                elif current_position == PositionType.FLAT and not self.long_only:
                    return "short"
                return "hold"
            
        except Exception as e:
            print(f"      {self.model_name} Exception: {e}")
        
        return "hold"


class AgentStrategy(BaseStrategy):
    """
    Strategy based on full agent decisions with RAG.
    
    This strategy uses the complete trading agent pipeline:
    - RAG search over trading books (vectorstore)
    - Dynamic tool selection via planner (price, technicals, sentiment, regimes, ML)
    - LLM synthesis of all inputs into a trading decision
    
    The agent considers:
    - Price action and technical indicators (RSI, MACD, Bollinger Bands, ATR)
    - News sentiment (FinBERT analysis)
    - Regime detection (Wasserstein volatility + HMM trend)
    - ML model predictions (Random Forest, XGBoost, etc.)
    - Trading book wisdom (risk management, position sizing, psychology)
    """
    
    def __init__(
        self,
        long_only: bool = True,
        use_rag: bool = True,
        timeout: int = 30
    ):
        name = f"Agent {'Long' if long_only else 'Long/Short'}"
        super().__init__(name, long_only)
        self.use_rag = use_rag
        self.timeout = timeout
        self.vectordb = None  # Lazy load
    
    def _extract_signal_from_verdict(self, analysis: str) -> str:
        """
        Extract trading signal from agent's verdict.
        
        More robust extraction that handles various formats.
        """
        analysis_upper = analysis.upper()
        
        # Look for verdict line with multiple patterns
        verdict_patterns = [
            r"VERDICT:\s*(.+?)(?:\n|$)",
            r"\*\*VERDICT\*\*:\s*(.+?)(?:\n|$)",
            r"VERDICT\s*[-–—:]\s*(.+?)(?:\n|$)",
        ]
        
        verdict = None
        for pattern in verdict_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                verdict = match.group(1).strip().upper()
                break
        
        # Fallback: look for key phrases in the full text
        if not verdict:
            if "NOT ATTRACTIVE" in analysis_upper:
                return "avoid"
            elif "UNCLEAR" in analysis_upper or "NEEDS MORE INFORMATION" in analysis_upper:
                return "unclear"
            elif "ATTRACTIVE IF STRICT RULES" in analysis_upper:
                # Look for directional hints
                if any(word in analysis_upper for word in ["BULLISH", "BUY", "LONG POSITION", "GO LONG"]):
                    return "buy"
                elif any(word in analysis_upper for word in ["BEARISH", "SHORT", "SELL"]):
                    return "short"
                return "buy"  # Default to buy if attractive
            return "unclear"
        
        # Parse the verdict
        if "NOT ATTRACTIVE" in verdict:
            return "avoid"
        elif "UNCLEAR" in verdict or "NEEDS MORE" in verdict:
            return "unclear"
        elif "ATTRACTIVE" in verdict:
            # Look for directional bias in verdict or surrounding context
            if any(word in verdict for word in ["SHORT", "SELL", "BEARISH"]):
                return "short"
            elif any(word in verdict for word in ["LONG", "BUY", "BULLISH"]):
                return "buy"
            else:
                # Check surrounding context for direction
                if "BEARISH" in analysis_upper or "DOWNSIDE" in analysis_upper:
                    return "short"
                return "buy"  # Default to buy if attractive
        
        return "unclear"
    
    def _build_query(
        self,
        symbol: str,
        current_position: PositionType,
        price: float,
        historical_data: Dict
    ) -> str:
        """Build a detailed query for the agent with context."""
        prices = historical_data.get("prices", [])
        
        # Calculate some basic context
        context_parts = []
        
        if len(prices) >= 2:
            recent_return = (prices[-1] - prices[-2]) / prices[-2] * 100
            context_parts.append(f"Today's return: {recent_return:+.2f}%")
        
        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            price_vs_ma = (price - ma20) / ma20 * 100
            trend = "above" if price > ma20 else "below"
            context_parts.append(f"Price is {trend} 20-day MA by {abs(price_vs_ma):.1f}%")
        
        if len(prices) >= 5:
            week_return = (prices[-1] - prices[-5]) / prices[-5] * 100
            context_parts.append(f"5-day return: {week_return:+.2f}%")
        
        context_str = ". ".join(context_parts) if context_parts else ""
        
        # Build position-specific query
        if current_position == PositionType.LONG:
            query = f"""I have a LONG position in {symbol} at current price ${price:.2f}. {context_str}
            
Should I HOLD this position or EXIT (sell)? Consider:
- Current market regime (volatility and trend)
- Technical indicators (RSI, MACD, Bollinger Bands)  
- ML model predictions for near-term direction
- News sentiment and any recent catalysts
- Risk management based on ATR for stop placement"""

        elif current_position == PositionType.SHORT:
            query = f"""I have a SHORT position in {symbol} at current price ${price:.2f}. {context_str}
            
Should I HOLD this short or COVER (exit)? Consider:
- Current market regime (volatility and trend)
- Technical indicators and momentum
- ML model predictions
- News sentiment"""

        else:  # FLAT
            query = f"""Analyzing {symbol} at ${price:.2f} for a potential new position. {context_str}

Should I initiate a position in {symbol}? If so, should it be LONG or SHORT? Consider:
- Current market regime (both volatility and trend)
- Technical setup (RSI, MACD, Bollinger Bands, trend strength)
- ML model consensus and confidence level
- News sentiment and recent catalysts
- Proper position sizing based on regime and ML confidence
- Stop loss placement using ATR"""

        return query
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> str:
        try:
            # Lazy load vectorstore
            if self.vectordb is None:
                print(f"  {symbol}: Loading RAG vectorstore...")
                self.vectordb = load_vectorstore()
            
            # Build detailed query with context
            query = self._build_query(symbol, current_position, price, historical_data)
            
            print(f"  {symbol}: Calling agent (this may take 30-60 seconds)...")
            
            # Call the full agent
            analysis = analyze_trade_agent(
                trading_idea=query,
                symbol=symbol,
                vectordb=self.vectordb,
                k_main=4,  # Reduced to speed up
                k_rules=4   # Reduced to speed up
            )
            
            # Extract signal from verdict
            raw_signal = self._extract_signal_from_verdict(analysis)
            print(f"  {symbol}: Agent verdict → {raw_signal}")
            
            # Map to portfolio action based on current position
            if raw_signal == "buy":
                if current_position == PositionType.FLAT:
                    return "buy"
                elif current_position == PositionType.SHORT and not self.long_only:
                    return "cover"
                return "hold"
            
            elif raw_signal == "short":
                if current_position == PositionType.LONG:
                    return "sell"  # Exit long before shorting
                elif current_position == PositionType.FLAT and not self.long_only:
                    return "short"
                return "hold"
            
            elif raw_signal == "avoid":
                # Exit if we have a position
                if current_position == PositionType.LONG:
                    return "sell"
                elif current_position == PositionType.SHORT:
                    return "cover"
                return "hold"
            
            else:  # unclear or unknown
                # Hold current position if any
                return "hold"
                
        except Exception as e:
            print(f"  {symbol}: ERROR in agent - {e}")
            import traceback
            traceback.print_exc()
        
        return "hold"


# Strategy factory
def create_strategy(strategy_config: Dict) -> BaseStrategy:
    """Create strategy instance from config"""
    strategy_type = strategy_config.get("type")
    long_only = strategy_config.get("long_only", True)
    
    if strategy_type == "baseline":
        return BuyAndHoldStrategy()
    
    elif strategy_type == "regime":
        model = strategy_config.get("model", "wasserstein")
        return RegimeStrategy(model_type=model, long_only=long_only)
    
    elif strategy_type == "ml":
        confidence = strategy_config.get("confidence_threshold", 0.6)
        return MLStrategy(long_only=long_only, confidence_threshold=confidence)
    
    elif strategy_type == "single_ml":
        model_name = strategy_config.get("model")
        confidence = strategy_config.get("confidence_threshold", 0.6)
        return MLSingleModelStrategy(model_name=model_name, long_only=long_only, confidence_threshold=confidence)
    
    elif strategy_type == "agent":
        use_rag = strategy_config.get("use_rag", True)
        timeout = strategy_config.get("timeout", 30)
        return AgentStrategy(long_only=long_only, use_rag=use_rag, timeout=timeout)
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
