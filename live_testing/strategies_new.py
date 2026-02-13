"""
Strategy Implementations - VIX ROC & Reflexion Agent

Updated January 2026 to focus on validated strategies:
- VIX ROC three-tier risk overlay
- Reflexion agent with self-critique
- Buy-and-hold benchmark

Deprecated: Wasserstein, HMM, ML models (no alpha generation)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Literal, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Suppress noisy logs from httpx and llama_index
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("analyze_trade_agent_reflexion").setLevel(logging.WARNING)

from portfolio_tracker import PositionType

# Import VIX ROC production module
try:
    from models.vix_roc_production import (
        VIXROCRiskOverlay,
        AssetClassifier,
        TIER_PARAMS,
        AssetTier
    )
    VIX_ROC_AVAILABLE = True
except ImportError:
    VIX_ROC_AVAILABLE = False
    print("Warning: VIX ROC module not available")

# Import vol prediction
try:
    from vol_prediction_tool import vol_prediction_tool_fn
    VOL_PREDICTION_AVAILABLE = True
except ImportError:
    VOL_PREDICTION_AVAILABLE = False
    print("Warning: Vol prediction module not available")

# Import reflexion agent
try:
    from analyze_trade_agent_reflexion import (
        analyze_trade_reflexion,
        load_vectorstore
    )
    REFLEXION_AVAILABLE = True
except ImportError:
    REFLEXION_AVAILABLE = False
    print("Warning: Reflexion agent not available")


from typing import Tuple

SignalResult = Tuple[str, str]  # (signal, reason)


class BaseStrategy:
    """Base class for all strategies"""
    
    def __init__(self, name: str, long_only: bool = True):
        self.name = name
        self.long_only = long_only
        self._last_reason = {}  # Cache reasons per symbol
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> SignalResult:
        """
        Generate trading signal with reasoning
        
        Args:
            symbol: Stock symbol
            current_position: Current position type (LONG, SHORT, FLAT)
            date: Current date
            price: Current price
            historical_data: Dictionary with price history and other data
            
        Returns:
            Tuple of (signal, reason) where signal is "buy", "sell", "short", "cover", "hold"
        """
        raise NotImplementedError
    
    def get_last_reason(self, symbol: str) -> str:
        """Get the last reason for a symbol"""
        return self._last_reason.get(symbol, "")


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold baseline - THE BENCHMARK"""
    
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
    ) -> SignalResult:
        # Buy once and hold forever
        if current_position == PositionType.FLAT:
            if symbol not in self.has_bought:
                self.has_bought[symbol] = True
                reason = "Buy & hold benchmark - initial purchase"
                self._last_reason[symbol] = reason
                return ("buy", reason)
        reason = "Buy & hold - maintaining position"
        self._last_reason[symbol] = reason
        return ("hold", reason)


class VIXROCStrategy(BaseStrategy):
    """
    VIX Rate-of-Change strategy with three-tier asset classification.
    
    Walk-forward validated: 15/15 wins (2020-2024)
    - Tier 1 (Value/Cyclical): +39% avg excess return
    - Tier 2 (Growth/Tech): +20% avg excess return
    - Tier 3 (Mega-Cap Tech): +140% avg excess return
    """
    
    def __init__(
        self,
        tier: Literal["auto", "tier1", "tier2", "tier3"] = "auto",
        long_only: bool = True
    ):
        name = f"VIX ROC ({tier})"
        super().__init__(name, long_only)
        self.tier_override = tier if tier != "auto" else None
        
        # Lazy initialization
        self._overlay = None
        self._vix_loaded = False
        self._position_state = {}  # Track if we're "out" per symbol
    
    def _ensure_overlay(self):
        """Lazy load VIX ROC overlay with current VIX data"""
        if self._overlay is not None and self._vix_loaded:
            return True
        
        if not VIX_ROC_AVAILABLE:
            return False
        
        try:
            import yfinance as yf
            
            self._overlay = VIXROCRiskOverlay()
            
            # Load recent VIX data
            vix = yf.download("^VIX", period="3mo", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            if not vix.empty:
                self._overlay.load_vix_data(vix)
                self._vix_loaded = True
                return True
            
        except Exception as e:
            print(f"Error loading VIX ROC overlay: {e}")
        
        return False
    
    def get_signal(self, symbol: str) -> Dict[str, Any]:
        """Get VIX ROC signal for symbol"""
        if not self._ensure_overlay():
            return {"signal": "hold", "error": "VIX ROC not available"}
        
        try:
            # Get tier classification
            if self.tier_override:
                # Force specific tier by setting the state before calling get_current_signal
                tier_map = {
                    "tier1": AssetTier.TIER1_VALUE,
                    "tier2": AssetTier.TIER2_GROWTH,
                    "tier3": AssetTier.TIER3_MEGACAP
                }
                tier = tier_map.get(self.tier_override, AssetTier.TIER1_VALUE)
                
                # Create/update state with forced tier
                from models.vix_roc_production import StrategyState
                if symbol not in self._overlay.strategy._states:
                    self._overlay.strategy._states[symbol] = StrategyState(
                        ticker=symbol,
                        tier=tier
                    )
                else:
                    self._overlay.strategy._states[symbol].tier = tier
            
            # Now call get_current_signal which uses the state
            signal_result = self._overlay.get_current_signal(symbol)
            
            return signal_result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"signal": "hold", "error": str(e)}
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> SignalResult:
        """Generate trading signal based on VIX ROC"""
        
        vix_signal = self.get_signal(symbol)
        signal = vix_signal.get("signal", "hold")
        vix_roc = vix_signal.get("vix_roc", 0)
        tier = vix_signal.get("tier", "unknown")
        message = vix_signal.get("message", "")
        
        # Initialize position state
        if symbol not in self._position_state:
            self._position_state[symbol] = "in"  # Start "in" market
        
        # State machine logic
        current_state = self._position_state[symbol]
        
        if signal == "exit":
            # VIX ROC says get out
            self._position_state[symbol] = "out"
            reason = f"VIX ROC EXIT: {tier} - VIX ROC={vix_roc:.1%}. {message}"
            self._last_reason[symbol] = reason
            if current_position == PositionType.LONG:
                return ("sell", reason)
            return ("hold", f"Already flat. {reason}")
        
        elif signal == "reenter":
            # VIX ROC says safe to re-enter
            self._position_state[symbol] = "in"
            reason = f"VIX ROC RE-ENTER: {tier} - VIX ROC={vix_roc:.1%}. {message}"
            self._last_reason[symbol] = reason
            if current_position == PositionType.FLAT:
                return ("buy", reason)
            return ("hold", f"Already long. {reason}")
        
        else:  # signal == "hold"
            if current_state == "in":
                # We're in the market, stay in or enter
                if current_position == PositionType.FLAT:
                    reason = f"VIX ROC SAFE: {tier} - VIX ROC={vix_roc:.1%}. Market stable, entering."
                    self._last_reason[symbol] = reason
                    return ("buy", reason)
                reason = f"VIX ROC HOLD: {tier} - VIX ROC={vix_roc:.1%}. Maintaining position."
                self._last_reason[symbol] = reason
                return ("hold", reason)
            else:
                # We're out, stay out
                reason = f"VIX ROC OUT: {tier} - VIX ROC={vix_roc:.1%}. Waiting for re-entry signal."
                self._last_reason[symbol] = reason
                return ("hold", reason)


class VIXROCVolAdjustedStrategy(VIXROCStrategy):
    """
    VIX ROC strategy with vol prediction position sizing.
    
    Combines:
    - VIX ROC for market timing (IN/OUT signal)
    - Vol prediction for position sizing (spike probability)
    """
    
    def __init__(
        self,
        tier: Literal["auto", "tier1", "tier2", "tier3"] = "auto",
        long_only: bool = True
    ):
        super().__init__(tier, long_only)
        self.name = f"VIX ROC + Vol ({tier})"
        self._vol_state = {}  # Track vol-adjusted sizing
    
    def get_vol_adjustment(self, symbol: str) -> float:
        """
        Get position size multiplier based on vol prediction.
        
        Returns:
            Multiplier: 1.0 = normal, 0.5-0.7 = reduced, 1.2 = can add
        """
        if not VOL_PREDICTION_AVAILABLE:
            return 1.0
        
        try:
            state = {"tool_results": {}}
            state = vol_prediction_tool_fn(state, {"symbol": symbol})
            result = state["tool_results"].get("vol_prediction", {})
            
            if result.get("error"):
                return 1.0
            
            regime = result.get("current_regime", "LOW")
            
            if regime == "LOW":
                spike_prob = result.get("spike_probability", 0)
                if spike_prob >= 0.6:
                    return 0.5  # Reduce 50%
                elif spike_prob >= 0.5:
                    return 0.7  # Reduce 30%
                else:
                    return 1.0  # Normal sizing
            else:  # HIGH regime
                calm_prob = result.get("calm_probability", 0)
                if calm_prob >= 0.5:
                    return 1.2  # Can add to position
                else:
                    return 0.6  # Stay reduced
                    
        except Exception as e:
            print(f"Vol prediction error for {symbol}: {e}")
            return 1.0
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> SignalResult:
        """Generate signal with vol-adjusted context"""
        # Get base VIX ROC signal
        signal, base_reason = super().generate_signal(
            symbol, current_position, date, price, historical_data
        )
        
        # Get vol adjustment
        vol_mult = self.get_vol_adjustment(symbol)
        self._vol_state[symbol] = vol_mult
        
        # Add vol adjustment info to reason
        if vol_mult != 1.0:
            vol_note = f" Vol sizing: {vol_mult:.0%}"
            reason = base_reason + vol_note
        else:
            reason = base_reason
        
        self._last_reason[symbol] = reason
        return (signal, reason)
    
    def get_position_size_multiplier(self, symbol: str) -> float:
        """Get the last computed vol multiplier for position sizing"""
        return self._vol_state.get(symbol, 1.0)


class ReflexionAgentStrategy(BaseStrategy):
    """
    Reflexion Agent strategy with self-critique pattern.
    
    4-step process:
    1. GENERATE: Initial analysis
    2. EVALUATE: Self-critique
    3. REFLECT: Extract learnings
    4. REFINE: Final decision
    """
    
    def __init__(self, long_only: bool = True, use_rag: bool = True):
        super().__init__("Reflexion Agent", long_only)
        self.use_rag = use_rag
        self._vectordb = None
        self._last_decisions = {}
    
    def _ensure_vectordb(self):
        """Lazy load vectordb for RAG"""
        if self._vectordb is None and self.use_rag:
            try:
                self._vectordb = load_vectorstore()
            except Exception as e:
                print(f"Warning: Could not load vectordb: {e}")
        return self._vectordb
    
    def generate_signal(
        self,
        symbol: str,
        current_position: PositionType,
        date: datetime,
        price: float,
        historical_data: Dict
    ) -> SignalResult:
        """Generate signal using reflexion agent with structured output."""
        
        if not REFLEXION_AVAILABLE:
            reason = "Reflexion agent not available"
            self._last_reason[symbol] = reason
            return ("hold", reason)
        
        try:
            # Build trading idea based on current position
            if current_position == PositionType.LONG:
                trading_idea = f"Should I continue holding my long position in {symbol} at ${price:.2f}?"
            elif current_position == PositionType.SHORT:
                trading_idea = f"Should I cover my short position in {symbol} at ${price:.2f}?"
            else:
                trading_idea = f"Is {symbol} a good buy at ${price:.2f}?"
            
            # Run reflexion agent
            vectordb = self._ensure_vectordb()
            result = analyze_trade_reflexion(
                trading_idea=trading_idea,
                symbol=symbol,
                vectordb=vectordb,
                verbose=False
            )
            
            # Parse the final decision (structured dict or legacy text)
            final_decision = result.get("final_decision", "")
            signal, reason = self._parse_decision(final_decision, current_position)
            
            # Cache the decision with structured data for stop-loss use
            self._last_decisions[symbol] = {
                "date": date,
                "decision": final_decision if isinstance(final_decision, dict) else final_decision[:500],
                "signal": signal,
                "reason": reason,
            }
            
            self._last_reason[symbol] = reason
            return (signal, reason)
            
        except Exception as e:
            reason = f"Reflexion agent error: {e}"
            self._last_reason[symbol] = reason
            return ("hold", reason)
    
    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """
        Return the stop-loss price from the last structured decision for a symbol.
        Returns None if no stop was set or decision was unstructured.
        """
        cached = self._last_decisions.get(symbol, {})
        decision = cached.get("decision")
        if isinstance(decision, dict):
            rm = decision.get("risk_management", {})
            return rm.get("stop_loss")
        return None
    
    def get_confidence(self, symbol: str) -> int:
        """Return the confidence score (1-10) from the last decision, default 5."""
        cached = self._last_decisions.get(symbol, {})
        decision = cached.get("decision")
        if isinstance(decision, dict):
            return decision.get("confidence", 5)
        return 5

    def _parse_decision(self, decision, current_position: PositionType) -> SignalResult:
        """
        Parse the agent's decision into a trading signal.
        
        Accepts either:
        - dict (structured JSON from the new schema)  
        - str (legacy free-text, fallback)
        """
        # ---- Structured dict path (preferred) ----
        if isinstance(decision, dict):
            verdict = decision.get("verdict", "UNCLEAR")
            confidence = decision.get("confidence", 5)
            summary = decision.get("summary", "")[:150]
            
            if verdict == "NOT_ATTRACTIVE":
                if current_position == PositionType.LONG:
                    return ("sell", f"Agent: NOT ATTRACTIVE (conf {confidence}/10) - {summary}")
                return ("hold", f"Agent: NOT ATTRACTIVE (already flat, conf {confidence}/10)")
            
            elif verdict == "ATTRACTIVE":
                if confidence <= 3:
                    # Low confidence = don't trade even if attractive
                    return ("hold", f"Agent: ATTRACTIVE but low confidence ({confidence}/10) - holding")
                if current_position == PositionType.FLAT:
                    return ("buy", f"Agent: ATTRACTIVE (conf {confidence}/10) - {summary}")
                return ("hold", f"Agent: ATTRACTIVE (already long, conf {confidence}/10)")
            
            else:  # UNCLEAR
                return ("hold", f"Agent: UNCLEAR (conf {confidence}/10) - {summary}")
        
        # ---- Legacy free-text path (fallback) ----
        decision_lower = decision.lower()
        
        reason_text = decision
        if "verdict:" in decision_lower:
            verdict_idx = decision_lower.find("verdict:")
            reason_text = decision[verdict_idx:verdict_idx+200]
        
        reason_text = reason_text.replace("\n", " ").strip()
        if len(reason_text) > 150:
            reason_text = reason_text[:147] + "..."
        
        if "verdict:" in decision_lower:
            if "not attractive" in decision_lower:
                if current_position == PositionType.LONG:
                    return ("sell", f"Agent: NOT ATTRACTIVE - {reason_text}")
                return ("hold", f"Agent: NOT ATTRACTIVE (already flat) - {reason_text}")
            
            elif "attractive" in decision_lower:
                if current_position == PositionType.FLAT:
                    return ("buy", f"Agent: ATTRACTIVE - {reason_text}")
                return ("hold", f"Agent: ATTRACTIVE (already long) - {reason_text}")
            
            elif "unclear" in decision_lower:
                return ("hold", f"Agent: UNCLEAR - {reason_text}")
        
        # Look for explicit signals with negation awareness
        _neg = r"(?<!not\s)(?<!don'?t\s)(?<!no\s)"
        if re.search(_neg + r"\b(exit|sell|close)\b", decision_lower):
            if current_position == PositionType.LONG:
                return ("sell", f"Agent exit signal - {reason_text}")
        
        if re.search(_neg + r"\b(buy|enter|go\s+long)\b", decision_lower):
            if current_position == PositionType.FLAT:
                return ("buy", f"Agent buy signal - {reason_text}")
        
        return ("hold", f"Agent: No clear signal - {reason_text}")


def create_strategy(config: Dict) -> BaseStrategy:
    """Factory function to create strategy from config"""
    strategy_type = config.get("type", "baseline")
    
    if strategy_type == "baseline":
        return BuyAndHoldStrategy()
    
    elif strategy_type == "vix_roc":
        tier = config.get("tier", "auto")
        long_only = config.get("long_only", True)
        return VIXROCStrategy(tier=tier, long_only=long_only)
    
    elif strategy_type == "vix_roc_vol":
        tier = config.get("tier", "auto")
        long_only = config.get("long_only", True)
        return VIXROCVolAdjustedStrategy(tier=tier, long_only=long_only)
    
    elif strategy_type == "reflexion_agent":
        long_only = config.get("long_only", True)
        use_rag = config.get("use_rag", True)
        return ReflexionAgentStrategy(long_only=long_only, use_rag=use_rag)
    
    else:
        print(f"Warning: Unknown strategy type '{strategy_type}', using buy & hold")
        return BuyAndHoldStrategy()


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing strategies...")
    print("=" * 60)
    
    # Test VIX ROC strategy
    vix_strategy = VIXROCStrategy(tier="auto")
    
    test_symbols = ["SPY", "QQQ", "NVDA", "JPM"]
    
    for symbol in test_symbols:
        signal = vix_strategy.get_signal(symbol)
        print(f"{symbol}: {signal.get('signal', 'N/A')} - {signal.get('message', '')[:50]}")
    
    print("\n" + "=" * 60)
    print("Strategy factory test:")
    
    from config_new import STRATEGIES
    
    for name, config in STRATEGIES.items():
        strategy = create_strategy(config)
        print(f"  {name}: {strategy.name}")
