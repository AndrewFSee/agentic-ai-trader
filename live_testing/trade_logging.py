"""
Trade Logging Module

Provides structured trade records with context for building trade memory.
Trades are logged to JSONL for easy ingestion into LlamaIndex.

Usage:
    from trade_logging import TradeRecord, log_trade, load_trade_records
    
    record = TradeRecord(
        trade_id="abc123",
        date="2026-01-13",
        symbol="AAPL",
        ...
    )
    log_trade(record)
    
    for record in load_trade_records():
        print(record)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Any

logger = logging.getLogger(__name__)

# Default log path relative to live_testing/
DEFAULT_LOG_PATH = Path(__file__).parent / "trade_log.jsonl"


@dataclass
class TradeRecord:
    """
    Structured record of a completed trade with full context.
    
    All fields serialize cleanly to JSON for LlamaIndex ingestion.
    """
    # Core identifiers
    trade_id: str
    date: str  # ISO format YYYY-MM-DD
    symbol: str
    side: Literal["long", "short"]
    
    # Position details
    size: float  # Dollar value
    shares: float
    entry_ts: str  # ISO timestamp
    exit_ts: str  # ISO timestamp
    entry_price: float
    exit_price: float
    
    # Performance
    pnl: float
    pnl_pct: float
    max_drawdown_pct: Optional[float] = None
    holding_days: Optional[int] = None
    
    # VIX ROC context
    vix_roc_tier: Optional[str] = None  # e.g. "tier1_value", "tier2_growth", "tier3_megacap"
    vix_roc_signal: Optional[str] = None  # e.g. "EXIT", "REENTER", "NEUTRAL", "IN_MARKET"
    vix_roc_value: Optional[float] = None  # The actual VIX ROC percentage
    
    # Volatility context
    vol_regime: Optional[str] = None  # e.g. "LOW", "HIGH"
    vol_sizing_mult: Optional[float] = None  # Position sizing multiplier
    
    # Agent rationale - the refined LLM reasoning
    agent_rationale: str = ""
    
    # Strategy that generated the trade
    strategy_name: str = ""
    
    # Full tool snapshot for detailed analysis
    tool_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Optional tags for categorization
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique trade ID."""
        return f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TradeRecord":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    def to_document_text(self) -> str:
        """
        Format as natural language for LlamaIndex indexing.
        This text will be embedded for similarity search.
        """
        lines = [
            f"Trade: {self.symbol} {self.side.upper()} on {self.date}",
            f"Strategy: {self.strategy_name}",
            "",
            f"Entry: ${self.entry_price:.2f} at {self.entry_ts}",
            f"Exit: ${self.exit_price:.2f} at {self.exit_ts}",
            f"Size: ${self.size:.2f} ({self.shares:.2f} shares)",
            "",
            f"P&L: ${self.pnl:.2f} ({self.pnl_pct:+.2f}%)",
        ]
        
        if self.max_drawdown_pct is not None:
            lines.append(f"Max Drawdown: {self.max_drawdown_pct:.2f}%")
        if self.holding_days is not None:
            lines.append(f"Holding Period: {self.holding_days} days")
        
        lines.append("")
        
        # VIX ROC context
        if self.vix_roc_tier:
            lines.append(f"VIX ROC Tier: {self.vix_roc_tier}")
        if self.vix_roc_signal:
            lines.append(f"VIX ROC Signal: {self.vix_roc_signal}")
        if self.vix_roc_value is not None:
            lines.append(f"VIX ROC Value: {self.vix_roc_value:.1f}%")
        
        # Vol context
        if self.vol_regime:
            lines.append(f"Vol Regime: {self.vol_regime}")
        if self.vol_sizing_mult is not None:
            lines.append(f"Vol Sizing Multiplier: {self.vol_sizing_mult:.2f}x")
        
        # Agent rationale
        if self.agent_rationale:
            lines.extend([
                "",
                "Agent Rationale:",
                self.agent_rationale
            ])
        
        return "\n".join(lines)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata dict for LlamaIndex Document.
        """
        return {
            "trade_id": self.trade_id,
            "date": self.date,
            "symbol": self.symbol,
            "side": self.side,
            "strategy": self.strategy_name,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "vix_roc_tier": self.vix_roc_tier,
            "vix_roc_signal": self.vix_roc_signal,
            "vol_regime": self.vol_regime,
            "is_winner": self.pnl > 0,
            "tags": self.tags,
        }


def log_trade(
    record: TradeRecord,
    path: Path = DEFAULT_LOG_PATH,
) -> None:
    """
    Append a trade record to the JSONL log file.
    
    Args:
        record: The TradeRecord to log
        path: Path to the JSONL file (default: live_testing/trade_log.jsonl)
    
    Raises:
        Exception: If write fails (logged but re-raised)
    """
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append as single JSON line
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")
        
        logger.info(
            "Logged trade: %s %s %s P&L=$%.2f (%.2f%%)",
            record.trade_id,
            record.symbol,
            record.side,
            record.pnl,
            record.pnl_pct,
        )
    except Exception as e:
        logger.exception("Failed to log trade %s: %s", record.trade_id, e)
        raise


def load_trade_records(
    path: Path = DEFAULT_LOG_PATH,
) -> Iterator[TradeRecord]:
    """
    Load all trade records from the JSONL log file.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        TradeRecord objects
        
    Note:
        Invalid lines are logged and skipped, not raised.
    """
    if not path.exists():
        logger.warning("Trade log not found at %s", path)
        return
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield TradeRecord.from_json(line)
            except Exception as e:
                logger.warning(
                    "Skipping invalid record at line %d: %s",
                    line_num,
                    e,
                )


def load_trade_records_list(
    path: Path = DEFAULT_LOG_PATH,
) -> List[TradeRecord]:
    """
    Load all trade records as a list.
    
    Convenience wrapper around load_trade_records() iterator.
    """
    return list(load_trade_records(path))


def get_trade_stats(records: List[TradeRecord]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from a list of trade records.
    
    Returns:
        Dict with sample_size, win_rate, avg_return_pct, avg_max_drawdown_pct, etc.
    """
    if not records:
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "total_pnl": 0.0,
        }
    
    winners = [r for r in records if r.pnl > 0]
    losers = [r for r in records if r.pnl <= 0]
    
    total_pnl = sum(r.pnl for r in records)
    avg_return = sum(r.pnl_pct for r in records) / len(records)
    
    # Drawdown stats
    dd_values = [r.max_drawdown_pct for r in records if r.max_drawdown_pct is not None]
    avg_dd = sum(dd_values) / len(dd_values) if dd_values else None
    
    # Holding period
    hold_days = [r.holding_days for r in records if r.holding_days is not None]
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else None
    
    return {
        "sample_size": len(records),
        "num_winners": len(winners),
        "num_losers": len(losers),
        "win_rate": len(winners) / len(records) if records else 0.0,
        "avg_return_pct": avg_return,
        "total_pnl": total_pnl,
        "avg_winner_pct": sum(r.pnl_pct for r in winners) / len(winners) if winners else 0.0,
        "avg_loser_pct": sum(r.pnl_pct for r in losers) / len(losers) if losers else 0.0,
        "avg_max_drawdown_pct": avg_dd,
        "avg_holding_days": avg_hold,
    }


# =============================================================================
# HELPER: Extract context from strategy for logging
# =============================================================================

def extract_vix_roc_context(strategy) -> Dict[str, Any]:
    """
    Extract VIX ROC context from a VIXROCStrategy or similar.
    
    Returns dict with vix_roc_tier, vix_roc_signal, vix_roc_value.
    """
    context = {}
    
    # Try to get last reason which contains VIX ROC info
    if hasattr(strategy, '_last_reason'):
        for symbol, reason in strategy._last_reason.items():
            if "VIX ROC" in reason:
                # Parse tier from reason
                if "tier1" in reason.lower():
                    context["vix_roc_tier"] = "tier1_value"
                elif "tier2" in reason.lower():
                    context["vix_roc_tier"] = "tier2_growth"
                elif "tier3" in reason.lower():
                    context["vix_roc_tier"] = "tier3_megacap"
                
                # Parse signal
                if "EXIT" in reason:
                    context["vix_roc_signal"] = "EXIT"
                elif "REENTER" in reason:
                    context["vix_roc_signal"] = "REENTER"
                elif "HOLD" in reason:
                    context["vix_roc_signal"] = "HOLD"
                
                # Parse VIX ROC value
                import re
                match = re.search(r"VIX ROC[=:]?\s*([-\d.]+)%", reason)
                if match:
                    context["vix_roc_value"] = float(match.group(1))
    
    return context


def extract_vol_context(strategy) -> Dict[str, Any]:
    """
    Extract volatility context from a VIXROCVolAdjustedStrategy or similar.
    
    Returns dict with vol_regime, vol_sizing_mult.
    """
    context = {}
    
    # Check for vol state
    if hasattr(strategy, '_vol_state'):
        # Get any recent vol multiplier
        for symbol, mult in strategy._vol_state.items():
            context["vol_sizing_mult"] = mult
            break
    
    return context


def extract_agent_rationale(strategy) -> str:
    """
    Extract the refined LLM rationale from a ReflexionAgentStrategy.
    
    Returns the agent's reasoning text.
    """
    if hasattr(strategy, '_last_decisions'):
        for symbol, decision in strategy._last_decisions.items():
            if "decision" in decision:
                return decision["decision"]
    
    if hasattr(strategy, '_last_reason'):
        for symbol, reason in strategy._last_reason.items():
            return reason
    
    return ""
