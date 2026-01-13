#!/usr/bin/env python
"""
Reflect on Trades - Extract Lessons via LLM

Groups trades by meaningful cohorts and uses an LLM to extract
actionable lessons that can be used for future trading decisions.

Usage:
    python scripts/reflect_on_trades.py
    python scripts/reflect_on_trades.py --min-samples 3
    python scripts/reflect_on_trades.py --db-dir db/trade_lessons

The lessons are stored in a separate LlamaIndex for retrieval
by the Reflexion agent.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI as OpenAIClient
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding

from live_testing.trade_logging import (
    TradeRecord,
    load_trade_records_list,
    get_trade_stats,
    DEFAULT_LOG_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_DIR = Path(__file__).parent.parent / "db" / "trade_lessons"
EMBED_MODEL = "text-embedding-3-large"
REFLECTION_MODEL = "gpt-4.1"


@dataclass
class TradeLesson:
    """
    A lesson extracted from analyzing a group of trades.
    """
    lesson_id: str
    group_key: str  # e.g., "AAPL_tier2_growth_LOW"
    
    # LLM-generated fields
    summary: str  # 1-2 sentence lesson
    pattern_condition: str  # Boolean-like condition
    verdict: str  # "avoid" | "size_down" | "ok" | "lean_in"
    confidence: float  # 0.0-1.0
    tags: List[str] = field(default_factory=list)
    
    # Group statistics
    sample_size: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    avg_max_drawdown_pct: Optional[float] = None
    total_pnl: float = 0.0
    
    # Metadata
    symbols: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    date_range: Tuple[str, str] = ("", "")
    
    def to_document_text(self) -> str:
        """Format as natural language for LlamaIndex indexing."""
        lines = [
            f"TRADE LESSON: {self.group_key}",
            "",
            f"Summary: {self.summary}",
            "",
            f"Pattern Condition: {self.pattern_condition}",
            f"Verdict: {self.verdict.upper()} (confidence: {self.confidence:.0%})",
            "",
            "Statistics:",
            f"  - Sample Size: {self.sample_size} trades",
            f"  - Win Rate: {self.win_rate:.1%}",
            f"  - Avg Return: {self.avg_return_pct:+.2f}%",
            f"  - Total P&L: ${self.total_pnl:,.2f}",
        ]
        
        if self.avg_max_drawdown_pct is not None:
            lines.append(f"  - Avg Max Drawdown: {self.avg_max_drawdown_pct:.2f}%")
        
        lines.extend([
            "",
            f"Symbols: {', '.join(self.symbols)}",
            f"Strategies: {', '.join(self.strategies)}",
            f"Date Range: {self.date_range[0]} to {self.date_range[1]}",
            f"Tags: {', '.join(self.tags)}",
        ])
        
        return "\n".join(lines)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict for LlamaIndex Document."""
        return {
            "lesson_id": self.lesson_id,
            "group_key": self.group_key,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "win_rate": self.win_rate,
            "sample_size": self.sample_size,
            "tags": self.tags,
        }


def group_trades(
    records: List[TradeRecord],
) -> Dict[str, List[TradeRecord]]:
    """
    Group trades by meaningful cohorts.
    
    Current grouping: symbol + vix_roc_tier + vol_regime
    
    This can be extended to include other dimensions like:
    - strategy_name
    - time of year
    - holding period bucket
    - entry signal type
    """
    groups = defaultdict(list)
    
    for record in records:
        # Build group key from available dimensions
        parts = []
        
        # Symbol
        parts.append(record.symbol or "UNKNOWN")
        
        # VIX ROC tier
        tier = record.vix_roc_tier or "unknown_tier"
        parts.append(tier)
        
        # Vol regime
        vol = record.vol_regime or "unknown_vol"
        parts.append(vol)
        
        group_key = "_".join(parts)
        groups[group_key].append(record)
    
    return dict(groups)


def build_reflection_prompt(
    group_key: str,
    records: List[TradeRecord],
    stats: Dict[str, Any],
) -> str:
    """
    Build a prompt for the LLM to reflect on a group of trades.
    
    Returns a system and user prompt tuple.
    """
    # Format individual trade summaries
    trade_summaries = []
    for i, r in enumerate(records[:10], 1):  # Limit to 10 for prompt length
        outcome = "WIN" if r.pnl > 0 else "LOSS"
        trade_summaries.append(
            f"{i}. {r.symbol} {r.side} on {r.date}: "
            f"{outcome} {r.pnl_pct:+.2f}%, held {r.holding_days or '?'} days"
        )
    
    trade_list = "\n".join(trade_summaries)
    if len(records) > 10:
        trade_list += f"\n... and {len(records) - 10} more trades"
    
    system_prompt = """You are a trading performance analyst reviewing a group of trades to extract lessons.

Your task is to analyze the trades and produce a structured lesson that can guide future trading decisions.

Be specific and actionable. The lesson should help a trading agent decide whether to take similar trades in the future."""

    user_prompt = f"""
GROUP: {group_key}

STATISTICS:
- Sample Size: {stats['sample_size']} trades
- Win Rate: {stats['win_rate']:.1%}
- Average Return: {stats['avg_return_pct']:+.2f}%
- Total P&L: ${stats['total_pnl']:,.2f}
- Avg Winner: {stats.get('avg_winner_pct', 0):+.2f}%
- Avg Loser: {stats.get('avg_loser_pct', 0):+.2f}%
- Avg Max Drawdown: {stats.get('avg_max_drawdown_pct', 'N/A')}%
- Avg Holding Days: {stats.get('avg_holding_days', 'N/A')}

INDIVIDUAL TRADES:
{trade_list}

TASK:
Analyze these trades and produce a JSON response with the following structure:

{{
    "summary": "A 1-2 sentence lesson summarizing the key insight from these trades",
    "pattern_condition": "A boolean-like condition describing when this pattern applies (e.g., 'vix_roc_tier == tier2_growth AND vol_regime == LOW AND symbol in [AAPL, MSFT]')",
    "verdict": "One of: avoid | size_down | ok | lean_in",
    "confidence": 0.0 to 1.0 based on sample size and consistency,
    "tags": ["list", "of", "relevant", "tags"]
}}

Guidelines:
- "avoid": Win rate < 40% or avg return < -2%
- "size_down": Win rate 40-50% or high drawdowns
- "ok": Win rate 50-60% with acceptable returns
- "lean_in": Win rate > 60% with positive avg return

Be conservative with confidence if sample size is small (< 10 trades).
Use tags to capture key characteristics (e.g., "high_vol", "momentum", "value", "earnings").

Respond with ONLY the JSON object, no other text.
"""

    return system_prompt, user_prompt


def call_reflection_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = REFLECTION_MODEL,
) -> Dict[str, Any]:
    """
    Call the LLM to generate a reflection on trades.
    
    Returns the parsed JSON response or a fallback if parsing fails.
    """
    client = OpenAIClient()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Lower temp for more consistent structured output
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        # Handle potential markdown code blocks
        if content.startswith("```"):
            # Extract JSON from code block
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)
        
        result = json.loads(content)
        
        # Validate required fields
        required = ["summary", "pattern_condition", "verdict", "confidence", "tags"]
        for field in required:
            if field not in result:
                result[field] = _get_default(field)
        
        # Validate verdict
        valid_verdicts = ["avoid", "size_down", "ok", "lean_in"]
        if result["verdict"] not in valid_verdicts:
            result["verdict"] = "ok"
        
        # Validate confidence
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        
        return result
        
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)
        return _get_fallback_response()
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return _get_fallback_response()


def _get_default(field: str) -> Any:
    """Get default value for a missing field."""
    defaults = {
        "summary": "Insufficient data for meaningful analysis.",
        "pattern_condition": "unknown",
        "verdict": "ok",
        "confidence": 0.3,
        "tags": [],
    }
    return defaults.get(field, None)


def _get_fallback_response() -> Dict[str, Any]:
    """Get fallback response when LLM fails."""
    return {
        "summary": "LLM analysis failed - manual review recommended.",
        "pattern_condition": "unknown",
        "verdict": "ok",
        "confidence": 0.1,
        "tags": ["needs_review"],
    }


def reflect_on_group(
    group_key: str,
    records: List[TradeRecord],
) -> TradeLesson:
    """
    Generate a lesson from a group of trades.
    """
    import uuid
    
    # Calculate stats
    stats = get_trade_stats(records)
    
    # Build and call LLM
    system_prompt, user_prompt = build_reflection_prompt(group_key, records, stats)
    llm_result = call_reflection_llm(system_prompt, user_prompt)
    
    # Extract unique values
    symbols = list(set(r.symbol for r in records if r.symbol))
    strategies = list(set(r.strategy_name for r in records if r.strategy_name))
    dates = sorted([r.date for r in records if r.date])
    date_range = (dates[0], dates[-1]) if dates else ("", "")
    
    # Create lesson
    lesson = TradeLesson(
        lesson_id=f"lesson_{uuid.uuid4().hex[:8]}",
        group_key=group_key,
        summary=llm_result["summary"],
        pattern_condition=llm_result["pattern_condition"],
        verdict=llm_result["verdict"],
        confidence=llm_result["confidence"],
        tags=llm_result["tags"],
        sample_size=stats["sample_size"],
        win_rate=stats["win_rate"],
        avg_return_pct=stats["avg_return_pct"],
        avg_max_drawdown_pct=stats.get("avg_max_drawdown_pct"),
        total_pnl=stats["total_pnl"],
        symbols=symbols,
        strategies=strategies,
        date_range=date_range,
    )
    
    return lesson


def build_lessons_index(
    lessons: List[TradeLesson],
    db_dir: Path = DEFAULT_DB_DIR,
) -> VectorStoreIndex:
    """
    Build a LlamaIndex from trade lessons.
    """
    logger.info("Building lessons index with %d lessons", len(lessons))
    
    # Convert to documents
    documents = [
        Document(
            text=lesson.to_document_text(),
            metadata=lesson.get_metadata(),
            doc_id=lesson.lesson_id,
        )
        for lesson in lessons
    ]
    
    # Configure embedding model
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    # Persist
    db_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(db_dir))
    
    logger.info("Lessons index saved to %s", db_dir)
    return index


def load_lessons_index(
    db_dir: Path = DEFAULT_DB_DIR,
) -> VectorStoreIndex:
    """Load an existing lessons index from disk."""
    if not db_dir.exists():
        raise FileNotFoundError(f"Lessons index not found at {db_dir}")
    
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    storage_context = StorageContext.from_defaults(persist_dir=str(db_dir))
    return VectorStoreIndex.from_storage_context(storage_context)


def query_lessons(
    query: str,
    k: int = 5,
    db_dir: Path = DEFAULT_DB_DIR,
) -> List[dict]:
    """Query the lessons index for relevant lessons."""
    index = load_lessons_index(db_dir)
    retriever = index.as_retriever(similarity_top_k=k)
    results = retriever.retrieve(query)
    
    return [
        {
            "text": r.node.get_content(),
            "score": r.score,
            "metadata": dict(r.node.metadata) if r.node.metadata else {},
        }
        for r in results
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Reflect on trades and extract lessons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to trade log JSONL (default: {DEFAULT_LOG_PATH})"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Directory to store lessons index (default: {DEFAULT_DB_DIR})"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum trades per group to generate lesson (default: 5)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional: Test query to run after building"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load trades
    logger.info("Loading trades from %s", args.log_path)
    records = load_trade_records_list(args.log_path)
    logger.info("Loaded %d trades", len(records))
    
    if not records:
        logger.warning("No trades found. Nothing to reflect on.")
        return
    
    # Group trades
    groups = group_trades(records)
    logger.info("Grouped into %d cohorts", len(groups))
    
    # Filter by minimum sample size
    valid_groups = {
        k: v for k, v in groups.items()
        if len(v) >= args.min_samples
    }
    logger.info(
        "%d groups meet minimum sample size (%d)",
        len(valid_groups),
        args.min_samples
    )
    
    if not valid_groups:
        logger.warning(
            "No groups meet minimum sample size. "
            "Try --min-samples %d or add more trades.",
            max(1, args.min_samples // 2)
        )
        return
    
    # Generate lessons
    lessons = []
    for group_key, group_records in valid_groups.items():
        logger.info(
            "Reflecting on %s (%d trades)...",
            group_key,
            len(group_records)
        )
        
        try:
            lesson = reflect_on_group(group_key, group_records)
            lessons.append(lesson)
            logger.info(
                "  → %s: %s (confidence: %.0f%%)",
                lesson.verdict.upper(),
                lesson.summary[:60] + "..." if len(lesson.summary) > 60 else lesson.summary,
                lesson.confidence * 100
            )
        except Exception as e:
            logger.error("Failed to reflect on %s: %s", group_key, e)
    
    if not lessons:
        logger.warning("No lessons generated.")
        return
    
    # Build index
    index = build_lessons_index(lessons, args.db_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("REFLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Lessons: {len(lessons)}")
    
    verdict_counts = defaultdict(int)
    for lesson in lessons:
        verdict_counts[lesson.verdict] += 1
    
    print("\nVerdicts:")
    for verdict, count in sorted(verdict_counts.items()):
        print(f"  {verdict.upper()}: {count}")
    
    print(f"\nLessons saved to: {args.db_dir}")
    
    # Optional test query
    if args.query:
        print(f"\n{'='*60}")
        print(f"Test Query: {args.query}")
        print(f"{'='*60}\n")
        
        results = query_lessons(
            query=args.query,
            k=3,
            db_dir=args.db_dir,
        )
        
        for i, r in enumerate(results, 1):
            print(f"--- Result {i} (score: {r['score']:.4f}) ---")
            print(r['text'][:500])
            print()


if __name__ == "__main__":
    main()
