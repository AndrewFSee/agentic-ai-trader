# analyze_trade_agent_reflexion.py

"""
Reflexion Agent for Trading Decisions

Implements the Reflexion pattern (Shinn et al., 2023):
1. GENERATE: Initial analysis and decision
2. EVALUATE: Self-critique the decision for weaknesses
3. REFLECT: Learn from the critique
4. REFINE: Generate improved final decision

This approach leads to better-calibrated trading decisions by catching
blind spots and overconfidence in the initial analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
import logging
import os
import time
import concurrent.futures

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI as OpenAIClient
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from agent_tools import run_tools
from planner import plan_tools  # Use streamlined planner

# Trade memory imports
from pathlib import Path
TRADE_MEMORY_DIR = Path(__file__).parent / "db" / "trades"
TRADE_LESSONS_DIR = Path(__file__).parent / "db" / "trade_lessons"

# LlamaIndex config - use absolute path for directory independence
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(_SCRIPT_DIR, "db", "books")
EMBED_MODEL = "text-embedding-3-large"
DECISION_MODEL = "gpt-4.1"

# Configure logging
logging.basicConfig(level=os.getenv("AGENT_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class RAGIndex:
    """Wrapper around LlamaIndex to provide similarity_search interface."""
    
    def __init__(self, index):
        self._index = index
        self._retriever = index.as_retriever(similarity_top_k=6)
    
    def similarity_search(self, query: str, k: int = 6) -> List:
        """Retrieve k most similar documents for the query."""
        self._retriever = self._index.as_retriever(similarity_top_k=k)
        nodes = self._retriever.retrieve(query)
        return [_NodeAsDoc(n) for n in nodes]


class _NodeAsDoc:
    """Adapter to make LlamaIndex nodes look like LangChain docs."""
    
    def __init__(self, node_with_score):
        self.node = node_with_score.node
        self.score = node_with_score.score
        self.page_content = self.node.get_content()
        self.metadata = dict(self.node.metadata) if self.node.metadata else {}


def load_vectorstore() -> Any:
    """Load the LlamaIndex vector store from disk."""
    from types import SimpleNamespace

    try:
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
        logger.info("Loaded LlamaIndex from %s", INDEX_DIR)
        return RAGIndex(index)
    except Exception as e:
        logger.warning("Could not load LlamaIndex: %s. Falling back to empty index.", e)

        def _empty_search(q, k=6):
            return []

        return SimpleNamespace(similarity_search=_empty_search)


# =============================================================================
# TRADE MEMORY INTEGRATION
# =============================================================================

# Track if we've already warned about missing indexes (to avoid spam)
_warned_no_trade_memory = False
_warned_no_trade_lessons = False


def load_trade_memory() -> Optional[Any]:
    """
    Load the trade memory index (past trades) from disk.
    
    Returns None if the index doesn't exist yet.
    """
    global _warned_no_trade_memory
    from types import SimpleNamespace
    
    # Check if index files actually exist (not just directory with .gitkeep)
    docstore_path = TRADE_MEMORY_DIR / "docstore.json"
    if not docstore_path.exists():
        if not _warned_no_trade_memory:
            logger.debug("Trade memory index not built yet - run scripts/build_trade_memory.py after closing trades")
            _warned_no_trade_memory = True
        return None
    
    try:
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=str(TRADE_MEMORY_DIR))
        index = load_index_from_storage(storage_context)
        logger.info("Loaded trade memory from %s", TRADE_MEMORY_DIR)
        return RAGIndex(index)
    except Exception as e:
        logger.warning("Could not load trade memory: %s", e)
        return None


def load_trade_lessons() -> Optional[Any]:
    """
    Load the trade lessons index (LLM-extracted insights) from disk.
    
    Returns None if the index doesn't exist yet.
    """
    global _warned_no_trade_lessons
    from types import SimpleNamespace
    
    # Check if index files actually exist (not just directory with .gitkeep)
    docstore_path = TRADE_LESSONS_DIR / "docstore.json"
    if not docstore_path.exists():
        if not _warned_no_trade_lessons:
            logger.debug("Trade lessons index not built yet - run scripts/reflect_on_trades.py after accumulating trades")
            _warned_no_trade_lessons = True
        return None
    
    try:
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=str(TRADE_LESSONS_DIR))
        index = load_index_from_storage(storage_context)
        logger.info("Loaded trade lessons from %s", TRADE_LESSONS_DIR)
        return RAGIndex(index)
    except Exception as e:
        logger.warning("Could not load trade lessons: %s", e)
        return None


def query_trade_memory(
    symbol: str,
    trading_idea: str,
    vix_roc_tier: Optional[str] = None,
    vol_regime: Optional[str] = None,
    k: int = 3,
) -> Tuple[List, List]:
    """
    Query trade memory and lessons for relevant past experiences.
    
    Args:
        symbol: The ticker symbol
        trading_idea: The current trading idea
        vix_roc_tier: Current VIX ROC tier (e.g., "tier2_growth")
        vol_regime: Current volatility regime (e.g., "LOW", "HIGH")
        k: Number of results to retrieve from each index
        
    Returns:
        Tuple of (past_trades, lessons) where each is a list of docs
    """
    past_trades = []
    lessons = []
    
    # Build query incorporating current context
    query_parts = [symbol, trading_idea]
    if vix_roc_tier:
        query_parts.append(f"vix_roc_tier {vix_roc_tier}")
    if vol_regime:
        query_parts.append(f"vol_regime {vol_regime}")
    
    query = " ".join(query_parts)
    
    # Query trade memory
    trade_memory = load_trade_memory()
    if trade_memory:
        try:
            past_trades = trade_memory.similarity_search(query, k=k)
            logger.info("Found %d relevant past trades", len(past_trades))
        except Exception as e:
            logger.warning("Trade memory query failed: %s", e)
    
    # Query lessons
    lessons_index = load_trade_lessons()
    if lessons_index:
        try:
            lessons = lessons_index.similarity_search(query, k=k)
            logger.info("Found %d relevant lessons", len(lessons))
        except Exception as e:
            logger.warning("Lessons query failed: %s", e)
    
    return past_trades, lessons


def _format_trade_memory_context(
    past_trades: List,
    lessons: List,
) -> str:
    """
    Format trade memory and lessons for inclusion in prompts.
    """
    sections = []
    
    if lessons:
        sections.append("=== RELEVANT PAST TRADE LESSONS ===\n")
        for i, doc in enumerate(lessons, 1):
            verdict = doc.metadata.get("verdict", "unknown")
            confidence = doc.metadata.get("confidence", 0)
            sections.append(f"--- Lesson {i} (Verdict: {verdict.upper()}, Confidence: {confidence:.0%}) ---")
            sections.append(doc.page_content.strip())
            sections.append("")
    
    if past_trades:
        sections.append("=== RELEVANT PAST TRADES ===\n")
        for i, doc in enumerate(past_trades, 1):
            outcome = "WIN" if doc.metadata.get("is_winner") else "LOSS"
            pnl_pct = doc.metadata.get("pnl_pct", 0)
            sections.append(f"--- Past Trade {i} ({outcome}, {pnl_pct:+.2f}%) ---")
            # Truncate long trade records
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            sections.append(content)
            sections.append("")
    
    if not sections:
        return ""
    
    return "\n".join(sections)


def _format_docs(docs: List) -> str:
    """Format document chunks for prompt."""
    parts = []
    for d in docs:
        book_name = d.metadata.get("book_name", d.metadata.get("file_name", d.metadata.get("source", "unknown")))
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        parts.append(
            f"[Book: {book_name}, page {page}]\n{d.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# TOOL RESULT FORMATTERS
# =============================================================================

def _format_price_summary(price_result: Dict[str, Any] | None) -> str:
    """Format price data for the prompt."""
    if not price_result:
        return "No price data available."

    if price_result.get("error"):
        return f"Price data unavailable. Error: {price_result['error']}"

    symbol = price_result["symbol"]
    latest_close = price_result.get("latest_close")
    latest_date = price_result.get("latest_date")
    
    lines = [f"Symbol: {symbol}"]
    
    if latest_close:
        lines.append(f"Latest Close: ${latest_close:.2f} ({latest_date})")
    
    # Calculate period change
    bars = price_result.get("bars", [])
    if len(bars) >= 2:
        first_close = bars[0].get('c')
        last_close = bars[-1].get('c')
        if first_close and last_close:
            pct_change = ((last_close - first_close) / first_close) * 100
            lines.append(f"Period change: {pct_change:+.2f}%")
    
    # Volume analysis
    vol = price_result.get("volume_analysis", {})
    if vol.get("avg_volume"):
        lines.append(f"Volume: {vol['volume_ratio']:.2f}x avg ({vol['volume_condition']})")
    
    # ATR
    atr = price_result.get("atr", {})
    if atr.get("atr_value") and latest_close:
        atr_pct = (atr['atr_value'] / latest_close * 100)
        lines.append(f"ATR(14): ${atr['atr_value']:.2f} ({atr_pct:.2f}% of price)")
        lines.append(f"  Stop distances: Conservative 2x=${atr['atr_value']*2:.2f}, Tight 1x=${atr['atr_value']:.2f}")

    return "\n".join(lines)


def _format_vix_roc_risk(result: Dict[str, Any] | None) -> str:
    """Format VIX ROC risk overlay results."""
    if not result:
        return "VIX ROC risk data not available."
    
    if result.get("error"):
        return f"VIX ROC risk unavailable: {result['error']}"
    
    lines = [
        "=" * 60,
        "🎯 VIX ROC RISK OVERLAY (PRIMARY MARKET TIMING)",
        "=" * 60,
        f"Symbol: {result.get('symbol', 'N/A')}",
        f"Tier: {result.get('tier_name', 'Unknown')}",
        f"Classification: {result.get('classification_source', 'Unknown')}",
        "",
        "Strategy Parameters:",
        f"  Exit when VIX ROC > {result.get('strategy_params', {}).get('exit_when_vix_roc_above', 'N/A')}",
        f"  Re-enter when VIX ROC < {result.get('strategy_params', {}).get('reenter_when_vix_roc_below', 'N/A')}",
        f"  Min days out: {result.get('strategy_params', {}).get('min_days_out', 'N/A')}",
        "",
        f"⚡ CURRENT SIGNAL: {result.get('current_signal', 'N/A').upper()}",
        f"📊 VIX ROC: {result.get('current_vix_roc', 'N/A')}",
        f"📍 Position Status: {result.get('position_status', 'N/A')}",
        "",
        f"💬 {result.get('message', '')}",
        "",
        f"Description: {result.get('tier_description', '')}",
        "=" * 60
    ]
    
    return "\n".join(lines)


def _format_vol_prediction(result: Dict[str, Any] | None) -> str:
    """Format volatility prediction results."""
    if not result:
        return "Volatility prediction not available."
    
    if result.get("error"):
        return f"Volatility prediction unavailable: {result['error']}"
    
    regime = result.get("current_regime", "UNKNOWN")
    risk_level = result.get("risk_level", "UNKNOWN")
    
    lines = [
        "=" * 60,
        "📊 VOLATILITY PREDICTION (POSITION SIZING)",
        "=" * 60,
        f"Current Regime: {regime}",
        f"Risk Level: {risk_level}",
    ]
    
    if regime == "LOW":
        spike_prob = result.get("spike_probability", 0)
        lines.append(f"Spike Probability (→HIGH): {spike_prob:.1%}")
        if spike_prob >= 0.6:
            lines.append("  ⚠️ ELEVATED - Reduce position 30-50%, tighten stops")
        elif spike_prob >= 0.5:
            lines.append("  ⚠️ WATCH - Consider reducing position 20-40%")
        else:
            lines.append("  ✅ Normal sizing appropriate")
    else:
        calm_prob = result.get("calm_probability", 0)
        lines.append(f"Calm Probability (→LOW): {calm_prob:.1%}")
        if calm_prob >= 0.5:
            lines.append("  ✅ Vol likely to subside, can add to winners")
        else:
            lines.append("  ⚠️ High vol may persist, maintain reduced size")
    
    vix_zscore = result.get("vix_zscore", 0)
    lines.append(f"VIX Z-Score: {vix_zscore:+.2f}")
    if vix_zscore > 1.0:
        lines.append("  ⚠️ Market fear elevated")
    elif vix_zscore < -1.0:
        lines.append("  ⚠️ Market complacent, watch for potential spike")
    
    lines.append(f"\nSuggested Action: {result.get('suggested_action', 'N/A')}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def _format_technical_summary(tool_results: Dict[str, Any]) -> str:
    """Format RSI, MACD, Bollinger Bands into a compact text block."""
    rsi = tool_results.get("polygon_technical_rsi")
    macd = tool_results.get("polygon_technical_macd")
    bb = tool_results.get("bollinger_bands")

    lines: List[str] = []

    if rsi and not rsi.get("error"):
        lines.append(f"RSI({rsi['window']}): {rsi['latest_value']:.2f} ({rsi['condition']})")

    if macd and not macd.get("error"):
        lines.append(f"MACD: {macd['latest_macd']:.4f}, Signal={macd['latest_signal']:.4f} ({macd['signal_type']})")

    if bb and not bb.get("error"):
        lines.append(f"Bollinger: Price={bb['latest_price']:.2f}, %B={bb['percent_b']:.3f}, {bb['position']}")

    return "\n".join(lines) if lines else "No technical indicator data available."


def _format_news_summary(news_result: Dict[str, Any] | None) -> str:
    """Format news sentiment summary."""
    if not news_result or news_result.get("num_articles", 0) == 0:
        return "No recent news articles found."

    if news_result.get("error"):
        return f"News unavailable: {news_result['error']}"

    lines = [
        f"Sentiment: {news_result['aggregate_label']}",
        f"Articles analyzed: {news_result['num_articles']}",
    ]
    
    for art in news_result.get("articles", [])[:3]:
        lines.append(f"  - {art['headline'][:60]}... ({art['sentiment']})")

    return "\n".join(lines)


def _call_llm(system: str, user: str, model: str = DECISION_MODEL) -> str:
    """Call OpenAI chat completion API with timeout handling."""
    client = OpenAIClient()
    timeout_sec = int(os.getenv('LLM_CALL_TIMEOUT_SEC', '120'))
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    timeout=timeout_sec,
                )
            )
            response = fut.result(timeout=timeout_sec + 10)
        
        return response.choices[0].message.content
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        raise


# =============================================================================
# REFLEXION AGENT IMPLEMENTATION
# =============================================================================

def _generate_initial_analysis(
    trading_idea: str,
    symbol: str,
    tool_results: Dict[str, Any],
    book_context: str,
    rules_context: str,
    trade_memory_context: str = "",
) -> str:
    """
    STEP 1: Generate initial analysis and decision.
    
    Now includes trade memory context (past trades and lessons) when available.
    """
    # Format tool results
    price_text = _format_price_summary(tool_results.get("polygon_price_data"))
    vix_roc_text = _format_vix_roc_risk(tool_results.get("vix_roc_risk"))
    vol_pred_text = _format_vol_prediction(tool_results.get("vol_prediction"))
    tech_text = _format_technical_summary(tool_results)
    news_text = _format_news_summary(tool_results.get("news_sentiment_finviz_finbert"))
    
    # Build system prompt - enhanced with trade memory instructions
    system_prompt = """You are an expert trading analyst. Your task is to analyze a trade idea 
using the provided market data, trading book knowledge, and lessons from past trades.

Focus on:
1. VIX ROC Risk Signal - the PRIMARY market timing indicator
2. Volatility Prediction - for position sizing
3. Technical Analysis - trend, momentum, overbought/oversold
4. News Sentiment - recent catalysts
5. PAST TRADE LESSONS - Learn from previous similar trades

IMPORTANT: If trade lessons are provided, you MUST:
- Explicitly consider each relevant lesson
- State whether you are FOLLOWING or OVERRIDING each lesson
- Justify any overrides with specific reasoning

Be direct and specific. Cite the trading books and lessons when making recommendations.
"""

    # Build trade memory section
    trade_memory_section = ""
    if trade_memory_context:
        trade_memory_section = f"""
PAST TRADE MEMORY
=================
{trade_memory_context}

"""

    user_prompt = f"""
TRADING IDEA
============
Symbol: {symbol}
Idea: {trading_idea}
{trade_memory_section}
VIX ROC RISK OVERLAY
====================
{vix_roc_text}

VOLATILITY PREDICTION
=====================
{vol_pred_text}

PRICE DATA
==========
{price_text}

TECHNICAL INDICATORS
====================
{tech_text}

NEWS & SENTIMENT
================
{news_text}

TRADING BOOK EXCERPTS (Idea-Related)
=====================================
{book_context}

TRADING BOOK EXCERPTS (Risk Management)
========================================
{rules_context}

TASK
====
Provide your initial analysis covering:

1. MARKET TIMING (VIX ROC):
   - Is the VIX ROC signal favorable? (IN MARKET vs OUT OF MARKET)
   - What tier is this asset and why?
   - Should we trade or wait?

2. POSITION SIZING (Vol Prediction):
   - What is the current volatility regime?
   - What position size adjustment is recommended?
   - Any spike/calm probability concerns?

3. TECHNICAL SETUP:
   - Is the trend favorable?
   - Are momentum indicators aligned?
   - Key support/resistance levels?

4. SENTIMENT & CATALYSTS:
   - What is the recent news sentiment?
   - Any notable catalysts?

5. TRADE LESSONS (if provided):
   - Review each relevant lesson from past trades
   - For each lesson, state: FOLLOWING or OVERRIDING
   - If overriding, explain why current conditions are different

6. INITIAL VERDICT:
   - VERDICT: ATTRACTIVE, NOT ATTRACTIVE, or UNCLEAR
   - Key reasons for your verdict
   - Suggested entry, stop, and target if attractive

Be specific and cite book wisdom and trade lessons where relevant.
"""

    return _call_llm(system_prompt, user_prompt)


def _evaluate_and_critique(
    initial_analysis: str,
    trading_idea: str,
    symbol: str,
) -> str:
    """
    STEP 2: Self-critique the initial analysis.
    Look for weaknesses, blind spots, and overconfidence.
    """
    system_prompt = """You are a skeptical trading risk manager reviewing an analyst's trade recommendation.

Your job is to find WEAKNESSES in the analysis:
- Overconfidence or confirmation bias
- Missing risk factors
- Ignored contrary signals
- Position sizing issues
- Unclear or unjustified assumptions
- Cherry-picked data

Be thorough but fair. A good trade idea should survive scrutiny.
"""

    user_prompt = f"""
TRADE IDEA UNDER REVIEW
=======================
Symbol: {symbol}
Idea: {trading_idea}

ANALYST'S INITIAL ANALYSIS
==========================
{initial_analysis}

CRITIQUE TASK
=============
Review this analysis critically. Identify:

1. CONFIRMATION BIAS:
   - Did the analyst ignore any contrary signals?
   - Are they selectively using data that supports their view?

2. RISK BLIND SPOTS:
   - What risks might they have underweighted?
   - Are stop losses and position sizes appropriate?

3. OVERCONFIDENCE:
   - Is the conviction level justified by the evidence?
   - Are there unjustified assumptions?

4. MISSING ANALYSIS:
   - What important factors weren't considered?
   - Would additional data change the conclusion?

5. VIX ROC & VOL PREDICTION:
   - Did they properly account for the VIX ROC signal?
   - Is position sizing aligned with volatility prediction?

6. VERDICT REVIEW:
   - Is the verdict (ATTRACTIVE/NOT ATTRACTIVE/UNCLEAR) justified?
   - Should it be downgraded due to identified weaknesses?

Be specific about each weakness found.
"""

    return _call_llm(system_prompt, user_prompt)


def _reflect_and_learn(
    initial_analysis: str,
    critique: str,
) -> str:
    """
    STEP 3: Reflect on the critique and extract learnings.
    """
    system_prompt = """You are synthesizing insights from a trade analysis and its critique.

Your job is to identify:
- Valid criticisms that should change the recommendation
- Criticisms that are too harsh or not applicable
- Key adjustments needed for the final analysis
"""

    user_prompt = f"""
INITIAL ANALYSIS
================
{initial_analysis}

CRITIQUE
========
{critique}

REFLECTION TASK
===============
Reflect on the critique and determine:

1. VALID CRITICISMS:
   - Which critiques are valid and important?
   - How should these change the recommendation?

2. OVERREACHING CRITICISMS:
   - Which critiques are too harsh or not applicable?
   - Why should we not let these change our view?

3. KEY ADJUSTMENTS:
   - What specific changes should be made to the final recommendation?
   - Should the verdict change? Why or why not?

4. CONFIDENCE CALIBRATION:
   - After considering the critique, what is the appropriate confidence level?
   - HIGH: Strong conviction despite critique
   - MEDIUM: Reasonable trade but with acknowledged uncertainties
   - LOW: Critique raised significant concerns

Be balanced - don't dismiss valid critiques but don't over-correct either.
"""

    return _call_llm(system_prompt, user_prompt)


def _generate_refined_decision(
    trading_idea: str,
    symbol: str,
    initial_analysis: str,
    critique: str,
    reflection: str,
    tool_results: Dict[str, Any],
) -> str:
    """
    STEP 4: Generate the final refined decision incorporating all insights.
    """
    vix_roc_result = tool_results.get("vix_roc_risk", {})
    vol_pred_result = tool_results.get("vol_prediction", {})
    
    system_prompt = """You are a senior trading decision maker generating the final recommendation.

You have access to:
1. The initial analysis
2. A critical review of that analysis  
3. Reflections on the valid and invalid criticisms

Your job is to synthesize everything into a FINAL, REFINED recommendation that:
- Addresses valid criticisms from the review
- Maintains strong points from the initial analysis
- Is properly calibrated for risk
- Provides actionable, specific guidance

NEVER IGNORE the VIX ROC signal - it is the PRIMARY market timing indicator.
"""

    user_prompt = f"""
TRADE IDEA
==========
Symbol: {symbol}
Idea: {trading_idea}

CURRENT VIX ROC STATUS
======================
Signal: {vix_roc_result.get('current_signal', 'N/A')}
Position Status: {vix_roc_result.get('position_status', 'N/A')}
Message: {vix_roc_result.get('message', 'N/A')}

CURRENT VOL PREDICTION
======================
Regime: {vol_pred_result.get('current_regime', 'N/A')}
Risk Level: {vol_pred_result.get('risk_level', 'N/A')}
Suggested Action: {vol_pred_result.get('suggested_action', 'N/A')}

INITIAL ANALYSIS
================
{initial_analysis}

CRITIQUE
========
{critique}

REFLECTION
==========
{reflection}

FINAL DECISION TASK
===================
Generate the FINAL trading recommendation that incorporates all insights.

Structure your response as:

## FINAL VERDICT

**VERDICT:** [ATTRACTIVE | NOT ATTRACTIVE | UNCLEAR] - Confidence: [HIGH | MEDIUM | LOW]

**Quick Summary:** One paragraph summary of the decision.

## VIX ROC MARKET TIMING

[Your interpretation of the VIX ROC signal and what it means for this trade]

## POSITION SIZING RECOMMENDATION

Based on volatility prediction:
- Suggested size: [X]% of account
- Rationale: [Why this size]

## TECHNICAL ANALYSIS SUMMARY

[Key technical points]

## RISK MANAGEMENT

- Entry: $[price] or [condition]
- Stop Loss: $[price] ([X] ATR away)
- Target: $[price] (R:R = [X]:1)
- Max Loss: [X]% of account

## ACKNOWLEDGED UNCERTAINTIES

[Valid concerns from the critique that traders should monitor]

## FINAL CHECKLIST

Before taking this trade, confirm:
☐ [Item 1]
☐ [Item 2]
☐ [Item 3]
☐ [Item 4]
"""

    return _call_llm(system_prompt, user_prompt)


def analyze_trade_reflexion(
    trading_idea: str,
    symbol: str | None,
    vectordb: Any | None = None,
    k_main: int = 6,
    k_rules: int = 6,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Full Reflexion Agent:
      1. GENERATE: Initial analysis
      2. EVALUATE: Self-critique
      3. REFLECT: Extract learnings
      4. REFINE: Final decision
    
    Returns dict with all steps for transparency.
    """
    if vectordb is None:
        vectordb = load_vectorstore()

    symbol = symbol or "N/A"

    # 1) RAG: book excerpts
    idea_query = f"{symbol} {trading_idea}"
    idea_docs = vectordb.similarity_search(idea_query, k=k_main)
    rules_docs = vectordb.similarity_search(
        "risk management position sizing stop loss max risk per trade drawdown "
        "trading rules trading plan psychology discipline",
        k=k_rules,
    )

    idea_context = _format_docs(idea_docs)
    rules_context = _format_docs(rules_docs)

    # 2) Planner: decide which tools to use
    tool_calls = plan_tools(trading_idea, symbol=symbol)
    
    if verbose:
        tool_names = [tc.get("tool_name", "unknown") for tc in tool_calls]
        print(f"\n[PLANNER] Selected tools: {', '.join(tool_names)}")

    # 3) Execute tools
    state: Dict[str, Any] = {"messages": [], "tool_results": {}}
    state = run_tools(state, tool_calls)
    tool_results = state["tool_results"]
    
    if verbose:
        print(f"[TOOLS] Results: {', '.join(tool_results.keys())}")

    # 4) Query trade memory for relevant past trades and lessons
    vix_roc_result = tool_results.get("vix_roc_risk", {})
    vol_result = tool_results.get("vol_prediction", {})
    
    past_trades, lessons = query_trade_memory(
        symbol=symbol,
        trading_idea=trading_idea,
        vix_roc_tier=vix_roc_result.get("tier_name"),
        vol_regime=vol_result.get("current_regime"),
        k=3,
    )
    
    trade_memory_context = _format_trade_memory_context(past_trades, lessons)
    
    if verbose and trade_memory_context:
        num_trades = len(past_trades)
        num_lessons = len(lessons)
        print(f"[TRADE MEMORY] Found {num_trades} past trades, {num_lessons} lessons")

    # =========================================================================
    # REFLEXION LOOP
    # =========================================================================
    
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: GENERATING INITIAL ANALYSIS...")
        print("=" * 70)
    
    initial_analysis = _generate_initial_analysis(
        trading_idea=trading_idea,
        symbol=symbol,
        tool_results=tool_results,
        book_context=idea_context,
        rules_context=rules_context,
        trade_memory_context=trade_memory_context,
    )
    
    if verbose:
        print("\n[INITIAL ANALYSIS]")
        print("-" * 40)
        print(initial_analysis[:500] + "..." if len(initial_analysis) > 500 else initial_analysis)
    
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: EVALUATING & CRITIQUING...")
        print("=" * 70)
    
    critique = _evaluate_and_critique(
        initial_analysis=initial_analysis,
        trading_idea=trading_idea,
        symbol=symbol,
    )
    
    if verbose:
        print("\n[CRITIQUE]")
        print("-" * 40)
        print(critique[:500] + "..." if len(critique) > 500 else critique)
    
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: REFLECTING & LEARNING...")
        print("=" * 70)
    
    reflection = _reflect_and_learn(
        initial_analysis=initial_analysis,
        critique=critique,
    )
    
    if verbose:
        print("\n[REFLECTION]")
        print("-" * 40)
        print(reflection[:500] + "..." if len(reflection) > 500 else reflection)
    
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING REFINED DECISION...")
        print("=" * 70)
    
    final_decision = _generate_refined_decision(
        trading_idea=trading_idea,
        symbol=symbol,
        initial_analysis=initial_analysis,
        critique=critique,
        reflection=reflection,
        tool_results=tool_results,
    )
    
    return {
        "initial_analysis": initial_analysis,
        "critique": critique,
        "reflection": reflection,
        "final_decision": final_decision,
        "tool_results": tool_results,
    }


def format_full_report(result: Dict[str, str]) -> str:
    """Format the full reflexion result as a markdown report."""
    return f"""
# Trading Analysis Report (Reflexion Agent)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Phase 1: Initial Analysis

{result['initial_analysis']}

---

## Phase 2: Critical Review

{result['critique']}

---

## Phase 3: Reflection

{result['reflection']}

---

## Phase 4: Final Decision (Refined)

{result['final_decision']}

---

*This analysis was generated using the Reflexion pattern: Initial analysis → Critical review → Reflection → Refined decision*
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Reflexion Trading Agent")
    parser.add_argument("--idea", type=str, help="Trading idea text")
    parser.add_argument("--symbol", type=str, help="Ticker symbol (e.g., NVDA)")
    parser.add_argument("--out", type=str, default=None, help="Output path for report")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    vectordb = load_vectorstore()

    if args.idea:
        # Non-interactive mode
        print("Running Reflexion Trading Agent...")
        result = analyze_trade_reflexion(
            trading_idea=args.idea,
            symbol=args.symbol,
            vectordb=vectordb,
            verbose=not args.quiet,
        )
        
        report = format_full_report(result)
        
        if args.out:
            out_path = args.out
        else:
            out_path = f"research_reports/reflexion_{args.symbol or 'NA'}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.md"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved report to {out_path}")
        
        print("\n" + "=" * 70)
        print("FINAL DECISION")
        print("=" * 70)
        print(result["final_decision"])
    else:
        # Interactive mode
        print("\n[Reflexion Trading Agent] Ctrl+C or type 'q' to exit.")
        print("This agent uses a 4-step process: Generate → Critique → Reflect → Refine\n")
        
        while True:
            idea = input("Describe your trading idea (or 'q' to quit): ").strip()
            if idea.lower() in {"q", "quit", "exit"}:
                break
            symbol = input("Symbol (e.g., AAPL): ").strip() or None

            print("\n[Analyzing with Reflexion pattern...]\n")
            
            result = analyze_trade_reflexion(
                trading_idea=idea,
                symbol=symbol,
                vectordb=vectordb,
                verbose=True,
            )
            
            print("\n" + "=" * 70)
            print("FINAL DECISION")
            print("=" * 70)
            print(result["final_decision"])
