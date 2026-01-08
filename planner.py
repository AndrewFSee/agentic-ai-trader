# planner.py

from typing import Dict, Any, List
import json
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import OpenAI as OpenAIClient

from polygon_tools import get_all_tools as get_polygon_tools
from sentiment_tools import get_all_tools as get_sentiment_tools
from research_tools import get_all_tools as get_research_tools
from tools import TOOL_REGISTRY as regime_tools_registry

PLANNER_MODEL = "gpt-4.1"  # Fast, reliable model for planning


def plan_tools(user_query: str, symbol: str | None = None) -> List[Dict[str, Any]]:
    """
    Returns a list of tool calls:
      [{"tool_name": "...", "arguments": {...}}, ...]
    based on the user's query and available tools.
    """
    client = OpenAIClient()

    # Combine Polygon.io, sentiment, research, and risk tools
    regime_tools_list = list(regime_tools_registry.values())
    # Exclude 'fn' key from regime tools for JSON serialization
    regime_tools_serializable = [
        {k: v for k, v in tool.items() if k != "fn"}
        for tool in regime_tools_list
    ]
    all_tools = get_polygon_tools() + get_sentiment_tools() + get_research_tools() + regime_tools_serializable
    
    tool_desc_str = json.dumps(all_tools, indent=2)
    symbol_info = symbol or "N/A"

    system_prompt = """
You are a planning assistant for a trading agent. Your job is to decide which tools
to call (if any) to help answer the user's question about a specific stock.

Available tools are described in JSON format. You must ONLY choose from these tools.
Do not invent tools.

==============================================================================
CORE MARKET DATA TOOLS
==============================================================================

- polygon_price_data
  - Use when recent price action / trend / volatility / context is relevant.
  - Provides up to 2 years of daily OHLCV data with volume analysis.
  - INCLUDES ATR(14) for volatility/stop placement - no separate tool needed.
  - REQUIRED for any trade evaluation.

- polygon_ticker_details
  - Use for company information and fundamental context.
  - Provides market cap, industry, sector, description.

- polygon_technical_rsi
  - Use for momentum / overbought / oversold analysis.
  - API-calculated RSI with classifications.

- polygon_technical_macd
  - Use for trend-following and momentum crossovers.
  - API-calculated MACD with histogram and signal.

- polygon_technical_sma / polygon_technical_ema
  - Use for moving average analysis and trend identification.

- bollinger_bands
  - Calculate Bollinger Bands from price data (no API call).
  - Use for overbought/oversold conditions and volatility analysis.

- polygon_earnings
  - Use for quarterly earnings data with revenue and EPS growth rates.
  - Shows last 8 quarters with QoQ and YoY growth trends.
  - Essential for fundamental analysis and growth assessment.

- news_sentiment_finviz_finbert
  - Use when the user cares about recent news/sentiment or catalysts.

==============================================================================
PRIMARY RISK MANAGEMENT TOOLS (MUST CALL FOR ANY TRADE EVALUATION)
==============================================================================

These two tools are the PRIMARY risk overlay for all trade decisions:

★ vix_roc_risk (REQUIRED - Market Timing)
  - VIX Rate-of-Change based market timing with automatic asset tier classification.
  - Walk-forward validated: 15/15 wins on tested assets (2020-2024).
  - Auto-classifies assets into three tiers:
    * TIER 1 (Value/Cyclical): SPY, DIA, IWM, XLF, XLE - conservative params
    * TIER 2 (Growth/Tech ETFs): QQQ, AAPL, AMZN, GOOGL - aggressive params  
    * TIER 3 (Mega-Cap Tech): NVDA, MSFT, META - ultra-conservative, extreme events only
  - Returns: tier classification, current_signal ('exit'|'reenter'|'hold'), position_status
  - INTERPRETATION:
    * 'exit' signal = DO NOT ENTER - market stress, wait for re-entry
    * 'hold' + IN MARKET = Safe to trade normally
    * 'reenter' signal = VIX calming, safe to re-enter
  - Performance (2020-2024 out-of-sample):
    * Tier 1: +39% avg excess return, 7/7 wins
    * Tier 2: +20% avg excess return, 5/5 wins  
    * Tier 3: +140% avg excess return, 3/3 wins

★ vol_prediction (REQUIRED - Position Sizing)
  - Predicts volatility regime transitions for position sizing and risk management.
  - Uses VIX-based universal features that work across all equities.
  - Returns:
    * current_regime: 'LOW' or 'HIGH' volatility
    * spike_probability: P(transition to HIGH) if currently LOW
    * calm_probability: P(transition to LOW) if currently HIGH
    * risk_level: 'LOW', 'WATCH', 'ELEVATED', 'CALMING', 'HIGH'
    * suggested_action: Position sizing recommendation
    * vix_zscore: VIX relative to 60-day mean
  - INTERPRETATION:
    * spike_probability >= 0.5: Reduce position size 20-40%
    * spike_probability >= 0.6: ELEVATED risk - reduce 30-50%, tighten stops
    * calm_probability >= 0.5: Vol likely to subside, can add to winners
  - Precision: ~62% at 0.6+ threshold (vs 23% base rate for spikes)

COMBINED USAGE:
- vix_roc_risk tells you IF you should trade (IN/OUT signal)
- vol_prediction tells you HOW MUCH to bet (position sizing)
- Together they provide complete risk management framework

==============================================================================
OPTIONAL TOOLS
==============================================================================

- vix_roc_portfolio_risk
  - Portfolio-wide VIX ROC risk assessment across multiple assets.
  - Only needed for portfolio-wide checks, not single stock analysis.

- polygon_dividends / polygon_splits
  - Use for corporate actions and dividend information.

- polygon_snapshot
  - Real-time price snapshot (may require paid tier).

DEEP RESEARCH TOOLS (SLOW - 2-5 minutes each, use only when explicitly requested):
- gpt_researcher_market_research
- gpt_researcher_company_analysis
- gpt_researcher_sector_trends
- gpt_researcher_economic_indicators

==============================================================================
DEPRECATED TOOLS (DO NOT USE)
==============================================================================

The following tools have been deprecated as they do not provide reliable alpha:
- regime_detection_wasserstein (use vol_prediction instead)
- regime_detection_hmm (use vix_roc_risk instead)
- regime_consensus_check (deprecated)
- ml_prediction (deprecated - not reliable for alpha generation)

==============================================================================
TOOL SELECTION GUIDE
==============================================================================

For comprehensive trading analysis, you MUST call:
1. polygon_price_data (price action, trend, ATR for stops)
2. vix_roc_risk (PRIMARY - market timing IN/OUT signal)
3. vol_prediction (PRIMARY - position sizing)
4. polygon_technical_rsi + polygon_technical_macd (momentum/trend)
5. bollinger_bands (volatility/overbought/oversold)
6. news_sentiment_finviz_finbert (catalysts and sentiment)

Optional additions:
7. polygon_ticker_details (company context)
8. polygon_earnings (fundamental growth trends)
9. polygon_technical_sma or polygon_technical_ema (trend analysis)

NOTE: The system uses Polygon.io with 100,000 calls/month, so select 6-9 tools
for a complete picture without worrying about rate limits.

If the query is purely theoretical and not about a specific stock or current market
conditions, you may not need any tools.

Return a JSON object with a single key "tool_calls" whose value is a list of objects.
Each object must have:
  - "tool_name": the name of the tool (string)
  - "arguments": a JSON object of arguments for that tool

If no tools are needed, return {"tool_calls": []}.
""".strip()

    user_prompt = f"""
USER QUERY
----------
{user_query}

SYMBOL
------
{symbol_info}

AVAILABLE TOOLS
---------------
{tool_desc_str}

INSTRUCTIONS
------------
Decide which tools to call to best help with this query. 

REQUIRED for any trade evaluation:
- polygon_price_data
- vix_roc_risk (PRIMARY market timing)
- vol_prediction (PRIMARY position sizing)

Examples of good behavior:

1. For "Is NVDA a good buy right now for a swing trade?":
  - polygon_price_data
  - vix_roc_risk
  - vol_prediction
  - polygon_technical_rsi
  - polygon_technical_macd
  - bollinger_bands
  - news_sentiment_finviz_finbert

2. For "Is MSFT a good long-term investment?":
  - polygon_price_data
  - vix_roc_risk
  - vol_prediction
  - polygon_ticker_details
  - polygon_earnings
  - polygon_dividends
  - news_sentiment_finviz_finbert

3. For "Should I exit my SPY position?":
  - polygon_price_data
  - vix_roc_risk
  - vol_prediction
  - polygon_technical_rsi
  - polygon_technical_macd

Return ONLY valid JSON with this structure:

{{
  "tool_calls": [
    {{
      "tool_name": "...",
      "arguments": {{}}
    }}
  ]
}}
""".strip()

    # Call OpenAI directly
    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content.strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ Planner returned invalid JSON, falling back to no tools.")
        return []

    tool_calls = obj.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []
    # Basic shape enforcement
    cleaned_calls: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("tool_name")
        args = call.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            cleaned_calls.append({"tool_name": name, "arguments": args})
    return cleaned_calls
