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

- stock_beta
  - Compute a stock's rolling beta and R² against SPY.
  - Use to assess whether a stock's market sensitivity fits the current BOCPD regime.
  - High beta (1.2+) amplifies upside in BULL regimes but amplifies losses in BEAR.
  - Low beta (<0.5) is defensive but underperforms in strong trends.
  - R² tells you how market-driven the stock is (low R² = regime signal less relevant).
  - ALWAYS call alongside bocpd_regime for regime-conditioned beta targeting.

- relative_strength
  - Compute a stock's excess return vs SPY and its sector ETF over 21d/63d/126d/252d.
  - Identifies market LEADERS (outperforming) and LAGGARDS (underperforming).
  - Composite RS score weighted toward longer timeframes for conviction.
  - In BULL regimes, prefer leaders (RS > 5). In BEAR, avoid laggards (they fall hardest).
  - Rising RS + bull transition = stock may lead the recovery.
  - ALWAYS call alongside stock_beta and bocpd_regime for complete stock selection.

- earnings_proximity
  - Check how close the next EARNINGS DATE is for a stock.
  - Flags binary event risk: earnings gaps can blow through stop-losses.
  - IMMINENT (<3d) = reduce to 25-50% sizing. HIGH (<7d) = reduce 30-50%.
  - Also detects recent earnings (<5 days) for post-earnings drift awareness.
  - ALWAYS call to avoid recommending swing trades right before earnings.

- polygon_earnings
  - Use for quarterly earnings data with revenue and EPS growth rates.
  - Shows last 8 quarters with QoQ and YoY growth trends.
  - Essential for fundamental analysis and growth assessment.

- news_sentiment_finviz_finbert
  - Use when the user cares about recent news/sentiment or catalysts.
  - Now includes topic classification (earnings, M&A, litigation, etc.) and
    topic-adjusted sentiment alongside raw FinBERT scores.

- topic_sentiment_newsdb
  - Deep sentiment analysis from local news database (210k+ S&P500 articles).
  - Classifies headlines by topic, computes topic-adjusted composite score.
  - Returns topic breakdown, high-signal alerts, and dual interpretation
    (event-driven for 7d window, contrarian for 20d window).
  - RECOMMENDED for any S&P 500 stock when sentiment context is needed.
  - Much richer signal than Finviz alone.

- earnings_topic_signal
  - Highest-alpha sentiment signal (IC ≈ +0.021, 52× raw FinBERT).
  - Filters to earnings/analyst news, checks FinBERT agreement.
  - Returns direction, confidence, expected alpha, and tradeable flag.
  - Use as a CONFIRMING factor alongside technical and price analysis.
  - Requires local news database.

==============================================================================
PRIMARY RISK MANAGEMENT TOOLS (MUST CALL FOR ANY TRADE EVALUATION)
==============================================================================

These two tools are the PRIMARY risk overlay for all trade decisions:

★ market_risk (REQUIRED - Drawdown Probability + Forward Volatility)
  - ML-based (GradientBoosting) market risk model with walk-forward validation.
  - Uses 21 VIX + SPY features to predict TWO things:
    1. P(SPY max drawdown > 3% in next 10 days) — continuous 0.0 to 1.0
    2. Predicted forward realised volatility (annualised %) — for position sizing
  - Returns:
    * drawdown_probability: Continuous probability (not binary)
    * drawdown_risk_level: LOW / MODERATE / ELEVATED / HIGH / EXTREME
    * predicted_fwd_vol_pct: Expected annualised vol for next 10 days
    * current_realized_vol_pct: Current 20-day vol for comparison
    * vol_trend: RISING / STABLE / FALLING
    * suggested_position_pct: Recommended size (1.0 = full, 0.2 = minimal)
    * suggested_stop_multiplier: How much to widen stops (1.0x to 2.0x)
    * variance_risk_premium: VIX / realised vol ratio (fear gauge)
    * top_risk_drivers: What features are driving current risk
  - INTERPRETATION:
    * drawdown_prob < 0.15: LOW risk — full position, normal stops
    * drawdown_prob 0.15-0.30: MODERATE — reduce to 85%, widen stops 1.2x
    * drawdown_prob 0.30-0.50: ELEVATED — reduce to 65%, widen stops 1.4x
    * drawdown_prob 0.50-0.70: HIGH — reduce to 40%, widen stops 1.7x
    * drawdown_prob > 0.70: EXTREME — reduce to 20% or skip trade
  - Risk is MARKET-WIDE (SPY/VIX based). For high-beta stocks, scale up.

★ vol_prediction (REQUIRED - Volatility Regime Transitions)
  - Predicts volatility regime transitions for additional risk context.
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

★ bocpd_regime (REQUIRED - Market Regime Context)
  - Bayesian Online Changepoint Detection (BOCPD) on SPY 21-day returns.
  - Classifies the current market into one of 6 regimes.
  - Returns:
    * current_regime: bull / bear / bull_transition / bear_transition / consolidation / crisis
    * risk_score: Instability score 0-1 (lower = more stable)
    * risk_level: LOW / MODERATE / ELEVATED / HIGH
    * expected_run_length: Days since last changepoint (higher = more stable)
    * trend_21d / trend_63d: Short and medium-term cumulative returns
    * volatility_ann: 21-day annualised volatility
    * interpretation: Actionable guidance for the regime
    * recent_regime_changes: Last regime transitions with dates
  - INTERPRETATION:
    * BULL: trend-following works, normal position sizing
    * BEAR: defensive, avoid breakout longs, consider hedging
    * BULL_TRANSITION: recovery in progress, smaller size, wider stops
    * BEAR_TRANSITION: weakness starting, reduce exposure, tighten stops
    * CONSOLIDATION: range-bound, low conviction, tight risk limits
    * CRISIS: capital preservation, minimal or zero equity exposure
  - Always runs on SPY (market-wide context), regardless of symbol being evaluated.

COMBINED USAGE:
- market_risk tells you HOW DANGEROUS the market is (drawdown probability + forward vol)
- vol_prediction tells you WHETHER VOL IS ABOUT TO CHANGE (regime transition probability)
- bocpd_regime tells you WHAT KIND OF MARKET we are in (bull/bear/transition/crisis)
- stock_beta tells you HOW SENSITIVE this stock is to the market (beta + R²)
- relative_strength tells you if this stock is a LEADER or LAGGARD vs market + sector
- earnings_proximity tells you if EARNINGS are imminent (binary event risk)
- Together they provide a complete risk management framework

==============================================================================
OPTIONAL TOOLS
==============================================================================

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
- regime_detection_hmm (use market_risk instead)
- regime_consensus_check (deprecated)
- ml_prediction (deprecated - not reliable for alpha generation)
- vix_roc_risk (replaced by market_risk — binary signal was uninformative 90%+ of the time)
- vix_roc_portfolio_risk (deprecated with vix_roc_risk)

==============================================================================
TOOL SELECTION GUIDE
==============================================================================

For comprehensive trading analysis, you MUST call:
1. polygon_price_data (price action, trend, ATR for stops)
2. market_risk (PRIMARY - drawdown probability + forward vol for position sizing)
3. vol_prediction (PRIMARY - volatility regime transition probabilities)
4. bocpd_regime (PRIMARY - market regime context: bull/bear/transition/crisis)
5. stock_beta (beta + R² vs SPY for regime-conditioned beta targeting)
6. relative_strength (excess returns vs SPY + sector — leader or laggard?)
7. earnings_proximity (days to next earnings — event risk check)
8. polygon_technical_rsi + polygon_technical_macd (momentum/trend)
9. bollinger_bands (volatility/overbought/oversold)
10. news_sentiment_finviz_finbert (live catalysts and sentiment)
11. topic_sentiment_newsdb (deep topic-classified sentiment from news database)
12. earnings_topic_signal (highest-alpha sentiment — confirming factor)

Optional additions:
9. polygon_ticker_details (company context)
10. polygon_earnings (fundamental growth trends)
11. polygon_technical_sma or polygon_technical_ema (trend analysis)

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
- market_risk (PRIMARY market timing + position sizing)
- vol_prediction (PRIMARY volatility regime transitions)
- bocpd_regime (PRIMARY market regime context)
- stock_beta (beta + R² for regime-conditioned sizing)
- relative_strength (leader/laggard stock selection)
- earnings_proximity (event risk check)

Examples of good behavior:

1. For "Is NVDA a good buy right now for a swing trade?":
  - polygon_price_data
  - market_risk
  - vol_prediction
  - bocpd_regime
  - stock_beta
  - relative_strength
  - earnings_proximity
  - polygon_technical_rsi
  - polygon_technical_macd
  - bollinger_bands
  - news_sentiment_finviz_finbert

2. For "Is MSFT a good long-term investment?":
  - polygon_price_data
  - market_risk
  - vol_prediction
  - bocpd_regime
  - stock_beta
  - relative_strength
  - earnings_proximity
  - polygon_ticker_details
  - polygon_earnings
  - polygon_dividends
  - news_sentiment_finviz_finbert

3. For "Should I exit my SPY position?":
  - polygon_price_data
  - market_risk
  - vol_prediction
  - bocpd_regime
  - stock_beta
  - relative_strength
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
