# planner.py

from typing import Dict, Any, List
import json
import warnings

# Suppress all deprecation warnings (including LangChain)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
try:
  from langchain_core.messages import HumanMessage  # type: ignore
except Exception:
  try:
    from langchain.schema import HumanMessage  # type: ignore
  except Exception:
    from langchain_core.messages import BaseMessage
    class HumanMessage(BaseMessage):
      def __init__(self, content: str):
        super().__init__(content=content, type="human")

from polygon_tools import get_all_tools as get_polygon_tools
from sentiment_tools import get_all_tools as get_sentiment_tools
from research_tools import get_all_tools as get_research_tools
from tools import TOOL_REGISTRY as regime_tools_registry

PLANNER_MODEL = "gpt-5.1"  # or another GPT model


def plan_tools(user_query: str, symbol: str | None = None) -> List[Dict[str, Any]]:
    """
    Returns a list of tool calls:
      [{"tool_name": "...", "arguments": {...}}, ...]
    based on the user's query and available tools.
    """
    llm = ChatOpenAI(model=PLANNER_MODEL)

    # Helper to call different langchain/langchain_openai interfaces
    class _LLMRespShim:
        def __init__(self, content: str):
            self.content = content

    def _call_llm(llm_obj, messages):
        # Try direct call first
        try:
            out = llm_obj(messages)
            if hasattr(out, "content"):
                return out
            if isinstance(out, str):
                return _LLMRespShim(out)
            if hasattr(out, "generations"):
                try:
                    text = out.generations[0][0].text
                except Exception:
                    try:
                        text = out.generations[0][0].message.content
                    except Exception:
                        text = str(out)
                return _LLMRespShim(text)
        except Exception:
            pass
        # Try other common langchain methods. Many implementations accept either
        # a list of prompt strings or a single combined string; try multiple
        # calling conventions to be robust across versions.
        combined_text = None
        try:
            combined_text = "\n\n".join([m.content for m in messages if hasattr(m, "content")])
        except Exception:
            combined_text = None

        for method in ("predict_messages", "predict", "generate", "invoke", "chat", "complete"):
            if hasattr(llm_obj, method):
                try:
                    fn = getattr(llm_obj, method)
                    out = None
                    # Try: list[str]
                    if combined_text is not None:
                        try:
                            out = fn([combined_text])
                        except Exception:
                            out = None
                    # Try: single str
                    if out is None and combined_text is not None:
                        try:
                            out = fn(combined_text)
                        except Exception:
                            out = None
                    # Try: original messages object
                    if out is None:
                        try:
                            out = fn(messages)
                        except Exception:
                            out = None

                    if isinstance(out, str):
                        return _LLMRespShim(out)
                    if hasattr(out, "content"):
                        return out
                    if hasattr(out, "generations"):
                        try:
                            text = out.generations[0][0].text
                        except Exception:
                            text = str(out)
                        return _LLMRespShim(text)
                    if isinstance(out, list) and out and hasattr(out[0], "content"):
                        return out[0]
                except Exception:
                    continue

        raise TypeError("LLM object is not callable and no compatible method found")

    # Combine Polygon.io, sentiment, research, and regime detection tools
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

You should especially consider:

- polygon_price_data
  - Use when recent price action / trend / volatility / context is relevant.
  - E.g., questions like "is this a good entry now?", "what's the recent trend?"
  - Provides up to 2 years of daily OHLCV data with volume analysis

- polygon_ticker_details
  - Use for company information and fundamental context.
  - E.g., "tell me about this company", "what sector is it in?"
  - Provides market cap, industry, sector, description

- polygon_technical_rsi
  - Use for momentum / overbought / oversold analysis.
  - E.g., "is it overbought?", "is momentum strong or weak?"
  - API-calculated RSI with classifications

- polygon_technical_macd
  - Use for trend-following and momentum crossovers.
  - E.g., "is the trend strengthening?", "is there a bullish/bearish crossover?"
  - API-calculated MACD with histogram and signal

- polygon_technical_sma / polygon_technical_ema
  - Use for moving average analysis and trend identification.
  - E.g., "is price above the 50-day MA?", "what's the trend?"

NOTE: ATR (Average True Range) for volatility/stop placement is automatically calculated
in polygon_price_data - no separate tool call needed. The price data includes ATR(14)
for risk management and stop loss placement.

- bollinger_bands
  - Calculate Bollinger Bands from price data (no API call, uses existing price data).
  - Use for overbought/oversold conditions and volatility analysis.
  - Price near upper band = overbought, near lower band = oversold.
  - Narrow bandwidth ("squeeze") often precedes volatility expansion.
  - E.g., "is it overbought?", "oversold?", "volatility compression?"

- polygon_snapshot
  - Use for real-time price snapshot and today's performance.
  - Quick way to get current price, day's range, volume

- polygon_dividends / polygon_splits
  - Use for corporate actions and dividend information.
  - E.g., "does this pay dividends?", "any recent stock splits?"

- polygon_earnings
  - Use for quarterly earnings data with revenue and EPS growth rates.
  - Shows last 8 quarters with QoQ and YoY growth trends.
  - Essential for fundamental analysis and growth assessment.
  - E.g., "is revenue growing?", "earnings momentum?", "growth trends?"

- news_sentiment_finviz_finbert
  - Use when the user cares about recent news/sentiment or catalysts.
  - E.g., "any recent news?", "is sentiment bullish or bearish lately?"

REGIME DETECTION TOOLS (choose one or both based on use case):
- regime_detection_wasserstein
  - Paper-faithful Wasserstein k-means for volatility regime detection (low/med/high)
  - **BEST FOR**: Tech/healthcare stocks, when adaptivity matters, distinct volatility shifts
  - **USE WHEN**: User asks about "volatility regime", "risk level", "position sizing for volatility"
  - **STRENGTHS**: Outperformed HMM on MSFT (+0.23 Sharpe), JNJ (+0.25 Sharpe), excels when regime changes rapidly
  - **WEAKNESSES**: Can get stuck in one regime (AAPL 96% in one regime), poor cluster separation (MMD ~1.0)
  - **EXAMPLE WINS**: "MSFT in high-vol regime" when it needed aggressive adaptation

- regime_detection_hmm  
  - Rolling HMM with forward filter for probabilistic regime (bearish/sideways/bullish)
  - **BEST FOR**: Smooth transitions, when you need transition probabilities, stable predictions
  - **USE WHEN**: User asks about "market regime", "trend", "bullish vs bearish", "regime transition risk"
  - **STRENGTHS**: Won on AAPL, more stable, provides transition probabilities, good for persistent regimes
  - **WEAKNESSES**: Too slow to adapt on volatile stocks (lost -0.23 on MSFT, -0.25 on JNJ)
  - **EXAMPLE WINS**: "AAPL sideways regime with 85% persistence" when stability was correct

- regime_consensus_check
  - ONLY call AFTER both regime tools to check if they agree
  - **USE WHEN**: Regime confidence is critical to the decision (e.g., large position, high conviction needed)
  - **AGREEMENT**: Both models converge → HIGH confidence regime classification
  - **DISAGREEMENT**: Models diverge → Uncertainty signal, potential regime transition, reduce size
  - **KEY INSIGHT**: Trading books emphasize avoiding uncertain regimes - disagreement is valuable signal

REGIME TOOL SELECTION GUIDE:
- **ALWAYS call BOTH regime tools for comprehensive analysis** (volatility + trend)
- Wasserstein provides: volatility regime (low/med/high) for position sizing
- HMM provides: trend regime (bearish/sideways/bullish) for directional bias
- Together they give complete picture: "medium volatility + bullish trend"
- **BOTH tools are fast** (< 5 seconds combined) so no downside to calling both
- **Consensus check**: Only call AFTER both regime tools to check agreement

BACKGROUND (for reference, not for skipping tools):
- Tech stocks (MSFT, NVDA): Wasserstein historically outperformed (+22% avg Sharpe)
- Healthcare (JNJ, PFE): Wasserstein also stronger (JNJ: Wass +0.24 vs HMM -0.01)
- Stable/Financial (AAPL): HMM won slightly (better for persistent regimes)
BUT: Call both anyway for complete analysis - performance differs by market condition

ML PREDICTION TOOL (Multi-Model with Performance Context):
- ml_prediction
  - Machine learning predictions from 4 trained models with comprehensive performance metrics
  - **BEST FOR**: Quantitative directional forecasts, data-driven entry/exit signals, validation
  - **MODELS**: Random Forest, XGBoost, Logistic Regression, Decision Tree (all 4 run in parallel)
  - **TRAINING**: 141 optimized features, 25 stocks, +26% vs baseline, Optuna-tuned hyperparameters
  - **USE WHEN**: User needs quantitative forecast, multiple model opinions, confidence levels
  - **STRENGTHS BY HORIZON**:
    - 3-day: Random Forest (Sharpe 1.52) - best for short-term predictions
    - 5-day: XGBoost (Sharpe 1.34) - best for medium-term predictions  
    - 10-day: XGBoost (Sharpe 1.45) - best for long-term predictions
  - **OUTPUT**: Individual predictions, consensus (STRONG UP/WEAK UP/WEAK DOWN/STRONG DOWN),
    probability distributions, historical win rates, Sharpe ratios, model agreement
  - **FAST**: ~2 seconds (loads pre-trained models, no retraining)
  - **OPTIONAL BUT RECOMMENDED**: Use when user asks about "direction", "forecast", "prediction",
    "should I buy", or wants quantitative signal alongside qualitative analysis
  - **INTEGRATION**: Complements technical/regime/sentiment - provides statistical validation
  - **EXAMPLE USE**: "ML says STRONG UP with 85% consensus (3/4 models agree), best model XGBoost
    predicts UP with 78% confidence (Sharpe 1.34)"

ML TOOL SELECTION GUIDE:
- **WHEN TO CALL**: Any trade idea evaluation, entry/exit timing, directional forecast
- **WHEN TO SKIP**: Pure fundamental deep dive without near-term trade (use research tools instead)
- **HORIZON CHOICE**:
  - 3-day: Day/swing trades, short-term moves, earnings plays
  - 5-day: Week-ahead forecasts, standard position trades (DEFAULT if unclear)
  - 10-day: Position trades, trend plays, longer holds
- **INTERPRETATION**: 
  - STRONG consensus (75%+ agreement) = high confidence
  - WEAK consensus (50-75%) = moderate confidence, watch for invalidation
  - Check best model probability for additional confidence gauge
  - Win rates and Sharpe ratios show historical performance context

DEEP RESEARCH TOOLS (GPT-Researcher - takes 2-5 minutes each):
- gpt_researcher_market_research
  - Autonomous deep research on ANY topic (scrapes 20+ sources, synthesizes report)
  - Use for: market analysis, competitive landscape, industry trends, technology disruption
  - Examples: "AI chip market dynamics", "impact of rising rates on tech", "EV supply chain"
  - ONLY use when comprehensive research is needed beyond quick market data

- gpt_researcher_company_analysis
  - Deep company research: business model, competitive advantages, financial trends, outlook
  - Use when ticker details and fundamentals are insufficient for deep understanding
  - Takes 2-5 minutes - only use when truly needed

- gpt_researcher_sector_trends
  - Sector/industry analysis: growth, players, disruption, regulation, themes
  - Use for understanding broader context, not for quick sector checks

- gpt_researcher_economic_indicators
  - Macro research: GDP, inflation, rates, policy, market implications
  - Use for macro backdrop analysis, not for quick indicator checks

NOTE: Research tools are OPTIONAL and SLOW (2-5 min each). Only use when:
- User explicitly asks for "research", "deep dive", "analysis", or "comprehensive"
- Quick data tools are insufficient for the query
- You need information beyond market data (e.g., industry dynamics, macro trends)

For comprehensive trading analysis, you MUST call:
- polygon_price_data (REQUIRED - always include this for any stock query - includes ATR for volatility)
- polygon_technical_rsi + polygon_technical_macd + bollinger_bands (technical analysis)
- polygon_ticker_details (company context)
- polygon_earnings (revenue and EPS growth trends - critical for fundamentals)
- news_sentiment_finviz_finbert (recent catalysts and sentiment)
- polygon_technical_sma or polygon_technical_ema (trend analysis)
- regime_detection_wasserstein (volatility regime for position sizing)
- regime_detection_hmm (trend regime for directional bias)

NOTE: ATR for risk management is automatically included in polygon_price_data.

Optional but recommended:
- polygon_dividends or polygon_splits if relevant to the analysis
- regime_consensus_check (ONLY if you called BOTH regime tools - checks agreement)
- Research tools ONLY if user requests deep research or analysis

Do NOT call polygon_snapshot (requires paid tier, will fail with 403).

The system uses Polygon.io with 100,000 calls/month (vs Alpha Vantage's 25/day),
so select 6-8 tools for a complete picture.

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
Decide which tools to call to best help with this query. Choose only the tools that are
most relevant to the user's intent (e.g., short-term technical trading, long-term investing,
news/sentiment-driven, etc.).

Examples of good behavior:
- For "Is NVDA a good buy right now for a swing trade?":
  - polygon_price_data
  - polygon_technical_rsi
  - polygon_technical_macd
  - bollinger_bands
  - regime_detection_wasserstein
  - regime_detection_hmm
  - news_sentiment_finviz_finbert

- For "Is MSFT a good long-term investment?":
  - polygon_price_data
  - polygon_ticker_details
  - polygon_technical_rsi
  - polygon_dividends
  - regime_detection_wasserstein
  - regime_detection_hmm
  - news_sentiment_finviz_finbert

- For "Show me recent news and sentiment on TSLA":
  - news_sentiment_finviz_finbert
  - polygon_snapshot
  - polygon_price_data
  - regime_detection_wasserstein
  - regime_detection_hmm

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

    messages = [HumanMessage(content=system_prompt + "\n\n" + user_prompt)]
    resp = _call_llm(llm, messages)
    text = resp.content.strip()

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
