# research_tools.py

"""
Deep research tools using GPT-Researcher for comprehensive market research.

GPT-Researcher conducts autonomous deep research on any topic by:
1. Planning research questions
2. Scraping multiple web sources (20+)
3. Synthesizing information into comprehensive reports
4. Providing citations and sources

Perfect for:
- Market analysis and sector trends
- Company deep dives and competitive analysis
- Economic indicators and macro research
- Technology/industry disruption research
- Regulatory and policy research
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import requests
import datetime as dt
import logging

load_dotenv()

# ============================================================================
# CONFIGURATION - Adjust these for testing vs production
# ============================================================================
# For TESTING (faster, cheaper):
#   MAX_ITERATIONS=1, MAX_SEARCH_RESULTS_PER_QUERY=3, TOTAL_WORDS=800
# For PRODUCTION (comprehensive):
#   MAX_ITERATIONS=3, MAX_SEARCH_RESULTS_PER_QUERY=5, TOTAL_WORDS=1200

# Set this to True for faster/cheaper testing, False for full research
TESTING_MODE = os.getenv("GPT_RESEARCHER_TESTING_MODE", "true").lower() == "true"

if TESTING_MODE:
    # Reduced settings for testing - ~1-2 minutes per research, ~$0.05 per report
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "1"))  # Default: 3
    MAX_SEARCH_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "3"))  # Default: 5
    TOTAL_WORDS = int(os.getenv("TOTAL_WORDS", "800"))  # Default: 1200
    MAX_SCRAPER_WORKERS = int(os.getenv("MAX_SCRAPER_WORKERS", "10"))  # Default: 15
    logger = logging.getLogger(__name__)
    logger.info("GPT-Researcher TESTING MODE: %s iterations, %s results/query, %s words",
                MAX_ITERATIONS, MAX_SEARCH_RESULTS_PER_QUERY, TOTAL_WORDS)
else:
    # Full research settings - 2-5 minutes per research, ~$0.10-0.30 per report
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    MAX_SEARCH_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "5"))
    TOTAL_WORDS = int(os.getenv("TOTAL_WORDS", "1200"))
    MAX_SCRAPER_WORKERS = int(os.getenv("MAX_SCRAPER_WORKERS", "15"))
    logger = logging.getLogger(__name__)
    logger.info("GPT-Researcher PRODUCTION MODE: %s iterations, %s results/query, %s words",
                MAX_ITERATIONS, MAX_SEARCH_RESULTS_PER_QUERY, TOTAL_WORDS)

# Tool registry
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(tool: Dict[str, Any]) -> None:
    """Register a research tool."""
    TOOL_REGISTRY[tool["name"]] = tool


def get_all_tools() -> List[Dict[str, Any]]:
    """Get all research tools in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOL_REGISTRY.values()
    ]


def get_tool_function(name: str):
    """Get the function implementation for a tool."""
    tool = TOOL_REGISTRY.get(name)
    return tool["fn"] if tool else None


# ==================== TOOL IMPLEMENTATIONS ====================

def gpt_researcher_market_research_tool_fn(state: dict, args: dict) -> dict:
    """
    Conduct deep market research using GPT-Researcher.
    
    Uses autonomous research agent to:
    - Generate research questions
    - Scrape 20+ web sources
    - Synthesize information
    - Provide comprehensive report with citations
    
    Returns detailed research report with sources and context.
    """
    try:
        from gpt_researcher import GPTResearcher
    except Exception as e:
        # Catch all import errors including binary/compiled-extension issues
        state["tool_results"]["gpt_researcher_market_research"] = {
            "error": (
                "Failed to import gpt-researcher or its dependencies: "
                + str(e)
                + ".\nPossible causes: gpt-researcher not installed, missing optional packages (spacy/thinc), or binary incompatibility with your installed numpy."
                + "\nSuggested fixes: run `pip install -r requirements_research.txt` in a fresh venv, or install a numpy version compatible with your compiled packages (e.g., try `pip install --force-reinstall --no-deps numpy==1.24.4`)."
            )
        }
        return state
    
    query = args.get("query", "")
    report_type = args.get("report_type", "research_report")
    
    if not query:
        state["tool_results"]["gpt_researcher_market_research"] = {
            "error": "Query is required"
        }
        return state
    
    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        state["tool_results"]["gpt_researcher_market_research"] = {
            "error": "OPENAI_API_KEY not set in environment"
        }
        return state
    
    if not os.environ.get("TAVILY_API_KEY"):
        state["tool_results"]["gpt_researcher_market_research"] = {
            "error": "TAVILY_API_KEY not set in environment (get one free at https://tavily.com)"
        }
        return state
    
    try:
        import asyncio
        
        async def do_research():
            # Initialize researcher with config
            # Pass config via environment variables (GPT-Researcher reads from os.environ)
            os.environ["MAX_ITERATIONS"] = str(MAX_ITERATIONS)
            os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = str(MAX_SEARCH_RESULTS_PER_QUERY)
            os.environ["TOTAL_WORDS"] = str(TOTAL_WORDS)
            os.environ["MAX_SCRAPER_WORKERS"] = str(MAX_SCRAPER_WORKERS)
            
            researcher = GPTResearcher(query=query, report_type=report_type)
            
            # Conduct research (scrapes web, generates questions, synthesizes)
            research_result = await researcher.conduct_research()
            
            # Generate report
            report = await researcher.write_report()
            
            # Get additional information
            sources = researcher.get_research_sources()
            source_urls = researcher.get_source_urls()
            costs = researcher.get_costs()
            images = researcher.get_research_images()
            
            return {
                "report": report,
                "research_result": research_result,
                "sources": sources[:10],  # Limit to top 10 sources
                "source_urls": source_urls[:10],
                "num_sources": len(sources),
                "num_images": len(images),
                "costs": costs
            }
        
        # Run async research
        result = asyncio.run(do_research())
        
        state["tool_results"]["gpt_researcher_market_research"] = {
            "query": query,
            "report_type": report_type,
            "report": result["report"],
            "sources": result["sources"],
            "source_urls": result["source_urls"],
            "num_sources": result["num_sources"],
            "num_images": result["num_images"],
            "costs": result["costs"]
        }
        
    except Exception as e:
        state["tool_results"]["gpt_researcher_market_research"] = {
            "query": query,
            "error": f"Research failed: {str(e)}"
        }
    
    return state


def gpt_researcher_company_analysis_tool_fn(state: dict, args: dict) -> dict:
    """
    Deep dive company analysis using GPT-Researcher.
    
    Researches:
    - Company background and history
    - Business model and revenue streams
    - Competitive landscape
    - Recent news and developments
    - Financial performance trends
    - Future outlook and catalysts
    """
    symbol = args.get("symbol", "")
    company_name = args.get("company_name", "")
    
    if not symbol and not company_name:
        state["tool_results"]["gpt_researcher_company_analysis"] = {
            "error": "Symbol or company_name is required"
        }
        return state
    
    # Construct detailed research query
    identifier = company_name if company_name else f"{symbol} stock"
    query = (
        f"Comprehensive analysis of {identifier}: "
        f"business model, competitive advantages, financial performance, "
        f"recent developments, risk factors, and growth outlook"
    )
    
    # Call the main research function
    temp_state = {"tool_results": {}}
    result_state = gpt_researcher_market_research_tool_fn(
        temp_state,
        {"query": query, "report_type": "research_report"}
    )
    
    # Copy result to company analysis key
    if "gpt_researcher_market_research" in result_state["tool_results"]:
        result = result_state["tool_results"]["gpt_researcher_market_research"]
        state["tool_results"]["gpt_researcher_company_analysis"] = {
            "symbol": symbol,
            "company_name": company_name,
            **result
        }
    
    return state


def gpt_researcher_sector_trends_tool_fn(state: dict, args: dict) -> dict:
    """
    Research sector/industry trends and dynamics.
    
    Analyzes:
    - Industry growth trends
    - Key players and market share
    - Disruption and innovation
    - Regulatory environment
    - Supply chain dynamics
    - Investment themes
    """
    sector = args.get("sector", "")
    timeframe = args.get("timeframe", "current and next 12 months")
    
    if not sector:
        state["tool_results"]["gpt_researcher_sector_trends"] = {
            "error": "Sector is required"
        }
        return state
    
    query = (
        f"Analysis of {sector} sector trends for {timeframe}: "
        f"market dynamics, key players, growth drivers, challenges, "
        f"innovation, disruption, and investment opportunities"
    )
    
    temp_state = {"tool_results": {}}
    result_state = gpt_researcher_market_research_tool_fn(
        temp_state,
        {"query": query, "report_type": "research_report"}
    )
    
    if "gpt_researcher_market_research" in result_state["tool_results"]:
        result = result_state["tool_results"]["gpt_researcher_market_research"]
        state["tool_results"]["gpt_researcher_sector_trends"] = {
            "sector": sector,
            "timeframe": timeframe,
            **result
        }
    
    return state


def gpt_researcher_economic_indicators_tool_fn(state: dict, args: dict) -> dict:
    """
    Research economic indicators and macro trends.
    
    Covers:
    - GDP, inflation, employment
    - Interest rates and monetary policy
    - Consumer sentiment
    - Manufacturing and services PMI
    - Trade and fiscal policy
    - Market implications
    """
    topic = args.get("topic", "current US economic conditions")
    
    query = (
        f"Economic analysis: {topic} - "
        f"key indicators, central bank policy, economic outlook, "
        f"and implications for financial markets"
    )
    
    temp_state = {"tool_results": {}}
    result_state = gpt_researcher_market_research_tool_fn(
        temp_state,
        {"query": query, "report_type": "research_report"}
    )
    
    if "gpt_researcher_market_research" in result_state["tool_results"]:
        result = result_state["tool_results"]["gpt_researcher_market_research"]
        state["tool_results"]["gpt_researcher_economic_indicators"] = {
            "topic": topic,
            **result
        }
    
    return state


# ==================== REGISTER TOOLS ====================

register_tool({
    "name": "gpt_researcher_market_research",
    "description": (
        "Conduct deep autonomous web research on any market-related topic. "
        "GPT-Researcher scrapes 20+ sources, generates research questions, "
        "and synthesizes comprehensive reports with citations. "
        "Use for: market analysis, competitive landscape, industry trends, "
        "technology disruption, regulatory changes, or any topic requiring "
        "comprehensive research. Takes 2-5 minutes to complete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Research query/question. Be specific and comprehensive. "
                    "Examples: 'AI chip market competitive dynamics and growth outlook', "
                    "'Impact of rising interest rates on tech valuations', "
                    "'Electric vehicle supply chain and battery technology trends'"
                )
            },
            "report_type": {
                "type": "string",
                "enum": ["research_report", "resource_report", "outline_report"],
                "description": (
                    "Type of report to generate. "
                    "research_report: Comprehensive analysis with synthesis (default). "
                    "resource_report: List of resources and summaries. "
                    "outline_report: Structured outline for further research."
                ),
                "default": "research_report"
            }
        },
        "required": ["query"]
    },
    "fn": gpt_researcher_market_research_tool_fn
})

register_tool({
    "name": "gpt_researcher_company_analysis",
    "description": (
        "Deep dive company research and analysis. "
        "Comprehensive report covering: business model, competitive advantages, "
        "financial performance, recent developments, risk factors, and outlook. "
        "Use when you need detailed company understanding beyond basic fundamentals."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'NVDA', 'TSLA')"
            },
            "company_name": {
                "type": "string",
                "description": "Company name (e.g., 'Nvidia Corp', 'Tesla Inc')"
            }
        },
        "required": []  # At least one of symbol or company_name needed (checked in function)
    },
    "fn": gpt_researcher_company_analysis_tool_fn
})

register_tool({
    "name": "gpt_researcher_sector_trends",
    "description": (
        "Research sector/industry trends and dynamics. "
        "Analyzes: growth trends, key players, disruption, regulation, "
        "supply chains, and investment themes. "
        "Use for understanding broader industry context for a trade."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sector": {
                "type": "string",
                "description": (
                    "Sector or industry to research. "
                    "Examples: 'semiconductors', 'cloud computing', 'biotechnology', "
                    "'renewable energy', 'fintech', 'e-commerce'"
                )
            },
            "timeframe": {
                "type": "string",
                "description": "Time horizon for analysis (default: 'current and next 12 months')",
                "default": "current and next 12 months"
            }
        },
        "required": ["sector"]
    },
    "fn": gpt_researcher_sector_trends_tool_fn
})

register_tool({
    "name": "gpt_researcher_economic_indicators",
    "description": (
        "Research economic indicators and macro trends. "
        "Covers: GDP, inflation, employment, rates, policy, sentiment, "
        "and market implications. "
        "Use for understanding macro backdrop for trading decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Economic topic to research. "
                    "Examples: 'current US economic conditions', 'inflation outlook', "
                    "'Federal Reserve policy and rate path', 'recession probability'"
                ),
                "default": "current US economic conditions"
            }
        },
        "required": []
    },
    "fn": gpt_researcher_economic_indicators_tool_fn
})


# ==================== FUNDAMENTALS & MACRO TOOLS ====================


def fetch_fundamentals_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch quick fundamentals for a symbol using multiple fallbacks:
      1) Alpha Vantage OVERVIEW (if available)
      2) Yahoo Finance quoteSummary JSON

    Returns a dict with common fields: symbol, marketCap, trailingPE, forwardPE,
    profitMargins, operatingMargins, revenueGrowth, revenue (latest), eps, and raw_source.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args.get("symbol") or args.get("ticker")
    if not symbol:
        state["tool_results"]["fetch_fundamentals"] = {"error": "symbol (ticker) is required"}
        return state

    symbol = symbol.upper()

    result = {"symbol": symbol, "raw_sources": []}

    # 1) Try Alpha Vantage OVERVIEW if API key present (best effort)
    av_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if av_key:
        try:
            av_url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": symbol, "apikey": av_key}
            r = requests.get(av_url, params=params, timeout=15)
            data = r.json()
            if data and not data.get("Note") and data.get("Symbol"):
                # Map useful fields
                result.update({
                    "marketCap": float(data.get("MarketCapitalization")) if data.get("MarketCapitalization") else None,
                    "pe_trailing": float(data.get("PERatio")) if data.get("PERatio") else None,
                    "peg": float(data.get("PEGRatio")) if data.get("PEGRatio") else None,
                    "eps": float(data.get("EPS")) if data.get("EPS") else None,
                    "profitMargin": float(data.get("ProfitMargin")) if data.get("ProfitMargin") else None,
                    "grossMargins": float(data.get("GrossMargins")) if data.get("GrossMargins") else None,
                })
                result["raw_sources"].append({"source": "alphavantage_overview", "data": data})
        except Exception as e:
            # Non-fatal; continue to other sources
            result.setdefault("warnings", []).append(f"AlphaVantage error: {e}")

    # 2) Yahoo Finance quoteSummary JSON (no API key required)
    try:
        y_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        modules = ",".join([
            "price",
            "summaryDetail",
            "financialData",
            "defaultKeyStatistics",
            "earnings",
            "incomeStatementHistory",
        ])
        r = requests.get(y_url, params={"modules": modules}, timeout=15)
        yj = r.json()
        quote = yj.get("quoteSummary", {}).get("result", [{}])[0]
        # Extract common fields safely
        def _get(path: List[str], default=None):
            node = quote
            for p in path:
                if not isinstance(node, dict):
                    return default
                node = node.get(p)
                if node is None:
                    return default
            return node

        price = _get(["price"])
        fin = _get(["financialData"]) or {}
        stats = _get(["defaultKeyStatistics"]) or {}

        result.update({
            "longName": _get(["price", "longName"]) or _get(["price", "shortName"]),
            "marketCap": _get(["price", "marketCap", "raw"]) or result.get("marketCap"),
            "currency": _get(["price", "currency"]),
            "pe_trailing": _get(["summaryDetail", "trailingPE", "raw"]) or _get(["defaultKeyStatistics", "trailingPE", "raw"]),
            "pe_forward": _get(["financialData", "forwardPE", "raw"]) or None,
            "profitMargins": _get(["financialData", "profitMargins", "raw"]) or None,
            "operatingMargins": _get(["financialData", "operatingMargins", "raw"]) or None,
            "revenueGrowth": _get(["financialData", "revenueGrowth", "raw"]) or None,
            "earningsPerShare": _get(["financialData", "earningsPerShare", "raw"]) or None,
        })

        result["raw_sources"].append({"source": "yahoo_quoteSummary", "data": quote})
    except Exception as e:
        result.setdefault("warnings", []).append(f"Yahoo summary error: {e}")

    # Timestamp
    result["fetched_at"] = dt.datetime.utcnow().isoformat() + "Z"

    state["tool_results"]["fetch_fundamentals"] = result
    return state


def fred_macro_indicators_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch macro indicators from FRED. Args:
      - series_ids: optional list of FRED series IDs (if omitted, a default set is used)
      - observation_start: optional YYYY-MM-DD to limit history (default: 2020-01-01)

    Default series fetched (best-effort):
      - DGS10 (10-year Treasury)
      - DGS2 (2-year Treasury)
      - TEDRATE (TED spread)
      - M2SL (M2 money stock)
      - BAMLC0A0CM (BofA US Corporate BBB Option-Adjusted Spread) -- if available

    Requires FRED_API_KEY in environment for reliable usage.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        state["tool_results"]["fred_macro_indicators"] = {"error": "FRED_API_KEY not set in environment"}
        return state

    series_ids = args.get("series_ids") or [
        "DGS10",
        "DGS2",
        "TEDRATE",
        "M2SL",
        "BAMLC0A0CM",
    ]

    obs_start = args.get("observation_start", "2020-01-01")
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    results = {"requested_series": series_ids, "observations": {}, "fetched_at": dt.datetime.utcnow().isoformat() + "Z"}

    for sid in series_ids:
        try:
            r = requests.get(base_url, params={"series_id": sid, "api_key": fred_key, "file_type": "json", "observation_start": obs_start}, timeout=15)
            j = r.json()
            if "observations" in j:
                # Keep last 30 observations and the latest value
                obs = j["observations"]
                latest = None
                # find latest non-null value from end
                for o in reversed(obs):
                    if o.get("value") not in (".", None, ""):
                        latest = {"date": o.get("date"), "value": o.get("value")}
                        break
                results["observations"][sid] = {"latest": latest, "count": len(obs)}
            else:
                results["observations"][sid] = {"error": j.get("error", "no observations returned")}
        except Exception as e:
            results["observations"][sid] = {"error": str(e)}

    # Compute term spread if DGS10 and DGS2 present
    try:
        d10 = results["observations"].get("DGS10", {}).get("latest", {}).get("value")
        d2 = results["observations"].get("DGS2", {}).get("latest", {}).get("value")
        if d10 and d2 and d10 not in (".",) and d2 not in (".",):
            try:
                term_spread = float(d10) - float(d2)
                results["term_spread_10y_2y"] = term_spread
            except Exception:
                results["term_spread_10y_2y"] = None
    except Exception:
        results.setdefault("warnings", []).append("Could not compute term spread")

    state["tool_results"]["fred_macro_indicators"] = results
    return state


register_tool({
    "name": "fetch_fundamentals",
    "description": (
        "Fetch quick fundamentals for a ticker using Alpha Vantage (if available) and Yahoo Finance fallback. "
        "Returns P/E, forward P/E, margins, revenue growth, EPS, market cap and raw sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "ticker": {"type": "string"}},
        "required": ["symbol"]
    },
    "fn": fetch_fundamentals_tool_fn,
})


register_tool({
    "name": "fred_macro_indicators",
    "description": (
        "Fetch macroeconomic indicators from FRED (St. Louis Fed). "
        "By default returns DGS10, DGS2, TEDRATE, M2SL, and BAMLC0A0CM. "
        "Requires FRED_API_KEY in environment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "series_ids": {"type": "array", "items": {"type": "string"}},
            "observation_start": {"type": "string", "format": "date"}
        },
        "required": []
    },
    "fn": fred_macro_indicators_tool_fn,
})

