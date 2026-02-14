# tools.py

import os
import datetime as dt
from typing import Callable, Any, Dict, TypedDict, List

from dotenv import load_dotenv

load_dotenv()  # load .env so ALPHAVANTAGE_API_KEY etc. are available

import requests
from bs4 import BeautifulSoup

# Optional ML imports for FinBERT sentiment
_TORCH_AVAILABLE = False
torch = None
F = None
AutoTokenizer = None
AutoModelForSequenceClassification = None
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
    # Optional: quieter tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except (ImportError, OSError):
    # OSError can occur with DLL loading issues on Windows
    pass


# =============================================================================
# Generic Tool Registry
# =============================================================================

class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict, dict], dict]  # (state, args) -> state


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(tool: ToolSpec):
    TOOL_REGISTRY[tool["name"]] = tool


# =============================================================================
# Alpha Vantage Core Config
# =============================================================================

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


# =============================================================================
# 1) Alpha Vantage: Daily Price Data
# =============================================================================

def alpha_vantage_price_data_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch daily OHLC data for a symbol using Alpha Vantage and store a summary
    in state["tool_results"]["alpha_vantage_price_data"].

    args:
      symbol: str (required)
      outputsize: "compact" | "full" (default: "compact")
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    if not ALPHAVANTAGE_API_KEY:
        state["tool_results"]["alpha_vantage_price_data"] = {
            "symbol": args.get("symbol"),
            "error": "ALPHAVANTAGE_API_KEY not set",
        }
        return state

    symbol = args["symbol"]
    outputsize = args.get("outputsize", "compact")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # Changed from TIME_SERIES_DAILY_ADJUSTED (premium) to free tier
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    time_key = "Time Series (Daily)"
    ts = data.get(time_key)
    if ts is None:
        state["tool_results"]["alpha_vantage_price_data"] = {
            "symbol": symbol,
            "error": data.get("Note") or data.get("Error Message") or "Unknown error",
        }
        return state

    # Convert to list of bars sorted ascending by date
    bars = []
    for date_str, row in ts.items():
        bars.append(
            {
                "timestamp": date_str,
                "open": float(row["1. open"]),
                "high": float(row["2. high"]),
                "low": float(row["3. low"]),
                "close": float(row["4. close"]),
                "volume": float(row["5. volume"]),  # Changed from "6. volume" (adjusted) to "5. volume" (regular)
            }
        )
    bars.sort(key=lambda x: x["timestamp"])

    if bars:
        first_close = bars[0]["close"]
        last_close = bars[-1]["close"]
        pct_change = (last_close / first_close - 1) * 100 if first_close != 0 else None
    else:
        first_close = last_close = pct_change = None

    # Extract close prices and dates for MACD calculation (newest first)
    sorted_bars = sorted(bars, key=lambda x: x["timestamp"], reverse=True)
    close_prices = [bar["close"] for bar in sorted_bars]
    dates = [bar["timestamp"] for bar in sorted_bars]
    volumes = [bar["volume"] for bar in sorted_bars]
    
    # Volume analysis
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    latest_volume = volumes[0] if volumes else 0
    volume_ratio = (latest_volume / avg_volume) if avg_volume > 0 else 1.0
    
    # Classify volume
    if volume_ratio > 2.0:
        volume_condition = "very high (2x+ average)"
    elif volume_ratio > 1.5:
        volume_condition = "high (1.5x+ average)"
    elif volume_ratio > 1.2:
        volume_condition = "above average"
    elif volume_ratio < 0.5:
        volume_condition = "very low (<0.5x average)"
    elif volume_ratio < 0.8:
        volume_condition = "below average"
    else:
        volume_condition = "average"
    
    # Find volume spikes in recent 5 bars
    recent_volume_spikes = []
    for i, bar in enumerate(sorted_bars[:5]):
        bar_vol_ratio = bar["volume"] / avg_volume if avg_volume > 0 else 1.0
        if bar_vol_ratio > 1.5:
            recent_volume_spikes.append({
                "date": bar["timestamp"],
                "volume": bar["volume"],
                "ratio": round(bar_vol_ratio, 2),
                "price_change_pct": round(((bar["close"] - bar["open"]) / bar["open"] * 100), 2) if bar["open"] != 0 else 0
            })

    summary = {
        "symbol": symbol.upper(),
        "interval": "daily",
        "num_bars": len(bars),
        "first_close": first_close,
        "last_close": last_close,
        "pct_change_over_period": pct_change,
        "recent_bars": bars[-5:],  # last 5 daily bars
        "close_prices": close_prices,  # for MACD calculation
        "dates": dates,  # for MACD calculation
        "volumes": volumes,  # for volume analysis
        # Volume analytics
        "avg_volume": round(avg_volume, 0),
        "latest_volume": round(latest_volume, 0),
        "volume_ratio": round(volume_ratio, 2),
        "volume_condition": volume_condition,
        "recent_volume_spikes": recent_volume_spikes,
    }

    state["tool_results"]["alpha_vantage_price_data"] = summary
    return state


register_tool(
    {
        "name": "alpha_vantage_price_data",
        "description": (
            "Fetch recent daily OHLC price data for a stock symbol using Alpha Vantage. "
            "Useful for understanding recent trend and volatility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "outputsize": {
                    "type": "string",
                    "enum": ["compact", "full"],
                    "default": "compact",
                },
            },
            "required": ["symbol"],
        },
        "fn": alpha_vantage_price_data_tool_fn,
    }
)


# =============================================================================
# 2) Alpha Vantage: Technical Indicators (RSI, MACD, ATR)
# =============================================================================

def _alpha_vantage_get_json(params: dict) -> dict:
    """
    Helper to call Alpha Vantage and return JSON or an error wrapper.
    """
    url = "https://www.alphavantage.co/query"
    try:
        r = requests.get(url, params=params, timeout=20)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def alpha_vantage_rsi_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch RSI indicator from Alpha Vantage and store in
    state["tool_results"]["alpha_vantage_rsi"].

    args:
      symbol: str (required)
      interval: "daily" | "weekly" | "monthly" (default: "daily")
      time_period: int (default: 14)
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    if not ALPHAVANTAGE_API_KEY:
        state["tool_results"]["alpha_vantage_rsi"] = {
            "symbol": args.get("symbol"),
            "error": "ALPHAVANTAGE_API_KEY not set",
        }
        return state

    symbol = args["symbol"]
    interval = args.get("interval", "daily")
    time_period = int(args.get("time_period", 14))

    params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    data = _alpha_vantage_get_json(params)

    ta_key = "Technical Analysis: RSI"
    ta = data.get(ta_key)
    if ta is None:
        state["tool_results"]["alpha_vantage_rsi"] = {
            "symbol": symbol,
            "error": data.get("Note") or data.get("Error Message") or "RSI unavailable",
        }
        return state

    latest_date = sorted(ta.keys())[-1]
    latest_rsi = float(ta[latest_date]["RSI"])

    summary = {
        "symbol": symbol.upper(),
        "indicator": "RSI",
        "interval": interval,
        "time_period": time_period,
        "latest_date": latest_date,
        "latest_rsi": latest_rsi,
    }
    state["tool_results"]["alpha_vantage_rsi"] = summary
    return state


register_tool(
    {
        "name": "alpha_vantage_rsi",
        "description": (
            "Fetch RSI (Relative Strength Index) from Alpha Vantage. "
            "Useful for overbought/oversold and momentum context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly"],
                    "default": "daily",
                },
                "time_period": {
                    "type": "integer",
                    "default": 14,
                },
            },
            "required": ["symbol"],
        },
        "fn": alpha_vantage_rsi_tool_fn,
    }
)


def _calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    # Calculate EMA for remaining values
    for price in prices[period:]:
        ema_value = (price - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_value)
    
    return ema


def _calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> dict:
    """
    Calculate Bollinger Bands from price data.
    Returns: {"middle": [SMA], "upper": [upper band], "lower": [lower band], "bandwidth": [%]}
    """
    if len(prices) < period:
        return {"middle": [], "upper": [], "lower": [], "bandwidth": []}
    
    middle = []  # SMA
    upper = []
    lower = []
    bandwidth = []
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        sma = sum(window) / period
        
        # Calculate standard deviation
        variance = sum((x - sma) ** 2 for x in window) / period
        std_dev = variance ** 0.5
        
        middle.append(sma)
        upper.append(sma + num_std * std_dev)
        lower.append(sma - num_std * std_dev)
        
        # Bandwidth as % of middle band (measures volatility)
        bw = (2 * num_std * std_dev / sma) * 100 if sma != 0 else 0
        bandwidth.append(bw)
    
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth
    }


def alpha_vantage_macd_tool_fn(state: dict, args: dict) -> dict:
    """
    Calculate MACD (12, 26, 9) from price data and store in
    state["tool_results"]["alpha_vantage_macd"].
    
    MACD is calculated as: EMA(12) - EMA(26)
    Signal line is: EMA(9) of MACD
    Histogram is: MACD - Signal

    args:
      symbol: str (required)
      interval: "daily" | "weekly" | "monthly" (default: "daily")
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    # Check if we already have price data in state
    price_result = state.get("tool_results", {}).get("alpha_vantage_price_data")
    
    if not price_result or "error" in price_result:
        # Need to fetch price data first
        symbol = args["symbol"]
        interval = args.get("interval", "daily")
        
        if not ALPHAVANTAGE_API_KEY:
            state["tool_results"]["alpha_vantage_macd"] = {
                "symbol": symbol,
                "error": "ALPHAVANTAGE_API_KEY not set",
            }
            return state
        
        # Fetch price data (compact = 100 bars, enough for MACD)
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # free tier only supports compact
            "apikey": ALPHAVANTAGE_API_KEY,
        }
        data = _alpha_vantage_get_json(params)
        
        ts_key = "Time Series (Daily)"
        ts = data.get(ts_key)
        if ts is None:
            state["tool_results"]["alpha_vantage_macd"] = {
                "symbol": symbol,
                "error": data.get("Note") or data.get("Error Message") or "Price data unavailable for MACD calculation",
            }
            return state
        
        # Extract close prices (most recent first, so reverse for calculation)
        sorted_dates = sorted(ts.keys(), reverse=True)
        closes = [float(ts[d]["4. close"]) for d in sorted_dates]
    else:
        # Use existing price data from state
        symbol = price_result["symbol"]
        interval = args.get("interval", "daily")
        closes = price_result.get("close_prices", [])
        sorted_dates = price_result.get("dates", [])
        
        if not closes or len(closes) < 35:  # Need at least 26 + 9 periods
            state["tool_results"]["alpha_vantage_macd"] = {
                "symbol": symbol,
                "error": "Insufficient price data for MACD calculation (need 35+ bars)",
            }
            return state
    
    # Reverse to oldest->newest for EMA calculation
    closes_reversed = list(reversed(closes))
    
    # Calculate MACD
    if len(closes_reversed) < 35:
        state["tool_results"]["alpha_vantage_macd"] = {
            "symbol": symbol,
            "error": "Insufficient price data for MACD calculation (need 35+ bars)",
        }
        return state
    
    # Calculate EMAs
    ema_12 = _calculate_ema(closes_reversed, 12)
    ema_26 = _calculate_ema(closes_reversed, 26)
    
    # MACD line = EMA(12) - EMA(26)
    # Start from index 26 (where both EMAs exist)
    macd_line = [ema_12[i + (26 - 12)] - ema_26[i] for i in range(len(ema_26))]
    
    # Signal line = EMA(9) of MACD
    signal_line = _calculate_ema(macd_line, 9)
    
    # Histogram = MACD - Signal (starting from where signal exists)
    histogram = [macd_line[i + (len(macd_line) - len(signal_line))] - signal_line[i] 
                 for i in range(len(signal_line))]
    
    # Get the most recent values
    latest_date = sorted_dates[0] if sorted_dates else "N/A"
    
    summary = {
        "symbol": symbol.upper(),
        "indicator": "MACD",
        "interval": interval,
        "latest_date": latest_date,
        "macd": round(macd_line[-1], 4),
        "signal": round(signal_line[-1], 4),
        "histogram": round(histogram[-1], 4),
    }
    state["tool_results"]["alpha_vantage_macd"] = summary
    return state


register_tool(
    {
        "name": "alpha_vantage_macd",
        "description": (
            "Fetch MACD (12, 26, 9) from Alpha Vantage. "
            "Useful for trend strength and momentum crossovers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly"],
                    "default": "daily",
                },
            },
            "required": ["symbol"],
        },
        "fn": alpha_vantage_macd_tool_fn,
    }
)


def alpha_vantage_atr_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch ATR (Average True Range) from Alpha Vantage and store in
    state["tool_results"]["alpha_vantage_atr"].

    args:
      symbol: str (required)
      interval: "daily" | "weekly" | "monthly" (default: "daily")
      time_period: int (default: 14)
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    if not ALPHAVANTAGE_API_KEY:
        state["tool_results"]["alpha_vantage_atr"] = {
            "symbol": args.get("symbol"),
            "error": "ALPHAVANTAGE_API_KEY not set",
        }
        return state

    symbol = args["symbol"]
    interval = args.get("interval", "daily")
    time_period = int(args.get("time_period", 14))

    params = {
        "function": "ATR",
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    data = _alpha_vantage_get_json(params)

    ta_key = "Technical Analysis: ATR"
    ta = data.get(ta_key)
    if ta is None:
        state["tool_results"]["alpha_vantage_atr"] = {
            "symbol": symbol,
            "error": data.get("Note") or data.get("Error Message") or "ATR unavailable",
        }
        return state

    latest_date = sorted(ta.keys())[-1]
    latest_atr = float(ta[latest_date]["ATR"])

    summary = {
        "symbol": symbol.upper(),
        "indicator": "ATR",
        "interval": interval,
        "time_period": time_period,
        "latest_date": latest_date,
        "latest_atr": latest_atr,
    }
    state["tool_results"]["alpha_vantage_atr"] = summary
    return state


register_tool(
    {
        "name": "alpha_vantage_atr",
        "description": (
            "Fetch ATR (Average True Range) from Alpha Vantage. "
            "Useful for volatility and position sizing / stop distance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly"],
                    "default": "daily",
                },
                "time_period": {
                    "type": "integer",
                    "default": 14,
                },
            },
            "required": ["symbol"],
        },
        "fn": alpha_vantage_atr_tool_fn,
    }
)


def bollinger_bands_tool_fn(state: dict, args: dict) -> dict:
    """
    Calculate Bollinger Bands from price data.
    Uses 20-period SMA ± 2 standard deviations by default.
    
    args:
      symbol: str (required)
      period: int (default 20)
      num_std: float (default 2.0)
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    symbol = args["symbol"]
    period = int(args.get("period", 20))
    num_std = float(args.get("num_std", 2.0))
    
    # Get price data from state or fetch it
    price_result = state.get("tool_results", {}).get("alpha_vantage_price_data")
    
    if not price_result or "error" in price_result:
        state["tool_results"]["bollinger_bands"] = {
            "symbol": symbol,
            "error": "Price data not available for Bollinger Bands calculation",
        }
        return state
    
    close_prices = price_result.get("close_prices", [])
    dates = price_result.get("dates", [])
    
    if not close_prices or len(close_prices) < period:
        state["tool_results"]["bollinger_bands"] = {
            "symbol": symbol,
            "error": f"Insufficient price data for Bollinger Bands (need {period}+ bars)",
        }
        return state
    
    # Reverse to oldest->newest for calculation
    closes_reversed = list(reversed(close_prices))
    
    # Calculate Bollinger Bands
    bb = _calculate_bollinger_bands(closes_reversed, period, num_std)
    
    # Get most recent values
    latest_price = close_prices[0]  # Most recent (already in newest-first order)
    latest_middle = bb["middle"][-1]
    latest_upper = bb["upper"][-1]
    latest_lower = bb["lower"][-1]
    latest_bandwidth = bb["bandwidth"][-1]
    
    # Calculate %B: where price is within the bands (0 = lower, 0.5 = middle, 1 = upper)
    band_range = latest_upper - latest_lower
    percent_b = ((latest_price - latest_lower) / band_range) if band_range > 0 else 0.5
    
    # Determine position
    if latest_price > latest_upper:
        position = "above upper band (overbought)"
    elif latest_price < latest_lower:
        position = "below lower band (oversold)"
    elif percent_b > 0.7:
        position = "near upper band"
    elif percent_b < 0.3:
        position = "near lower band"
    else:
        position = "middle of bands"
    
    summary = {
        "symbol": symbol.upper(),
        "indicator": "Bollinger Bands",
        "period": period,
        "num_std": num_std,
        "latest_date": dates[0] if dates else "N/A",
        "latest_price": round(latest_price, 2),
        "middle_band": round(latest_middle, 2),
        "upper_band": round(latest_upper, 2),
        "lower_band": round(latest_lower, 2),
        "bandwidth_pct": round(latest_bandwidth, 2),
        "percent_b": round(percent_b, 3),
        "position": position,
    }
    
    state["tool_results"]["bollinger_bands"] = summary
    return state


register_tool(
    {
        "name": "bollinger_bands",
        "description": (
            "Calculate Bollinger Bands from price data. "
            "Useful for identifying overbought/oversold conditions and volatility compression/expansion. "
            "Price near upper band suggests overbought, near lower band suggests oversold. "
            "Narrow bandwidth (squeeze) often precedes volatility expansion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {
                    "type": "integer",
                    "default": 20,
                },
                "num_std": {
                    "type": "number",
                    "default": 2.0,
                },
            },
            "required": ["symbol"],
        },
        "fn": bollinger_bands_tool_fn,
    }
)


# =============================================================================
# 3) Alpha Vantage: Fundamentals (OVERVIEW)
# =============================================================================

def alpha_vantage_fundamentals_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch basic fundamental data for a symbol using Alpha Vantage OVERVIEW endpoint.

    args:
      symbol: str (required)
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    if not ALPHAVANTAGE_API_KEY:
        state["tool_results"]["alpha_vantage_fundamentals"] = {
            "symbol": args.get("symbol"),
            "error": "ALPHAVANTAGE_API_KEY not set",
        }
        return state

    symbol = args["symbol"]

    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    data = _alpha_vantage_get_json(params)

    if not data or "Symbol" not in data:
        state["tool_results"]["alpha_vantage_fundamentals"] = {
            "symbol": symbol,
            "error": data.get("Note") or data.get("Error Message") or "Overview unavailable",
        }
        return state

    # Extract a concise set of fields useful for decisions
    summary = {
        "symbol": data.get("Symbol"),
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": float(data["MarketCapitalization"]) if data.get("MarketCapitalization") not in (None, "", "None") else None,
        "pe_ratio": float(data["PERatio"]) if data.get("PERatio") not in (None, "", "None") else None,
        "peg_ratio": float(data["PEGRatio"]) if data.get("PEGRatio") not in (None, "", "None") else None,
        "eps_ttm": float(data["EPS"]) if data.get("EPS") not in (None, "", "None") else None,
        "dividend_yield": float(data["DividendYield"]) if data.get("DividendYield") not in (None, "", "None") else None,
        "profit_margin": float(data["ProfitMargin"]) if data.get("ProfitMargin") not in (None, "", "None") else None,
        "roe_ttm": float(data["ReturnOnEquityTTM"]) if data.get("ReturnOnEquityTTM") not in (None, "", "None") else None,
        "revenue_ttm": float(data["RevenueTTM"]) if data.get("RevenueTTM") not in (None, "", "None") else None,
        "quarterly_earnings_growth_yoy": float(data["QuarterlyEarningsGrowthYOY"]) if data.get("QuarterlyEarningsGrowthYOY") not in (None, "", "None") else None,
        "quarterly_revenue_growth_yoy": float(data["QuarterlyRevenueGrowthYOY"]) if data.get("QuarterlyRevenueGrowthYOY") not in (None, "", "None") else None,
        # Avoid dumping the whole description into the prompt; you can add if you want.
    }

    state["tool_results"]["alpha_vantage_fundamentals"] = summary
    return state


# NOTE: OVERVIEW endpoint is PREMIUM ONLY - commenting out for free tier
# register_tool(
#     {
#         "name": "alpha_vantage_fundamentals",
#         "description": (
#             "Fetch basic fundamentals for a stock (market cap, P/E, EPS, margins, growth, etc.) "
#             "using the Alpha Vantage OVERVIEW endpoint."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "symbol": {"type": "string"},
#             },
#             "required": ["symbol"],
#         },
#         "fn": alpha_vantage_fundamentals_tool_fn,
#     }
# )


# =============================================================================
# 4) Finviz News Scraping + FinBERT Sentiment
# =============================================================================

_FINBERT_MODEL_NAME = "ProsusAI/finbert"
_finbert_tokenizer = None
_finbert_model = None
_FINBERT_AVAILABLE = _TORCH_AVAILABLE  # only available if torch is installed


def _ensure_finbert_loaded() -> bool:
    """
    Lazily load FinBERT model/tokenizer the first time we need it.
    Downloads from HuggingFace on first run and caches locally.
    Requires torch and transformers to be installed.
    """
    global _finbert_tokenizer, _finbert_model, _FINBERT_AVAILABLE

    if not _TORCH_AVAILABLE:
        return False
    
    if not _FINBERT_AVAILABLE:
        return False

    if _finbert_model is not None and _finbert_tokenizer is not None:
        return True

    try:
        print(f"📥 Downloading FinBERT model '{_FINBERT_MODEL_NAME}' from HuggingFace (first time only)...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained(
            _FINBERT_MODEL_NAME,
            force_download=False,  # use cache if available
            resume_download=True    # resume partial downloads
        )
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(
            _FINBERT_MODEL_NAME,
            force_download=False,
            resume_download=True
        )
        print(f"✅ FinBERT model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ FinBERT model could not be loaded: {e}")
        print(f"   Falling back to 'unknown' sentiment for news analysis")
        _FINBERT_AVAILABLE = False
        return False


def _finbert_sentiment(text: str) -> dict:
    """
    Run FinBERT sentiment on a short text. If FinBERT is unavailable,
    return a neutral/unknown sentiment instead of crashing.
    """
    if not _ensure_finbert_loaded():
        return {
            "label": "unknown",
            "probs": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
        }

    inputs = _finbert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    labels = ["negative", "neutral", "positive"]
    label_idx = int(torch.tensor(probs).argmax())
    label = labels[label_idx]
    return {
        "label": label,
        "probs": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        },
    }


def _get_recent_news_from_finviz(
    symbol: str, window_days: int = 3, max_articles: int = 20
) -> List[dict]:
    """
    Scrape recent news for a symbol from finviz.com.
    Returns a list of:
      {"headline": ..., "summary": "", "url": ..., "published_at": iso_timestamp}
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    news_table = soup.find("table", id="news-table")
    if not news_table:
        return []

    now = dt.datetime.utcnow()
    cutoff = now - dt.timedelta(days=window_days)

    articles: List[dict] = []
    last_date_str = None

    for row in news_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        date_time_str = cols[0].get_text(strip=True)
        headline_link = cols[1].find("a")
        headline = (
            headline_link.get_text(strip=True)
            if headline_link
            else cols[1].get_text(strip=True)
        )
        href = headline_link["href"] if headline_link else None

        # finviz formats date/time as "Dec-12-24 08:35AM" or "08:35AM"
        if " " in date_time_str:
            date_part, time_part = date_time_str.split(" ", 1)
            last_date_str = date_part
        else:
            time_part = date_time_str  # reuse last_date_str

        if last_date_str is None:
            continue

        try:
            dt_str = f"{last_date_str} {time_part}"
            pub_dt = dt.datetime.strptime(dt_str, "%b-%d-%y %I:%M%p")
        except ValueError:
            continue

        if pub_dt < cutoff:
            # Older than window; Finviz is reverse-chronological
            break

        articles.append(
            {
                "headline": headline,
                "summary": "",
                "url": href,
                "published_at": pub_dt.isoformat() + "Z",
            }
        )
        if len(articles) >= max_articles:
            break

    return articles


def news_sentiment_finviz_finbert_tool_fn(state: dict, args: dict) -> dict:
    """
    Scrape recent news for a symbol from Finviz, run FinBERT sentiment, and store summary in
    state["tool_results"]["news_sentiment_finviz_finbert"].
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args["symbol"]
    window_days = int(args.get("window_days", 3))
    max_articles = int(args.get("max_articles", 20))

    articles = _get_recent_news_from_finviz(
        symbol, window_days=window_days, max_articles=max_articles
    )

    results = []
    pos_scores: List[float] = []

    for art in articles:
        headline = art["headline"] or ""
        text = headline
        if not text.strip():
            continue
        s = _finbert_sentiment(text)
        results.append(
            {
                "headline": headline,
                "summary": "",
                "url": art["url"],
                "published_at": art["published_at"],
                "sentiment": s["label"],
                "probs": s["probs"],
            }
        )
        pos_scores.append(s["probs"]["positive"])

    if pos_scores:
        avg_pos = sum(pos_scores) / len(pos_scores)
    else:
        avg_pos = None

    if avg_pos is None:
        agg_label = "unknown"
    elif avg_pos > 0.6:
        agg_label = "bullish"
    elif avg_pos < 0.4:
        agg_label = "bearish"
    else:
        agg_label = "mixed"

    state["tool_results"]["news_sentiment_finviz_finbert"] = {
        "symbol": symbol.upper(),
        "window_days": window_days,
        "num_articles": len(results),
        "avg_positive_prob": avg_pos,
        "aggregate_label": agg_label,
        "articles": results[:10],
    }
    return state


register_tool(
    {
        "name": "news_sentiment_finviz_finbert",
        "description": (
            "Scrape recent news for a symbol from finviz.com and run FinBERT sentiment analysis "
            "on the headlines. Useful for understanding near-term news sentiment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "window_days": {"type": "integer", "default": 3},
                "max_articles": {"type": "integer", "default": 20},
            },
            "required": ["symbol"],
        },
        "fn": news_sentiment_finviz_finbert_tool_fn,
    }
)


# =============================================================================
# Regime Detection Tools
# =============================================================================

# Lazy imports to avoid loading heavy dependencies if not needed
_REGIME_DETECTORS_LOADED = False
_RollingPaperWassersteinDetector = None
_RollingWindowHMM = None
_fetch_polygon_bars = None
_calculate_features = None


def _ensure_regime_detectors_loaded():
    """Lazy load regime detection models."""
    global _REGIME_DETECTORS_LOADED, _RollingPaperWassersteinDetector, _RollingWindowHMM
    global _fetch_polygon_bars, _calculate_features
    
    if _REGIME_DETECTORS_LOADED:
        return
    
    try:
        from models.paper_wasserstein_regime_detection import (
            RollingPaperWassersteinDetector,
            fetch_polygon_bars,
            calculate_features
        )
        from models.rolling_hmm_regime_detection import RollingWindowHMM
        
        _RollingPaperWassersteinDetector = RollingPaperWassersteinDetector
        _RollingWindowHMM = RollingWindowHMM
        _fetch_polygon_bars = fetch_polygon_bars
        _calculate_features = calculate_features
        _REGIME_DETECTORS_LOADED = True
    except ImportError as e:
        print(f"Warning: Could not load regime detection models: {e}")
        _REGIME_DETECTORS_LOADED = False


def regime_detection_wasserstein_tool_fn(state: dict, args: dict) -> dict:
    """
    Paper-faithful Wasserstein k-means regime detection.
    
    Best for:
    - Tech/healthcare stocks with distinct volatility regimes
    - Detecting regime changes when adaptivity is important
    - When cluster separation quality matters (MMD scores)
    
    Returns current regime (low/medium/high volatility) with confidence metrics.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    _ensure_regime_detectors_loaded()
    
    if not _REGIME_DETECTORS_LOADED:
        state["tool_results"]["regime_wasserstein"] = {
            "symbol": args.get("symbol"),
            "error": "Regime detection models not available (install requirements_hmm.txt)"
        }
        return state
    
    symbol = args["symbol"]
    n_regimes = args.get("n_regimes", 3)
    window_size = args.get("window_size", 20)
    training_window = args.get("training_window", 500)
    
    try:
        # Fetch data and calculate features
        df = _fetch_polygon_bars(symbol, "2020-01-01", dt.datetime.now().strftime("%Y-%m-%d"))
        df = _calculate_features(df, window=window_size)
        
        # Initialize detector
        detector = _RollingPaperWassersteinDetector(
            n_regimes=n_regimes,
            window_size=window_size,
            training_window_days=training_window,
            retrain_frequency_days=126,  # Quarterly
            feature_cols=['realized_vol', 'trend_strength', 'volume_momentum']
        )
        
        # Train and predict
        train_end = df.index[training_window]
        detector.train_on_window(df, train_end, verbose=False)
        
        # Get current regime
        latest_features = df[['realized_vol', 'trend_strength', 'volume_momentum']].iloc[-window_size:].values
        current_regime = detector.model.predict_distribution(latest_features)
        
        # Map regime to interpretation
        regime_mapping = {
            0: "Low Volatility",
            1: "Medium Volatility", 
            2: "High Volatility"
        }
        
        # Interpretation based on regime
        if current_regime == 2:
            interpretation = "High volatility regime - consider reducing position size and wider stops"
            confidence = "medium"  # Wasserstein is good at identifying high vol
        elif current_regime == 1:
            interpretation = "Medium volatility - normal position sizing appropriate"
            confidence = "medium"
        else:
            interpretation = "Low volatility regime - opportunity for slightly larger positions"
            confidence = "medium"
        
        state["tool_results"]["regime_wasserstein"] = {
            "symbol": symbol,
            "method": "Wasserstein k-means (paper-faithful)",
            "regime": int(current_regime),
            "regime_name": regime_mapping.get(current_regime, "Unknown"),
            "confidence": confidence,
            "interpretation": interpretation,
            "note": "Best for stocks with distinct volatility regimes (tech, healthcare). Performs better when adaptive (low label consistency)."
        }
        
    except Exception as e:
        state["tool_results"]["regime_wasserstein"] = {
            "symbol": symbol,
            "error": f"Failed to detect regime: {str(e)}"
        }
    
    return state


def regime_detection_hmm_tool_fn(state: dict, args: dict) -> dict:
    """
    Rolling HMM regime detection with forward filter (no look-ahead bias).
    
    Best for:
    - Smooth regime transitions
    - Probabilistic predictions with confidence intervals
    - When you need transition probabilities
    
    Returns current regime (bearish/sideways/bullish) with probability distribution.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    _ensure_regime_detectors_loaded()
    
    if not _REGIME_DETECTORS_LOADED:
        state["tool_results"]["regime_hmm"] = {
            "symbol": args.get("symbol"),
            "error": "Regime detection models not available (install requirements_hmm.txt)"
        }
        return state
    
    symbol = args["symbol"]
    n_regimes = args.get("n_regimes", 3)
    training_window = args.get("training_window", 500)
    
    try:
        # Fetch data and calculate features
        df = _fetch_polygon_bars(symbol, "2020-01-01", dt.datetime.now().strftime("%Y-%m-%d"))
        df = _calculate_features(df, window=20)
        
        # Initialize HMM detector
        detector = _RollingWindowHMM(
            n_regimes=n_regimes,
            training_window_days=training_window,
            retrain_frequency_days=126,
            persistence_prior=0.8,
            feature_columns=['realized_vol', 'trend_strength', 'volume_momentum']
        )
        
        # Train on recent window
        train_features = df[['realized_vol', 'trend_strength', 'volume_momentum']].iloc[-training_window:].values
        train_end = df.index[-1]
        detector.train_on_window(train_features, train_end)
        
        # Predict on latest observation
        latest_features = df[['realized_vol', 'trend_strength', 'volume_momentum']].iloc[-1:].values
        result = detector.predict_forward_filter(latest_features)
        
        current_regime = result['most_likely_state']
        regime_name = result['regime_name']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Calculate transition probabilities
        trans_matrix = detector.model.transmat_
        current_trans = trans_matrix[current_regime]
        
        # Interpretation based on regime and transitions
        if current_regime == 0:  # Bearish
            interpretation = f"Bearish regime - {current_trans[current_regime]:.0%} chance of staying bearish. Consider defensive positioning."
        elif current_regime == 1:  # Sideways
            interpretation = f"Sideways/choppy regime - {current_trans[current_regime]:.0%} persistence. Range-bound trading likely."
        else:  # Bullish
            interpretation = f"Bullish regime - {current_trans[current_regime]:.0%} chance of continuing. Favorable for long positions."
        
        # Warn about potential transitions
        max_transition_idx = max((i for i in range(n_regimes) if i != current_regime), key=lambda i: current_trans[i])
        max_transition_prob = current_trans[max_transition_idx]
        
        if max_transition_prob > 0.15:
            transition_name = detector.regime_names[max_transition_idx]
            interpretation += f" Warning: {max_transition_prob:.0%} transition probability to {transition_name}."
        
        state["tool_results"]["regime_hmm"] = {
            "symbol": symbol,
            "method": "Rolling HMM (forward filter)",
            "regime": int(current_regime),
            "regime_name": regime_name,
            "confidence": round(confidence, 3),
            "probabilities": {
                detector.regime_names[0]: round(probabilities[0], 3),
                detector.regime_names[1]: round(probabilities[1], 3),
                detector.regime_names[2]: round(probabilities[2], 3)
            },
            "persistence_probability": round(current_trans[current_regime], 3),
            "interpretation": interpretation,
            "note": "Best for smooth transitions and probabilistic predictions. More stable than Wasserstein."
        }
        
    except Exception as e:
        state["tool_results"]["regime_hmm"] = {
            "symbol": symbol,
            "error": f"Failed to detect regime: {str(e)}"
        }
    
    return state


def regime_consensus_check_tool_fn(state: dict, args: dict) -> dict:
    """
    Checks agreement between Wasserstein and HMM regime predictions.
    Only works if both regime tools were called first.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    wass = state["tool_results"].get("regime_wasserstein")
    hmm = state["tool_results"].get("regime_hmm")
    
    if not wass or not hmm:
        state["tool_results"]["regime_consensus"] = {
            "error": "Both regime_wasserstein and regime_hmm must be called first"
        }
        return state
    
    if wass.get("error") or hmm.get("error"):
        state["tool_results"]["regime_consensus"] = {
            "error": "One or both regime detection tools failed"
        }
        return state
    
    # Check agreement (rough mapping since they use different scales)
    wass_regime = wass["regime"]  # 0=low-vol, 1=med-vol, 2=high-vol
    hmm_regime = hmm["regime"]    # 0=bearish, 1=sideways, 2=bullish
    
    # Consider agreement if both indicate similar risk level
    # High-vol (2) roughly maps to bearish (0)
    # Med-vol (1) roughly maps to sideways (1)  
    # Low-vol (0) roughly maps to bullish (2)
    agreement = (
        (wass_regime == 2 and hmm_regime == 0) or  # Both see high risk
        (wass_regime == 1 and hmm_regime == 1) or  # Both see medium
        (wass_regime == 0 and hmm_regime == 2)     # Both see low risk
    )
    
    if agreement:
        recommendation = (
            "✓ Both models AGREE on regime classification - HIGH CONFIDENCE. "
            "Wasserstein and HMM independently converged on similar risk assessment."
        )
        confidence_level = "HIGH"
    else:
        recommendation = (
            "⚠ Models DISAGREE - UNCERTAINTY SIGNAL. "
            f"Wasserstein sees {wass['regime_name']}, HMM sees {hmm['regime_name']}. "
            "This divergence often precedes regime transitions. Exercise caution, reduce position size, "
            "or wait for consensus. Consider the trading books' emphasis on avoiding uncertain regimes."
        )
        confidence_level = "LOW"
    
    state["tool_results"]["regime_consensus"] = {
        "agreement": agreement,
        "confidence_level": confidence_level,
        "wasserstein_regime": wass["regime_name"],
        "hmm_regime": hmm["regime_name"],
        "wasserstein_confidence": wass["confidence"],
        "hmm_confidence": hmm["confidence"],
        "recommendation": recommendation
    }
    
    return state


# =============================================================================
# DEPRECATED REGIME DETECTION TOOLS
# =============================================================================
# These tools have been deprecated as they do not provide reliable alpha.
# Use vix_roc_risk and vol_prediction instead for market timing and position sizing.

# DEPRECATED: register_tool({
#     "name": "regime_detection_wasserstein",
#     "description": (
#         "Detect market regime using paper-faithful Wasserstein k-means clustering. "
#         "Returns low/medium/high volatility regime with MMD quality metrics. "
#         "DEPRECATED: Use vol_prediction instead for volatility regime detection."
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "symbol": {"type": "string"},
#             "n_regimes": {"type": "integer", "default": 3},
#             "window_size": {"type": "integer", "default": 20},
#             "training_window": {"type": "integer", "default": 500}
#         },
#         "required": ["symbol"]
#     },
#     "fn": regime_detection_wasserstein_tool_fn
# })

# DEPRECATED: register_tool({
#     "name": "regime_detection_hmm",
#     "description": (
#         "Detect market regime using Rolling HMM with forward filter. "
#         "DEPRECATED: Use vix_roc_risk instead for market timing."
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "symbol": {"type": "string"},
#             "n_regimes": {"type": "integer", "default": 3},
#             "training_window": {"type": "integer", "default": 500}
#         },
#         "required": ["symbol"]
#     },
#     "fn": regime_detection_hmm_tool_fn
# })

# DEPRECATED: register_tool({
#     "name": "regime_consensus_check",
#     "description": (
#         "Check if Wasserstein and HMM regime detectors agree. "
#         "DEPRECATED: No longer needed as both regime tools are deprecated."
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "symbol": {"type": "string"}
#         },
#         "required": ["symbol"]
#     },
#     "fn": regime_consensus_check_tool_fn
# })

# =============================================================================
# DEPRECATED ML Prediction Tool
# =============================================================================
# ML prediction has been deprecated - does not provide reliable alpha.
# Use vix_roc_risk and vol_prediction for risk management instead.

# Import the ML prediction function (kept for backward compatibility)
try:
    from ml_prediction_tool import ml_prediction_tool_fn
    _ML_TOOL_AVAILABLE = True
except ImportError:
    _ML_TOOL_AVAILABLE = False
    
    # Fallback if import fails
    def ml_prediction_tool_fn(state: dict, args: dict) -> dict:
        if "tool_results" not in state:
            state["tool_results"] = {}
        state["tool_results"]["ml_prediction"] = {
            "error": "ML prediction tool not available (import failed)",
            "symbol": args.get("symbol")
        }
        return state

# DEPRECATED: ML prediction tool registration
# if _ML_TOOL_AVAILABLE:
#     register_tool({
#         "name": "ml_prediction",
#         "description": (
#             "DEPRECATED: Get ML model predictions for stock direction. "
#             "Use vix_roc_risk and vol_prediction instead for more reliable signals."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "symbol": {
#                     "type": "string",
#                     "description": "Stock ticker symbol (e.g., AAPL, TSLA)"
#                 },
#                 "horizon": {
#                     "type": "integer",
#                     "description": "Prediction horizon in days (3, 5, or 10). Default: 5",
#                     "enum": [3, 5, 10]
#                 },
#                 "models": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "Optional: Specific models to use. Default: all 4 models"
#                 }
#             },
#             "required": ["symbol"]
#         },
#         "fn": ml_prediction_tool_fn
#     })

# =============================================================================
# BOCPD Regime Detection Tool (Bayesian Online Changepoint Detection)
# =============================================================================
# Replaces Wasserstein/HMM regime tools with a Bayesian changepoint model
# that produces 6-regime classification (bull/bear/transition/crisis/consolidation).
# Runs on SPY to detect market-wide regime for contextual trade decisions.

try:
    from models.bocpd_regime.detector import detect_current_regime as _bocpd_detect
    _BOCPD_REGIME_AVAILABLE = True
except ImportError as e:
    _BOCPD_REGIME_AVAILABLE = False
    print(f"Warning: BOCPD regime detection not available: {e}")


def bocpd_regime_tool_fn(state: dict, args: dict) -> dict:
    """
    Detect current market regime using BOCPD on SPY.

    Returns regime label, risk score, ERL, trends, and actionable interpretation.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    if not _BOCPD_REGIME_AVAILABLE:
        state["tool_results"]["bocpd_regime"] = {
            "error": "BOCPD regime detection not available (import failed)"
        }
        return state

    # Always run on SPY for market regime context
    symbol = "SPY"
    try:
        result = _bocpd_detect(symbol=symbol, lookback_years=3)
        state["tool_results"]["bocpd_regime"] = result
    except Exception as e:
        state["tool_results"]["bocpd_regime"] = {
            "symbol": symbol,
            "error": f"BOCPD regime detection failed: {str(e)}"
        }

    return state


if _BOCPD_REGIME_AVAILABLE:
    register_tool({
        "name": "bocpd_regime",
        "description": (
            "Detect the current MARKET REGIME using Bayesian Online Changepoint Detection (BOCPD) "
            "on SPY. Provides qualitative market context that complements the quantitative "
            "market_risk tool."
            "\n\nReturns:"
            "\n• current_regime: bull / bear / bull_transition / bear_transition / consolidation / crisis"
            "\n• risk_score: Continuous instability score (0=stable, 1=unstable)"
            "\n• risk_level: LOW / MODERATE / ELEVATED / HIGH"
            "\n• expected_run_length: Days since last detected changepoint (higher = more stable)"
            "\n• trend_21d / trend_63d: Short and medium-term cumulative returns"
            "\n• volatility_ann: 21-day annualised volatility"
            "\n• interpretation: Actionable guidance for the current regime"
            "\n• recent_regime_changes: Last 5 regime transitions with dates"
            "\n\nUse this to understand WHAT KIND of market we are in, then adapt "
            "strategy selection accordingly. Bull regimes favour trend-following; "
            "bear regimes favour defensive positioning; crisis = capital preservation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ignored — always runs on SPY for market regime."
                }
            },
            "required": ["symbol"]
        },
        "fn": bocpd_regime_tool_fn,
    })


# =============================================================================
# Stock Beta Tool (Regime-Conditioned Beta Targeting)
# =============================================================================
# Computes rolling beta + R² of a stock against SPY so the decision agent
# can combine market regime (from BOCPD) with a stock's market sensitivity.
# High-beta in bull regimes amplifies upside; low-beta in bear preserves capital.

def stock_beta_tool_fn(state: dict, args: dict) -> dict:
    """
    Compute rolling beta, R², and regime-appropriate guidance for a stock vs SPY.
    """
    import warnings as _w
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args.get("symbol", "").upper()
    window = int(args.get("window", 60))

    if not symbol:
        state["tool_results"]["stock_beta"] = {"error": "No symbol provided"}
        return state

    try:
        import yfinance as yf
    except ImportError:
        state["tool_results"]["stock_beta"] = {
            "symbol": symbol,
            "error": "yfinance not installed — required for stock beta"
        }
        return state

    import numpy as np
    import pandas as pd

    # Need enough history for rolling window + warm-up
    lookback_days = max(window * 4, 300)
    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        spy = yf.download("SPY", start=start, progress=False, auto_adjust=True)
        if symbol == "SPY":
            stock = spy.copy()
        else:
            stock = yf.download(symbol, start=start, progress=False, auto_adjust=True)

    # Flatten multi-level columns (yfinance >= 0.2.31)
    for df in [spy, stock]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if spy is None or len(spy) < window + 20:
        state["tool_results"]["stock_beta"] = {
            "symbol": symbol, "error": "Insufficient SPY data"
        }
        return state
    if stock is None or len(stock) < window + 20:
        state["tool_results"]["stock_beta"] = {
            "symbol": symbol, "error": f"Insufficient data for {symbol}"
        }
        return state

    # Align dates and compute daily returns
    spy_close = spy["Close"].squeeze()
    stock_close = stock["Close"].squeeze()
    combined = pd.DataFrame({
        "spy": spy_close, "stock": stock_close
    }).dropna()
    combined["spy_ret"] = combined["spy"].pct_change()
    combined["stock_ret"] = combined["stock"].pct_change()
    combined = combined.dropna()

    if len(combined) < window + 5:
        state["tool_results"]["stock_beta"] = {
            "symbol": symbol,
            "error": f"Only {len(combined)} overlapping bars — need at least {window + 5}"
        }
        return state

    # ---- Rolling beta + R² ----
    spy_ret = combined["spy_ret"]
    stock_ret = combined["stock_ret"]

    rolling_cov = stock_ret.rolling(window).cov(spy_ret)
    rolling_var = spy_ret.rolling(window).var()
    rolling_beta = (rolling_cov / rolling_var).dropna()

    rolling_corr = stock_ret.rolling(window).corr(spy_ret).dropna()
    rolling_r2 = rolling_corr ** 2

    if len(rolling_beta) == 0:
        state["tool_results"]["stock_beta"] = {
            "symbol": symbol, "error": "Rolling beta computation returned empty"
        }
        return state

    current_beta = float(rolling_beta.iloc[-1])
    current_r2 = float(rolling_r2.iloc[-1])

    # Full-period (OLS) beta for reference
    full_cov = stock_ret.cov(spy_ret)
    full_var = spy_ret.var()
    full_beta = float(full_cov / full_var) if full_var > 0 else float("nan")
    full_corr = stock_ret.corr(spy_ret)
    full_r2 = float(full_corr ** 2)

    # Beta stability: coefficient of variation of rolling betas
    beta_mean = float(rolling_beta.mean())
    beta_std = float(rolling_beta.std())
    beta_cv = abs(beta_std / beta_mean) if abs(beta_mean) > 0.01 else float("nan")
    beta_min = float(rolling_beta.min())
    beta_max = float(rolling_beta.max())

    # Recent trend: is beta rising or falling?
    if len(rolling_beta) >= 20:
        recent_beta_mean = float(rolling_beta.iloc[-20:].mean())
        older_beta_mean = float(rolling_beta.iloc[-40:-20].mean()) if len(rolling_beta) >= 40 else beta_mean
        beta_trend = "rising" if recent_beta_mean > older_beta_mean + 0.05 else \
                     "falling" if recent_beta_mean < older_beta_mean - 0.05 else "stable"
    else:
        recent_beta_mean = current_beta
        beta_trend = "unknown"

    # ---- Interpret beta ----
    if current_beta >= 1.5:
        beta_label = "very high beta (aggressive)"
    elif current_beta >= 1.2:
        beta_label = "high beta (growth/momentum)"
    elif current_beta >= 0.8:
        beta_label = "market beta (broad market)"
    elif current_beta >= 0.5:
        beta_label = "low beta (defensive)"
    elif current_beta >= 0.0:
        beta_label = "very low beta (bond-proxy / utility)"
    else:
        beta_label = "negative beta (hedge / inverse)"

    # R² interpretation
    if current_r2 >= 0.6:
        r2_label = "strongly market-driven — regime signal very relevant"
    elif current_r2 >= 0.3:
        r2_label = "moderately market-driven — regime signal somewhat relevant"
    elif current_r2 >= 0.1:
        r2_label = "weakly market-driven — regime signal has limited relevance"
    else:
        r2_label = "idiosyncratic — regime signal mostly irrelevant, stock moves on its own"

    # Stability interpretation
    if beta_cv is not None and not np.isnan(beta_cv):
        if beta_cv < 0.15:
            stability_label = "very stable beta"
        elif beta_cv < 0.30:
            stability_label = "moderately stable beta"
        elif beta_cv < 0.50:
            stability_label = "unstable beta — sensitivity shifts"
        else:
            stability_label = "highly unstable beta — unreliable for regime targeting"
    else:
        stability_label = "insufficient data for stability assessment"

    # ---- Regime-conditioned guidance ----
    regime_guidance = {
        "bull": {
            "preferred_beta": "1.2+ (amplify upside with high-beta names)",
            "stock_fit": "good fit" if current_beta >= 1.2 else
                         "neutral" if current_beta >= 0.8 else "poor fit — too defensive for bull"
        },
        "bull_transition": {
            "preferred_beta": "0.8–1.2 (market beta, balanced exposure)",
            "stock_fit": "good fit" if 0.8 <= current_beta <= 1.2 else
                         "acceptable" if current_beta >= 0.5 else "caution — beta mismatch"
        },
        "consolidation": {
            "preferred_beta": "0.6–1.0 (neutral to slightly defensive)",
            "stock_fit": "good fit" if 0.6 <= current_beta <= 1.0 else
                         "acceptable" if current_beta <= 1.2 else "caution — too aggressive for range-bound"
        },
        "bear_transition": {
            "preferred_beta": "0.3–0.8 (defensive, limit drawdown)",
            "stock_fit": "good fit" if 0.3 <= current_beta <= 0.8 else
                         "acceptable" if current_beta <= 1.0 else "poor fit — too aggressive for weakening market"
        },
        "bear": {
            "preferred_beta": "0–0.5 or cash (capital preservation)",
            "stock_fit": "good fit" if current_beta <= 0.5 else
                         "caution" if current_beta <= 0.8 else "poor fit — high beta in bear = amplified losses"
        },
        "crisis": {
            "preferred_beta": "0 (cash) or negative beta (hedges)",
            "stock_fit": "good fit" if current_beta <= 0.2 else
                         "poor fit — almost all equities lose in crisis"
        },
    }

    result = {
        "symbol": symbol,
        "window": window,
        "current_beta": round(current_beta, 3),
        "beta_label": beta_label,
        "current_r2": round(current_r2, 3),
        "r2_label": r2_label,
        "full_period_beta": round(full_beta, 3),
        "full_period_r2": round(full_r2, 3),
        "beta_range": f"{beta_min:.2f} to {beta_max:.2f}",
        "beta_trend": beta_trend,
        "beta_stability_cv": round(beta_cv, 3) if not np.isnan(beta_cv) else None,
        "stability_label": stability_label,
        "regime_beta_guidance": regime_guidance,
        "data_points": len(combined),
    }
    state["tool_results"]["stock_beta"] = result
    return state


register_tool({
    "name": "stock_beta",
    "description": (
        "Compute a stock's rolling BETA and R² against SPY for regime-conditioned "
        "beta targeting. Combines with bocpd_regime to select stocks appropriate "
        "for the current market environment."
        "\n\nReturns:"
        "\n• current_beta: Rolling {window}-day beta vs SPY"
        "\n• beta_label: Interpretation (very high / high / market / low / very low / negative)"
        "\n• current_r2: How much of the stock's variance is explained by SPY"
        "\n• r2_label: Whether regime signal is relevant for this stock"
        "\n• full_period_beta / full_period_r2: Longer-term reference values"
        "\n• beta_range: Min–max over the rolling window history"
        "\n• beta_trend: Rising / falling / stable"
        "\n• beta_stability_cv: Coefficient of variation (lower = more stable)"
        "\n• regime_beta_guidance: For each BOCPD regime, the preferred beta range "
        "and whether this stock fits"
        "\n\nStrategy: In BULL regimes, prefer beta 1.2+ to amplify upside. "
        "In BEAR regimes, prefer beta <0.5 or cash. In TRANSITION, stay near "
        "market beta (0.8–1.2). Low R² means the stock is idiosyncratic and "
        "regime signal matters less."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker to compute beta for."
            },
            "window": {
                "type": "integer",
                "description": "Rolling window in trading days (default 60)."
            },
        },
        "required": ["symbol"]
    },
    "fn": stock_beta_tool_fn,
})


# =============================================================================
# Relative Strength Tool (Stock vs Sector vs Market)
# =============================================================================
# Compares a stock's return against its sector ETF and SPY across multiple
# timeframes.  Leaders outperform both; laggards underperform.
# Combined with BOCPD regime + beta → full stock-selection framework:
#   regime (context) + beta (sensitivity) + RS (leadership) = pick right stock.

# GICS Sector → Primary ETF mapping (Select Sector SPDRs + a few extras)
_SECTOR_ETF_MAP = {
    # SIC-based broad buckets Polygon often returns
    "technology": "XLK",
    "information technology": "XLK",
    "software": "XLK",
    "electronic": "XLK",
    "semiconductors": "XLK",
    "communication services": "XLC",
    "communications": "XLC",
    "media": "XLC",
    "telecommunications": "XLC",
    "health care": "XLV",
    "healthcare": "XLV",
    "pharmaceutical": "XLV",
    "biotechnology": "XLV",
    "financials": "XLF",
    "financial": "XLF",
    "banks": "XLF",
    "insurance": "XLF",
    "consumer discretionary": "XLY",
    "retail": "XLY",
    "consumer staples": "XLP",
    "food": "XLP",
    "beverage": "XLP",
    "household": "XLP",
    "industrials": "XLI",
    "industrial": "XLI",
    "aerospace": "XLI",
    "defense": "XLI",
    "machinery": "XLI",
    "energy": "XLE",
    "oil": "XLE",
    "gas": "XLE",
    "petroleum": "XLE",
    "utilities": "XLU",
    "electric": "XLU",
    "real estate": "XLRE",
    "materials": "XLB",
    "chemicals": "XLB",
    "mining": "XLB",
    "metals": "XLB",
}

# Ticker → well-known sector override (for popular tickers where SIC is vague)
_TICKER_SECTOR_OVERRIDE = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "META": "XLC", "AMZN": "XLY", "TSLA": "XLY", "NFLX": "XLC", "DIS": "XLC",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "V": "XLK", "MA": "XLK",
    "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "ABBV": "XLV", "LLY": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "BA": "XLI", "CAT": "XLI", "HON": "XLI", "UPS": "XLI", "GE": "XLI",
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU",
    "AMD": "XLK", "AVGO": "XLK", "INTC": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "ORCL": "XLK", "CSCO": "XLK", "TXN": "XLK", "QCOM": "XLK", "MU": "XLK",
}


def _resolve_sector_etf(symbol: str, sic_description: str | None = None) -> str | None:
    """Resolve a ticker to its sector ETF using overrides → SIC description → None."""
    symbol = symbol.upper()
    if symbol in _TICKER_SECTOR_OVERRIDE:
        return _TICKER_SECTOR_OVERRIDE[symbol]
    if sic_description:
        desc_lower = sic_description.lower()
        for keyword, etf in _SECTOR_ETF_MAP.items():
            if keyword in desc_lower:
                return etf
    return None


def relative_strength_tool_fn(state: dict, args: dict) -> dict:
    """
    Compute relative strength of a stock vs its sector ETF and SPY
    across 21d, 63d, 126d, and 252d timeframes.
    """
    import warnings as _w
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args.get("symbol", "").upper()
    if not symbol:
        state["tool_results"]["relative_strength"] = {"error": "No symbol provided"}
        return state

    try:
        import yfinance as yf
    except ImportError:
        state["tool_results"]["relative_strength"] = {
            "symbol": symbol,
            "error": "yfinance not installed — required for relative strength"
        }
        return state

    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------
    # 1. Resolve sector ETF
    # ------------------------------------------------------------------
    # Try ticker_details already in state (from polygon_ticker_details call)
    ticker_details = state.get("tool_results", {}).get("polygon_ticker_details")
    sic_desc = None
    if ticker_details and not ticker_details.get("error"):
        sic_desc = ticker_details.get("sic_description")

    sector_etf = _resolve_sector_etf(symbol, sic_desc)

    # ------------------------------------------------------------------
    # 2. Download price data (need ~260 trading days + buffer)
    # ------------------------------------------------------------------
    start = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    tickers_to_fetch = ["SPY", symbol]
    if sector_etf and sector_etf != symbol:
        tickers_to_fetch.append(sector_etf)

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        raw = yf.download(tickers_to_fetch, start=start, progress=False,
                          auto_adjust=True, group_by="ticker")

    if raw is None or raw.empty:
        state["tool_results"]["relative_strength"] = {
            "symbol": symbol, "error": "Failed to download price data"
        }
        return state

    # Extract close prices, handling single-ticker vs multi-ticker shapes
    def _extract_close(df, ticker):
        try:
            if isinstance(df.columns, pd.MultiIndex):
                series = df[(ticker, "Close")].dropna()
            else:
                series = df["Close"].dropna()
            return series.squeeze()
        except (KeyError, TypeError):
            return None

    spy_close = _extract_close(raw, "SPY")
    stock_close = _extract_close(raw, symbol)
    sector_close = _extract_close(raw, sector_etf) if sector_etf else None

    if spy_close is None or len(spy_close) < 30:
        state["tool_results"]["relative_strength"] = {
            "symbol": symbol, "error": "Insufficient SPY data"
        }
        return state
    if stock_close is None or len(stock_close) < 30:
        state["tool_results"]["relative_strength"] = {
            "symbol": symbol, "error": f"Insufficient data for {symbol}"
        }
        return state

    # ------------------------------------------------------------------
    # 3. Compute returns over multiple windows
    # ------------------------------------------------------------------
    windows = {"21d": 21, "63d": 63, "126d": 126, "252d": 252}
    vs_spy = {}
    vs_sector = {}

    for label, w in windows.items():
        if len(stock_close) >= w and len(spy_close) >= w:
            stock_ret = float((stock_close.iloc[-1] / stock_close.iloc[-w]) - 1)
            spy_ret = float((spy_close.iloc[-1] / spy_close.iloc[-w]) - 1)
            excess = stock_ret - spy_ret
            vs_spy[label] = {
                "stock_return": round(stock_ret * 100, 2),
                "spy_return": round(spy_ret * 100, 2),
                "excess_return": round(excess * 100, 2),
            }

            if sector_close is not None and len(sector_close) >= w:
                sec_ret = float((sector_close.iloc[-1] / sector_close.iloc[-w]) - 1)
                sec_excess = stock_ret - sec_ret
                vs_sector[label] = {
                    "sector_return": round(sec_ret * 100, 2),
                    "excess_vs_sector": round(sec_excess * 100, 2),
                }

    if not vs_spy:
        state["tool_results"]["relative_strength"] = {
            "symbol": symbol, "error": "Not enough overlapping data for RS calculation"
        }
        return state

    # ------------------------------------------------------------------
    # 4. Composite RS score (weighted: 252d=40%, 126d=30%, 63d=20%, 21d=10%)
    # ------------------------------------------------------------------
    weights = {"252d": 0.40, "126d": 0.30, "63d": 0.20, "21d": 0.10}
    weighted_excess = 0.0
    total_weight = 0.0
    for label, w in weights.items():
        if label in vs_spy:
            weighted_excess += vs_spy[label]["excess_return"] * w
            total_weight += w
    composite_rs = round(weighted_excess / total_weight, 2) if total_weight > 0 else None

    # ------------------------------------------------------------------
    # 5. RS percentile rank (using 63d excess return vs SPY)
    #    Simplified: classify into buckets by excess return magnitude
    # ------------------------------------------------------------------
    if "63d" in vs_spy:
        excess_63 = vs_spy["63d"]["excess_return"]
        if excess_63 > 20:
            rs_rank_label = "top-tier leader (RS > 20%)"
        elif excess_63 > 10:
            rs_rank_label = "strong leader (RS 10-20%)"
        elif excess_63 > 3:
            rs_rank_label = "mild outperformer (RS 3-10%)"
        elif excess_63 > -3:
            rs_rank_label = "in-line with market (RS ±3%)"
        elif excess_63 > -10:
            rs_rank_label = "mild underperformer (RS -3 to -10%)"
        elif excess_63 > -20:
            rs_rank_label = "laggard (RS -10 to -20%)"
        else:
            rs_rank_label = "severe laggard (RS < -20%)"
    else:
        excess_63 = None
        rs_rank_label = "insufficient data for 63d ranking"

    # ------------------------------------------------------------------
    # 6. Trend: is RS improving or deteriorating?
    # ------------------------------------------------------------------
    if "21d" in vs_spy and "63d" in vs_spy:
        short_excess = vs_spy["21d"]["excess_return"]
        med_excess = vs_spy["63d"]["excess_return"]
        # Normalise 21d to quarterly scale for comparison
        if short_excess > med_excess + 2:
            rs_trend = "improving — short-term RS accelerating"
        elif short_excess < med_excess - 2:
            rs_trend = "deteriorating — short-term RS decelerating"
        else:
            rs_trend = "stable — RS consistent across timeframes"
    else:
        rs_trend = "unknown"

    # ------------------------------------------------------------------
    # 7. Regime fit guidance
    # ------------------------------------------------------------------
    regime_rs_guidance = {}
    if composite_rs is not None:
        is_leader = composite_rs > 5
        is_laggard = composite_rs < -5
        regime_rs_guidance = {
            "bull": "ideal — leaders amplify bull moves" if is_leader else
                    "acceptable but consider stronger names" if not is_laggard else
                    "poor — laggards tend to underperform even in bulls",
            "bear": "caution — even leaders sell off in bears, but may recover first" if is_leader else
                    "avoid — laggards fall hardest in bear markets" if is_laggard else
                    "neutral",
            "transition": "promising — leaders with rising RS often lead recoveries" if is_leader and rs_trend.startswith("improving") else
                          "wait for confirmation" if not is_laggard else
                          "avoid — laggards rarely lead turns",
        }

    result = {
        "symbol": symbol,
        "sector_etf": sector_etf or "unknown",
        "vs_spy": vs_spy,
        "vs_sector": vs_sector if vs_sector else None,
        "composite_rs_score": composite_rs,
        "rs_rank_label": rs_rank_label,
        "rs_trend": rs_trend,
        "regime_rs_guidance": regime_rs_guidance,
    }
    state["tool_results"]["relative_strength"] = result
    return state


register_tool({
    "name": "relative_strength",
    "description": (
        "Compute a stock's RELATIVE STRENGTH vs SPY and its sector ETF across "
        "multiple timeframes (21d, 63d, 126d, 252d). Identifies market leaders "
        "and laggards — critical for stock selection alongside regime + beta."
        "\n\nReturns:"
        "\n• vs_spy: Excess return over SPY at each timeframe"
        "\n• vs_sector: Excess return over sector ETF (if resolved)"
        "\n• composite_rs_score: Weighted RS score (252d=40%, 126d=30%, 63d=20%, 21d=10%)"
        "\n• rs_rank_label: Qualitative ranking (top-tier leader → severe laggard)"
        "\n• rs_trend: Improving / deteriorating / stable"
        "\n• regime_rs_guidance: Whether this stock's RS fits the current regime"
        "\n\nStrategy: In BULL regimes, buy leaders (composite RS > 5). "
        "In BEAR regimes, avoid laggards (they fall hardest). "
        "Rising RS in a transition often signals early recovery leadership."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker to compute relative strength for."
            },
        },
        "required": ["symbol"]
    },
    "fn": relative_strength_tool_fn,
})


# =============================================================================
# Earnings Proximity Tool (Catalyst Awareness)
# =============================================================================
# Flags upcoming earnings dates so the agent can adjust position sizing
# and warn the user about binary event risk.
# Uses Polygon's reference/tickers endpoint for next earnings date.

def earnings_proximity_tool_fn(state: dict, args: dict) -> dict:
    """
    Check distance to next and last earnings date for risk awareness.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args.get("symbol", "").upper()
    if not symbol:
        state["tool_results"]["earnings_proximity"] = {"error": "No symbol provided"}
        return state

    import pandas as pd
    from datetime import datetime, timedelta

    # Try yfinance first (more reliable for earnings dates)
    next_earnings_date = None
    last_earnings_date = None
    source = None

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None:
            # yfinance calendar returns dict with datetime.date values
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        next_earnings_date = pd.Timestamp(ed[0])
                    else:
                        next_earnings_date = pd.Timestamp(ed)
            elif hasattr(cal, "iloc"):
                # DataFrame format (older yfinance)
                if "Earnings Date" in cal.columns:
                    next_earnings_date = pd.Timestamp(cal["Earnings Date"].iloc[0])
                elif len(cal.columns) > 0:
                    next_earnings_date = pd.Timestamp(cal.iloc[0, 0])

        # Get last earnings from earnings_dates (more complete history)
        try:
            edates = ticker.earnings_dates
            if edates is not None and len(edates) > 0:
                now_tz = pd.Timestamp.now(tz="UTC") if edates.index.tz else pd.Timestamp.now()
                past = edates.index[edates.index <= now_tz]
                future = edates.index[edates.index > now_tz]
                if len(past) > 0:
                    last_earnings_date = pd.Timestamp(past[0])  # Most recent past
                if len(future) > 0 and next_earnings_date is None:
                    next_earnings_date = pd.Timestamp(future[-1])  # Soonest future
        except Exception:
            pass

        source = "yfinance"
    except Exception:
        pass

    # Fallback: check polygon_earnings in state for last reported date
    if last_earnings_date is None:
        poly_earnings = state.get("tool_results", {}).get("polygon_earnings")
        if poly_earnings and not poly_earnings.get("error"):
            end_date = poly_earnings.get("latest_quarter_end")
            if end_date:
                try:
                    last_earnings_date = pd.Timestamp(end_date)
                    if source is None:
                        source = "polygon_earnings"
                except Exception:
                    pass

    now = pd.Timestamp.now()

    # ------------------------------------------------------------------
    # Compute days-to-earnings and risk classification
    # ------------------------------------------------------------------
    days_to_next = None
    days_since_last = None
    earnings_risk = "unknown"
    guidance = ""

    if next_earnings_date is not None:
        # Strip timezone if present
        if next_earnings_date.tz is not None:
            next_earnings_date = next_earnings_date.tz_localize(None)
        days_to_next = (next_earnings_date - now).days

        if days_to_next < 0:
            # Date is in the past — earnings already happened, yfinance stale
            days_to_next = None
        elif days_to_next <= 3:
            earnings_risk = "IMMINENT"
            guidance = (
                "Earnings within 3 days — binary event risk is EXTREME. "
                "Reduce position to 25-50% of normal or avoid entry. "
                "Post-earnings gap risk can exceed stop-loss levels."
            )
        elif days_to_next <= 7:
            earnings_risk = "HIGH"
            guidance = (
                "Earnings within 1 week — elevated event risk. "
                "Reduce position by 30-50% or set tight stops. "
                "Consider waiting until after the report."
            )
        elif days_to_next <= 14:
            earnings_risk = "MODERATE"
            guidance = (
                "Earnings within 2 weeks — be aware of rising implied vol. "
                "Normal sizing OK but have an exit plan before the report."
            )
        elif days_to_next <= 30:
            earnings_risk = "LOW"
            guidance = (
                "Earnings 2-4 weeks away — minimal event risk for now. "
                "Swing trades can proceed normally but monitor as date approaches."
            )
        else:
            earnings_risk = "NONE"
            guidance = "Earnings date well into the future — no event risk concern."

    if last_earnings_date is not None:
        if last_earnings_date.tz is not None:
            last_earnings_date = last_earnings_date.tz_localize(None)
        days_since_last = (now - last_earnings_date).days

    # Post-earnings drift note
    post_earnings_note = ""
    if days_since_last is not None and days_since_last <= 5:
        post_earnings_note = (
            "Earnings released within the last 5 days — post-earnings drift "
            "is active. Positive surprise + momentum = continuation likely. "
            "Negative surprise = beware of further selling."
        )

    result = {
        "symbol": symbol,
        "next_earnings_date": str(next_earnings_date.date()) if next_earnings_date and days_to_next is not None else None,
        "days_to_next_earnings": days_to_next,
        "last_earnings_date": str(last_earnings_date.date()) if last_earnings_date else None,
        "days_since_last_earnings": days_since_last,
        "earnings_risk": earnings_risk,
        "guidance": guidance,
        "post_earnings_note": post_earnings_note,
        "source": source or "none",
    }
    state["tool_results"]["earnings_proximity"] = result
    return state


register_tool({
    "name": "earnings_proximity",
    "description": (
        "Check how close the next EARNINGS DATE is for a stock. "
        "Flags binary event risk so the agent can adjust position sizing "
        "or recommend waiting until after the report."
        "\n\nReturns:"
        "\n• next_earnings_date: Date of next expected earnings release"
        "\n• days_to_next_earnings: Trading days until earnings"
        "\n• earnings_risk: IMMINENT (<3d) / HIGH (<7d) / MODERATE (<14d) / LOW (<30d) / NONE"
        "\n• guidance: Actionable position sizing recommendation"
        "\n• last_earnings_date + days_since: For post-earnings drift awareness"
        "\n• post_earnings_note: If earnings just happened, drift guidance"
        "\n\nUSE THIS before any swing trade to avoid getting caught by "
        "earnings gaps that can blow through stop-loss levels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker to check earnings date for."
            },
        },
        "required": ["symbol"]
    },
    "fn": earnings_proximity_tool_fn,
})


# =============================================================================
# Volatility Prediction Tool (Early Warning System)
# =============================================================================

# Import the vol prediction function
try:
    from vol_prediction_tool import vol_prediction_tool_fn
    _VOL_PREDICTION_AVAILABLE = True
except ImportError:
    _VOL_PREDICTION_AVAILABLE = False
    
    def vol_prediction_tool_fn(state: dict, args: dict) -> dict:
        if "tool_results" not in state:
            state["tool_results"] = {}
        state["tool_results"]["vol_prediction"] = {
            "error": "Vol prediction tool not available (import failed)",
            "symbol": args.get("symbol")
        }
        return state

if _VOL_PREDICTION_AVAILABLE:
    register_tool({
        "name": "vol_prediction",
        "description": (
            "Predict volatility regime transitions for position sizing and risk management. "
            "Uses VIX-based universal features (work across all equities) plus asset-specific volatility. "
            "\n\nReturns:"
            "\n• current_regime: 'LOW' or 'HIGH' volatility"
            "\n• spike_probability: P(transitioning to HIGH) if currently LOW (vol spike warning)"
            "\n• calm_probability: P(transitioning to LOW) if currently HIGH (vol calming signal)"
            "\n• risk_level: 'LOW', 'WATCH', 'ELEVATED', 'CALMING', 'HIGH'"
            "\n• suggested_action: Position sizing recommendation"
            "\n• vix_zscore: Current VIX relative to 60-day history"
            "\n\nPrecision:"
            "\n• Spike predictions (0.6+ threshold): ~62% precision vs 23% base rate"
            "\n• Calm predictions (0.6+ threshold): ~62% precision vs 42% base rate"
            "\n\nUse this for position sizing, not directional timing. "
            "Low spike probability = normal sizing. High spike probability = reduce exposure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, TSLA)"
                }
            },
            "required": ["symbol"]
        },
        "fn": vol_prediction_tool_fn
    })


# =============================================================================
# VIX ROC Risk Overlay Tool (Three-Tier Market Timing)
# =============================================================================

# Import the VIX ROC overlay
try:
    from models.vix_roc_production import VIXROCRiskOverlay, AssetTier, TIER_PARAMS
    _VIX_ROC_AVAILABLE = True
    _vix_roc_overlay = None  # Lazy initialization
except ImportError:
    _VIX_ROC_AVAILABLE = False


def _get_vix_roc_overlay():
    """Lazy initialization of VIX ROC overlay with current VIX data."""
    global _vix_roc_overlay
    
    if _vix_roc_overlay is None:
        try:
            import yfinance as yf
            import pandas as pd
            
            _vix_roc_overlay = VIXROCRiskOverlay()
            
            # Load recent VIX data (last 60 days for ROC calculation)
            vix = yf.download("^VIX", period="3mo", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            if not vix.empty:
                _vix_roc_overlay.load_vix_data(vix)
        except Exception as e:
            print(f"Warning: Could not initialize VIX ROC overlay: {e}")
            return None
    
    return _vix_roc_overlay


def vix_roc_risk_tool_fn(state: dict, args: dict) -> dict:
    """
    VIX ROC-based risk overlay tool for market timing.
    
    Classifies asset into tier and provides current risk signal.
    Walk-forward validated: 15/15 wins on tested assets (2020-2024).
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    if not _VIX_ROC_AVAILABLE:
        state["tool_results"]["vix_roc_risk"] = {
            "error": "VIX ROC tool not available (import failed)",
            "symbol": args.get("symbol")
        }
        return state
    
    symbol = args.get("symbol", "").upper()
    if not symbol:
        state["tool_results"]["vix_roc_risk"] = {
            "error": "symbol is required"
        }
        return state
    
    overlay = _get_vix_roc_overlay()
    if overlay is None:
        state["tool_results"]["vix_roc_risk"] = {
            "symbol": symbol,
            "error": "Could not initialize VIX ROC overlay (check yfinance)"
        }
        return state
    
    try:
        # Get classification
        classification = overlay.classify_asset(symbol)
        
        # Get current signal
        signal = overlay.get_current_signal(symbol)
        
        # Build result
        result = {
            "symbol": symbol,
            "tier": classification["tier"],
            "tier_name": classification["tier_name"],
            "classification_source": classification["classification_source"],
            "strategy_params": {
                "exit_when_vix_roc_above": f"{classification['params']['exit_threshold']*100:.0f}%",
                "reenter_when_vix_roc_below": f"{classification['params']['reentry_threshold']*100:+.0f}%",
                "min_days_out": classification["params"]["min_exit_days"],
                "roc_lookback_days": classification["params"]["roc_lookback"]
            },
            "current_signal": signal["signal"],
            "current_vix_roc": f"{signal['vix_roc']:.1%}",
            "position_status": "IN MARKET" if signal["position_pct"] == 1.0 else "OUT OF MARKET",
            "message": signal["message"],
            "tier_description": classification["description"]
        }
        
        # Add sector/beta if available
        if classification.get("sector"):
            result["sector"] = classification["sector"]
        if classification.get("beta"):
            result["beta"] = classification["beta"]
        
        state["tool_results"]["vix_roc_risk"] = result
        
    except Exception as e:
        state["tool_results"]["vix_roc_risk"] = {
            "symbol": symbol,
            "error": f"Error computing VIX ROC signal: {str(e)}"
        }
    
    return state


def vix_roc_portfolio_risk_fn(state: dict, args: dict) -> dict:
    """
    Get portfolio-wide VIX ROC risk assessment.
    
    Checks multiple assets and provides overall market risk level.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    if not _VIX_ROC_AVAILABLE:
        state["tool_results"]["vix_roc_portfolio_risk"] = {
            "error": "VIX ROC tool not available"
        }
        return state
    
    symbols = args.get("symbols", [])
    if not symbols:
        # Default to major indices/ETFs
        symbols = ["SPY", "QQQ", "IWM"]
    
    overlay = _get_vix_roc_overlay()
    if overlay is None:
        state["tool_results"]["vix_roc_portfolio_risk"] = {
            "error": "Could not initialize VIX ROC overlay"
        }
        return state
    
    try:
        assessment = overlay.get_risk_assessment([s.upper() for s in symbols])
        
        # Simplify signals for output
        simplified_signals = []
        for sig in assessment.get("signals", []):
            simplified_signals.append({
                "ticker": sig["ticker"],
                "signal": sig["signal"],
                "tier": sig["tier"],
                "message": sig.get("message", "")
            })
        
        result = {
            "timestamp": assessment["timestamp"],
            "current_vix": assessment["current_vix"],
            "risk_level": assessment["risk_level"],
            "risk_message": assessment["risk_message"],
            "exit_signals_active": assessment["exit_signals"],
            "assets_out_of_market": assessment["assets_out_of_market"],
            "total_assets_checked": assessment["total_assets"],
            "individual_signals": simplified_signals
        }
        
        state["tool_results"]["vix_roc_portfolio_risk"] = result
        
    except Exception as e:
        state["tool_results"]["vix_roc_portfolio_risk"] = {
            "error": f"Error computing portfolio risk: {str(e)}"
        }
    
    return state


if _VIX_ROC_AVAILABLE:
    # DEPRECATED: vix_roc_risk replaced by market_risk (ML-based drawdown + vol prediction)
    # Functions still exist but are no longer registered as tools.
    # register_tool({
    #     "name": "vix_roc_risk",
    #     "description": "DEPRECATED - Use market_risk instead.",
    #     "parameters": {...},
    #     "fn": vix_roc_risk_tool_fn,
    # })
    # register_tool({
    #     "name": "vix_roc_portfolio_risk",
    #     "description": "DEPRECATED - Use market_risk instead.",
    #     "parameters": {...},
    #     "fn": vix_roc_portfolio_risk_fn,
    # })
    pass


# =============================================================================
# MARKET RISK MODEL (ML-based drawdown probability + forward volatility)
# Replaces vix_roc_risk binary signal with continuous risk probabilities.
# =============================================================================

try:
    from market_risk_model import market_risk_tool_fn
    _MARKET_RISK_AVAILABLE = True
except ImportError:
    _MARKET_RISK_AVAILABLE = False

if _MARKET_RISK_AVAILABLE:
    register_tool({
        "name": "market_risk",
        "description": (
            "ML-based market risk assessment — the PRIMARY market timing and position sizing tool. "
            "Predicts DRAWDOWN PROBABILITY and FORWARD VOLATILITY using GradientBoosting "
            "trained on VIX + SPY features with walk-forward validation."
            "\n\nReturns:"
            "\n• drawdown_probability: P(SPY max DD > 3% in next 10 days) — continuous 0-1"
            "\n• drawdown_risk_level: LOW / MODERATE / ELEVATED / HIGH / EXTREME"
            "\n• predicted_fwd_vol_pct: Predicted annualised forward volatility"
            "\n• current_realized_vol_pct: Current 20-day realised vol for comparison"
            "\n• vol_trend: RISING / STABLE / FALLING"
            "\n• suggested_position_pct: Recommended position size (1.0 = full, 0.2 = minimal)"
            "\n• suggested_stop_multiplier: How much to widen stops (1.0x to 2.0x)"
            "\n• variance_risk_premium: VIX / realised vol ratio"
            "\n• top_risk_drivers: Top 5 features driving current risk"
            "\n• validation: Walk-forward AUC, Brier score, vol R²"
            "\n\nUSE THIS before entering any position to assess market-wide risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker being evaluated (for context; risk is market-wide)."
                }
            },
            "required": ["symbol"]
        },
        "fn": market_risk_tool_fn,
    })