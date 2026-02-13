"""
Polygon.io Market Data Tools for Trading Agent
==============================================

This module provides market data tools using Polygon.io's API.
Replaces Alpha Vantage with 100k calls/month on Massive tier vs 25/day free tier.

Free Tier Features:
- All US Stocks Tickers  
- 5 API Calls / Minute
- 2 Years Historical Data
- 100% Market Coverage
- End of Day Data
- Reference Data
- Corporate Actions  
- Technical Indicators
- Minute Aggregates

Tools:
1. polygon_price_data - Daily OHLCV bars with volume analysis
2. polygon_ticker_details - Company information and metadata
3. polygon_technical_sma - Simple Moving Average
4. polygon_technical_ema - Exponential Moving Average  
5. polygon_technical_rsi - Relative Strength Index
6. polygon_technical_macd - MACD indicator
7. polygon_previous_close - Previous trading day's data
8. polygon_snapshot - Real-time snapshot of ticker
9. polygon_dividends - Dividend history
10. polygon_splits - Stock split history
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict, Optional
import warnings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Use unified tool registry
from tool_registry import ToolSpec, ToolRegistry

# ==================== CONFIGURATION ====================

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# ==================== REGISTRY ====================

# Use unified ToolRegistry with O(1) lookup
_registry = ToolRegistry()


def register_tool(spec: ToolSpec):
    """Register a tool in the global registry."""
    _registry.register(spec)


def get_all_tools() -> List[Dict]:
    """Get all registered tools (without the fn field for JSON serialization)."""
    return _registry.get_all_specs()


def get_tool_function(tool_name: str) -> Optional[callable]:
    """Get the function associated with a tool name."""
    return _registry.get_function(tool_name)


# ==================== HELPER FUNCTIONS ====================

def _check_api_key() -> Optional[dict]:
    """Check if Polygon API key is configured. Returns error dict if missing, None if OK."""
    if not POLYGON_API_KEY:
        return {"error": "POLYGON_API_KEY environment variable not set. Get one at https://polygon.io"}
    return None


def _polygon_request(endpoint: str, params: dict = None, max_retries: int = 3) -> dict:
    """Make a request to Polygon.io API with error handling and retry logic."""
    key_error = _check_api_key()
    if key_error:
        return key_error
    
    if params is None:
        params = {}
    
    params['apiKey'] = POLYGON_API_KEY
    url = f"{POLYGON_BASE_URL}{endpoint}"
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if data.get('status') == 'ERROR':
                error_msg = data.get('error', 'Unknown API error')
                return {"error": error_msg}
            
            return data
            
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2.0 * (2 ** attempt)
                import logging
                logging.getLogger(__name__).warning(
                    "Polygon API retry %d/%d for %s: %s (waiting %.0fs)",
                    attempt + 1, max_retries, endpoint, e, wait
                )
                import time
                time.sleep(wait)
        except ValueError as e:
            return {"error": f"JSON parsing failed: {str(e)}"}
    
    return {"error": f"Request failed after {max_retries} retries: {str(last_error)}"}


def _calculate_atr(bars: List[Dict], window: int = 14) -> Dict:
    """
    Calculate Average True Range (ATR) from OHLCV bars.
    ATR measures volatility - critical for stop loss placement.
    """
    if not bars or len(bars) < window + 1:
        return {
            "atr_value": None,
            "window": window,
            "error": f"Need at least {window + 1} bars for ATR calculation"
        }
    
    true_ranges = []
    for i in range(1, len(bars)):
        current = bars[i]
        previous = bars[i-1]
        
        high = current.get('h', 0)
        low = current.get('l', 0)
        prev_close = previous.get('c', 0)
        
        # True Range is max of:
        # 1. Current High - Current Low
        # 2. abs(Current High - Previous Close)
        # 3. abs(Current Low - Previous Close)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < window:
        return {
            "atr_value": None,
            "window": window,
            "error": f"Insufficient data for ATR{window}"
        }
    
    # Calculate ATR as simple moving average of True Range
    # (Technically should use Wilder's smoothing, but SMA is close enough)
    recent_trs = true_ranges[-window:]
    atr_value = sum(recent_trs) / len(recent_trs)
    
    return {
        "atr_value": atr_value,
        "window": window,
        "num_bars_used": len(bars)
    }


def _calculate_volume_analysis(bars: List[Dict]) -> Dict:
    """Calculate volume analysis metrics from OHLCV bars."""
    if not bars or len(bars) < 2:
        return {
            "avg_volume": None,
            "latest_volume": None,
            "volume_ratio": None,
            "volume_condition": "insufficient data",
            "recent_volume_spikes": []
        }
    
    volumes = [bar['v'] for bar in bars if 'v' in bar]
    if not volumes:
        return {
            "avg_volume": None,
            "latest_volume": None,
            "volume_ratio": None,
            "volume_condition": "no volume data",
            "recent_volume_spikes": []
        }
    
    avg_volume = sum(volumes) / len(volumes)
    latest_volume = volumes[-1]
    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
    
    # Classify volume condition
    if volume_ratio > 2.0:
        condition = "very high"
    elif volume_ratio > 1.5:
        condition = "high"
    elif volume_ratio > 1.2:
        condition = "above average"
    elif volume_ratio > 0.8:
        condition = "average"
    elif volume_ratio > 0.5:
        condition = "below average"
    else:
        condition = "very low"
    
    # Find recent volume spikes (>1.5x avg in last 10 bars)
    spikes = []
    recent_bars = bars[-min(10, len(bars)):]
    for i, bar in enumerate(recent_bars):
        bar_volume = bar.get('v', 0)
        if bar_volume > avg_volume * 1.5:
            spikes.append({
                "date": datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d') if 't' in bar else f"bar_{i}",
                "volume": bar_volume,
                "ratio": bar_volume / avg_volume if avg_volume > 0 else 0,
                "close": bar.get('c')
            })
    
    return {
        "avg_volume": avg_volume,
        "latest_volume": latest_volume,
        "volume_ratio": volume_ratio,
        "volume_condition": condition,
        "recent_volume_spikes": spikes
    }


# ==================== TOOL IMPLEMENTATIONS ====================

def polygon_price_data_tool_fn(state: dict, args: dict) -> dict:
    """
    Get daily OHLCV price data from Polygon.io with volume analysis.
    
    Returns:
        - Daily bars with open, high, low, close, volume
        - Volume analysis (avg, ratio, condition, spikes)
        - Price data for technical indicator calculations
    """
    symbol = args.get("symbol", "").upper()
    days = args.get("days", 100)
    
    if not symbol:
        state["tool_results"]["polygon_price_data"] = {"error": "Symbol is required"}
        return state
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Polygon aggregates endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000  # Polygon supports up to 50k results
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_price_data"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', [])
    if not results:
        state["tool_results"]["polygon_price_data"] = {
            "symbol": symbol,
            "error": "No price data available"
        }
        return state
    
    # Calculate volume analysis
    volume_analysis = _calculate_volume_analysis(results)
    
    # Calculate ATR (14-day by default)
    atr_analysis = _calculate_atr(results, window=14)
    
    # Get latest bar
    latest = results[-1]
    latest_date = datetime.fromtimestamp(latest['t'] / 1000).strftime('%Y-%m-%d')
    latest_close = latest.get('c', 0)
    
    # Format result
    result = {
        "symbol": symbol,
        "interval": "daily",
        "latest_date": latest_date,
        "latest_open": latest.get('o'),
        "latest_high": latest.get('h'),
        "latest_low": latest.get('l'),
        "latest_close": latest_close,
        "latest_volume": latest.get('v'),
        "num_bars": len(results),
        "date_range": {
            "from": datetime.fromtimestamp(results[0]['t'] / 1000).strftime('%Y-%m-%d'),
            "to": latest_date
        },
        "volume_analysis": volume_analysis,
        "atr": atr_analysis,  # ATR for stop loss placement
        "bars": results  # Full bar data for calculations
    }
    
    state["tool_results"]["polygon_price_data"] = result
    return state


def polygon_ticker_details_tool_fn(state: dict, args: dict) -> dict:
    """
    Get company details and metadata from Polygon.io.
    
    Returns:
        - Company name, description
        - Market cap, shares outstanding
        - Primary exchange, locale
        - Industry, sector, SIC code
        - Homepage URL, logo
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_ticker_details"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v3/reference/tickers/{symbol}"
    params = {"date": datetime.now().strftime('%Y-%m-%d')}
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_ticker_details"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', {})
    if not results:
        state["tool_results"]["polygon_ticker_details"] = {
            "symbol": symbol,
            "error": "No ticker details available"
        }
        return state
    
    # Format result
    result = {
        "symbol": symbol,
        "name": results.get('name'),
        "description": results.get('description'),
        "market_cap": results.get('market_cap'),
        "shares_outstanding": results.get('share_class_shares_outstanding'),
        "primary_exchange": results.get('primary_exchange'),
        "locale": results.get('locale'),
        "type": results.get('type'),
        "currency": results.get('currency_name'),
        "sic_code": results.get('sic_code'),
        "sic_description": results.get('sic_description'),
        "homepage_url": results.get('homepage_url'),
        "list_date": results.get('list_date'),
        "branding": results.get('branding', {})
    }
    
    state["tool_results"]["polygon_ticker_details"] = result
    return state


def polygon_technical_sma_tool_fn(state: dict, args: dict) -> dict:
    """
    Get Simple Moving Average (SMA) from Polygon.io.
    
    Technical indicator from Polygon's free tier.
    """
    symbol = args.get("symbol", "").upper()
    window = args.get("window", 50)
    
    if not symbol:
        state["tool_results"]["polygon_technical_sma"] = {"error": "Symbol is required"}
        return state
    
    # SMA endpoint: /v1/indicators/sma/{ticker}
    endpoint = f"/v1/indicators/sma/{symbol}"
    params = {
        "timespan": "day",
        "adjusted": "true",
        "window": window,
        "series_type": "close",
        "order": "desc",
        "limit": 120  # Get enough data for analysis
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_technical_sma"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', {}).get('values', [])
    if not results:
        state["tool_results"]["polygon_technical_sma"] = {
            "symbol": symbol,
            "error": "No SMA data available"
        }
        return state
    
    # Get latest SMA value
    latest = results[0]  # Results are desc order
    
    result = {
        "symbol": symbol,
        "indicator": "SMA",
        "window": window,
        "latest_timestamp": latest.get('timestamp'),
        "latest_value": latest.get('value'),
        "num_values": len(results),
        "values": results[:20]  # Return recent 20 values
    }
    
    state["tool_results"]["polygon_technical_sma"] = result
    return state


def polygon_technical_ema_tool_fn(state: dict, args: dict) -> dict:
    """
    Get Exponential Moving Average (EMA) from Polygon.io.
    
    Technical indicator from Polygon's free tier.
    """
    symbol = args.get("symbol", "").upper()
    window = args.get("window", 12)
    
    if not symbol:
        state["tool_results"]["polygon_technical_ema"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v1/indicators/ema/{symbol}"
    params = {
        "timespan": "day",
        "adjusted": "true",
        "window": window,
        "series_type": "close",
        "order": "desc",
        "limit": 120
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_technical_ema"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', {}).get('values', [])
    if not results:
        state["tool_results"]["polygon_technical_ema"] = {
            "symbol": symbol,
            "error": "No EMA data available"
        }
        return state
    
    latest = results[0]
    
    result = {
        "symbol": symbol,
        "indicator": "EMA",
        "window": window,
        "latest_timestamp": latest.get('timestamp'),
        "latest_value": latest.get('value'),
        "num_values": len(results),
        "values": results[:20]
    }
    
    state["tool_results"]["polygon_technical_ema"] = result
    return state


def polygon_technical_rsi_tool_fn(state: dict, args: dict) -> dict:
    """
    Get Relative Strength Index (RSI) from Polygon.io.
    
    Technical indicator from Polygon's free tier.
    """
    symbol = args.get("symbol", "").upper()
    window = args.get("window", 14)
    
    if not symbol:
        state["tool_results"]["polygon_technical_rsi"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v1/indicators/rsi/{symbol}"
    params = {
        "timespan": "day",
        "adjusted": "true",
        "window": window,
        "series_type": "close",
        "order": "desc",
        "limit": 120
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_technical_rsi"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', {}).get('values', [])
    if not results:
        state["tool_results"]["polygon_technical_rsi"] = {
            "symbol": symbol,
            "error": "No RSI data available"
        }
        return state
    
    latest = results[0]
    latest_value = latest.get('value')
    
    # Classify RSI
    if latest_value is not None:
        if latest_value >= 70:
            condition = "overbought"
        elif latest_value <= 30:
            condition = "oversold"
        else:
            condition = "neutral"
    else:
        condition = "unknown"
    
    result = {
        "symbol": symbol,
        "indicator": "RSI",
        "window": window,
        "latest_timestamp": latest.get('timestamp'),
        "latest_value": latest_value,
        "condition": condition,
        "num_values": len(results),
        "values": results[:20]
    }
    
    state["tool_results"]["polygon_technical_rsi"] = result
    return state


def polygon_technical_macd_tool_fn(state: dict, args: dict) -> dict:
    """
    Get MACD (Moving Average Convergence Divergence) from Polygon.io.
    
    Technical indicator from Polygon's free tier.
    """
    symbol = args.get("symbol", "").upper()
    short_window = args.get("short_window", 12)
    long_window = args.get("long_window", 26)
    signal_window = args.get("signal_window", 9)
    
    if not symbol:
        state["tool_results"]["polygon_technical_macd"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v1/indicators/macd/{symbol}"
    params = {
        "timespan": "day",
        "adjusted": "true",
        "short_window": short_window,
        "long_window": long_window,
        "signal_window": signal_window,
        "series_type": "close",
        "order": "desc",
        "limit": 120
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_technical_macd"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', {}).get('values', [])
    if not results:
        state["tool_results"]["polygon_technical_macd"] = {
            "symbol": symbol,
            "error": "No MACD data available"
        }
        return state
    
    latest = results[0]
    macd_value = latest.get('value')
    signal = latest.get('signal')
    histogram = macd_value - signal if (macd_value is not None and signal is not None) else None
    
    # Determine signal
    if histogram is not None:
        if histogram > 0:
            signal_type = "bullish"
        elif histogram < 0:
            signal_type = "bearish"
        else:
            signal_type = "neutral"
    else:
        signal_type = "unknown"
    
    result = {
        "symbol": symbol,
        "indicator": "MACD",
        "parameters": {
            "short_window": short_window,
            "long_window": long_window,
            "signal_window": signal_window
        },
        "latest_timestamp": latest.get('timestamp'),
        "latest_macd": macd_value,
        "latest_signal": signal,
        "latest_histogram": histogram,
        "signal_type": signal_type,
        "num_values": len(results),
        "values": results[:20]
    }
    
    state["tool_results"]["polygon_technical_macd"] = result
    return state


# NOTE: ATR is calculated from price data in polygon_price_data_tool_fn
# No separate API call needed - it's more efficient and works on free tier


def polygon_previous_close_tool_fn(state: dict, args: dict) -> dict:
    """
    Get previous day's close data from Polygon.io.
    
    Quick way to get latest price action.
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_previous_close"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v2/aggs/ticker/{symbol}/prev"
    params = {"adjusted": "true"}
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_previous_close"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', [])
    if not results:
        state["tool_results"]["polygon_previous_close"] = {
            "symbol": symbol,
            "error": "No previous close data available"
        }
        return state
    
    prev = results[0]
    
    result = {
        "symbol": symbol,
        "date": datetime.fromtimestamp(prev['t'] / 1000).strftime('%Y-%m-%d'),
        "open": prev.get('o'),
        "high": prev.get('h'),
        "low": prev.get('l'),
        "close": prev.get('c'),
        "volume": prev.get('v'),
        "vwap": prev.get('vw'),
        "num_trades": prev.get('n')
    }
    
    state["tool_results"]["polygon_previous_close"] = result
    return state


def polygon_snapshot_tool_fn(state: dict, args: dict) -> dict:
    """
    Get real-time snapshot of ticker from Polygon.io.
    
    Includes current price, day's range, volume, and more.
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_snapshot"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
    
    data = _polygon_request(endpoint)
    
    if "error" in data:
        state["tool_results"]["polygon_snapshot"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    ticker = data.get('ticker', {})
    if not ticker:
        state["tool_results"]["polygon_snapshot"] = {
            "symbol": symbol,
            "error": "No snapshot data available"
        }
        return state
    
    day = ticker.get('day', {})
    prev_day = ticker.get('prevDay', {})
    
    result = {
        "symbol": symbol,
        "updated": ticker.get('updated'),
        "today": {
            "open": day.get('o'),
            "high": day.get('h'),
            "low": day.get('l'),
            "close": day.get('c'),
            "volume": day.get('v'),
            "vwap": day.get('vw')
        },
        "previous_day": {
            "close": prev_day.get('c'),
            "volume": prev_day.get('v')
        },
        "change": {
            "price": ticker.get('todaysChange'),
            "percent": ticker.get('todaysChangePerc')
        }
    }
    
    state["tool_results"]["polygon_snapshot"] = result
    return state


def polygon_dividends_tool_fn(state: dict, args: dict) -> dict:
    """
    Get dividend history from Polygon.io.
    
    Corporate actions data from free tier.
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_dividends"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v3/reference/dividends"
    params = {
        "ticker": symbol,
        "order": "desc",
        "limit": 100
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_dividends"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', [])
    
    result = {
        "symbol": symbol,
        "num_dividends": len(results),
        "recent_dividends": results[:10] if results else [],
        "latest_dividend": results[0] if results else None
    }
    
    state["tool_results"]["polygon_dividends"] = result
    return state


def polygon_splits_tool_fn(state: dict, args: dict) -> dict:
    """
    Get stock split history from Polygon.io.
    
    Corporate actions data from free tier.
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_splits"] = {"error": "Symbol is required"}
        return state
    
    endpoint = f"/v3/reference/splits"
    params = {
        "ticker": symbol,
        "order": "desc",
        "limit": 100
    }
    
    data = _polygon_request(endpoint, params)
    
    if "error" in data:
        state["tool_results"]["polygon_splits"] = {
            "symbol": symbol,
            "error": data["error"]
        }
        return state
    
    results = data.get('results', [])
    
    result = {
        "symbol": symbol,
        "num_splits": len(results),
        "splits": results[:10] if results else [],
        "latest_split": results[0] if results else None
    }
    
    state["tool_results"]["polygon_splits"] = result
    return state


# ==================== REGISTER TOOLS ====================

# Register price data tool
register_tool({
    "name": "polygon_price_data",
    "description": "Get daily OHLCV price data with volume analysis. Returns up to 100 daily bars with open, high, low, close, volume. Includes volume analysis metrics (average, ratio, condition, spikes). Data used for technical calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA')"
            },
            "days": {
                "type": "integer",
                "description": "Number of days of history to retrieve (default: 100, max: 730)",
                "default": 100
            }
        },
        "required": ["symbol"]
    },
    "fn": polygon_price_data_tool_fn
})

# Register ticker details tool
register_tool({
    "name": "polygon_ticker_details",
    "description": "Get company information and metadata. Returns company name, description, market cap, shares outstanding, industry, sector, exchange info, and more. Use for fundamental context.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol"
            }
        },
        "required": ["symbol"]
    },
    "fn": polygon_ticker_details_tool_fn
})

# Register technical indicator tools
register_tool({
    "name": "polygon_technical_sma",
    "description": "Get Simple Moving Average (SMA). Returns recent SMA values for trend analysis. Common windows: 50, 100, 200 days.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "window": {"type": "integer", "description": "SMA period (default: 50)", "default": 50}
        },
        "required": ["symbol"]
    },
    "fn": polygon_technical_sma_tool_fn
})

register_tool({
    "name": "polygon_technical_ema",
    "description": "Get Exponential Moving Average (EMA). Returns recent EMA values for trend analysis. Common windows: 12, 26 days.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "window": {"type": "integer", "description": "EMA period (default: 12)", "default": 12}
        },
        "required": ["symbol"]
    },
    "fn": polygon_technical_ema_tool_fn
})

register_tool({
    "name": "polygon_technical_rsi",
    "description": "Get Relative Strength Index (RSI). Returns RSI values with overbought/oversold classification. Values >70 = overbought, <30 = oversold.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "window": {"type": "integer", "description": "RSI period (default: 14)", "default": 14}
        },
        "required": ["symbol"]
    },
    "fn": polygon_technical_rsi_tool_fn
})

register_tool({
    "name": "polygon_technical_macd",
    "description": "Get MACD indicator. Returns MACD line, signal line, histogram with bullish/bearish classification.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "short_window": {"type": "integer", "description": "Short EMA period (default: 12)", "default": 12},
            "long_window": {"type": "integer", "description": "Long EMA period (default: 26)", "default": 26},
            "signal_window": {"type": "integer", "description": "Signal line period (default: 9)", "default": 9}
        },
        "required": ["symbol"]
    },
    "fn": polygon_technical_macd_tool_fn
})

# NOTE: ATR is now calculated automatically in polygon_price_data (no separate tool needed)
# This is more efficient and doesn't require an extra API call

# Register quick data tools
register_tool({
    "name": "polygon_previous_close",
    "description": "Get previous trading day's OHLCV data. Quick way to get latest price action.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    },
    "fn": polygon_previous_close_tool_fn
})

register_tool({
    "name": "polygon_snapshot",
    "description": "Get real-time snapshot with current price, day's range, volume, and change vs previous close.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    },
    "fn": polygon_snapshot_tool_fn
})

# Register corporate actions tools
register_tool({
    "name": "polygon_dividends",
    "description": "Get dividend payment history. Returns recent dividend amounts, ex-dates, and payment dates.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    },
    "fn": polygon_dividends_tool_fn
})

register_tool({
    "name": "polygon_splits",
    "description": "Get stock split history. Returns split ratios and execution dates.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    },
    "fn": polygon_splits_tool_fn
})


# ==================== EARNINGS DATA ====================

def polygon_earnings_tool_fn(state: dict, args: dict) -> dict:
    """
    Fetch quarterly earnings data including revenue and EPS with growth rates.
    Provides last 8 quarters to show growth trends (2 years of data).
    """
    symbol = args.get("symbol", "").upper()
    
    if not symbol:
        state["tool_results"]["polygon_earnings"] = {
            "symbol": symbol,
            "error": "Symbol is required"
        }
        return state
    
    try:
        # Fetch financials from Polygon (vX endpoint)
        endpoint = f"/vX/reference/financials"
        params = {
            "ticker": symbol,
            "limit": 8,  # Last 8 quarters
            "timeframe": "quarterly",
            "order": "desc"
        }
        
        data = _polygon_request(endpoint, params)
        
        if data.get("status") != "OK" or not data.get("results"):
            state["tool_results"]["polygon_earnings"] = {
                "symbol": symbol,
                "error": "No earnings data available"
            }
            return state
        
        # Parse earnings data
        quarters = []
        for result in data["results"]:
            financials = result.get("financials", {})
            income_statement = financials.get("income_statement", {})
            balance_sheet = financials.get("balance_sheet", {})
            
            quarter_data = {
                "fiscal_period": result.get("fiscal_period"),
                "fiscal_year": result.get("fiscal_year"),
                "start_date": result.get("start_date"),
                "end_date": result.get("end_date"),
                "revenue": income_statement.get("revenues", {}).get("value"),
                "net_income": income_statement.get("net_income_loss", {}).get("value"),
                "eps_basic": income_statement.get("basic_earnings_per_share", {}).get("value"),
                "eps_diluted": income_statement.get("diluted_earnings_per_share", {}).get("value"),
                "gross_profit": income_statement.get("gross_profit", {}).get("value"),
                "operating_income": income_statement.get("operating_income_loss", {}).get("value"),
            }
            quarters.append(quarter_data)
        
        # Calculate growth rates (QoQ and YoY)
        growth_metrics = []
        for i, q in enumerate(quarters):
            metrics = {
                "period": f"{q['fiscal_year']}-{q['fiscal_period']}",
                "end_date": q["end_date"],
                "revenue": q["revenue"],
                "eps_diluted": q["eps_diluted"],
                "net_income": q["net_income"],
            }
            
            # Quarter-over-Quarter growth
            if i < len(quarters) - 1:
                prev_q = quarters[i + 1]
                if q["revenue"] and prev_q["revenue"]:
                    metrics["revenue_qoq_growth"] = ((q["revenue"] - prev_q["revenue"]) / abs(prev_q["revenue"])) * 100
                if q["eps_diluted"] and prev_q["eps_diluted"]:
                    metrics["eps_qoq_growth"] = ((q["eps_diluted"] - prev_q["eps_diluted"]) / abs(prev_q["eps_diluted"])) * 100
                if q["net_income"] and prev_q["net_income"]:
                    metrics["net_income_qoq_growth"] = ((q["net_income"] - prev_q["net_income"]) / abs(prev_q["net_income"])) * 100
            
            # Year-over-Year growth (compare to 4 quarters ago)
            if i < len(quarters) - 4:
                yoy_q = quarters[i + 4]
                if q["revenue"] and yoy_q["revenue"]:
                    metrics["revenue_yoy_growth"] = ((q["revenue"] - yoy_q["revenue"]) / abs(yoy_q["revenue"])) * 100
                if q["eps_diluted"] and yoy_q["eps_diluted"]:
                    metrics["eps_yoy_growth"] = ((q["eps_diluted"] - yoy_q["eps_diluted"]) / abs(yoy_q["eps_diluted"])) * 100
                if q["net_income"] and yoy_q["net_income"]:
                    metrics["net_income_yoy_growth"] = ((q["net_income"] - yoy_q["net_income"]) / abs(yoy_q["net_income"])) * 100
            
            growth_metrics.append(metrics)
        
        # Calculate average growth rates
        revenue_qoq_rates = [m["revenue_qoq_growth"] for m in growth_metrics if "revenue_qoq_growth" in m]
        revenue_yoy_rates = [m["revenue_yoy_growth"] for m in growth_metrics if "revenue_yoy_growth" in m]
        eps_qoq_rates = [m["eps_qoq_growth"] for m in growth_metrics if "eps_qoq_growth" in m]
        eps_yoy_rates = [m["eps_yoy_growth"] for m in growth_metrics if "eps_yoy_growth" in m]
        
        avg_metrics = {
            "avg_revenue_qoq_growth": sum(revenue_qoq_rates) / len(revenue_qoq_rates) if revenue_qoq_rates else None,
            "avg_revenue_yoy_growth": sum(revenue_yoy_rates) / len(revenue_yoy_rates) if revenue_yoy_rates else None,
            "avg_eps_qoq_growth": sum(eps_qoq_rates) / len(eps_qoq_rates) if eps_qoq_rates else None,
            "avg_eps_yoy_growth": sum(eps_yoy_rates) / len(eps_yoy_rates) if eps_yoy_rates else None,
        }
        
        # Latest quarter data
        latest = quarters[0] if quarters else {}
        
        result = {
            "symbol": symbol,
            "latest_quarter": f"{latest.get('fiscal_year')}-{latest.get('fiscal_period')}",
            "latest_quarter_end": latest.get("end_date"),
            "latest_revenue": latest.get("revenue"),
            "latest_eps": latest.get("eps_diluted"),
            "latest_net_income": latest.get("net_income"),
            "quarters_count": len(quarters),
            "quarterly_data": growth_metrics,
            "average_growth": avg_metrics,
        }
        
        state["tool_results"]["polygon_earnings"] = result
        
    except Exception as e:
        state["tool_results"]["polygon_earnings"] = {
            "symbol": symbol,
            "error": f"Failed to fetch earnings: {str(e)}"
        }
    
    return state


register_tool({
    "name": "polygon_earnings",
    "description": "Get quarterly earnings reports with revenue and EPS growth rates. Returns last 8 quarters of financial data including quarter-over-quarter and year-over-year growth trends. Essential for analyzing company growth trajectory and earnings momentum.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    },
    "fn": polygon_earnings_tool_fn
})
