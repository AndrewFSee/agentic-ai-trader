"""
Data collection module for ML models.
Fetches historical data from Polygon.io and caches it locally.
Falls back to yfinance for longer historical data.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import requests
from dotenv import load_dotenv

load_dotenv()

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from config import (
    STOCK_UNIVERSE, 
    START_DATE_STR, 
    END_DATE_STR,
    DATA_DIR,
    VERBOSE
)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_daily_bars_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance.
    Used as fallback when Polygon doesn't have enough historical data.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    if VERBOSE:
        print(f"  Fetching {symbol} from Yahoo Finance...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Add placeholder columns that Polygon has but yfinance doesn't
        if "vwap" not in df.columns:
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        if "transactions" not in df.columns:
            df["transactions"] = 0
        
        # Select relevant columns
        df = df[["open", "high", "low", "close", "volume", "vwap", "transactions"]]
        
        # Ensure index is named 'date'
        df.index.name = "date"
        
        if VERBOSE:
            print(f"    Fetched {len(df)} bars for {symbol} from yfinance")
        
        return df
        
    except Exception as e:
        if VERBOSE:
            print(f"    Error fetching {symbol} from yfinance: {e}")
        return pd.DataFrame()


def fetch_daily_bars(symbol: str, start_date: str, end_date: str, use_yfinance: bool = False) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Polygon.io or yfinance.
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_yfinance: If True, use yfinance instead of Polygon
    
    Returns:
        DataFrame with OHLCV data
    """
    # Check cache first
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        # Check if cached data actually has the requested date range
        if len(df) > 0:
            actual_start = df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10]
            requested_start = start_date
            # If cached data starts much later than requested, it's probably incomplete
            # Allow 30 days slack for weekends/holidays
            if actual_start <= requested_start or (pd.to_datetime(actual_start) - pd.to_datetime(requested_start)).days < 30:
                if VERBOSE:
                    print(f"  Loading {symbol} from cache...")
                return df
            else:
                if VERBOSE:
                    print(f"  Cache for {symbol} starts at {actual_start}, need {requested_start} - refetching...")
    
    # Use yfinance if requested or Polygon not available
    if use_yfinance or not POLYGON_API_KEY:
        df = fetch_daily_bars_yfinance(symbol, start_date, end_date)
        if not df.empty:
            df.to_parquet(cache_file)
        return df
    
    if VERBOSE:
        print(f"  Fetching {symbol} from Polygon.io...")
    
    # Fetch from API
    endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY
    }
    
    url = f"{POLYGON_BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Accept both OK and DELAYED status (delayed data is still valid historical data)
        status = data.get("status")
        if status not in ["OK", "DELAYED"]:
            raise ValueError(f"API error: {status}")
        
        results = data.get("results", [])
        
        if not results:
            raise ValueError(f"No data returned for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Rename columns to standard format
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions"
        })
        
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date")
        
        # Select relevant columns
        df = df[["open", "high", "low", "close", "volume", "vwap", "transactions"]]
        
        # Check if Polygon returned enough data
        # If data starts much later than requested, try yfinance
        if len(df) > 0:
            actual_start = df.index[0]
            requested_start = pd.to_datetime(start_date)
            days_missing = (actual_start - requested_start).days
            
            if days_missing > 60 and YFINANCE_AVAILABLE:  # More than 2 months missing
                if VERBOSE:
                    print(f"    Polygon data starts at {actual_start.strftime('%Y-%m-%d')}, missing {days_missing} days")
                    print(f"    Falling back to yfinance for complete history...")
                
                df_yf = fetch_daily_bars_yfinance(symbol, start_date, end_date)
                if not df_yf.empty and len(df_yf) > len(df):
                    df = df_yf
        
        # Cache the data
        df.to_parquet(cache_file)
        
        if VERBOSE:
            print(f"    Fetched {len(df)} bars for {symbol}")
        
        return df
        
    except Exception as e:
        if VERBOSE:
            print(f"    Error fetching {symbol} from Polygon: {e}")
        
        # Fallback to yfinance on error
        if YFINANCE_AVAILABLE:
            if VERBOSE:
                print(f"    Falling back to yfinance...")
            df = fetch_daily_bars_yfinance(symbol, start_date, end_date)
            if not df.empty:
                df.to_parquet(cache_file)
            return df
        
        return pd.DataFrame()


def fetch_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY (S&P 500 ETF) for market-relative features."""
    return fetch_daily_bars("SPY", start_date, end_date)


def collect_all_stocks() -> Dict[str, pd.DataFrame]:
    """
    Collect data for all stocks in the universe.
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    print("=" * 80)
    print(f"COLLECTING DATA FOR {sum(len(v) for v in STOCK_UNIVERSE.values())} STOCKS")
    print("=" * 80)
    print(f"Date range: {START_DATE_STR} to {END_DATE_STR}")
    print()
    
    all_data = {}
    
    # Collect data for each category
    for category, symbols in STOCK_UNIVERSE.items():
        print(f"\n{category.upper()} stocks:")
        
        for symbol in symbols:
            try:
                df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
                
                if not df.empty:
                    all_data[symbol] = df
                else:
                    print(f"    WARNING: No data for {symbol}")
                    
            except Exception as e:
                print(f"    ERROR: Failed to fetch {symbol}: {e}")
    
    # Fetch SPY for market-relative features
    print("\n\nMARKET BENCHMARK:")
    try:
        spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
        if not spy_df.empty:
            all_data["SPY"] = spy_df
            print(f"  OK: SPY fetched ({len(spy_df)} bars)")
    except Exception as e:
        print(f"  WARNING: Failed to fetch SPY: {e}")
    
    print("\n" + "=" * 80)
    print(f"COLLECTION COMPLETE: {len(all_data)} stocks")
    print("=" * 80)
    
    return all_data


def get_stock_category(symbol: str) -> str:
    """Get the category for a given symbol."""
    for category, symbols in STOCK_UNIVERSE.items():
        if symbol in symbols:
            return category
    return "unknown"


if __name__ == "__main__":
    # Test data collection
    data = collect_all_stocks()
    
    # Print summary
    print("\n\nDATA SUMMARY:")
    print("-" * 80)
    for symbol, df in data.items():
        category = get_stock_category(symbol)
        print(f"{symbol:6s} ({category:10s}): {len(df):4d} bars, "
              f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
