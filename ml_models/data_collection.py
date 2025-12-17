"""
Data collection module for ML models.
Fetches historical data from Polygon.io and caches it locally.
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


def fetch_daily_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Polygon.io.
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set in environment")
    
    # Check cache first
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file):
        if VERBOSE:
            print(f"  Loading {symbol} from cache...")
        return pd.read_parquet(cache_file)
    
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
        
        # Cache the data
        df.to_parquet(cache_file)
        
        if VERBOSE:
            print(f"    Fetched {len(df)} bars for {symbol}")
        
        return df
        
    except Exception as e:
        if VERBOSE:
            print(f"    Error fetching {symbol}: {e}")
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
