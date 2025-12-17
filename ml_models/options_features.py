"""
Options Features for ML Trading Models

Uses Polygon.io Starter tier to fetch options data:
- Implied Volatility (IV) and IV percentile
- Put/Call ratios
- Options volume and open interest
- Options flow and unusual activity

Requires POLYGON_API_KEY and Starter tier ($29/month).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import requests
import warnings

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
POLYGON_BASE_URL = 'https://api.polygon.io'


def _polygon_request(endpoint: str, params: dict = None) -> dict:
    """Make request to Polygon API with error handling."""
    if not POLYGON_API_KEY:
        return {'error': 'POLYGON_API_KEY not set'}
    
    if params is None:
        params = {}
    
    params['apiKey'] = POLYGON_API_KEY
    url = f"{POLYGON_BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'ERROR':
            return {'error': data.get('error', 'Unknown API error')}
        
        return data
        
    except requests.exceptions.RequestException as e:
        return {'error': f"Request failed: {str(e)}"}


def get_options_chain(symbol: str, expiration_date: Optional[str] = None) -> Dict:
    """
    Get options chain using snapshot endpoint for actual market data.
    
    Args:
        symbol: Stock ticker
        expiration_date: Optional (not used, kept for compatibility)
    
    Returns:
        dict with options market data (volume, IV, greeks)
    """
    # Use snapshot endpoint which has actual market data
    endpoint = f"/v3/snapshot/options/{symbol}"
    
    params = {
        'limit': 250  # Get top 250 contracts by volume
    }
    
    result = _polygon_request(endpoint, params)
    
    if result.get('error'):
        return result
    
    # Transform to consistent format
    if 'results' in result and result['results']:
        return result
    
    return {'error': 'No options data available'}


def calculate_put_call_ratio(options_data: List[Dict]) -> float:
    """Calculate put/call ratio from options snapshot data."""
    if not options_data:
        return np.nan
    
    # Snapshot data has 'details' with contract info and 'day' with market data
    puts_volume = 0
    calls_volume = 0
    
    for opt in options_data:
        details = opt.get('details', {})
        day_data = opt.get('day', {})
        
        contract_type = details.get('contract_type')
        volume = day_data.get('volume', 0)
        
        if contract_type == 'put':
            puts_volume += volume
        elif contract_type == 'call':
            calls_volume += volume
    
    if calls_volume == 0:
        return np.nan
    
    return puts_volume / calls_volume


def calculate_iv_percentile(current_iv: float, historical_iv: List[float], window: int = 252) -> float:
    """Calculate IV percentile (where current IV ranks in historical range)."""
    if not historical_iv or len(historical_iv) < 2:
        return 50.0  # Default to median
    
    recent_iv = historical_iv[-window:] if len(historical_iv) > window else historical_iv
    
    percentile = (sum(1 for iv in recent_iv if iv < current_iv) / len(recent_iv)) * 100
    
    return percentile


def get_options_snapshot(symbol: str, date: Optional[str] = None) -> Dict:
    """
    Get options market snapshot for a symbol using Polygon snapshot endpoint.
    
    Includes:
    - Implied volatility metrics
    - Put/call ratios
    - Options volume
    - Open interest
    """
    # Get options snapshot (actual market data)
    options_data = get_options_chain(symbol, expiration_date=date)
    
    if options_data.get('error'):
        return options_data
    
    contracts = options_data.get('results', [])
    
    if not contracts:
        return {'error': 'No options contracts found'}
    
    # Calculate metrics from snapshot data
    put_call_ratio = calculate_put_call_ratio(contracts)
    
    # Extract metrics from snapshot structure
    ivs = []
    total_volume = 0
    total_oi = 0
    call_volume = 0
    put_volume = 0
    call_oi = 0
    put_oi = 0
    
    for opt in contracts:
        details = opt.get('details', {})
        day_data = opt.get('day', {})
        greeks = opt.get('greeks', {})
        
        # Implied volatility from greeks
        iv = greeks.get('implied_volatility')
        if iv and iv > 0:
            ivs.append(iv)
        
        # Volume and OI from day data
        volume = day_data.get('volume', 0)
        oi = day_data.get('open_interest', 0)
        
        total_volume += volume
        total_oi += oi
        
        # Split by type
        contract_type = details.get('contract_type')
        if contract_type == 'call':
            call_volume += volume
            call_oi += oi
        elif contract_type == 'put':
            put_volume += volume
            put_oi += oi
    
    avg_iv = np.mean(ivs) if ivs else np.nan
    
    return {
        'symbol': symbol,
        'date': date,
        'put_call_ratio': put_call_ratio,
        'put_call_ratio_oi': put_oi / call_oi if call_oi > 0 else np.nan,
        'avg_iv': avg_iv,
        'total_volume': total_volume,
        'total_open_interest': total_oi,
        'call_volume': call_volume,
        'put_volume': put_volume,
        'call_oi': call_oi,
        'put_oi': put_oi,
        'num_contracts': len(contracts)
    }


def add_options_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add options-based features to dataframe.
    
    Since Polygon Starter tier doesn't include options snapshot API,
    we'll use historical volatility as a proxy for IV features.
    
    Features added (calculated from price data):
    - realized_vol_30d: 30-day annualized volatility
    - vol_percentile: Volatility rank (0-100)
    - vol_regime: Binary (1 if vol > 75th percentile)
    - vol_expansion: Binary (1 if vol increasing)
    - vol_contraction: Binary (1 if vol decreasing)
    - vol_spike: Binary (1 if vol > 2x median)
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        symbol: Stock ticker
    
    Returns:
        DataFrame with options-proxy features added
    """
    print(f"  Adding volatility-based options proxy features for {symbol}...")
    
    # Note: Real options data requires higher Polygon tier
    # We'll use historical volatility which is highly correlated with IV
    
    # Calculate realized volatility (rolling windows)
    returns = df['close'].pct_change()
    
    # 30-day realized vol (annualized)
    df['realized_vol_30d'] = returns.rolling(30).std() * np.sqrt(252)
    
    # 10-day short-term vol
    df['realized_vol_10d'] = returns.rolling(10).std() * np.sqrt(252)
    
    # 60-day longer-term vol
    df['realized_vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
    
    # Volatility percentile (where is current vol in historical range)
    vol_rolling = df['realized_vol_30d'].rolling(window=252, min_periods=60)
    df['vol_percentile'] = vol_rolling.apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 1 else 50.0,
        raw=False
    )
    
    # Volatility regimes
    df['vol_regime_high'] = (df['vol_percentile'] > 75).astype(int)  # Expensive options
    df['vol_regime_low'] = (df['vol_percentile'] < 25).astype(int)  # Cheap options
    
    # Volatility trends
    df['vol_expansion'] = (df['realized_vol_10d'] > df['realized_vol_30d']).astype(int)
    df['vol_contraction'] = (df['realized_vol_10d'] < df['realized_vol_30d']).astype(int)
    
    # Volatility term structure (short vs long term)
    df['vol_term_structure'] = df['realized_vol_10d'] / (df['realized_vol_60d'] + 1e-6)
    
    # Volatility spike detection
    vol_median = df['realized_vol_30d'].rolling(window=252, min_periods=60).median()
    df['vol_spike'] = (df['realized_vol_30d'] > 2 * vol_median).astype(int)
    
    # Vol-adjusted returns (Sharpe-like measure)
    df['vol_adjusted_return'] = returns / (df['realized_vol_30d'] / np.sqrt(252) + 1e-6)
    
    # Interaction features
    # High vol + negative returns = fear (proxy for high put demand)
    df['vol_fear'] = ((df['vol_regime_high'] == 1) & (returns < 0)).astype(int)
    
    # Low vol + positive returns = complacency (potential reversal)
    df['vol_complacency'] = ((df['vol_regime_low'] == 1) & (returns > 0)).astype(int)
    
    # Fill missing values (forward fill first 30 days of rolling calcs)
    vol_cols = [col for col in df.columns if 'vol' in col.lower() and col not in ['volume', 'vix_volume']]
    
    for col in vol_cols:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].ffill().bfill().fillna(0)
        elif df[col].dtype in [np.int64, np.int32]:
            df[col] = df[col].fillna(0)
    
    print(f"    Added {len(vol_cols)} volatility features (options proxy)")
    
    return df


if __name__ == "__main__":
    # Test options features
    import pandas as pd
    from datetime import datetime
    
    print("Testing options features...")
    
    # Create test dataframe
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    
    # Simulate price movements
    np.random.seed(42)
    returns = np.random.randn(60) * 0.02
    prices = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 60)
    }, index=dates)
    
    # Add options features
    df = add_options_features(df, 'AAPL')
    
    print("\nSample data with options features:")
    options_cols = [col for col in df.columns if any(
        term in col.lower() for term in ['option', 'put_call', 'iv_']
    )]
    print(df[options_cols].head())
    
    print("\nOptions feature summary:")
    print(df[options_cols].describe())
