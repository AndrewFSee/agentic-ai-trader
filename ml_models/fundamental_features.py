"""
Fundamental Features for ML Trading Models

Uses yfinance to fetch fundamental ratios and financial metrics.
Free, unofficial API with good coverage of US stocks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("Warning: yfinance not installed. Run: pip install yfinance")
    YFINANCE_AVAILABLE = False


def get_fundamentals(symbol: str) -> Dict:
    """
    Fetch fundamental data for a symbol using yfinance.
    
    Returns dict with:
    - Valuation ratios (P/E, P/B, P/S)
    - Profitability metrics (margins, ROE)
    - Growth metrics (revenue growth, earnings growth)
    - Financial health (debt/equity, current ratio)
    """
    if not YFINANCE_AVAILABLE:
        return {'error': 'yfinance not available'}
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            # Valuation ratios
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            
            # Profitability
            'profit_margin': info.get('profitMargins', np.nan),
            'operating_margin': info.get('operatingMargins', np.nan),
            'gross_margin': info.get('grossMargins', np.nan),
            'roe': info.get('returnOnEquity', np.nan),
            'roa': info.get('returnOnAssets', np.nan),
            
            # Growth
            'revenue_growth': info.get('revenueGrowth', np.nan),
            'earnings_growth': info.get('earningsGrowth', np.nan),
            'revenue_per_share': info.get('revenuePerShare', np.nan),
            'earnings_per_share': info.get('trailingEps', np.nan),
            
            # Financial health
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),
            'total_cash': info.get('totalCash', np.nan),
            'total_debt': info.get('totalDebt', np.nan),
            
            # Market metrics
            'market_cap': info.get('marketCap', np.nan),
            'enterprise_value': info.get('enterpriseValue', np.nan),
            'shares_outstanding': info.get('sharesOutstanding', np.nan),
            'float_shares': info.get('floatShares', np.nan),
            
            # Dividend info
            'dividend_yield': info.get('dividendYield', np.nan),
            'payout_ratio': info.get('payoutRatio', np.nan),
            
            # Analyst metrics
            'target_price': info.get('targetMeanPrice', np.nan),
            'recommendation': info.get('recommendationKey', 'none'),
            'num_analyst_opinions': info.get('numberOfAnalystOpinions', 0),
            
            # Sector/industry for relative valuation
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
        }
        
        return fundamentals
        
    except Exception as e:
        print(f"    Warning: Could not fetch fundamentals for {symbol}: {e}")
        return {'error': str(e)}


def add_fundamental_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add fundamental features to dataframe.
    
    Note: Fundamentals update quarterly, so we forward-fill values.
    
    Features added:
    - Valuation ratios (P/E, P/B, P/S, PEG)
    - Profitability (margins, ROE, ROA)
    - Growth (revenue/earnings growth)
    - Financial health (debt ratios, liquidity)
    - Relative valuation (vs market median)
    - Dividend metrics
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex
        symbol: Stock ticker
    
    Returns:
        DataFrame with fundamental features added
    """
    print(f"  Adding fundamental features for {symbol}...")
    
    if not YFINANCE_AVAILABLE:
        print("    Warning: yfinance not available, skipping fundamental features")
        return df
    
    # Fetch fundamentals (these are current values)
    fundamentals = get_fundamentals(symbol)
    
    if fundamentals.get('error'):
        print(f"    Warning: Could not fetch fundamentals: {fundamentals['error']}")
        # Add placeholder columns with NaN
        fundamental_cols = [
            'pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio',
            'profit_margin', 'operating_margin', 'gross_margin', 'roe', 'roa',
            'revenue_growth', 'earnings_growth', 'debt_to_equity',
            'current_ratio', 'dividend_yield', 'payout_ratio'
        ]
        for col in fundamental_cols:
            df[col] = np.nan
        return df
    
    # Add fundamental ratios (static values, forward-filled)
    df['pe_ratio'] = fundamentals.get('pe_ratio', np.nan)
    df['forward_pe'] = fundamentals.get('forward_pe', np.nan)
    df['pb_ratio'] = fundamentals.get('pb_ratio', np.nan)
    df['ps_ratio'] = fundamentals.get('ps_ratio', np.nan)
    df['peg_ratio'] = fundamentals.get('peg_ratio', np.nan)
    
    # Profitability metrics
    df['profit_margin'] = fundamentals.get('profit_margin', np.nan)
    df['operating_margin'] = fundamentals.get('operating_margin', np.nan)
    df['gross_margin'] = fundamentals.get('gross_margin', np.nan)
    df['roe'] = fundamentals.get('roe', np.nan)
    df['roa'] = fundamentals.get('roa', np.nan)
    
    # Growth metrics
    df['revenue_growth'] = fundamentals.get('revenue_growth', np.nan)
    df['earnings_growth'] = fundamentals.get('earnings_growth', np.nan)
    
    # Financial health
    df['debt_to_equity'] = fundamentals.get('debt_to_equity', np.nan)
    df['current_ratio'] = fundamentals.get('current_ratio', np.nan)
    
    # Dividend metrics
    df['dividend_yield'] = fundamentals.get('dividend_yield', np.nan)
    df['payout_ratio'] = fundamentals.get('payout_ratio', np.nan)
    
    # Derived features (interactions with price action)
    
    # Price vs fair value (using P/E and growth)
    if not pd.isna(df['pe_ratio'].iloc[0]) and not pd.isna(df['earnings_growth'].iloc[0]):
        df['pe_to_growth'] = df['pe_ratio'] / (df['earnings_growth'] * 100 + 1e-6)
    else:
        df['pe_to_growth'] = np.nan
    
    # Quality score (profitability + growth - debt)
    df['quality_score'] = (
        df['roe'].fillna(0) * 10 +  # ROE weighted heavily
        df['profit_margin'].fillna(0) * 10 +  # Margins important
        df['revenue_growth'].fillna(0) * 5 -  # Growth is good
        df['debt_to_equity'].fillna(100) / 100  # Low debt is good
    )
    
    # Relative valuation (cheap = good)
    # Lower P/E and P/B = more attractive
    df['valuation_score'] = 0
    if not pd.isna(df['pe_ratio'].iloc[0]):
        df['valuation_score'] -= df['pe_ratio'] / 20  # Normalize PE around 20
    if not pd.isna(df['pb_ratio'].iloc[0]):
        df['valuation_score'] -= df['pb_ratio'] / 3  # Normalize PB around 3
    
    # Financial health score
    df['financial_health'] = (
        df['current_ratio'].fillna(1) * 0.5 +  # Liquidity
        (100 - df['debt_to_equity'].fillna(100)) / 100  # Low debt
    )
    
    # Dividend attractiveness
    df['dividend_attractive'] = (
        df['dividend_yield'].fillna(0) * 100 > 2  # Yield > 2%
    ).astype(int)
    
    # Growth stock indicator (high growth, high P/E)
    df['is_growth_stock'] = (
        (df['earnings_growth'].fillna(0) > 0.15) &  # >15% growth
        (df['pe_ratio'].fillna(0) > 20)  # High valuation
    ).astype(int)
    
    # Value stock indicator (low P/E, low P/B)
    df['is_value_stock'] = (
        (df['pe_ratio'].fillna(100) < 15) &  # Low P/E
        (df['pb_ratio'].fillna(100) < 2)  # Low P/B
    ).astype(int)
    
    # Handle missing values (fill with median or 0)
    fundamental_cols = [col for col in df.columns if any(
        term in col.lower() for term in ['ratio', 'margin', 'growth', 'roe', 'roa', 'debt', 'quality', 'valuation', 'health']
    )]
    
    for col in fundamental_cols:
        if df[col].isna().all():
            df[col] = 0  # If all NaN, fill with 0
        else:
            df[col] = df[col].fillna(df[col].median())  # Fill with median
    
    print(f"    Added {len(fundamental_cols)} fundamental features")
    
    return df


if __name__ == "__main__":
    # Test fundamental features
    import pandas as pd
    from datetime import datetime
    
    print("Testing fundamental features...")
    
    # Create test dataframe
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    df = pd.DataFrame({
        'close': np.random.randn(30).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 30)
    }, index=dates)
    
    # Add fundamental features
    df = add_fundamental_features(df, 'AAPL')
    
    print("\nSample data with fundamental features:")
    fundamental_cols = [col for col in df.columns if any(
        term in col.lower() for term in ['ratio', 'margin', 'growth', 'quality']
    )]
    print(df[fundamental_cols].head())
    
    print("\nFundamental feature summary:")
    print(df[fundamental_cols].describe())
