"""
Test Enhanced Features on Single Stock

Quick test to verify all new feature modules work correctly.
"""

import sys
import os

# Make sure we're in the ml_models directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_collection import fetch_daily_bars, fetch_spy_data
from feature_engineering import engineer_features, get_feature_columns
from config import START_DATE_STR, END_DATE_STR

def test_enhanced_features(symbol='AAPL'):
    """Test enhanced feature engineering on a single stock."""
    
    print("="*80)
    print(f"TESTING ENHANCED FEATURES ON {symbol}")
    print("="*80)
    
    # 1. Fetch data
    print("\n[1] Fetching price data...")
    df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
    spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
    
    print(f"    Fetched {len(df)} bars for {symbol}")
    print(f"    Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Engineer features (this will call all new feature modules)
    print("\n[2] Engineering features (including new modules)...")
    df_features = engineer_features(symbol, df, spy_df)
    
    # 3. Analyze results
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Engineered columns: {len(df_features.columns)}")
    print(f"Final shape: {df_features.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    print(f"\nTotal features: {len(feature_cols)}")
    
    # Categorize features by type
    sentiment_features = [col for col in feature_cols if 'sentiment' in col.lower()]
    regime_features = [col for col in feature_cols if 'regime' in col.lower() or 'hmm' in col.lower() or 'wass' in col.lower()]
    fundamental_features = [col for col in feature_cols if any(term in col.lower() for term in ['ratio', 'margin', 'growth', 'roe', 'quality', 'debt'])]
    options_features = [col for col in feature_cols if 'option' in col.lower() or 'put_call' in col.lower() or 'iv_' in col.lower()]
    technical_features = [col for col in feature_cols if any(term in col.lower() for term in ['rsi', 'macd', 'sma', 'ema', 'bb_', 'atr'])]
    
    print("\nFeature Breakdown:")
    print(f"  Technical indicators: {len(technical_features)}")
    print(f"  Sentiment features: {len(sentiment_features)}")
    print(f"  Regime features: {len(regime_features)}")
    print(f"  Fundamental features: {len(fundamental_features)}")
    print(f"  Options features: {len(options_features)}")
    print(f"  Other features: {len(feature_cols) - len(sentiment_features) - len(regime_features) - len(fundamental_features) - len(options_features) - len(technical_features)}")
    
    # Show sample of each feature category
    print("\n" + "-"*80)
    print("SAMPLE FEATURES BY CATEGORY")
    print("-"*80)
    
    if sentiment_features:
        print("\nSentiment Features:")
        for feat in sentiment_features[:5]:
            print(f"  - {feat}")
    
    if regime_features:
        print("\nRegime Features:")
        for feat in regime_features[:5]:
            print(f"  - {feat}")
    
    if fundamental_features:
        print("\nFundamental Features:")
        for feat in fundamental_features[:5]:
            print(f"  - {feat}")
    
    if options_features:
        print("\nOptions Features:")
        for feat in options_features[:5]:
            print(f"  - {feat}")
    
    # Check for missing values
    print("\n" + "-"*80)
    print("DATA QUALITY CHECK")
    print("-"*80)
    
    missing_pct = (df_features[feature_cols].isnull().sum() / len(df_features) * 100).sort_values(ascending=False)
    
    if missing_pct.max() > 0:
        print("\nFeatures with missing values:")
        for feat, pct in missing_pct.head(10).items():
            if pct > 0:
                print(f"  {feat}: {pct:.1f}% missing")
    else:
        print("\n✓ No missing values!")
    
    # Show sample data
    print("\n" + "-"*80)
    print("SAMPLE DATA (Last 5 rows)")
    print("-"*80)
    
    # Show a mix of features
    display_cols = ['close']
    if sentiment_features:
        display_cols.extend(sentiment_features[:2])
    if regime_features:
        display_cols.extend(regime_features[:2])
    if fundamental_features:
        display_cols.extend(fundamental_features[:2])
    if options_features:
        display_cols.extend(options_features[:2])
    
    print(df_features[display_cols].tail())
    
    # Show target distribution
    print("\n" + "-"*80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("-"*80)
    
    target_cols = [col for col in df_features.columns if 'target_' in col]
    for target in target_cols:
        target_dist = df_features[target].value_counts()
        print(f"\n{target}:")
        print(f"  Up (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_features)*100:.1f}%)")
        print(f"  Down (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_features)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    
    return df_features


if __name__ == "__main__":
    df = test_enhanced_features('AAPL')
