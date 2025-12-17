"""Test NaN handling for failed stocks"""

from data_collection import fetch_daily_bars
from feature_engineering import engineer_features
from train_models import split_data

failed_stocks = ['GME', 'AMC', 'COIN', 'MSTR', 'PLTR']

for symbol in failed_stocks:
    print(f"\n{'='*60}")
    print(f"Testing {symbol}")
    print('='*60)
    
    try:
        # Fetch data
        df = fetch_daily_bars(symbol, '2020-12-01', '2025-12-15')
        print(f"[OK] Fetched {len(df)} bars")
        
        # Engineer features
        df = engineer_features(symbol, df)
        print(f"[OK] Engineered features: {df.shape}")
        
        # Check for NaN
        nan_count = df.isna().sum().sum()
        print(f"  NaN values before prep: {nan_count}")
        
        if nan_count > 0:
            nan_features = df.isna().sum()[df.isna().sum() > 0]
            print(f"  Features with NaN: {len(nan_features)}")
            if len(nan_features) > 0:
                print(f"  Top 5: {list(nan_features.head().index)}")
        
        # Prepare data - use proper feature selection
        from feature_engineering import get_feature_columns
        feature_cols = get_feature_columns(df)
        splits = split_data(df, feature_cols, 'target_class_3d')
        
        # Check NaN in splits (they're numpy arrays)
        import numpy as np
        nan_train = np.isnan(splits['X_train']).sum()
        nan_val = np.isnan(splits['X_val']).sum()
        nan_test = np.isnan(splits['X_test']).sum()
        
        print(f"[OK] Data preparation SUCCESS!")
        print(f"  Train: {splits['X_train'].shape} (NaN: {nan_train})")
        print(f"  Val:   {splits['X_val'].shape} (NaN: {nan_val})")
        print(f"  Test:  {splits['X_test'].shape} (NaN: {nan_test})")
        
        if nan_train + nan_val + nan_test == 0:
            print(f"[PASS] {symbol} FIXED - No NaN in training data!")
        else:
            print(f"[WARN] {symbol} still has {nan_train + nan_val + nan_test} NaN values")
            
    except Exception as e:
        print(f"[FAIL] {symbol} FAILED: {e}")

print(f"\n{'='*60}")
print("Test complete!")
print('='*60)
