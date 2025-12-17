"""Debug object dtype issues in failed stocks"""

from data_collection import fetch_daily_bars
from feature_engineering import engineer_features
from train_models import split_data

# Test just one stock
symbol = 'GME'

print(f"Testing {symbol}")
df = fetch_daily_bars(symbol, '2020-12-01', '2025-12-15')
print(f"Fetched {len(df)} bars")

df = engineer_features(symbol, df)
print(f"Engineered features: {df.shape}")

# Check dtypes
feature_cols = [col for col in df.columns if 'target' not in col]
target_cols = [col for col in df.columns if 'target' in col]

print(f"\nFeature columns: {len(feature_cols)}")
print(f"Target columns: {len(target_cols)}")

# Find object dtypes
object_cols = df[feature_cols].select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"\n[WARN] Found {len(object_cols)} feature columns with object dtype:")
    for col in object_cols:
        print(f"  - {col}: {df[col].dtype}")
        print(f"    Sample values: {df[col].dropna().head(3).tolist()}")
else:
    print("\n[OK] No object dtypes in features")

# Check target columns
target_object_cols = df[target_cols].select_dtypes(include=['object']).columns
if len(target_object_cols) > 0:
    print(f"\n[WARN] Found {len(target_object_cols)} target columns with object dtype:")
    for col in target_object_cols:
        print(f"  - {col}: {df[col].dtype}")
        print(f"    Sample values: {df[col].dropna().head(3).tolist()}")
else:
    print("\n[OK] No object dtypes in targets")

print("\nNow trying split_data...")
try:
    splits = split_data(df, feature_cols, 'target_class_3d')
    print("[PASS] split_data succeeded!")
    print(f"  Train shape: {splits['X_train'].shape}")
    print(f"  Val shape: {splits['X_val'].shape}")
    print(f"  Test shape: {splits['X_test'].shape}")
except Exception as e:
    print(f"[FAIL] split_data failed: {e}")
    import traceback
    traceback.print_exc()
