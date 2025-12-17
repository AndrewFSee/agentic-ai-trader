#!/usr/bin/env python3
"""Quick debug script to find the dtype error"""

import traceback
from models.paper_wasserstein_regime_detection import (
    RollingPaperWassersteinDetector,
    fetch_polygon_bars,
    calculate_features
)

try:
    print("Fetching data...")
    df = fetch_polygon_bars("AAPL", "2020-01-01", "2023-12-31")
    print(f"Fetched {len(df)} days")
    
    print("\nCalculating features...")
    df = calculate_features(df, window=20)
    print(f"Features calculated, {len(df)} days")
    
    print("\nSplitting data...")
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("\nInitializing detector...")
    detector = RollingPaperWassersteinDetector(
        n_regimes=3,
        window_size=20,
        training_window_days=500,
        retrain_frequency_days=126,
        feature_cols=['realized_vol', 'trend_strength', 'volume_momentum']
    )
    
    print("\nTraining...")
    detector.train_on_window(train_df, train_df.index[-1], verbose=True)
    
    print("\nPredicting...")
    predictions = detector.predict_forward_rolling(
        df,
        test_df.index[20],
        test_df.index[-1],
        verbose=True
    )
    
    print(f"\nSuccess! Generated {len(predictions)} predictions")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"Predictions index dtype: {predictions.index.dtype}")
    print(f"First 5 predictions:")
    print(predictions.head())
    
except Exception as e:
    print(f"\nError occurred:")
    traceback.print_exc()
