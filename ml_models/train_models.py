"""
Model training pipeline for supervised learning.
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    MODELS, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO,
    PREDICTION_HORIZONS, MODELS_DIR, RESULTS_DIR, VERBOSE
)

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def split_data(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
    """
    Split data into train/validation/test sets (time-series aware).
    
    Args:
        df: DataFrame with features and targets
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        Dictionary with train/val/test splits
    """
    # Remove rows with missing targets
    df = df.dropna(subset=[target_col])
    
    # Remove rows with too many missing features
    df = df.dropna(subset=feature_cols, thresh=len(feature_cols) * 0.8)
    
    # Ensure all feature columns are numeric (convert any object dtypes)
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Forward fill remaining missing values using updated pandas syntax
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # Final check: replace any remaining NaN with median (robust fallback)
    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
    
    # Time-series split (no shuffling)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))
    
    # Split
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    if VERBOSE:
        print(f"  Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
        print(f"  Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Target distribution (train): {np.bincount(y_train.astype(int)) / len(y_train)}")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'train_df': train_df,
        'X_val': X_val, 'y_val': y_val, 'val_df': val_df,
        'X_test': X_test, 'y_test': y_test, 'test_df': test_df,
        'feature_cols': feature_cols
    }


def train_logistic_regression(X_train, y_train, X_val, y_val) -> Tuple:
    """Train Logistic Regression baseline."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    return model, scaler, y_pred, y_proba


def train_decision_tree(X_train, y_train, X_val, y_val) -> Tuple:
    """Train Decision Tree baseline."""
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    return model, None, y_pred, y_proba


def train_random_forest(X_train, y_train, X_val, y_val) -> Tuple:
    """Train Random Forest."""
    params = MODELS['random_forest']
    
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        min_samples_split=params.get('min_samples_split', 50),
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    return model, None, y_pred, y_proba


def train_xgboost(X_train, y_train, X_val, y_val) -> Tuple:
    """Train XGBoost."""
    params = MODELS['xgboost']
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    return model, None, y_pred, y_proba


def evaluate_model(y_true, y_pred, y_proba, model_name: str) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5
    }
    
    if VERBOSE:
        print(f"\n  {model_name} Metrics:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def train_all_models(symbol: str, df: pd.DataFrame, feature_cols: List[str], 
                     horizon: int) -> Dict:
    """
    Train all models for a given symbol and prediction horizon.
    
    Args:
        symbol: Stock ticker
        df: DataFrame with features and targets
        feature_cols: List of feature column names
        horizon: Prediction horizon (5 or 10 days)
    
    Returns:
        Dictionary with trained models and results
    """
    target_col = f'target_direction_{horizon}d'
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODELS: {symbol} - {horizon}-day prediction")
    print(f"{'='*80}")
    
    # Split data
    data_splits = split_data(df, feature_cols, target_col)
    
    results = {
        'symbol': symbol,
        'horizon': horizon,
        'models': {},
        'feature_importance': {}
    }
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression (baseline)...")
    lr_model, lr_scaler, lr_pred, lr_proba = train_logistic_regression(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    results['models']['logistic'] = {
        'model': lr_model,
        'scaler': lr_scaler,
        'metrics': evaluate_model(data_splits['y_val'], lr_pred, lr_proba, 'Logistic Regression')
    }
    
    # Train Decision Tree
    print("\nTraining Decision Tree (baseline)...")
    dt_model, _, dt_pred, dt_proba = train_decision_tree(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    results['models']['decision_tree'] = {
        'model': dt_model,
        'scaler': None,
        'metrics': evaluate_model(data_splits['y_val'], dt_pred, dt_proba, 'Decision Tree')
    }
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model, _, rf_pred, rf_proba = train_random_forest(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    results['models']['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'metrics': evaluate_model(data_splits['y_val'], rf_pred, rf_proba, 'Random Forest')
    }
    results['feature_importance']['random_forest'] = dict(zip(
        feature_cols, rf_model.feature_importances_
    ))
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model, _, xgb_pred, xgb_proba = train_xgboost(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    results['models']['xgboost'] = {
        'model': xgb_model,
        'scaler': None,
        'metrics': evaluate_model(data_splits['y_val'], xgb_pred, xgb_proba, 'XGBoost')
    }
    results['feature_importance']['xgboost'] = dict(zip(
        feature_cols, xgb_model.feature_importances_
    ))
    
    # Store data splits for backtesting
    results['data_splits'] = data_splits
    
    # Save models
    model_dir = os.path.join(MODELS_DIR, f"{symbol}_{horizon}d")
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, model_data in results['models'].items():
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump({
            'model': model_data['model'],
            'scaler': model_data['scaler'],
            'feature_cols': feature_cols,
            'metrics': model_data['metrics']
        }, model_path)
    
    print(f"\nOK: Models saved to {model_dir}")
    
    return results


if __name__ == "__main__":
    # Test training pipeline
    from data_collection import fetch_daily_bars, fetch_spy_data
    from feature_engineering import engineer_features, get_feature_columns
    from config import START_DATE_STR, END_DATE_STR
    
    print("Testing training pipeline...")
    
    # Fetch and engineer features
    symbol = "AAPL"
    df = fetch_daily_bars(symbol, START_DATE_STR, END_DATE_STR)
    spy_df = fetch_spy_data(START_DATE_STR, END_DATE_STR)
    df_features = engineer_features(symbol, df, spy_df)
    feature_cols = get_feature_columns(df_features)
    
    # Train models for 5-day prediction
    results = train_all_models(symbol, df_features, feature_cols, 5)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
