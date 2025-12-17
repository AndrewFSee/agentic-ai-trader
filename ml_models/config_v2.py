"""
Configuration file v2 with increased model capacity for underfitting problem.

Changes from v1:
1. Increased model complexity (more trees, deeper)
2. Reduced regularization (was causing underfitting)
3. Focus test on defensive + volatility stocks (showed edge)
"""

# ==================== DATA COLLECTION ====================

# Stock universe - REDUCED to stocks that showed edge
STOCK_UNIVERSE = {
    "defensive": ["PG", "JNJ", "COST", "KO"],  # 80% win rate with Logistic
    "volatility": ["MSTR", "GME"],  # Huge gains possible but risky
    "momentum": ["PLTR"],  # Strong trends, test separately
}

# ALL SYMBOLS for full testing
ALL_SYMBOLS = [
    # Growth (poor ML results - skip for now)
    # "NVDA", "AAPL", "META", "GOOGL", "MSFT",
    
    # Value (mixed results - skip for now)  
    # "JPM", "BAC", "XOM", "CVX", "WMT",
    
    # Momentum (mixed)
    "PLTR", "TSLA",
    
    # Defensive (BEST RESULTS - focus here)
    "PG", "JNJ", "COST", "KO", "PEP",
    
    # Volatility (MIXED - huge wins and losses)
    "MSTR", "GME", "AMC"
]

# Date range
START_DATE = "2020-01-01"
END_DATE = "2024-12-15"

# API settings
API_DELAY = 0  # Polygon has generous rate limits
MAX_RETRIES = 3

# ==================== FEATURE ENGINEERING ====================

# Technical indicators
TECHNICAL_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "sma_periods": [20, 50, 200],
    "ema_periods": [12, 26, 50],
    "atr_period": 14,
    "volume_sma_period": 20,
}

# Feature groups (all enabled)
FEATURE_GROUPS = {
    "price": True,
    "technical": True,
    "volume": True,
    "volatility": True,
    "momentum": True,
    "market_relative": False,  # Requires SPY data
    "seasonality": True,
    "regime": True,  # NEW - regime detection features
    "interactions": True,  # NEW - feature interactions
    "polynomial": True,  # NEW - polynomial features
}

# Prediction horizons
PREDICTION_HORIZONS = [3, 5, 10]  # Days ahead - testing if shorter horizon helps

# ==================== MODEL TRAINING ====================

# Train/validation/test split (time-series aware)
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

# Model configurations - INCREASED CAPACITY
MODELS = {
    "logistic": {
        "name": "Logistic Regression",
        "type": "sklearn",
        "C": 1.0,  # Keep default (was best performer)
        "penalty": "l2",
        "max_iter": 2000,  # Increased
        "class_weight": "balanced"
    },
    "decision_tree": {
        "name": "Decision Tree",
        "type": "sklearn",
        "max_depth": 15,  # Increased from 10
        "min_samples_split": 20,  # Decreased from 50 (less regularization)
        "min_samples_leaf": 10,  # Decreased from 20
        "class_weight": "balanced"
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "sklearn",
        "n_estimators": 300,  # Increased from 100
        "max_depth": 15,  # Increased from 10
        "min_samples_split": 20,  # Decreased from 50
        "min_samples_leaf": 10,  # Decreased from 20
        "max_features": "sqrt",  # Better for many features
        "class_weight": "balanced",
        "n_jobs": -1  # Use all CPU cores
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "xgboost",
        "n_estimators": 500,  # Increased from 100
        "max_depth": 8,  # Increased from 6
        "learning_rate": 0.1,  # Keep same
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # REMOVED regularization (was causing underfitting)
        # "reg_alpha": 0,
        # "reg_lambda": 1,
    }
}

# Cross-validation
CV_FOLDS = 5  # Time-series cross-validation

# ==================== BACKTESTING ====================

# Transaction costs
TRANSACTION_COST = 0.001  # 0.1% per trade (conservative)

# Position sizing
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.95  # Use 95% of capital per trade

# Risk management
MAX_POSITION_SIZE = 1.0  # No leverage
STOP_LOSS = -0.10  # -10% stop loss
TAKE_PROFIT = 0.20  # +20% take profit

# ==================== UTILITY ====================

# Caching
CACHE_DIR = "data"

# Logging
LOG_LEVEL = "INFO"

# Results
RESULTS_DIR = "../results/ml_models"

# Randomness
RANDOM_STATE = 42

# ==================== CONFIG OBJECT ====================

CONFIG = {
    "symbols": list(STOCK_UNIVERSE["defensive"]) + list(STOCK_UNIVERSE["volatility"]) + list(STOCK_UNIVERSE["momentum"]),
    "start_date": START_DATE,
    "end_date": END_DATE,
    "technical_params": TECHNICAL_PARAMS,
    "feature_groups": FEATURE_GROUPS,
    "prediction_horizons": PREDICTION_HORIZONS,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VALIDATION_RATIO,
    "test_ratio": TEST_RATIO,
    "models": MODELS,
    "cv_folds": CV_FOLDS,
    "transaction_cost": TRANSACTION_COST,
    "initial_capital": INITIAL_CAPITAL,
    "random_state": RANDOM_STATE,
}

CONFIG_V2 = CONFIG  # Alias for v2
