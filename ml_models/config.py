"""
Configuration for ML-based return prediction models.
"""
from datetime import datetime, timedelta

# ==================== DATA COLLECTION ====================

# Stock universe - diversified across categories
STOCK_UNIVERSE = {
    "growth": ["NVDA", "TSLA", "GOOGL", "META", "AMZN"],
    "value": ["JPM", "BAC", "WFC", "XOM", "CVX"],
    "momentum": ["MSFT", "AAPL", "V", "MA", "COST"],
    "defensive": ["JNJ", "PG", "KO", "PEP", "WMT"],
    "volatility": ["GME", "AMC", "COIN", "MSTR", "PLTR"]
}

# Date range for historical data
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365 * 5)  # 5 years of data
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")

# ==================== FEATURE ENGINEERING ====================

# Technical indicator parameters
TECHNICAL_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "ema_periods": [12, 26, 50],
    "atr_period": 14,
    "volume_sma": 20
}

# Feature groups to create
FEATURE_GROUPS = {
    "price_features": True,      # Returns, log returns, price changes
    "technical": True,            # RSI, MACD, Bollinger Bands, SMAs, EMAs
    "volume": True,               # Volume ratios, volume momentum
    "volatility": True,           # Realized vol, ATR, Bollinger bandwidth
    "regime": True,               # HMM and Wasserstein regime states
    "momentum": True,             # Multi-period momentum
    "market_relative": True,      # Performance vs SPY
    "seasonality": True,          # Day of week, month effects
}

# ==================== TARGET VARIABLES ====================

# Prediction horizons
PREDICTION_HORIZONS = [3, 5, 10]  # 3-day, 5-day and 10-day forward returns

# Classification threshold (for binary classification)
# Positive class = return > threshold
RETURN_THRESHOLD = 0.0  # 0% = predict up vs down

# ==================== MODEL TRAINING ====================

# Train/validation/test split (time-series aware)
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

# Models to train
MODELS = {
    "logistic": {
        "name": "Logistic Regression",
        "type": "sklearn",
        "baseline": True
    },
    "decision_tree": {
        "name": "Decision Tree",
        "type": "sklearn",
        "baseline": True
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "sklearn",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 50
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "xgboost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
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
STOP_LOSS = -0.05  # -5% stop loss
TAKE_PROFIT = 0.10  # 10% take profit

# ==================== OUTPUT ====================

# Results directory
RESULTS_DIR = "results/ml_models"
MODELS_DIR = "saved_models"  # Relative to ml_models directory
DATA_DIR = "data"  # Relative to ml_models directory

# Logging
VERBOSE = True
SAVE_PREDICTIONS = True
SAVE_FEATURE_IMPORTANCE = True
