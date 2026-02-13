"""
Live Testing Configuration

Defines strategies, symbols, and testing parameters for simulation backtests.
"""

from datetime import datetime, timedelta

# ============================================================================
# TESTING PARAMETERS
# ============================================================================

# Date range for backtesting
START_DATE = "2023-01-01"  # Adjust to your preference
# Polygon free tier has 15-min delayed data, use yesterday
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Universe of stocks to test
# Limited to 5 stocks to reduce LLM API calls
TEST_SYMBOLS = [
    "AAPL",  # Tech (large cap)
    "MSFT",  # Tech (large cap)
    "JPM",   # Financial
    "JNJ",   # Healthcare (defensive)
    "XOM",   # Energy (cyclical)
]

# Full universe (for future testing)
FULL_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", 
    "JNJ", "UNH", "XOM", "PG"
]

# For compatibility with ml_models imports
STOCK_UNIVERSE = TEST_SYMBOLS
POLYGON_API_KEY = None  # Will be loaded from .env
USE_POLYGON_IO = True
DATA_LOOKBACK_YEARS = 5
START_DATE_STR = START_DATE
END_DATE_STR = END_DATE
DATA_DIR = "data"
VERBOSE = False

# Initial capital per strategy
INITIAL_CAPITAL = 100000.0

# Rebalancing frequency
REBALANCE_FREQUENCY = "daily"  # "daily", "weekly", "monthly"

# Transaction costs
COMMISSION_PER_TRADE = 0.0  # Commission per trade
SLIPPAGE_BPS = 5  # Slippage in basis points (5 = 0.05%)

# Position sizing
MAX_POSITION_SIZE = 0.20  # Max 20% per position
MIN_POSITION_SIZE = 0.05  # Min 5% per position

# Regime detection parameters
REGIME_LOOKBACK_DAYS = 252  # 1 year
REGIME_UPDATE_FREQUENCY = 5  # Update regime every N days

# ML prediction parameters
ML_PREDICTION_HORIZON = 5  # 5-day predictions
ML_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to act

# Agent parameters
AGENT_USE_RAG = True  # Whether to use RAG search (slower but more informed)
AGENT_TIMEOUT = 30  # Max seconds per agent decision

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================

STRATEGIES = {
    # Baseline
    "buy_and_hold": {
        "type": "baseline",
        "long_only": True,
        "description": "Simple buy and hold"
    },
    
    # Wasserstein Regime
    "wasserstein_long": {
        "type": "regime",
        "model": "wasserstein",
        "long_only": True,
        "description": "Wasserstein regime detection (long-only)"
    },
    "wasserstein_long_short": {
        "type": "regime",
        "model": "wasserstein",
        "long_only": False,
        "description": "Wasserstein regime detection (long/short)"
    },
    
    # HMM Regime
    "hmm_long": {
        "type": "regime",
        "model": "hmm",
        "long_only": True,
        "description": "HMM regime detection (long-only)"
    },
    "hmm_long_short": {
        "type": "regime",
        "model": "hmm",
        "long_only": False,
        "description": "HMM regime detection (long/short)"
    },
    
    # ML Models - Consensus
    "ml_consensus_long": {
        "type": "ml",
        "long_only": True,
        "description": "ML consensus predictions (long-only)"
    },
    "ml_consensus_long_short": {
        "type": "ml",
        "long_only": False,
        "description": "ML consensus predictions (long/short)"
    },
    
    # ML Models - Individual Models
    "rf_long": {
        "type": "single_ml",
        "model": "Random Forest",
        "long_only": True,
        "description": "Random Forest predictions (long-only)"
    },
    "rf_long_short": {
        "type": "single_ml",
        "model": "Random Forest",
        "long_only": False,
        "description": "Random Forest predictions (long/short)"
    },
    "xgb_long": {
        "type": "single_ml",
        "model": "XGBoost",
        "long_only": True,
        "description": "XGBoost predictions (long-only)"
    },
    "xgb_long_short": {
        "type": "single_ml",
        "model": "XGBoost",
        "long_only": False,
        "description": "XGBoost predictions (long/short)"
    },
    "lr_long": {
        "type": "single_ml",
        "model": "Logistic Regression",
        "long_only": True,
        "description": "Logistic Regression predictions (long-only)"
    },
    "lr_long_short": {
        "type": "single_ml",
        "model": "Logistic Regression",
        "long_only": False,
        "description": "Logistic Regression predictions (long/short)"
    },
    "dt_long": {
        "type": "single_ml",
        "model": "Decision Tree",
        "long_only": True,
        "description": "Decision Tree predictions (long-only)"
    },
    "dt_long_short": {
        "type": "single_ml",
        "model": "Decision Tree",
        "long_only": False,
        "description": "Decision Tree predictions (long/short)"
    },
    
    # Agent Decisions
    "agent_long": {
        "type": "agent",
        "long_only": True,
        "use_rag": AGENT_USE_RAG,
        "description": "Full agent decisions (long-only)"
    },
    "agent_long_short": {
        "type": "agent",
        "long_only": False,
        "use_rag": AGENT_USE_RAG,
        "description": "Full agent decisions (long/short)"
    },
}

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

METRICS_TO_TRACK = [
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "num_trades",
    "avg_trade_duration",
    "volatility",
    "calmar_ratio",
]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Where to save results
RESULTS_DIR = "simulation_results"
LOGS_DIR = "trading_logs"

# Output formats
SAVE_TRADE_LOG = True  # Save detailed trade log
SAVE_EQUITY_CURVE = True  # Save daily equity values
SAVE_SUMMARY_CSV = True  # Save summary metrics
SAVE_COMPARISON_PLOT = True  # Generate comparison plots

# Verbosity
VERBOSE = True  # Print progress during simulation
DEBUG = False  # Print detailed debug info
