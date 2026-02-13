"""
Live Testing Configuration - VIX ROC & Reflexion Agent

Updated January 2026 to focus on validated strategies:
- VIX ROC three-tier risk overlay (15/15 wins in walk-forward testing)
- Reflexion agent with self-critique pattern
- Buy-and-hold benchmark

Deprecated (not generating alpha):
- Wasserstein regime detection
- HMM regime detection
- ML prediction models
"""

from datetime import datetime, timedelta

# ============================================================================
# TESTING PARAMETERS
# ============================================================================

# Date range - RESET for new testing period
# Start fresh from today
START_DATE = datetime.now().strftime("%Y-%m-%d")
# End date will be rolling (always yesterday for Polygon delay)
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Universe of stocks to test (same as before for comparison)
TEST_SYMBOLS = [
    "AAPL",  # Tech (large cap) - Tier 2 Growth
    "MSFT",  # Tech (large cap) - Tier 3 Mega-Cap
    "JPM",   # Financial - Tier 1 Value
    "JNJ",   # Healthcare (defensive) - Tier 1 Value
    "XOM",   # Energy (cyclical) - Tier 1 Value
]

# Extended universe for comprehensive testing
EXTENDED_SYMBOLS = [
    # Tier 1 (Value/Cyclical)
    "SPY", "DIA", "IWM", "XLF", "XLE", "JPM", "JNJ", "XOM",
    # Tier 2 (Growth)
    "QQQ", "AAPL", "AMZN", "GOOGL",
    # Tier 3 (Mega-Cap Tech)
    "NVDA", "MSFT", "META",
]

# For compatibility
STOCK_UNIVERSE = TEST_SYMBOLS
POLYGON_API_KEY = None  # Will be loaded from .env
USE_POLYGON_IO = True
DATA_LOOKBACK_YEARS = 2
START_DATE_STR = START_DATE
END_DATE_STR = END_DATE
DATA_DIR = "data"

# Initial capital per strategy
INITIAL_CAPITAL = 100000.0

# Rebalancing frequency
REBALANCE_FREQUENCY = "daily"

# Transaction costs
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_BPS = 5  # 0.05%

# Position sizing
MAX_POSITION_SIZE = 0.20  # Max 20% per position
MIN_POSITION_SIZE = 0.05  # Min 5% per position

# VIX ROC parameters
VIX_LOOKBACK_DAYS = 60  # Days of VIX history for ROC calculation

# Reflexion agent parameters
AGENT_USE_RAG = True
AGENT_TIMEOUT = 120  # Allow more time for reflexion steps

# ============================================================================
# STRATEGIES TO TEST (UPDATED)
# ============================================================================

STRATEGIES = {
    # =========================================================================
    # BENCHMARKS
    # =========================================================================
    "buy_and_hold": {
        "type": "baseline",
        "long_only": True,
        "description": "Simple buy and hold - THE BENCHMARK"
    },
    
    # =========================================================================
    # VIX ROC STRATEGIES (Primary - Walk-forward validated 15/15 wins)
    # =========================================================================
    "vix_roc_auto": {
        "type": "vix_roc",
        "tier": "auto",  # Auto-classify based on asset
        "long_only": True,
        "description": "VIX ROC with automatic tier classification"
    },
    
    "vix_roc_tier1": {
        "type": "vix_roc",
        "tier": "tier1",  # Force Tier 1 (conservative)
        "long_only": True,
        "description": "VIX ROC Tier 1 (Value/Cyclical params)"
    },
    
    "vix_roc_tier2": {
        "type": "vix_roc",
        "tier": "tier2",  # Force Tier 2 (aggressive)
        "long_only": True,
        "description": "VIX ROC Tier 2 (Growth/Tech params)"
    },
    
    "vix_roc_tier3": {
        "type": "vix_roc",
        "tier": "tier3",  # Force Tier 3 (mega-cap)
        "long_only": True,
        "description": "VIX ROC Tier 3 (Mega-Cap Tech params)"
    },
    
    # =========================================================================
    # REFLEXION AGENT (Self-critiquing agent)
    # =========================================================================
    "reflexion_agent": {
        "type": "reflexion_agent",
        "long_only": True,
        "use_rag": AGENT_USE_RAG,
        "description": "Reflexion agent (generate→critique→reflect→refine)"
    },
    
    # =========================================================================
    # VOL PREDICTION OVERLAY (Position sizing)
    # =========================================================================
    "vix_roc_vol_adjusted": {
        "type": "vix_roc_vol",
        "tier": "auto",
        "long_only": True,
        "use_vol_prediction": True,
        "description": "VIX ROC with vol prediction position sizing"
    },
}

# ============================================================================
# DEPRECATED STRATEGIES (kept for reference but not active)
# ============================================================================

DEPRECATED_STRATEGIES = {
    # These did not generate alpha in testing
    "wasserstein_long": {"deprecated": True, "reason": "No alpha generation"},
    "wasserstein_long_short": {"deprecated": True, "reason": "No alpha generation"},
    "hmm_long": {"deprecated": True, "reason": "No alpha generation"},
    "hmm_long_short": {"deprecated": True, "reason": "No alpha generation"},
    "ml_consensus_long": {"deprecated": True, "reason": "No alpha generation"},
    "ml_consensus_long_short": {"deprecated": True, "reason": "No alpha generation"},
    "rf_long": {"deprecated": True, "reason": "No alpha generation"},
    "xgb_long": {"deprecated": True, "reason": "No alpha generation"},
    "agent_long": {"deprecated": True, "reason": "Replaced by reflexion_agent"},
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
    "excess_return_vs_buyhold",  # NEW: Key metric
]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

RESULTS_DIR = "simulation_results"
LOGS_DIR = "trading_logs"

SAVE_TRADE_LOG = True
SAVE_EQUITY_CURVE = True
SAVE_SUMMARY_CSV = True
SAVE_COMPARISON_PLOT = True

VERBOSE = True
DEBUG = False
