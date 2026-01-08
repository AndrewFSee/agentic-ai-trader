"""
Pytest fixtures for Agentic AI Trader tests.

Provides shared test data, mocks, and configuration.
"""
import pytest
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def test_symbols():
    """Standard symbols for testing."""
    return ["AAPL", "MSFT", "JPM"]


@pytest.fixture(scope="session")
def test_date_range():
    """Standard date range for testing."""
    return {
        "start": "2023-01-01",
        "end": "2024-01-01"
    }


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    np.random.seed(42)
    
    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, len(dates))
    })
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def sample_features(sample_ohlcv_data):
    """Generate sample feature data for ML models."""
    df = sample_ohlcv_data.copy()
    
    # Add basic features
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['rsi_14'] = _calculate_rsi(df['close'], 14)
    
    # Add target
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    return df.dropna()


def _calculate_rsi(prices, period=14):
    """Helper to calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@pytest.fixture
def sample_tool_results():
    """Sample tool results for testing agent decisions."""
    return {
        "polygon_price_data": {
            "symbol": "AAPL",
            "num_bars": 100,
            "last_close": 175.50,
            "last_volume": 50000000,
            "avg_volume": 45000000,
            "volume_ratio": 1.11
        },
        "polygon_technical_rsi": {
            "symbol": "AAPL",
            "latest_rsi": 55.5,
            "rsi_signal": "neutral"
        },
        "polygon_technical_macd": {
            "symbol": "AAPL",
            "macd": 1.25,
            "signal": 0.95,
            "histogram": 0.30,
            "macd_signal": "bullish"
        },
        "bollinger_bands": {
            "symbol": "AAPL",
            "latest_price": 175.50,
            "upper_band": 180.00,
            "middle_band": 172.00,
            "lower_band": 164.00,
            "percent_b": 0.71,
            "position": "upper_half"
        },
        "regime_detection_wasserstein": {
            "symbol": "AAPL",
            "current_regime": "medium_volatility",
            "confidence": 0.85
        },
        "regime_detection_hmm": {
            "symbol": "AAPL",
            "current_regime": "bullish",
            "trend_persistence": 0.92
        },
        "ml_prediction": {
            "symbol": "AAPL",
            "direction": "UP",
            "confidence": 0.68,
            "model_agreement": 3
        }
    }


# ============================================================================
# Mock Fixtures for API Calls
# ============================================================================

@pytest.fixture
def mock_polygon_api():
    """Mock Polygon API responses."""
    def _mock_response(symbol: str, endpoint: str):
        if "aggs" in endpoint:
            # Price data
            return {
                "status": "OK",
                "results": [
                    {
                        "t": int(datetime.now().timestamp() * 1000),
                        "o": 175.0,
                        "h": 177.0,
                        "l": 174.0,
                        "c": 176.5,
                        "v": 50000000
                    }
                    for _ in range(100)
                ]
            }
        elif "rsi" in endpoint:
            return {
                "status": "OK",
                "results": {
                    "values": [{"value": 55.5, "timestamp": int(datetime.now().timestamp() * 1000)}]
                }
            }
        return {"status": "ERROR", "message": "Unknown endpoint"}
    
    return _mock_response


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API responses."""
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "tool_calls": [
                            {"tool_name": "polygon_price_data", "arguments": {"symbol": "AAPL"}},
                            {"tool_name": "polygon_technical_rsi", "arguments": {"symbol": "AAPL"}}
                        ]
                    })
                )
            )
        ]
    )


# ============================================================================
# Trading Fixtures
# ============================================================================

@pytest.fixture
def sample_portfolio_state():
    """Sample portfolio state for testing."""
    return {
        "cash": 80000.0,
        "positions": {
            "AAPL": {
                "shares": 100,
                "entry_price": 170.0,
                "position_type": "long",
                "entry_date": "2024-01-01"
            }
        },
        "equity": 97500.0,
        "initial_capital": 100000.0
    }


@pytest.fixture
def sample_trade_signals():
    """Sample trade signals for testing."""
    return {
        "AAPL": {"action": "hold", "confidence": 0.6},
        "MSFT": {"action": "buy", "confidence": 0.75},
        "JPM": {"action": "sell", "confidence": 0.8}
    }


# ============================================================================
# Walk-Forward Fixtures
# ============================================================================

@pytest.fixture
def walk_forward_dates():
    """Standard walk-forward validation date splits."""
    return {
        "train": {"start": "2020-01-01", "end": "2022-12-31"},
        "validate": {"start": "2023-01-01", "end": "2023-12-31"},
        "test": {"start": "2024-01-01", "end": "2024-06-30"}
    }


# ============================================================================
# Environment & Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    # Use test API keys if real ones not present
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    if not os.getenv("POLYGON_API_KEY"):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for test outputs."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# ============================================================================
# Skip Conditions
# ============================================================================

def pytest_configure(config):
    """Configure custom skip conditions."""
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring real API access"
    )
    config.addinivalue_line(
        "markers", "requires_ml: mark test as requiring ML libraries"
    )


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if API keys not available."""
    if not os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY") == "test-key-not-real":
        pytest.skip("Polygon API key not available")


@pytest.fixture
def skip_if_no_ml():
    """Skip test if ML libraries not available."""
    try:
        import sklearn
        # Don't import torch directly - it has DLL issues on Windows
        # Just check if sklearn is available
    except ImportError:
        pytest.skip("ML libraries (sklearn) not available")
    except OSError:
        pytest.skip("ML libraries have DLL loading issues")
