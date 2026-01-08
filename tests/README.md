# Testing Infrastructure

This document describes the testing infrastructure for the Agentic AI Trader project.

## Directory Structure

```
tests/
├── conftest.py          # Shared pytest fixtures
├── __init__.py
├── unit/                # Fast unit tests (no external dependencies)
│   ├── __init__.py
│   ├── test_tools_unit.py
│   └── test_validation_unit.py
├── integration/         # Tests that call external APIs
│   ├── __init__.py
│   └── test_agent_integration.py
└── backtests/          # Historical backtesting tests
    ├── __init__.py
    └── test_strategy_backtests.py
```

## Running Tests

### Quick Unit Tests (< 1 second)
```bash
pytest tests/unit/ -v
```

### All Tests
```bash
pytest tests/ -v
```

### By Marker
```bash
# Only unit tests
pytest -m unit

# Only integration tests (requires API keys)
pytest -m integration

# Only backtest tests (slower)
pytest -m backtest

# Skip slow tests
pytest -m "not slow"

# Tests that require ML libraries
pytest -m ml
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests (fast, no external dependencies) |
| `integration` | Integration tests (may call external APIs) |
| `backtest` | Backtesting tests (longer running) |
| `slow` | Slow tests (> 10 seconds) |
| `api` | Tests that require API keys |
| `ml` | Tests that require ML models (torch, sklearn) |
| `rag` | Tests that require RAG vector store |
| `live` | Tests that interact with live trading |

## Fixtures

Common fixtures are defined in `conftest.py`:

### Data Fixtures
- `sample_ohlcv_data` - Sample price data for testing
- `sample_features` - Sample ML features
- `sample_tool_results` - Mock tool output
- `sample_portfolio_state` - Mock portfolio state
- `sample_trade_signals` - Mock trading signals

### Configuration Fixtures
- `project_root` - Project root directory
- `test_symbols` - Standard test symbols
- `test_date_range` - Standard date range
- `walk_forward_dates` - Walk-forward validation splits

### Skip Fixtures
- `skip_if_no_api_key` - Skip if no API keys available
- `skip_if_no_ml` - Skip if ML libraries not available

## Validation Framework

The project includes a comprehensive walk-forward validation framework:

### Walk-Forward Validation (`live_testing/walk_forward_validation.py`)

```python
from walk_forward_validation import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_start="2020-01-01",
    train_end="2022-12-31",
    validate_start="2023-01-01",
    validate_end="2023-12-31",
    test_start="2024-01-01",
    test_end="2024-12-31"
)

validator = WalkForwardValidator(config)
results = validator.run_full_validation("my_strategy", prices, signal_generator)
```

### Validation Gate (`live_testing/validation_gate.py`)

Enforces minimum requirements before live trading:

```python
from validation_gate import ValidationGate

gate = ValidationGate()

# Check if strategy is ready
if gate.is_live_ready("my_strategy"):
    print("Safe to trade!")
else:
    status = gate.get_validation_status("my_strategy")
    print(f"Next steps: {status['next_steps']}")
```

### Backtest Integration (`live_testing/backtest_integration.py`)

Run comprehensive backtests:

```bash
# Single strategy
python backtest_integration.py --strategy buy_and_hold --symbols AAPL MSFT

# All strategies
python backtest_integration.py --symbols AAPL MSFT JPM
```

## Validation Pipeline

Before any strategy goes live, it must pass:

1. **Backtest (2020-2022)** - Train on historical data
   - Sharpe Ratio >= 0.5
   - Max Drawdown >= -25%
   - Min 30 trades

2. **Out-of-Sample (2023-2024)** - Validate on unseen data
   - Sharpe Ratio >= 0.3
   - Max Drawdown >= -30%
   - Positive returns preferred

3. **Paper Trading (30+ days)** - Validate in real-time
   - Performance similar to backtest
   - No major deviations

4. **Live Trading** - Only after all validation passes

## CI/CD Integration

Add to your CI pipeline:

```yaml
test:
  script:
    - pip install pytest pytest-cov
    - pytest tests/unit/ -v --tb=short
    - pytest tests/integration/ -v --tb=short -m "not slow"
```

## Known Issues

1. **Torch DLL on Windows**: Some tests may fail on Windows due to torch DLL loading issues. Tests gracefully skip when this occurs.

2. **API Rate Limits**: Integration tests may fail due to API rate limits. Use `--tb=short` for cleaner output.

3. **Large Feature Lookback**: The `prepare_features()` function drops ~200 rows due to rolling window requirements. Ensure sufficient historical data.
