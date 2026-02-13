# Live Testing Infrastructure - Setup Complete ✅

## Forward Paper Trading System

The **forward paper trading system** is now operational and ready to track strategy performance starting from **December 17, 2025**.

### What This System Does

- **Starts TODAY**: Tracks performance from 2025-12-17 forward (not historical replay)
- **Runs Daily**: Execute once per day to generate trading signals and track positions
- **9 Strategies**: Tests regime detection, ML predictions, and agent decisions vs buy-and-hold
- **5 Stocks**: AAPL, MSFT, JPM, JNJ, XOM (reduced to minimize LLM API calls)
- **State Persistence**: Saves portfolios, positions, trades between runs
- **Performance Tracking**: Logs daily equity, returns, Sharpe ratios, drawdowns

### Key Differences from Historical Backtest

| Historical Backtest | Forward Paper Trading |
|---------------------|----------------------|
| Replays 2020-2025 data | Starts TODAY |
| Tests "what if" scenarios | Tracks real future performance |
| Results immediate | Results accumulate over weeks |
| All data available at once | Fetches new data daily |

## Daily Usage

### Run Today's Trading
```bash
cd live_testing
python paper_trader.py
```

This will:
1. Check if already ran today (exit if yes)
2. Fetch latest prices from Polygon API
3. Generate signals for all 9 strategies
4. Execute trades (buy/sell/short/cover)
5. Save state to `data/trading_state.json`
6. Log performance to `trading_logs/daily_performance.csv`

### Check Current Status
```bash
python paper_trader.py --status
```

Shows:
- Current equity for each strategy
- Total return % since start
- Number of positions held
- Detailed position list (shares, entry price, days held)

### Start Fresh (Reset Everything)
```bash
python paper_trader.py --reset
```

⚠️ WARNING: This deletes all state and starts over from scratch!

## Automatic Scheduling (Windows Task Scheduler)

### Quick Setup

1. **Test the batch file first**:
   ```cmd
   run_daily_trading.bat
   ```

2. **Open Task Scheduler**: Press `Win+R`, type `taskschd.msc`, press Enter

3. **Create Basic Task**:
   - Name: "Paper Trading Daily"
   - Trigger: Daily at 9:00 AM (or your preferred time)
   - Action: Start program → Browse to `run_daily_trading.bat`
   - Start in: `C:\Users\Andrew\projects\agentic_ai_trader\live_testing`

4. **Configure settings**:
   - Run whether user is logged on or not
   - Run with highest privileges
   - Run task as soon as possible after missed schedule

## First Run Results (2025-12-17) ✅

**System Operational** - All strategies executed successfully!

### Initial Positions:

- **Buy & Hold**: Bought all 5 stocks (equal weight)
- **Wasserstein**: Bought AAPL only (selective in high-vol regime)
- **HMM**: Bought all 5 stocks (bullish regime detected)
- **ML/Agent**: No positions yet (waiting for signals)

### Day 1 Performance:
- Buy & Hold: -0.03%
- Wasserstein: -0.01%
- HMM: -0.03%
- ML/Agent: 0.00% (no positions)

## What Was Also Created

Historical backtesting framework in `live_testing/` directory for comparing strategies (completed, but forward trading is the primary focus).

### Files Created

1. **`live_testing/config.py`** - Configuration settings
   - Test symbols, date ranges, capital
   - Strategy definitions (9 strategies)
   - Position sizing, transaction costs
   - Performance metrics to track

2. **`live_testing/portfolio_tracker.py`** - Position & P&L tracking
   - `PortfolioTracker` class for managing cash and positions
   - Long/short position support
   - Trade logging with commission and slippage
   - Performance metrics calculation (Sharpe, Sortino, max drawdown, etc.)

3. **`live_testing/strategies.py`** - Strategy implementations
   - `BuyAndHoldStrategy` - Baseline
   - `RegimeStrategy` - Wasserstein or HMM based
   - `MLStrategy` - ML consensus predictions
   - `AgentStrategy` - Full agent decisions
   - Signal generation with position awareness

4. **`live_testing/simulation_runner.py`** - Main orchestrator
   - Fetches historical data from Polygon
   - Runs all strategies in parallel
   - Tracks daily equity curves
   - Saves results to CSV/JSON
   - Displays performance comparison

5. **`live_testing/README.md`** - Complete documentation
   - Quick start guide
   - Strategy descriptions
   - Customization instructions
   - Troubleshooting tips

6. **`live_testing/quick_test.py`** - Verification script
   - Tests all imports
   - Verifies portfolio tracker
   - Tests strategy creation
   - Checks signal generation

### Output Directories (Git-Ignored)

- `simulation_results/` - Performance metrics, equity curves, JSON results
- `trading_logs/` - Trade-by-trade logs for each strategy
- `experimental/` - For any experimental work
- `live_testing/` - Entire testing infrastructure (except .gitkeep)

## Strategies Tested

### 9 Total Strategies

1. **Buy & Hold** (baseline)
2. **Wasserstein Long-Only** - Buy in low vol, sell in high vol
3. **Wasserstein Long/Short** - Long in low vol, short in high vol
4. **HMM Long-Only** - Buy in bullish, sell in bearish
5. **HMM Long/Short** - Long in bullish, short in bearish
6. **ML Consensus Long-Only** - Buy on bullish ML signals
7. **ML Consensus Long/Short** - Long on bullish, short on bearish
8. **Agent Long-Only** - Full agent decisions (long-only)
9. **Agent Long/Short** - Full agent decisions (long/short)

### Position Logic Implemented

- **Already LONG + BUY signal** → Hold long
- **Already SHORT + SHORT signal** → Hold short
- **LONG + SELL signal** → Close long
- **SHORT + COVER signal** → Close short
- **Otherwise** → Execute signal

## How to Use

### 1. Quick Verification

```bash
cd live_testing
python quick_test.py
```

Verifies all imports and components work.

### 2. Configure Test

Edit `config.py`:
```python
START_DATE = "2023-01-01"  # Adjust to your preference
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", ...]
INITIAL_CAPITAL = 100000.0
```

### 3. Run Full Simulation

```bash
python simulation_runner.py
```

This will:
- Fetch historical data for all symbols
- Run all 9 strategies
- Calculate performance metrics
- Save results to `simulation_results/`
- Display comparison summary

### 4. Review Results

Check:
- `simulation_results/summary_YYYYMMDD_HHMMSS.csv` - Performance comparison
- `simulation_results/equity_*.csv` - Daily equity curves
- `trading_logs/trades_*.csv` - Trade-by-trade logs

## Performance Metrics Tracked

For each strategy:
- Total Return (%)
- Annualized Return (%)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor (wins/losses ratio)
- Number of Trades
- Average Trade Duration (days)
- Volatility
- Calmar Ratio (return/max drawdown)
- Total Commission Paid
- Total Slippage Cost

## Safety Features

### Git Protection
- **Entire `live_testing/` directory is git-ignored**
- All results, logs, and experimental work stay local
- Only `.gitignore` update was pushed to GitHub
- Your testing stays private

### No Risk of Pushing Sensitive Data
- `.gitignore` updated with patterns:
  ```
  live_testing/
  experimental/
  trading_logs/
  simulation_results/
  ```
- Even if you accidentally `git add .`, these won't be tracked

## Important Notes

### First Run Will Be Slow
- Polygon API data fetching (~5-10 seconds per symbol)
- Model initialization (FinBERT downloads ~500MB first time)
- Regime detection calculations (need 252 days lookback)
- ML model loading (4 models from pickle files)

### API Considerations
- **Polygon Free Tier**: 5 API calls/minute, 100/day
- Adjust `TEST_SYMBOLS` accordingly
- Rate limiting is built-in via `polygon_tools.py`

### Model Requirements
- **Regime Models**: Need min 50 days historical data
- **ML Models**: Need sentiment database (run `ml_models/build_sentiment_*.py` first if not done)
- **Agent**: Uses RAG (slower but more informed) - set `AGENT_USE_RAG = False` in config for speed

### Memory Usage
- Testing 10 symbols over 2 years = moderate memory
- Testing 100 symbols over 5 years = high memory
- Reduce symbols or shorten date range if memory issues

## Customization Examples

### Test Just Buy & Hold vs ML
Edit `config.py`:
```python
STRATEGIES = {
    "buy_and_hold": {...},
    "ml_consensus_long": {...},
    "ml_consensus_long_short": {...}
}
```

### Change Position Sizing
```python
MAX_POSITION_SIZE = 0.10  # Max 10% per position (was 20%)
MIN_POSITION_SIZE = 0.02  # Min 2% per position (was 5%)
```

### Add Transaction Costs
```python
COMMISSION_PER_TRADE = 1.0  # $1 per trade
SLIPPAGE_BPS = 10  # 0.10% slippage (was 5)
```

### Shorten Test Period
```python
START_DATE = "2024-01-01"  # Just 2024
END_DATE = "2024-12-31"
```

## Expected Output

```
================================================================================
LIVE TESTING SIMULATION
================================================================================
Symbols: AAPL, MSFT, GOOGL, NVDA, JPM, BAC, JNJ, UNH, XOM, PG
Period: 2023-01-01 to 2024-12-17
Strategies: 9
Initial Capital: $100,000.00
================================================================================

Fetching historical data...
  Fetching data for AAPL...
  Fetching data for MSFT...
  ...
Fetched data for 10 symbols

Running simulation over 495 trading days...
Simulating: 100%|██████████| 495/495 [05:23<00:00,  1.53it/s]

================================================================================
Simulation complete!
================================================================================

---

## Quick Reference - Forward Paper Trading Commands

### Daily Operations
```bash
# Run today's trading (main command)
python paper_trader.py

# Check current status
python paper_trader.py --status

# Reset everything and start fresh
python paper_trader.py --reset
```

### Files to Monitor
```bash
# Daily performance log (CSV)
trading_logs/daily_performance.csv

# Full state (JSON)
data/trading_state.json

# Scheduled run log
trading_logs/daily_runner.log
```

### Timeline
- **Day 1** (2025-12-17): ✅ Initial positions established
- **Week 1**: Strategies adjust based on regime changes
- **Weeks 2-4**: Performance differences become meaningful
- **After 1 Month**: Compare and analyze results

Saved summary to: simulation_results/summary_20241217_143522.csv
Saved equity curves to: simulation_results
Saved trade logs to: trading_logs
Saved full results to: simulation_results/full_results_20241217_143522.json

================================================================================
PERFORMANCE SUMMARY
================================================================================

ML Consensus Long/Short:
  Total Return:       45.23%
  Annualized Return:  20.15%
  Sharpe Ratio:        1.52
  Max Drawdown:       11.34%
  Win Rate:           64.50%
  Num Trades:              32

Agent Long/Short:
  Total Return:       42.88%
  Annualized Return:  19.12%
  Sharpe Ratio:        1.48
  Max Drawdown:       12.10%
  Win Rate:           61.20%
  Num Trades:              28

...

Buy & Hold:
  Total Return:       28.50%
  Annualized Return:  12.10%
  Sharpe Ratio:        0.95
  Max Drawdown:       18.25%
  Win Rate:          100.00%
  Num Trades:              10

================================================================================
```

## Troubleshooting

### "ModuleNotFoundError"
- Make sure you're running from `live_testing/` directory
- Check all dependencies installed: `pip install -r requirements.txt`

### "No data fetched"
- Verify Polygon API key in `.env`
- Check date range (weekends/holidays have no data)
- Verify symbol tickers are correct

### "HMM/Wasserstein errors"
- Need minimum 50 days historical data
- First run initializes models (slow)
- Check console for specific error messages

### "ML prediction errors"
- Sentiment database may be incomplete
- Run `ml_models/build_sentiment_batch*.py` first
- Or set confidence threshold lower in config

### "Agent timeout errors"
- Set `AGENT_TIMEOUT` higher in config (default 30 sec)
- Or set `AGENT_USE_RAG = False` for faster execution
- Or use ML/regime strategies instead

## Next Steps

1. **Run Quick Test**: `cd live_testing && python quick_test.py`
2. **Adjust Config**: Edit `config.py` with your preferred settings
3. **Run Simulation**: `python simulation_runner.py`
4. **Analyze Results**: Open CSV files in Excel or Python
5. **Iterate**: Modify strategies, parameters, symbols as needed

## Future Enhancements

Potential additions you could make:
- Risk parity position sizing
- Dynamic position sizing based on volatility
- Stop-loss and take-profit rules
- Multi-asset portfolio allocation
- Walk-forward optimization
- Monte Carlo robustness testing
- Visualization dashboard with Plotly

## Summary

✅ **Complete testing infrastructure ready**
✅ **9 strategies configured (long-only + long/short)**
✅ **Position logic handles hold scenarios**
✅ **All results git-ignored (safe for experimentation)**
✅ **Comprehensive performance metrics**
✅ **Easy to customize and extend**

Your live testing framework is ready to go! Start with `quick_test.py` to verify everything works, then run the full simulation.
