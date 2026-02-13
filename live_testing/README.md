# Live Testing Infrastructure

**Complete backtesting framework for comparing strategies against buy-and-hold.**

## Overview

This directory contains infrastructure for testing:
- **Regime Detection Models** (Wasserstein & HMM)
- **ML Prediction Models** (consensus of 4 models)
- **Full Agent Decisions** (with RAG search)
- **Buy & Hold Baseline**

Each strategy is tested in **long-only** and **long/short** versions.

## Quick Start

### 1. Configure Test Parameters

Edit `config.py` to set:
```python
START_DATE = "2023-01-01"  # Test period
TEST_SYMBOLS = ["AAPL", "MSFT", ...]  # Stocks to test
INITIAL_CAPITAL = 100000.0  # Starting capital
```

### 2. Run Simulation

```bash
cd live_testing
python simulation_runner.py
```

This will:
- Fetch historical data for all symbols
- Run all strategies in parallel
- Calculate performance metrics
- Save results to `simulation_results/`
- Display comparison summary

### 3. Review Results

Results are saved in:
- `simulation_results/summary_YYYYMMDD_HHMMSS.csv` - Performance metrics
- `simulation_results/equity_*.csv` - Daily equity curves
- `simulation_results/full_results_*.json` - Complete results
- `trading_logs/trades_*.csv` - Trade-by-trade logs

## Strategies Tested

### Baseline
1. **Buy & Hold** - Simple buy and hold forever

### Regime-Based
2. **Wasserstein Long-Only** - Buy in low volatility, sell in high volatility
3. **Wasserstein Long/Short** - Long in low vol, short in high vol
4. **HMM Long-Only** - Buy in bullish regime, sell in bearish
5. **HMM Long/Short** - Long in bullish, short in bearish

### ML-Based
6. **ML Consensus Long-Only** - Buy on bullish ML signals, sell on bearish
7. **ML Consensus Long/Short** - Long on bullish, short on bearish

### Agent-Based
8. **Agent Long-Only** - Full agent decisions with RAG (long-only)
9. **Agent Long/Short** - Full agent decisions (long/short)

## Position Logic

- **If already LONG and signal is BUY**: Hold the long position
- **If already SHORT and signal is SHORT**: Hold the short position
- **If LONG and signal is SELL**: Close long position
- **If SHORT and signal is COVER**: Close short position
- **Otherwise**: Execute the signal

## Performance Metrics

Each strategy is evaluated on:
- Total Return (%)
- Annualized Return (%)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor (wins/losses ratio)
- Number of Trades
- Average Trade Duration
- Calmar Ratio (return/drawdown)

## File Structure

```
live_testing/
├── config.py                  # Test configuration
├── portfolio_tracker.py       # Position and P&L tracking
├── strategies.py              # Strategy implementations
├── simulation_runner.py       # Main orchestrator
├── README.md                 # This file
│
└── simulation_results/        # Results output
    ├── summary_*.csv         # Performance comparison
    ├── equity_*.csv          # Equity curves
    └── full_results_*.json   # Complete results

└── trading_logs/             # Trade logs
    └── trades_*.csv          # Trade-by-trade records
```

## Customization

### Add a New Strategy

1. Create strategy class in `strategies.py`:
```python
class MyStrategy(BaseStrategy):
    def generate_signal(self, symbol, current_position, date, price, historical_data):
        # Your logic here
        return "buy" or "sell" or "short" or "cover" or "hold"
```

2. Add to `config.py`:
```python
STRATEGIES = {
    "my_strategy": {
        "type": "custom",
        "long_only": True,
        "description": "My custom strategy"
    }
}
```

3. Update strategy factory in `strategies.py`

### Modify Test Parameters

In `config.py`:
- `MAX_POSITION_SIZE` - Max allocation per position (default 20%)
- `COMMISSION_PER_TRADE` - Commission costs
- `SLIPPAGE_BPS` - Slippage in basis points
- `REBALANCE_FREQUENCY` - How often to rebalance
- `REGIME_LOOKBACK_DAYS` - Regime detection window
- `ML_CONFIDENCE_THRESHOLD` - Minimum ML confidence to act

## Performance Comparison

After running, you'll see output like:

```
PERFORMANCE SUMMARY
================================================================================

Agent Long/Short:
  Total Return:       42.15%
  Annualized Return:  18.32%
  Sharpe Ratio:        1.45
  Max Drawdown:       12.34%
  Win Rate:           62.50%
  Num Trades:              24

ML Consensus Long:
  Total Return:       38.72%
  Annualized Return:  16.88%
  Sharpe Ratio:        1.38
  ...

Buy & Hold:
  Total Return:       28.50%
  Annualized Return:  12.10%
  Sharpe Ratio:        0.95
  Max Drawdown:       18.25%
  Win Rate:          100.00%
  Num Trades:              10
```

## Important Notes

1. **This directory is git-ignored** - Your test results won't be pushed to GitHub
2. **API Rate Limits** - Polygon API has rate limits; adjust symbols/dates accordingly
3. **Model Loading** - First run will be slow as models load (especially FinBERT)
4. **Memory Usage** - Testing many symbols over long periods uses significant memory
5. **No Look-Ahead Bias** - Strategies only use data available at each point in time

## Troubleshooting

### "No data fetched"
- Check Polygon API key in `.env`
- Verify date range is valid (not weekends/holidays)
- Check symbol tickers are correct

### "Import errors"
- Make sure you're in the project root when running
- Check that all dependencies are installed: `pip install -r requirements.txt`

### "Model loading errors"
- Wasserstein/HMM models need historical data (min 50 days)
- ML models need sentiment database (run `ml_models/build_sentiment_*.py` first)
- FinBERT model downloads on first use (requires internet)

### "Slow execution"
- Reduce number of symbols in `TEST_SYMBOLS`
- Shorten date range
- Set `AGENT_USE_RAG = False` in config (faster but less informed)
- Use regime/ML strategies instead of full agent

## Future Enhancements

Potential additions:
- [ ] Multi-asset portfolio allocation
- [ ] Risk parity weighting
- [ ] Dynamic position sizing based on volatility
- [ ] Stop-loss and take-profit rules
- [ ] Ensemble strategy (combine multiple signals)
- [ ] Walk-forward optimization
- [ ] Monte Carlo robustness testing
- [ ] Transaction cost impact analysis
- [ ] Visualization dashboard (plotly/dash)

## Credits

Built on top of the agentic_ai_trader framework.
