# ML-Based Return Prediction Pipeline

## Overview

Supervised learning pipeline for predicting short-term stock returns (5-day and 10-day horizons) using gradient boosting, random forests, and baseline models. Based on Lopez de Prado's *Advances in Financial Machine Learning* and Jansen's *Machine Learning for Algorithmic Trading*.

## Features

### Data Collection
- Fetches 5 years of daily OHLCV data from Polygon.io
- Diversified stock universe across categories:
  - **Growth**: NVDA, TSLA, GOOGL, META, AMZN
  - **Value**: JPM, BAC, WFC, XOM, CVX
  - **Momentum**: MSFT, AAPL, V, MA, COST
  - **Defensive**: JNJ, PG, KO, PEP, WMT
  - **Volatility**: GME, AMC, COIN, MSTR, PLTR
- Automatic caching to avoid redundant API calls

### Feature Engineering (60+ features)
- **Price Features**: Returns, log returns, momentum
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs, EMAs
- **Volume Features**: Volume ratios, OBV, volume-price correlation
- **Volatility Features**: Realized vol, ATR, Parkinson estimator
- **Momentum Features**: ROC, drawdowns, consecutive up/down days
- **Market-Relative**: Beta, alpha, relative strength vs SPY
- **Seasonality**: Day of week, month, quarter effects

### Models
1. **Logistic Regression** (baseline with L2 regularization)
2. **Decision Tree** (baseline with max_depth=10)
3. **Random Forest** (100 trees, class-balanced)
4. **XGBoost** (gradient boosting with early stopping)

All models trained with:
- Time-series aware train/val/test split (60/20/20)
- Class balancing for imbalanced targets
- Feature scaling where appropriate
- Proper validation to avoid look-ahead bias

### Backtesting
- Compares ML models vs buy-and-hold benchmark
- Metrics: Total return, annualized return, Sharpe ratio, max drawdown, win rate
- Transaction costs (0.1% per trade)
- Identifies best model per stock and horizon

## Quick Start

### 1. Install Dependencies
```bash
cd ml_models
pip install -r requirements.txt
```

### 2. Set API Key
Ensure `POLYGON_API_KEY` is set in `.env` file at project root.

### 3. Run Quick Test (Single Stock)
```bash
python run_pipeline.py --test
```

This runs the full pipeline on AAPL with 5-day prediction horizon (~5 minutes).

### 4. Run Full Pipeline (All Stocks)
```bash
python run_pipeline.py
```

This trains models on 25 stocks across 2 horizons (~2 hours).

## File Structure

```
ml_models/
├── config.py                 # Configuration (stocks, parameters, features)
├── data_collection.py        # Fetch and cache data from Polygon.io
├── feature_engineering.py    # Create 60+ features
├── train_models.py          # Train 4 models with validation
├── backtest.py              # Backtest vs buy-and-hold
├── run_pipeline.py          # Main orchestrator
├── requirements.txt         # Dependencies
└── README.md               # This file

Generated files:
├── data/                    # Cached OHLCV data (parquet)
├── saved_models/            # Trained models (pickle)
└── results/                 # JSON results with metrics
```

## Configuration

Edit `config.py` to customize:

### Stock Universe
```python
STOCK_UNIVERSE = {
    "growth": ["NVDA", "TSLA", "GOOGL"],
    "value": ["JPM", "BAC", "WFC"],
    # ...
}
```

### Prediction Horizons
```python
PREDICTION_HORIZONS = [5, 10]  # Days
```

### Model Parameters
```python
MODELS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        # ...
    }
}
```

### Feature Groups
```python
FEATURE_GROUPS = {
    "technical": True,
    "volume": True,
    "volatility": True,
    # ...
}
```

## Output Format

Results saved to `results/ml_pipeline_results_TIMESTAMP.json`:

```json
{
  "timestamp": "20251215_143022",
  "config": {
    "symbols": ["AAPL", "MSFT", ...],
    "horizons": [5, 10]
  },
  "individual_results": [
    {
      "symbol": "AAPL",
      "category": "momentum",
      "horizon": 5,
      "training_metrics": {...},
      "backtest_metrics": {...},
      "best_model": "XGBoost"
    }
  ],
  "aggregate": {
    "5": {
      "models": {
        "XGBoost": {
          "mean_return": 0.15,
          "mean_sharpe": 1.2,
          "num_stocks": 25
        }
      }
    }
  }
}
```

## Example Usage

### Test Individual Components

**Data Collection:**
```bash
python data_collection.py
```

**Feature Engineering:**
```bash
python feature_engineering.py
```

**Model Training:**
```bash
python train_models.py
```

**Backtesting:**
```bash
python backtest.py
```

### Custom Pipeline

```python
from run_pipeline import run_full_pipeline

# Run on specific stocks and horizons
symbols = ["AAPL", "MSFT", "GOOGL"]
horizons = [5, 10]
results, aggregate = run_full_pipeline(symbols, horizons)
```

## Performance Expectations

Based on Lopez de Prado and Jansen's research:

- **Expected Sharpe**: 0.5 - 1.5 (realistic for daily predictions)
- **Win Rate**: 52-58% (slight edge over random)
- **Feature Importance**: Technical indicators + volatility often dominate
- **Best Models**: XGBoost and Random Forest typically outperform baselines
- **Overfitting Risk**: Use validation set to detect; expect test performance < validation

## Key Considerations

1. **Data Leakage**: All features use only past data (no look-ahead bias)
2. **Transaction Costs**: 0.1% per trade (conservative estimate)
3. **Class Imbalance**: Models use class weighting to handle ~50/50 up/down split
4. **Feature Scaling**: Logistic Regression uses StandardScaler; tree models don't need it
5. **Time-Series Split**: No shuffling; respects temporal order

## Next Steps

1. **Hyperparameter Tuning**: Grid search on validation set
2. **Ensemble Methods**: Stack models for improved performance
3. **Alternative Targets**: Try regression (predict magnitude) vs classification
4. **Regime-Conditional Models**: Train separate models per market regime
5. **Feature Selection**: Use feature importance to reduce dimensionality
6. **Walk-Forward Optimization**: Retrain periodically on rolling window

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*
- Tulchinsky, I. (2015). *Finding Alphas*

## Troubleshooting

**Issue**: "POLYGON_API_KEY not set"
- **Solution**: Add `POLYGON_API_KEY=your_key` to `.env` file

**Issue**: Models all predict same class
- **Solution**: Check class balance; may need to adjust `RETURN_THRESHOLD` in config

**Issue**: Poor performance on all stocks
- **Solution**: Markets may be efficient at 5-10 day horizon; try different features or longer horizons

**Issue**: Out of memory
- **Solution**: Reduce number of stocks or use smaller `n_estimators` for ensemble models
