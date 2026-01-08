# Agentic AI Trader

A RAG-enhanced trading decision agent that combines retrieval-augmented generation with real-time market analysis tools. Features a validated VIX Rate-of-Change strategy and a self-critiquing Reflexion agent pattern.

## Overview

This project implements an intelligent trading assistant that:

1. **Retrieves wisdom from trading books** using a vector store (LlamaIndex)
2. **Analyzes market conditions** with multiple data tools (Polygon.io, FinBERT sentiment)
3. **Assesses risk** using a walk-forward validated VIX ROC strategy
4. **Makes decisions** through a Reflexion agent pattern with self-critique

## Key Features

### VIX ROC Three-Tier Strategy ⭐

A market timing overlay based on VIX Rate-of-Change, not absolute VIX level.

**Core Insight**: Exit when VIX is *accelerating* (panic phase), re-enter when VIX is *decelerating* (even if still elevated).

| Tier | Asset Type | Exit Threshold | Re-entry | Validation |
|------|------------|----------------|----------|------------|
| **Tier 1** | Value/Cyclical (SPY, XLF, XLE) | VIX ROC > 50% | < +15% | 7/7 wins |
| **Tier 2** | Growth/Tech (QQQ, AAPL, GOOGL) | VIX ROC > 20% | < 0% | 5/5 wins |
| **Tier 3** | Mega-Cap Tech (NVDA, MSFT) | VIX ROC > 75% | < -10% | 3/3 wins |

**Walk-Forward Results (2020-2024 Out-of-Sample)**:
- **15/15 wins** across all tested assets
- Average excess return: **+39%** (Tier 1), **+20%** (Tier 2), **+140%** (Tier 3)
- Significant drawdown reduction during COVID crash, 2022 bear market

### Volatility Prediction Tool

Predicts volatility regime transitions for position sizing:

- **VIX z-score** as primary predictor (works across all assets)
- **HIGH→LOW transitions**: 65-71% precision
- **LOW→HIGH (spikes)**: 45-62% precision at 0.6 threshold
- Asset-agnostic: trained on SPY, transfers to other equities

### Reflexion Agent Pattern

Implements the 4-step Reflexion pattern (Shinn et al., 2023):

1. **GENERATE**: Initial analysis with RAG + market data
2. **EVALUATE**: Self-critique for blind spots and overconfidence
3. **REFLECT**: Extract learnings from the critique
4. **REFINE**: Generate improved final decision

## Architecture

```
User Query → RAG Search (trading books) → Planner (tool selection)
           → Tool Execution (market data) → Reflexion Agent
           → Self-Critique → Final Decision
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Reflexion Agent** | `analyze_trade_agent_reflexion.py` | Main decision engine with self-critique |
| **Tool Registry** | `tools.py` | All market data and analysis tools |
| **Planner** | `planner.py` | LLM-based dynamic tool selection |
| **VIX ROC Strategy** | `models/vix_roc_production.py` | Production risk overlay |
| **Vol Prediction** | `vol_prediction_tool.py` | Volatility regime prediction |
| **Paper Trading** | `live_testing/paper_trader_new.py` | Forward testing system |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agentic_ai_trader.git
cd agentic_ai_trader

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - POLYGON_API_KEY
```

### Build Vector Store (Optional)

If you have trading books in `data/books/`:

```bash
python build_vectorstore.py
```

### Run the Agent

```bash
# Interactive trading analysis
python analyze_trade_agent_reflexion.py

# Example session:
# > Enter trading idea: Should I buy NVDA on this dip?
# > Enter symbol: NVDA
# [Agent analyzes with VIX ROC, vol prediction, sentiment, RAG...]
# [4-step Reflexion: Generate → Critique → Reflect → Refine]
# > VERDICT: ATTRACTIVE IF STRICT RULES ARE FOLLOWED
```

### Paper Trading

```bash
cd live_testing

# Reset and start fresh
python paper_trader_new.py --reset

# Run daily trading decisions
python paper_trader_new.py

# Check status
python paper_trader_new.py --status
```

## Test Results

### VIX ROC Strategy Performance (2020-2024)

**Tier 1: Value/Cyclical Assets**
| Asset | Buy & Hold | Strategy | Excess Return | Max DD Improvement |
|-------|------------|----------|---------------|-------------------|
| SPY | +94.6% | +111.5% | **+17.0%** | +6.4% |
| DIA | +62.0% | +88.5% | **+26.4%** | +11.8% |
| IWM | +42.2% | +81.0% | **+38.8%** | +5.4% |
| XLF | +71.6% | +114.9% | **+43.3%** | +15.7% |
| XLE | +76.3% | +133.0% | **+56.7%** | +20.2% |

**Tier 2: Growth/Tech Assets**
| Asset | Buy & Hold | Strategy | Excess Return |
|-------|------------|----------|---------------|
| QQQ | +143.9% | +169.2% | **+25.4%** |
| AAPL | +244.0% | +263.7% | **+19.7%** |
| GOOGL | +177.6% | +200.4% | **+22.8%** |
| AMZN | +131.2% | +158.7% | **+27.6%** |

**Tier 3: Mega-Cap Tech (Ultra-Conservative)**
| Asset | Buy & Hold | Strategy | Excess Return |
|-------|------------|----------|---------------|
| NVDA | +2148.4% | +2445.3% | **+296.9%** |
| MSFT | +174.4% | +234.1% | **+59.7%** |
| META | +180.2% | +242.8% | **+62.7%** |

### Volatility Prediction Accuracy

| Transition | Precision | Recall | Notes |
|------------|-----------|--------|-------|
| HIGH → LOW | 65-71% | ~60% | Most reliable signal |
| LOW → HIGH | 45-62% | ~40% | At 0.6 probability threshold |
| Any transition | ~34% | - | Not just "stay same" |

**Key Features Used**:
1. VIX z-score (primary predictor)
2. VIX momentum (5-day)
3. Asset realized volatility
4. Recent drawdown

## Available Tools

| Tool | Description |
|------|-------------|
| `vix_roc_risk` | VIX ROC signal for single asset |
| `vix_roc_portfolio_risk` | Portfolio-level VIX ROC assessment |
| `vol_prediction` | Volatility regime transition probabilities |
| `alpha_vantage_price_data` | OHLCV with volume analysis |
| `alpha_vantage_rsi` | Relative Strength Index |
| `alpha_vantage_atr` | Average True Range |
| `bollinger_bands` | Bollinger Bands with %B |
| `news_sentiment_finviz_finbert` | FinBERT sentiment from Finviz |

## Project Structure

```
agentic_ai_trader/
├── analyze_trade_agent_reflexion.py  # Main Reflexion agent
├── planner.py                         # LLM tool selection
├── tools.py                           # Tool registry (~1400 lines)
├── vol_prediction_tool.py             # Volatility prediction
├── agent_tools.py                     # Tool execution hub
├── build_vectorstore.py               # Vector store setup
├── models/
│   ├── vix_roc_production.py          # VIX ROC strategy (~950 lines)
│   ├── rolling_hmm_regime_detection.py # HMM regime (archived)
│   └── paper_wasserstein_regime_detection.py # Wasserstein (archived)
├── live_testing/
│   ├── paper_trader_new.py            # Forward paper trading
│   ├── strategies_new.py              # Strategy implementations
│   ├── config_new.py                  # Configuration
│   └── portfolio_tracker.py           # Portfolio management
├── docs/
│   ├── VIX_ROC_STRATEGY.md            # Strategy documentation
│   ├── RESEARCH_FEATURE.md            # Research capabilities
│   └── POLYGON_MIGRATION.md           # Data provider setup
└── tests/                             # Test suite
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # For LLM calls
POLYGON_API_KEY=...            # For market data

# Optional
ALPHAVANTAGE_API_KEY=...       # Alternative data source
AGENT_LOG_LEVEL=INFO           # Logging level
```

## Dependencies

- **LlamaIndex**: Vector store and RAG
- **OpenAI**: LLM for planning and decisions
- **Polygon.io**: Real-time market data
- **yfinance**: VIX data and fallback
- **scikit-learn**: Volatility prediction models
- **PyTorch/Transformers**: FinBERT sentiment (optional)

## Development Notes

### Adding New Tools

```python
# In tools.py
def my_tool_fn(state: dict, args: dict) -> dict:
    result = do_analysis(args["symbol"])
    state["tool_results"]["my_tool"] = result
    return state

register_tool({
    "name": "my_tool",
    "description": "What this tool does",
    "parameters": {"symbol": "Stock ticker"},
    "fn": my_tool_fn
})
```

### Rate Limiting

The system handles API rate limits automatically:
- Alpha Vantage: 5 calls/minute (free tier)
- Polygon.io: Varies by plan
- OpenAI: Automatic retry with backoff

## Archived Models

The following models are kept for reference but not actively used (deprecated due to lack of alpha generation):

- **HMM Regime Detection** (`models/rolling_hmm_regime_detection.py`)
- **Wasserstein Regime Detection** (`models/paper_wasserstein_regime_detection.py`)
- **ML Prediction Models** (`ml_models/`)

The VIX ROC strategy proved more effective and simpler to maintain.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## Author

Agentic AI Trader - January 2026
