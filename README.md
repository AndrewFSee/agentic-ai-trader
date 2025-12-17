# Agentic AI Trader

RAG-enhanced trading decision agent that combines vector store of trading books with LLM-based planning and tool execution.

## Project Structure

```
agentic_ai_trader/
├── analyze_trade_agent.py      # Main trading agent entry point
├── planner.py                   # LLM-based dynamic tool selection
├── tools.py                     # Tool registry (Alpha Vantage, FinBERT)
├── agent_tools.py              # Tool execution hub
├── build_vectorstore.py        # Trading books vector store builder
├── polygon_tools.py            # Polygon.io market data tools
├── research_tools.py           # Research agent integration
├── sentiment_tools.py          # Sentiment analysis tools
│
├── models/                     # Regime detection models
│   ├── paper_wasserstein_regime_detection.py   # Paper-faithful Wasserstein k-means
│   └── rolling_hmm_regime_detection.py         # Professional rolling HMM
│
├── data/                       # Trading books and source data
├── db/                         # Vector store (Chroma DB)
│   └── books/                  # Trading book embeddings
│
├── docs/                       # Documentation
│   ├── WASSERSTEIN_VS_HMM_VERDICT.md          # Comprehensive comparison results
│   ├── GPT_RESEARCHER_INTEGRATION.md          # Research agent docs
│   ├── PROFESSIONAL_HMM_APPROACH.md           # HMM methodology
│   ├── POLYGON_MIGRATION.md                   # API migration notes
│   └── RESEARCH_FEATURE.md                    # Research features
│
├── tests/                      # Test scripts and comparisons
├── results/                    # Test results, logs, JSON outputs
├── scratch/                    # Development experiments
├── scripts/                    # Utility scripts
└── research_reports/           # Generated research reports

```

## Core Components

### Main Agent
- **analyze_trade_agent.py**: Orchestrates RAG search → planner → tool execution → decision
- **planner.py**: GPT-based tool selection (returns JSON tool calls)
- **tools.py**: Tool registry with market data, sentiment, **and regime detection**
- **agent_tools.py**: Rate limiting and tool execution

### Regime Detection Models (Integrated as Agent Tools)

**New**: Both regime detection methods are now available as tools in the main agent. The LLM dynamically selects which method(s) to use based on:
- Stock type (tech/healthcare → Wasserstein, stable → HMM)
- Question type (volatility → Wasserstein, trend → HMM)
- Confidence needs (both + consensus check for high conviction)

See [docs/REGIME_DETECTION_AGENT_GUIDE.md](docs/REGIME_DETECTION_AGENT_GUIDE.md) for detailed decision framework.

#### Paper-Faithful Wasserstein (`models/paper_wasserstein_regime_detection.py`)
- Implements Horvath et al. (2021) Algorithm 1
- **Median barycenters** (not mean) for robustness
- Fast 1D Wasserstein distance (O(N log N))
- MMD cluster quality evaluation
- Empirical distributions approach

**Performance**:
- Mean Sharpe improvement: +0.013 (12 stocks)
- Best on: MSFT (+0.233), JNJ (+0.246), BAC (+0.127)
- Mixed vs HMM: 50% win rate, +22% average Sharpe

#### Rolling HMM (`models/rolling_hmm_regime_detection.py`)
- Forward-filter only (no retroactive relabeling)
- Frozen parameters after training
- Persistent transition matrix initialization
- Volatility-based state mapping

**Performance**:
- Competitive with Wasserstein
- Better on some stocks (AAPL)
- More stable label consistency

### Vector Store
- **Location**: `db/books/`
- **Source**: PDF trading books in `data/books/`
- **Embeddings**: OpenAI `text-embedding-3-large`
- **Rebuild**: `python build_vectorstore.py`

## Environment Setup

### Required API Keys (`.env`)
```
OPENAI_API_KEY=sk-...
POLYGON_API_KEY=...
ALPHAVANTAGE_API_KEY=... (optional)
```

### Installation
```bash
pip install -r requirements.txt
pip install -r requirements_hmm.txt  # For HMM models
```

## Usage

### Run Main Agent
```bash
python analyze_trade_agent.py
```
Interactive loop: Enter trading idea → symbol → get analysis

### Test Regime Models
```bash
# Wasserstein stability test
python tests/test_paper_wasserstein_stability.py --symbol AAPL --start-date 2020-01-01

# HMM vs Wasserstein comparison
python tests/final_wasserstein_vs_hmm.py --symbol MSFT

# Multi-stock backtest
python tests/test_paper_wasserstein_backtest.py --sectors --start-date 2020-01-01
```

## Key Findings

### Wasserstein vs HMM Verdict
**Neither is universally superior.** Use ensemble approach.

- **Wasserstein excels**: Tech/healthcare stocks with distinct volatility regimes
- **HMM excels**: Smooth transitions, probabilistic predictions
- **Performance paradox**: Better when label consistency is LOW (adaptive)
- **MMD issue**: All stocks show poor cluster separation (~1.0 ratio)

See [docs/WASSERSTEIN_VS_HMM_VERDICT.md](docs/WASSERSTEIN_VS_HMM_VERDICT.md) for full analysis.

## Development

### Testing
All test scripts in `tests/` folder:
- `test_paper_wasserstein_*.py` - Wasserstein tests
- `test_rolling_hmm_*.py` - HMM tests
- `compare_*.py` - Head-to-head comparisons
- `debug_*.py` - Debugging utilities

### Results
All outputs in `results/` folder:
- JSON files: Test results, backtests, comparisons
- Log files: Detailed execution logs
- PNG files: Visualization outputs

## Architecture Notes

### Tool Registry Pattern
All tools follow: `fn(state: dict, args: dict) -> dict`
- State stored in `state["tool_results"][tool_name]`
- Registration via `register_tool()` with `ToolSpec`

### Rate Limiting
Smart rate limiter in `agent_tools.py`:
- Tracks API call timestamps
- Only delays when necessary
- Respects 5 calls/min limits

### RAG Strategy
Two searches per query:
1. **Idea-specific**: k=6 chunks matching query + symbol
2. **Risk management**: k=6 chunks about discipline/psychology

## License

[Your License]

## References

- Horvath et al. (2021): "Clustering Market Regimes using the Wasserstein Distance"
- Trading books in `data/books/`
