# Agentic AI Trader - AI Agent Instructions

## Architecture Overview

This is a **RAG-enhanced trading decision agent** that combines:
1. **Vector store of trading books** (`db/books/` LlamaIndex) for retrieval-augmented generation
2. **LLM-based planner** (`planner.py`) that dynamically selects market data tools
3. **Tool registry pattern** (`tools.py`) for Alpha Vantage API calls and FinBERT sentiment
4. **Decision agent** (`analyze_trade_agent.py`) that synthesizes all inputs into trade recommendations

### Data Flow
```
User Query → RAG Search (trading books) → Planner (selects tools) 
→ Tool Execution (market data) → Decision LLM → Structured Verdict
```

## Core Components

### 1. Tool Registry (`tools.py`)
- **Pattern**: All tools follow `fn(state: dict, args: dict) -> dict` signature
- **State Management**: Tools store results in `state["tool_results"][tool_name]`
- **Registration**: Use `register_tool()` with `ToolSpec` TypedDict (name, description, parameters, fn)
- **Available Tools**:
  - `alpha_vantage_price_data` - Daily OHLC bars with volume analysis
  - `alpha_vantage_rsi` - Relative Strength Index (free tier)
  - `alpha_vantage_macd` - MACD calculated from price data (not API, works on free tier)
  - `alpha_vantage_atr` - Average True Range (free tier)
  - `bollinger_bands` - Calculated from price data (no API call, instant)
  - `alpha_vantage_fundamentals` - Company overview (PREMIUM ONLY, disabled)
  - `news_sentiment_finviz_finbert` - Scrapes Finviz + runs FinBERT sentiment model
  - `bocpd_regime` - BOCPD market regime detection (bull/bear/transition/crisis/consolidation) on SPY
  - `market_risk` - ML-based drawdown probability + forward volatility (isotonic calibration)
  - `vol_prediction` - Volatility regime transition probabilities
  - `regime_detection_wasserstein` - DEPRECATED (commented out)
  - `regime_detection_hmm` - DEPRECATED (commented out)
  - `regime_consensus_check` - DEPRECATED (commented out)

### 2. Planner (`planner.py`)
- **Model**: Uses `gpt-5.1` (configurable via `PLANNER_MODEL`)
- **Input**: User query + symbol → JSON list of tool calls
- **Output Format**: `{"tool_calls": [{"tool_name": "...", "arguments": {...}}]}`
- **Tool Selection**: Selects all relevant tools for comprehensive analysis (typically 5-6 tools)

### 3. Vector Store (`build_vectorstore.py`)
- **Framework**: LlamaIndex with VectorStoreIndex
- **Location**: `db/books/` (persisted to disk)
- **Source**: PDF trading books in `data/books/`
- **Embeddings**: OpenAI `text-embedding-3-large`
- **Chunking**: SentenceSplitter with 1200 chars, 150 overlap
- **Metadata**: Each chunk has `book_name`, `page_label`, `file_name`
- **Rebuild**: Run `python build_vectorstore.py` when adding new books
- **Loading**: Use `load_index_from_storage()` in `analyze_trade_agent.py`

### 4. Decision Agent (`analyze_trade_agent.py`)
- **RAG Strategy**: Two searches per query:
  - Idea-specific: `k=6` chunks matching user query + symbol
  - Risk management: `k=6` chunks about position sizing, psychology, discipline
- **Final Verdict**: MUST be one of three structured outputs:
  - `VERDICT: NOT ATTRACTIVE based on the books and current data`
  - `VERDICT: UNCLEAR / NEEDS MORE INFORMATION`
  - `VERDICT: ATTRACTIVE IF STRICT RULES ARE FOLLOWED`

## Environment Setup

### Required API Keys (`.env`)
```
OPENAI_API_KEY=sk-...
ALPHAVANTAGE_API_KEY=...
```

### Dependencies
- LlamaIndex: `llama-index`, `llama-index-embeddings-openai`, `llama-index-llms-openai`
- OpenAI SDK: `openai` (for LLM calls)
- ML: `torch`, `transformers` (for FinBERT sentiment)
- Web scraping: `beautifulsoup4`, `requests`
- Document loading: `pypdf` (for PDF parsing)

### FinBERT Model
- **Model**: `ProsusAI/finbert`
- **Lazy Loading**: Model loads on first use via `_ensure_finbert_loaded()`
- **Graceful Degradation**: Returns "unknown" sentiment if model fails to load

## Development Patterns

### Adding a New Tool
1. Define function: `def my_tool_fn(state: dict, args: dict) -> dict`
2. Store result: `state["tool_results"]["my_tool"] = {...}`
3. Register: `register_tool({"name": "my_tool", "description": "...", "parameters": {...}, "fn": my_tool_fn})`
4. Update planner prompt in `planner.py` to include guidance on when to use it

### Formatting Tool Results
- All `alpha_vantage_*` tools return structured dicts with `symbol`, `interval`, `latest_*` fields
- Volume analysis included in price data: `avg_volume`, `volume_ratio`, `volume_condition`, `recent_volume_spikes`
- Bollinger Bands return: `latest_price`, `middle_band`, `upper_band`, `lower_band`, `percent_b`, `bandwidth_pct`, `position`
- Error handling: Always check for `result.get("error")` before accessing data
- Use private `_format_*_summary()` functions in `analyze_trade_agent.py` to convert tool results to prompt text

### Rate Limiting
- **Smart Rate Limiter** in `agent_tools.py`: `RateLimiter` class tracks API call timestamps
- Only delays when necessary to respect 5 calls/min limit (not fixed 12-second delays)
- Automatically removes expired calls from tracking window
- Usage: `_alpha_vantage_limiter.wait_if_needed("Alpha Vantage")` before each API call

### Error Handling Convention
- API errors stored as: `{"symbol": "...", "error": "error message"}`
- Formatters check `if result.get("error")` and return user-friendly message
- Never raise exceptions in tools; always return state with error field

## Testing & Debugging

### Manual Testing
- Run main agent: `python analyze_trade_agent.py`
- Interactive loop prompts for: trading idea → symbol → displays analysis
- Quit with 'q', 'quit', or 'exit'

### Common Issues
- **Rate Limits**: Alpha Vantage free tier = 25 requests/day and 5 requests/minute
  - `agent_tools.py` adds 12-second delays between Alpha Vantage calls to respect rate limits
  - Free tier uses `TIME_SERIES_DAILY` (not `TIME_SERIES_DAILY_ADJUSTED` which is premium)
  - Volume field is `"5. volume"` for free tier (not `"6. volume"` for premium)
  - **Premium Endpoints** (NOT available on free tier): MACD, fundamentals (OVERVIEW)
  - **Free Endpoints**: TIME_SERIES_DAILY, RSI, possibly ATR (needs testing)
- **Missing Dependencies**: torch and transformers are NOT in base conda environment
  - `tools.py` now has conditional imports so non-ML tools work without torch
  - FinBERT sentiment will return "unknown" if torch/transformers not installed
  - News scraping from Finviz works regardless of torch availability
- **Missing Books**: If RAG returns empty, check `data/books/*.pdf` exists and rebuild vectorstore
- **Model Errors**: Check `DECISION_MODEL` and `PLANNER_MODEL` are valid OpenAI models (currently `gpt-5.1`)
- **Deprecation Warnings**: Suppressed via `warnings.filterwarnings("ignore", category=DeprecationWarning)` in main files

## Project Structure
```
agentic_ai_trader/
├── analyze_trade_agent.py      # Main entry point
├── planner.py                   # LLM-based tool selection
├── tools.py                     # Tool registry
├── agent_tools.py              # Tool execution hub
├── market_risk_model.py        # ML drawdown probability + forward vol
├── build_vectorstore.py        # Vector store setup
├── models/                     # ML and regime detection models
│   ├── bocpd_regime/           # BOCPD market regime package (from allocator project)
│   │   ├── bocpd.py            # Adams & MacKay BOCPD with dynamic hazard
│   │   ├── labeling.py         # 6-regime classifier (multi-timeframe)
│   │   ├── detector.py         # High-level detect_current_regime() for tool
│   │   ├── config.py           # BOCPDConfig + RegimeLabelConfig
│   │   └── utils.py            # Smoothing, thresholding helpers
│   ├── paper_wasserstein_regime_detection.py  # DEPRECATED
│   └── rolling_hmm_regime_detection.py        # DEPRECATED
├── docs/                       # Documentation
├── tests/                      # Test scripts
├── results/                    # Test outputs
└── scratch/                    # Development experiments
```

## Key Files Reference
- `analyze_trade_agent.py` - Main entry point and orchestration
- `tools.py` - All market data, sentiment, and regime detection tools
- `planner.py` - LLM-based dynamic tool selection with regime/risk guidance
- `agent_tools.py` - Tool execution hub (`run_tools` function)
- `market_risk_model.py` - Isotonic calibration DD probability + HistGBR vol prediction
- `build_vectorstore.py` - One-time setup for trading book embeddings
- `models/bocpd_regime/` - BOCPD market regime package (adapted from regime_aware_portfolio_allocator)
- `models/paper_wasserstein_regime_detection.py` - DEPRECATED Wasserstein k-means
- `models/rolling_hmm_regime_detection.py` - DEPRECATED rolling HMM
- `tests/` - All test and comparison scripts
- `docs/` - Comprehensive documentation
