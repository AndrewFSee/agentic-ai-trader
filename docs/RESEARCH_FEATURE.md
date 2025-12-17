# Deep Research Feature - GPT-Researcher Integration

## Quick Start for Testing

**TL;DR**: Research tools are in **TESTING MODE** by default (faster, cheaper). Perfect for development!

```bash
# 1. Install
pip install -r requirements_research.txt

# 2. Get Tavily API key (free, 1000 searches/month)
# Visit: https://tavily.com

# 3. Add to .env
TAVILY_API_KEY="tvly-dev-YOUR_KEY"

# 4. Test (will use TESTING MODE automatically)
python analyze_trade_agent.py
> "comprehensive analysis of AI chip market"
> NVDA
# Takes ~1-2 minutes, costs ~$0.03-0.05

# 5. Switch to production mode when needed (optional)
GPT_RESEARCHER_TESTING_MODE="false"  # Add to .env
```

---

## Overview

The trading agent now includes **GPT-Researcher** integration for autonomous deep research on any market-related topic. This complements the real-time market data tools with comprehensive web research capabilities.

## What is GPT-Researcher?

GPT-Researcher is an autonomous research agent that:
- 📊 Plans research questions automatically
- 🌐 Scrapes 20+ web sources in parallel
- 🧠 Synthesizes information using LLMs
- 📝 Generates comprehensive reports with citations
- ⚡ Completes research in 2-5 minutes

## Installation

1. Install the package:
```bash
pip install -r requirements_research.txt
```

2. Get a free Tavily API key:
   - Visit https://tavily.com
   - Sign up for free tier
   - Copy your API key

3. Add to `.env`:
```bash
TAVILY_API_KEY="your_tavily_api_key_here"
```

## Available Research Tools

### 1. `gpt_researcher_market_research`
General-purpose deep research on any market topic.

**Use for:**
- Market analysis and trends
- Competitive landscape
- Industry dynamics
- Technology disruption
- Regulatory changes

**Example queries:**
- "AI chip market competitive dynamics and growth outlook"
- "Impact of rising interest rates on tech valuations"
- "Electric vehicle supply chain and battery technology trends"

### 2. `gpt_researcher_company_analysis`
Deep company research beyond basic fundamentals.

**Covers:**
- Business model and revenue streams
- Competitive advantages
- Financial performance trends
- Recent developments
- Risk factors
- Growth outlook

**Example:**
```python
args = {"symbol": "NVDA", "company_name": "Nvidia Corp"}
```

### 3. `gpt_researcher_sector_trends`
Sector/industry analysis.

**Analyzes:**
- Industry growth trends
- Key players and market share
- Disruption and innovation
- Regulatory environment
- Investment themes

**Example:**
```python
args = {"sector": "semiconductors", "timeframe": "current and next 12 months"}
```

### 4. `gpt_researcher_economic_indicators`
Macro research and economic analysis.

**Covers:**
- GDP, inflation, employment
- Interest rates and monetary policy
- Consumer sentiment
- Market implications

**Example:**
```python
args = {"topic": "Federal Reserve policy and rate path"}
```

## How It Works in the Trading Agent

### Automatic Tool Selection

The planner intelligently decides when to use research tools:

**✅ Will use research tools when:**
- User explicitly asks for "research", "analysis", "deep dive"
- Query requires information beyond market data
- Need industry dynamics, competitive landscape, or macro context

**❌ Won't use research tools for:**
- Quick price checks
- Standard technical analysis
- Real-time market data queries
- Fast trading decisions

### Integration Flow

```
User Query
    ↓
Planner (decides which tools)
    ↓
Standard Tools (price, technicals, sentiment) - Fast
    ↓
Research Tools (if needed) - Slow (2-5 min)
    ↓
Analysis Agent (synthesizes all data)
    ↓
Comprehensive Report
```

### Example Usage

```python
from analyze_trade_agent import analyze_trade_agent

# Research is automatically included when appropriate
analysis = analyze_trade_agent(
    trading_idea="long position in AI infrastructure stocks",
    symbol="NVDA"
)

# The agent will:
# 1. Get price/technical data (Polygon.io) - Fast
# 2. Get sentiment (FinBERT) - Fast
# 3. IF appropriate: Deep research on AI infrastructure market
# 4. Synthesize everything into trading recommendation
```

## Performance & Costs

### Timing
- **Standard analysis** (no research): 10-30 seconds
- **With research (TESTING MODE)**: ~1-2 minutes per research tool
- **With research (PRODUCTION MODE)**: 2-5 minutes per research tool

### API Costs
**TESTING MODE** (recommended for development):
- **Tavily**: ~3-5 searches per research (~0.5% of free tier per research)
- **OpenAI**: ~$0.03-$0.05 per research report
- **Time**: 1-2 minutes per research

**PRODUCTION MODE** (comprehensive research):
- **Tavily**: Free tier = 1,000 searches/month (~10-15 searches per research)
- **OpenAI**: ~$0.10-0.30 per research report (depending on model and depth)
- **Polygon.io**: No extra cost (uses existing API calls)
- **Time**: 2-5 minutes per research

### Rate Limits
- Research tools run independently (no rate limit conflicts)
- Can run multiple research tools in parallel if needed

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY="sk-..."           # For LLM synthesis
TAVILY_API_KEY="tvly-dev-..."     # For web search

# Testing Mode (RECOMMENDED during development/testing)
GPT_RESEARCHER_TESTING_MODE="true"   # Default: true (faster, cheaper)
# Set to "false" for production (comprehensive research)

# Fine-grained control (optional, auto-set by testing mode)
MAX_ITERATIONS="1"                    # Default: 1 (testing), 3 (production)
MAX_SEARCH_RESULTS_PER_QUERY="3"     # Default: 3 (testing), 5 (production)
TOTAL_WORDS="800"                     # Default: 800 (testing), 1200 (production)
MAX_SCRAPER_WORKERS="10"              # Default: 10 (testing), 15 (production)

# Already configured
POLYGON_API_KEY="..."             # For market data
```

### Testing Mode vs Production Mode

**TESTING MODE (Default)** - Recommended during development:
- **Iterations**: 1 (vs 3 in production)
- **Search Results**: 3 per query (vs 5 in production)
- **Report Length**: 800 words (vs 1200 in production)
- **Time**: 1-2 minutes per research
- **Cost**: ~$0.03-0.05 per research (3-6x cheaper)
- **Quality**: Good enough for testing and validation
- **Use when**: Testing integration, debugging, rapid iteration, cost-conscious exploration

**PRODUCTION MODE** - Use for final analysis:
- **Iterations**: 3 (deeper, more comprehensive)
- **Search Results**: 5 per query (broader coverage)
- **Report Length**: 1200 words (detailed)
- **Time**: 2-5 minutes per research
- **Cost**: ~$0.10-0.30 per research
- **Quality**: Comprehensive, publication-ready
- **Use when**: Critical investment decisions, client reports, final recommendations

**How to Switch:**
```bash
# In your .env file:

# Testing (fast/cheap) - DEFAULT
GPT_RESEARCHER_TESTING_MODE="true"

# Production (comprehensive)
GPT_RESEARCHER_TESTING_MODE="false"
```

### Planner Guidance

The planner is configured to:
- Prioritize fast market data tools
- Only add research tools when truly valuable
- Avoid research for time-sensitive trading decisions
- Use research for strategic/analytical queries

## Examples

### Example 1: Quick Trade Analysis (No Research)
```python
idea = "swing trade TSLA"
symbol = "TSLA"
# Result: Uses only fast tools (price, RSI, MACD, sentiment)
# Time: ~15 seconds
```

### Example 2: Deep Analysis (With Research)
```python
idea = "comprehensive analysis of EV market leaders"
symbol = "TSLA"
# Result: Adds gpt_researcher_sector_trends for EV market
# Time: ~3 minutes
# Output: Includes deep sector research report
```

### Example 3: Macro-Informed Trading
```python
idea = "tech stocks under rising rate environment"
symbol = "AAPL"
# Result: Adds gpt_researcher_economic_indicators for rate analysis
# Time: ~3 minutes
# Output: Includes macro research on rate impact
```

## Advanced: Manual Research Tool Usage

You can also use research tools directly:

```python
from research_tools import gpt_researcher_market_research_tool_fn

state = {"tool_results": {}}
args = {
    "query": "quantum computing market size and key players",
    "report_type": "research_report"
}

state = gpt_researcher_market_research_tool_fn(state, args)
result = state["tool_results"]["gpt_researcher_market_research"]

print(result["report"])  # Full research report
print(result["sources"])  # List of sources used
print(result["costs"])    # Token usage and costs
```

## Report Types

GPT-Researcher supports three report types:

1. **`research_report`** (default): Comprehensive analysis with synthesis
2. **`resource_report`**: List of resources and summaries
3. **`outline_report`**: Structured outline for further research

## Troubleshooting

### "gpt-researcher not installed"
```bash
pip install gpt-researcher
```

### "TAVILY_API_KEY not set"
- Get free key at https://tavily.com
- Add to `.env` file

### Research taking too long / too expensive
**Problem**: Research taking 3-5 minutes and using too many API calls during testing.

**Solution**: Enable testing mode (should be default):
```bash
# In .env file
GPT_RESEARCHER_TESTING_MODE="true"
```

This reduces:
- Iterations: 3 → 1
- Search results: 5 → 3 per query
- Words: 1200 → 800
- Time: 2-5 min → 1-2 min
- Cost: $0.10-0.30 → $0.03-0.05

**When to use production mode**: Only for final, critical analyses.

### Research quality not good enough
**Problem**: Testing mode giving insufficient results.

**Solution**: Either:
1. Run specific research again with production settings:
```bash
MAX_ITERATIONS=3 MAX_SEARCH_RESULTS_PER_QUERY=5 python analyze_trade_agent.py
```

2. Switch to production mode permanently:
```bash
# In .env file
GPT_RESEARCHER_TESTING_MODE="false"
```

### "Import could not be resolved"
- This is a linting warning (expected before installation)
- Will work after `pip install gpt-researcher`

## Future Enhancements

Potential additions for next phase:
- Custom research sources (local documents, specific websites)
- Research result caching
- Parallel research execution
- Research quality scoring
- Multi-agent research workflows

## References

- GPT-Researcher GitHub: https://github.com/assafelovic/gpt-researcher
- Documentation: https://docs.gptr.dev/
- Tavily API: https://tavily.com
