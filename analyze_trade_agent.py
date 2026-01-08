# analyze_trade_agent.py

from typing import List, Dict, Any
from datetime import datetime
import warnings
import logging
import os
import time
import concurrent.futures

# Suppress all deprecation warnings (including LangChain)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# If a project venv was created (.venv_research), prefer running inside it to avoid ABI/import issues
# DISABLED FOR PAPER TRADING: Use conda base environment which has all dependencies
# try:
#     import sys
#     import os
#     venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv_research"))
#     if os.path.isdir(venv_path):
#         venv_python = os.path.join(venv_path, "Scripts", "python.exe")
#         # If we're not already running under the venv python, re-exec using it, unless in WSL.
#         if not sys.executable.lower().startswith(os.path.normcase(venv_path).lower()):
#             if os.path.exists(venv_python) and sys.executable.lower() != os.path.normcase(venv_python).lower():
#                 if not os.getenv("WSL_DISTRO_NAME"):
#                     print(f"Re-executing under project venv: {venv_python}")
#                     os.execv(venv_python, [venv_python] + sys.argv)
# except Exception:
#     # Best-effort only; if this fails, continue with the current interpreter
#     pass

# Compatibility shim: some Python envs (conda/mixed installs) can cause
# stdlib importlib.metadata.version(...) to raise or return None for some
# distributions. That breaks packages (transformers / huggingface_hub) which
# call importlib.metadata.version(). Patch the stdlib function at runtime to
# fall back to the importlib_metadata backport when needed.
try:
    import importlib.metadata as _stdlib_meta
    import importlib_metadata as _backport_meta
    _orig_version_fn = _stdlib_meta.version

    def _safe_version(name: str) -> str | None:
        try:
            return _orig_version_fn(name)
        except Exception:
            try:
                return _backport_meta.version(name)
            except Exception:
                return None

    # Replace the stdlib version function with the safe wrapper
    _stdlib_meta.version = _safe_version
except Exception:
    # If anything goes wrong, allow the import path to continue and fail
    # naturally; this shim is best-effort for problematic envs.
    pass

from openai import OpenAI as OpenAIClient
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from agent_tools import run_tools
from planner import plan_tools

# LlamaIndex config
INDEX_DIR = "db/books"
EMBED_MODEL = "text-embedding-3-large"
# Use a stable, lower-latency model by default to avoid stalls if a premium model is unavailable.
DECISION_MODEL = "gpt-5.1"

# Configure basic logging. Set AGENT_LOG_LEVEL=DEBUG to enable verbose logs.
logging.basicConfig(level=os.getenv("AGENT_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class RAGIndex:
    """Wrapper around LlamaIndex to provide similarity_search interface."""
    
    def __init__(self, index):
        self._index = index
        self._retriever = index.as_retriever(similarity_top_k=6)
    
    def similarity_search(self, query: str, k: int = 6) -> List:
        """Retrieve k most similar documents for the query."""
        self._retriever = self._index.as_retriever(similarity_top_k=k)
        nodes = self._retriever.retrieve(query)
        # Convert NodeWithScore to a doc-like format
        return [_NodeAsDoc(n) for n in nodes]


class _NodeAsDoc:
    """Adapter to make LlamaIndex nodes look like LangChain docs."""
    
    def __init__(self, node_with_score):
        self.node = node_with_score.node
        self.score = node_with_score.score
        self.page_content = self.node.get_content()
        self.metadata = dict(self.node.metadata) if self.node.metadata else {}


def load_vectorstore() -> Any:
    """Load the LlamaIndex vector store from disk."""
    from types import SimpleNamespace

    try:
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
        logger.info("Loaded LlamaIndex from %s", INDEX_DIR)
        return RAGIndex(index)
    except Exception as e:
        logger.warning("Could not load LlamaIndex: %s. Falling back to empty index.", e)

        def _empty_search(q, k=6):
            return []

        return SimpleNamespace(similarity_search=_empty_search)


def _format_docs(docs: List) -> str:
    parts = []
    for d in docs:
        book_name = d.metadata.get("book_name", d.metadata.get("file_name", d.metadata.get("source", "unknown")))
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        parts.append(
            f"[Book: {book_name}, page {page}]\n{d.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _format_price_summary(price_result: Dict[str, Any] | None) -> str:
    if not price_result:
        return "No price data available."

    if price_result.get("error"):
        return f"Price data unavailable. Error: {price_result['error']}"

    symbol = price_result["symbol"]
    interval = price_result["interval"]
    num_bars = price_result["num_bars"]
    
    # Get latest bar data
    latest_close = price_result.get("latest_close")
    latest_open = price_result.get("latest_open")
    latest_high = price_result.get("latest_high")
    latest_low = price_result.get("latest_low")
    latest_date = price_result.get("latest_date")
    date_range = price_result.get("date_range", {})

    lines = [f"Symbol: {symbol}", f"Interval: {interval}", f"Bars: {num_bars}"]
    if date_range:
        lines.append(f"Date range: {date_range.get('from')} to {date_range.get('to')}")
    
    # Calculate price change over the period
    bars = price_result.get("bars", [])
    if len(bars) >= 2:
        first_close = bars[0].get('c')
        last_close = bars[-1].get('c')
        if first_close and last_close:
            pct_change = ((last_close - first_close) / first_close) * 100
            lines.append(f"Period change: {pct_change:+.2f}% (from ${first_close:.2f} to ${last_close:.2f})")
    
    if latest_close is not None:
        lines.append(f"\nLatest bar ({latest_date}):")
        lines.append(f"  Open: ${latest_open:.2f}" if latest_open else "  Open: N/A")
        lines.append(f"  High: ${latest_high:.2f}" if latest_high else "  High: N/A")
        lines.append(f"  Low: ${latest_low:.2f}" if latest_low else "  Low: N/A")
        lines.append(f"  Close: ${latest_close:.2f}")
        
        # Calculate intraday range
        if latest_high and latest_low:
            range_pct = ((latest_high - latest_low) / latest_low) * 100
            lines.append(f"  Day Range: {range_pct:.2f}%")
    
    # Add recent price action (last 10 bars summary)
    if len(bars) >= 10:
        lines.append(f"\nRecent 10 days:")
        recent_bars = bars[-10:]
        for bar in recent_bars:
            bar_date = datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d')
            bar_close = bar.get('c', 0)
            bar_volume = bar.get('v', 0)
            lines.append(f"  {bar_date}: ${bar_close:.2f} (vol: {bar_volume:,.0f})")
    
    # Volume analysis
    volume_analysis = price_result.get("volume_analysis", {})
    if volume_analysis and volume_analysis.get("avg_volume"):
        lines.append(f"\nVolume Analysis:")
        lines.append(f"  Average volume: {volume_analysis['avg_volume']:,.0f}")
        lines.append(f"  Latest volume: {volume_analysis['latest_volume']:,.0f}")
        lines.append(f"  Volume ratio: {volume_analysis['volume_ratio']:.2f}x")
        lines.append(f"  Condition: {volume_analysis['volume_condition']}")
        
        spikes = volume_analysis.get("recent_volume_spikes", [])
        if spikes:
            lines.append(f"  Recent volume spikes:")
            for spike in spikes:
                lines.append(
                    f"    {spike['date']}: {spike['ratio']:.2f}x avg "
                    f"(close: ${spike['close']:.2f})"
                )
    
    # ATR analysis
    atr_data = price_result.get("atr", {})
    if atr_data and atr_data.get("atr_value"):
        atr_value = atr_data['atr_value']
        lines.append(f"\nATR Analysis (volatility for stop placement):")
        lines.append(f"  ATR({atr_data['window']}): ${atr_value:.2f}")
        lines.append(f"  ATR as % of price: {(atr_value / latest_close * 100):.2f}%" if latest_close else "")
        lines.append(f"  Suggested stop distances:")
        lines.append(f"    Conservative (2.0x ATR): ${atr_value * 2.0:.2f}")
        lines.append(f"    Moderate (1.5x ATR): ${atr_value * 1.5:.2f}")
        lines.append(f"    Tight (1.0x ATR): ${atr_value:.2f} (risk of noise stop-out)")

    return "\n".join(lines)


def _format_news_summary(news_result: Dict[str, Any] | None) -> str:
    if not news_result:
        return "No news data available."

    if news_result.get("num_articles", 0) == 0:
        return "No recent news articles found in the window."

    symbol = news_result["symbol"]
    agg_label = news_result["aggregate_label"]
    avg_pos = news_result.get("avg_positive_prob")

    lines = [
        f"Symbol: {symbol}",
        f"Aggregate FinBERT sentiment label: {agg_label}",
        (
            f"Average positive probability: {avg_pos:.3f}"
            if avg_pos is not None
            else ""
        ),
        f"Articles analyzed: {news_result['num_articles']}",
        "",
        "Sample articles (headline | date | FinBERT label):",
    ]

    for art in news_result.get("articles", [])[:5]:
        lines.append(
            f"- {art['headline']} | {art['published_at']} "
            f"| FinBERT: {art['sentiment']}"
        )

    return "\n".join([l for l in lines if l])


def _format_technical_summary(tool_results: Dict[str, Any]) -> str:
    """
    Format RSI, MACD, SMA, EMA, Bollinger Bands into a compact text block.
    NOTE: ATR is now in price data formatter, not here.
    """
    rsi = tool_results.get("polygon_technical_rsi")
    macd = tool_results.get("polygon_technical_macd")
    sma = tool_results.get("polygon_technical_sma")
    ema = tool_results.get("polygon_technical_ema")
    bb = tool_results.get("bollinger_bands")

    lines: List[str] = []

    if not rsi and not macd and not sma and not ema and not bb:
        return "No technical indicator data available."

    if rsi:
        if rsi.get("error"):
            lines.append(f"RSI: unavailable (Error: {rsi['error']})")
        else:
            lines.append(
                f"RSI (period={rsi['window']}): "
                f"{rsi['latest_value']:.2f} ({rsi['condition']})"
            )

    if macd:
        if macd.get("error"):
            lines.append(f"MACD: unavailable (Error: {macd['error']})")
        else:
            lines.append(
                f"MACD: {macd['latest_macd']:.4f}, "
                f"Signal={macd['latest_signal']:.4f}, Hist={macd['latest_histogram']:.4f} "
                f"({macd['signal_type']})"
            )

    if sma:
        if sma.get("error"):
            lines.append(f"SMA: unavailable (Error: {sma['error']})")
        else:
            lines.append(
                f"SMA (period={sma['window']}): "
                f"{sma['latest_value']:.2f}"
            )
    
    if ema:
        if ema.get("error"):
            lines.append(f"EMA: unavailable (Error: {ema['error']})")
        else:
            lines.append(
                f"EMA (period={ema['window']}): "
                f"{ema['latest_value']:.2f}"
            )
    
    if bb:
        if bb.get("error"):
            lines.append(f"Bollinger Bands: unavailable (Error: {bb['error']})")
        else:
            lines.append(
                f"Bollinger Bands (period={bb['period']}, {bb['num_std']}σ): "
                f"Price={bb['latest_price']:.2f}, "
                f"Middle={bb['middle_band']:.2f}, "
                f"Upper={bb['upper_band']:.2f}, "
                f"Lower={bb['lower_band']:.2f}"
            )
            lines.append(
                f"  %B={bb['percent_b']:.3f}, Bandwidth={bb['bandwidth_pct']:.2f}%, "
                f"Position: {bb['position']}"
            )

    return "\n".join(lines) if lines else "No technical indicator data available."


def _format_ticker_details(ticker_result: Dict[str, Any] | None) -> str:
    """Format Polygon ticker details (company info, market cap, sector)."""
    if not ticker_result:
        return "No ticker details available."

    if ticker_result.get("error"):
        return f"Ticker details unavailable. Error: {ticker_result['error']}"

    lines = []
    lines.append(f"Symbol: {ticker_result.get('symbol', 'N/A')}")
    name = ticker_result.get("name")
    if name:
        lines.append(f"Name: {name}")
    
    market_cap = ticker_result.get("market_cap")
    if market_cap:
        lines.append(f"Market Cap: ${market_cap:,.0f}")
    
    sector = ticker_result.get("sic_description")
    if sector:
        lines.append(f"Industry: {sector}")
    
    description = ticker_result.get("description")
    if description:
        # Truncate long descriptions
        desc_short = description[:200] + "..." if len(description) > 200 else description
        lines.append(f"Description: {desc_short}")

    return "\n".join(lines)


def _format_snapshot(snapshot_result: Dict[str, Any] | None) -> str:
    """Format Polygon snapshot (real-time price, day's range, volume)."""
    if not snapshot_result:
        return "No snapshot data available."

    if snapshot_result.get("error"):
        return f"Snapshot unavailable. Error: {snapshot_result['error']}"

    lines = []
    lines.append(f"Symbol: {snapshot_result.get('symbol', 'N/A')}")
    
    current_price = snapshot_result.get("current_price")
    if current_price:
        lines.append(f"Current Price: ${current_price:.2f}")
    
    day_change = snapshot_result.get("day_change")
    day_change_pct = snapshot_result.get("day_change_percent")
    if day_change is not None and day_change_pct is not None:
        lines.append(f"Day Change: ${day_change:+.2f} ({day_change_pct:+.2f}%)")
    
    day_high = snapshot_result.get("day_high")
    day_low = snapshot_result.get("day_low")
    if day_high and day_low:
        lines.append(f"Day Range: ${day_low:.2f} - ${day_high:.2f}")
    
    volume = snapshot_result.get("volume")
    if volume:
        lines.append(f"Volume: {volume:,}")

    return "\n".join(lines) if lines else "Fundamental data present but no key fields extracted."


def _format_fundamentals(fund_result: Dict[str, Any] | None) -> str:
    """Format quick fundamentals fetched by fetch_fundamentals tool."""
    if not fund_result:
        return "No fundamentals available."

    if fund_result.get("error"):
        return f"Fundamentals unavailable. Error: {fund_result['error']}"

    lines = []
    lines.append(f"Symbol: {fund_result.get('symbol')}")
    long_name = None
    # Try to read longName from Yahoo fallback
    if isinstance(fund_result.get('raw_sources'), list):
        for s in fund_result['raw_sources']:
            if s.get('source') == 'alphavantage_overview' and isinstance(s.get('data'), dict):
                long_name = s['data'].get('Name')
                break

    if long_name:
        lines.append(f"Name: {long_name}")

    # Key numeric fields
    for field, label, fmt in [
        (fund_result.get('marketCap'), 'Market Cap', '${:,.0f}'),
        (fund_result.get('pe_trailing'), 'Trailing P/E', '{:.2f}'),
        (fund_result.get('pe_forward'), 'Forward P/E', '{:.2f}'),
        (fund_result.get('peg'), 'PEG', '{:.3f}'),
        (fund_result.get('eps'), 'EPS (ttm)', '{:.2f}'),
        (fund_result.get('profitMargin'), 'Profit Margin', '{:.3f}'),
    ]:
        if field is not None:
            try:
                lines.append(f"{label}: " + fmt.format(field))
            except Exception:
                lines.append(f"{label}: {field}")

    # Any warnings
    if fund_result.get('warnings'):
        lines.append('Warnings: ' + '; '.join(fund_result.get('warnings')))

    return "\n".join(lines)


def _format_earnings(earnings_result: Dict[str, Any] | None) -> str:
    """Format quarterly earnings data with growth rates."""
    if not earnings_result:
        return "No earnings data available."
    
    if earnings_result.get("error"):
        return f"Earnings data unavailable. Error: {earnings_result['error']}"
    
    lines = []
    lines.append(f"Symbol: {earnings_result.get('symbol')}")
    lines.append(f"Latest Quarter: {earnings_result.get('latest_quarter')} (ended {earnings_result.get('latest_quarter_end')})")
    
    # Latest quarter metrics
    latest_rev = earnings_result.get('latest_revenue')
    latest_eps = earnings_result.get('latest_eps')
    latest_income = earnings_result.get('latest_net_income')
    
    if latest_rev:
        lines.append(f"Latest Revenue: ${latest_rev:,.0f}")
    if latest_eps:
        lines.append(f"Latest EPS (diluted): ${latest_eps:.2f}")
    if latest_income:
        lines.append(f"Latest Net Income: ${latest_income:,.0f}")
    
    # Average growth rates
    avg_growth = earnings_result.get('average_growth', {})
    lines.append("\nAverage Growth Rates:")
    
    if avg_growth.get('avg_revenue_qoq_growth') is not None:
        lines.append(f"  Revenue (QoQ): {avg_growth['avg_revenue_qoq_growth']:+.2f}%")
    if avg_growth.get('avg_revenue_yoy_growth') is not None:
        lines.append(f"  Revenue (YoY): {avg_growth['avg_revenue_yoy_growth']:+.2f}%")
    if avg_growth.get('avg_eps_qoq_growth') is not None:
        lines.append(f"  EPS (QoQ): {avg_growth['avg_eps_qoq_growth']:+.2f}%")
    if avg_growth.get('avg_eps_yoy_growth') is not None:
        lines.append(f"  EPS (YoY): {avg_growth['avg_eps_yoy_growth']:+.2f}%")
    
    # Recent quarterly trend (last 4 quarters)
    quarterly = earnings_result.get('quarterly_data', [])
    if quarterly and len(quarterly) >= 4:
        lines.append("\nRecent Quarterly Performance (last 4 quarters):")
        for q in quarterly[:4]:
            period = q.get('period', 'N/A')
            rev = q.get('revenue')
            eps = q.get('eps_diluted')
            rev_qoq = q.get('revenue_qoq_growth')
            eps_qoq = q.get('eps_qoq_growth')
            
            q_line = f"  {period}: "
            if rev:
                q_line += f"Rev=${rev/1e9:.2f}B"
            if eps:
                q_line += f", EPS=${eps:.2f}"
            if rev_qoq is not None:
                q_line += f", Rev growth={rev_qoq:+.1f}%"
            if eps_qoq is not None:
                q_line += f", EPS growth={eps_qoq:+.1f}%"
            lines.append(q_line)
    
    return "\n".join(lines)


def _format_macro_summary(fred_result: Dict[str, Any] | None) -> str:
    """Format FRED macro indicators into a compact summary for the decision LLM."""
    if not fred_result:
        return "No macro indicators available."

    if fred_result.get("error"):
        return f"Macro indicators unavailable. Error: {fred_result['error']}"

    lines = ["Macro & liquidity indicators from FRED:"]

    obs = fred_result.get("observations", {})
    dgs10 = obs.get("DGS10", {}).get("latest")
    dgs2 = obs.get("DGS2", {}).get("latest")
    ted = obs.get("TEDRATE", {}).get("latest")
    m2 = obs.get("M2SL", {}).get("latest")
    bbb = obs.get("BAMLC0A0CM", {}).get("latest")

    if dgs10:
        lines.append(f"  10y Treasury (DGS10) {dgs10.get('date')}: {dgs10.get('value')}%")
    if dgs2:
        lines.append(f"  2y Treasury (DGS2) {dgs2.get('date')}: {dgs2.get('value')}%")
    if "term_spread_10y_2y" in fred_result:
        lines.append(f"  10y-2y term spread: {fred_result.get('term_spread_10y_2y'):+.2f} percentage points")
    if ted:
        lines.append(f"  TED spread (TEDRATE) {ted.get('date')}: {ted.get('value')}")
    if bbb:
        lines.append(f"  BofA BBB OAS (BAMLC0A0CM) {bbb.get('date')}: {bbb.get('value')}")
    if m2:
        lines.append(f"  M2 Money Stock (M2SL) {m2.get('date')}: {m2.get('value')}")

    return "\n".join(lines)


def _format_ml_predictions(ml_result: Dict[str, Any] | None) -> str:
    """Format ML model predictions for the decision LLM."""
    if not ml_result:
        return "ML predictions not available."
    
    if ml_result.get("error"):
        return f"ML predictions: {ml_result['error']}"
    
    symbol = ml_result.get("symbol", "N/A")
    horizon = ml_result.get("horizon", "N/A")
    predictions = ml_result.get("predictions", {})
    consensus = ml_result.get("consensus", {})
    best_model_info = ml_result.get("best_model", {})
    horizon_perf = ml_result.get("horizon_performance", {})
    
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("🤖 MACHINE LEARNING MODEL PREDICTIONS")
    lines.append(f"Symbol: {symbol} | Horizon: {horizon} | Models Trained on 25 Stocks (2020-2025)")
    lines.append("=" * 80)
    
    # Individual model predictions with detailed breakdown
    lines.append("\n📊 INDIVIDUAL MODEL FORECASTS:")
    lines.append("-" * 80)
    
    for model_name, pred_data in predictions.items():
        if pred_data.get("error"):
            lines.append(f"\n❌ {model_name}: ERROR - {pred_data['error']}")
            continue
            
        direction = pred_data.get("prediction", "N/A")
        prob_data = pred_data.get("probability", {})
        perf = pred_data.get("performance", {})
        
        # Get confidence from probability data
        if prob_data and isinstance(prob_data, dict):
            up_prob = prob_data.get("up", 0)
            down_prob = prob_data.get("down", 0)
            confidence = prob_data.get("confidence", max(up_prob, down_prob))
        else:
            up_prob = down_prob = confidence = 0
        
        # Add directional emoji
        direction_icon = "📈" if "UP" in direction else "📉"
        
        lines.append(f"\n{direction_icon} **{model_name}**:")
        lines.append(f"   Prediction: {direction}")
        lines.append(f"   Confidence: {confidence:.1%}")
        lines.append(f"   Probability Breakdown: UP {up_prob:.1%} | DOWN {down_prob:.1%}")
        
        if perf:
            sharpe = perf.get('sharpe', 0)
            win_rate = perf.get('win_rate', 0)
            lines.append(f"   Historical Performance: Sharpe Ratio {sharpe:.2f}, Win Rate {win_rate:.1%}")
    
    # Consensus - the most important part
    if consensus:
        lines.append("\n" + "=" * 80)
        lines.append("🎯 CONSENSUS VERDICT:")
        lines.append("=" * 80)
        
        direction = consensus.get('direction', 'N/A')
        up_votes = consensus.get('up_votes', 0)
        total_votes = consensus.get('total_votes', 0)
        down_votes = total_votes - up_votes
        confidence = consensus.get('confidence', 0)
        
        # Add consensus emoji based on strength
        if confidence >= 0.75:
            strength = "STRONG ⚡"
            emoji = "🔴" if "DOWN" in direction else "🟢"
        elif confidence >= 0.5:
            strength = "MODERATE ⚠️"
            emoji = "🟡"
        else:
            strength = "WEAK ❓"
            emoji = "⚪"
        
        lines.append(f"\n{emoji} Consensus Direction: {direction}")
        lines.append(f"   Agreement Level: {confidence:.0%} ({strength})")
        lines.append(f"   Vote Distribution: {up_votes} models UP, {down_votes} models DOWN (Total: {total_votes})")
        
        # Add interpretation for the agent
        lines.append(f"\n💡 Interpretation:")
        if confidence >= 0.75:
            lines.append(f"   - STRONG consensus = High confidence. Models are aligned.")
            lines.append(f"   - Suggested sizing: NORMAL position size (e.g., 2-3% of account)")
        elif confidence >= 0.5:
            lines.append(f"   - MODERATE consensus = Some disagreement among models.")
            lines.append(f"   - Suggested sizing: REDUCED by 30-50% (e.g., 1-1.5% of account)")
        else:
            lines.append(f"   - WEAK consensus = High disagreement. Low confidence signal.")
            lines.append(f"   - Suggested sizing: SKIP trade or MINIMAL exposure (<1% of account)")
    
    # Best model for this horizon
    if best_model_info:
        lines.append("\n" + "-" * 80)
        lines.append("🏆 BEST PERFORMING MODEL (This Horizon):")
        model_name = best_model_info.get('name', 'N/A')
        best_pred = best_model_info.get('prediction', 'N/A')
        lines.append(f"   Model: {model_name}")
        lines.append(f"   Prediction: {best_pred}")
        
        best_perf = best_model_info.get('performance', {})
        if best_perf:
            sharpe = best_perf.get('sharpe', 0)
            ret = best_perf.get('return', 0)
            win_rate = best_perf.get('win_rate', 0)
            lines.append(f"   Historical Stats: Sharpe {sharpe:.2f} | Return {ret:.1%} | Win Rate {win_rate:.1%}")
    
    # Context and limitations
    lines.append("\n" + "-" * 80)
    lines.append("⚠️  IMPORTANT CONTEXT & LIMITATIONS:")
    lines.append("-" * 80)
    if horizon_perf and horizon_perf.get('models'):
        best_name = horizon_perf.get('best_model', 'N/A')
        lines.append(f"   • Best model historically for {horizon}: {best_name}")
    lines.append(f"   • Training data: 25 stocks, 2020-2025 (includes COVID crash, bull market, 2022 bear)")
    lines.append(f"   • Sharpe ratios range: 0.78 to 1.52 across all models")
    lines.append(f"   • These are SHORT-TERM tactical signals, not long-term forecasts")
    lines.append(f"   • ML is FEATURE-DRIVEN, not news-reactive (may miss recent catalysts)")
    lines.append(f"   • Always cross-check ML with: technicals, sentiment, regime, fundamentals")
    
    lines.append("=" * 80 + "\n")
    
    return "\n".join(lines)


def _format_regime_wasserstein(wass_result: Dict[str, Any] | None) -> str:
    """Format Wasserstein regime detection results for the decision LLM."""
    if not wass_result:
        return "Wasserstein regime detection not available."
    
    if wass_result.get("error"):
        return f"Wasserstein regime: {wass_result['error']}"
    
    symbol = wass_result.get("symbol", "N/A")
    regime_name = wass_result.get("regime_name", "Unknown")
    confidence = wass_result.get("confidence", "unknown")
    mmd_ratio = wass_result.get("mmd_quality_ratio")
    interpretation = wass_result.get("interpretation", "")
    note = wass_result.get("note", "")
    
    lines = [f"Wasserstein Volatility Regime for {symbol}:"]
    lines.append(f"  Regime: {regime_name}")
    lines.append(f"  Confidence: {confidence}")
    if mmd_ratio is not None:
        lines.append(f"  MMD Quality: {mmd_ratio:.3f} {'(poor separation)' if mmd_ratio < 1.1 else '(good separation)' if mmd_ratio > 1.5 else '(moderate)'}")
    lines.append(f"  Interpretation: {interpretation}")
    if note:
        lines.append(f"  Note: {note}")
    
    return "\n".join(lines)


def _format_regime_hmm(hmm_result: Dict[str, Any] | None) -> str:
    """Format HMM regime detection results for the decision LLM."""
    if not hmm_result:
        return "HMM regime detection not available."
    
    if hmm_result.get("error"):
        return f"HMM regime: {hmm_result['error']}"
    
    symbol = hmm_result.get("symbol", "N/A")
    regime = hmm_result.get("regime")  # The integer regime index
    regime_name = hmm_result.get("regime_name", "Unknown")
    confidence = hmm_result.get("confidence")
    persistence = hmm_result.get("persistence_probability")
    probabilities = hmm_result.get("probabilities", {})
    interpretation = hmm_result.get("interpretation", "")
    note = hmm_result.get("note", "")
    
    lines = [f"HMM Trend Regime for {symbol}:"]
    lines.append(f"  Current Regime: {regime_name} (state {regime})")
    if confidence is not None:
        lines.append(f"  Model Confidence: {confidence:.2f}")
    if persistence is not None:
        lines.append(f"  Persistence Probability: {persistence:.2f} ({'stable' if persistence > 0.85 else 'unstable' if persistence < 0.70 else 'moderate'})")
    if probabilities:
        lines.append("  Current State Probabilities (Forward Filter):")
        for regime_label, prob in probabilities.items():
            marker = " ← CURRENT" if regime_label == regime_name else ""
            lines.append(f"    {regime_label}: {prob:.2%}{marker}")
    lines.append(f"  Interpretation: {interpretation}")
    if note:
        lines.append(f"  Note: {note}")
    
    return "\n".join(lines)


def _format_regime_consensus(consensus_result: Dict[str, Any] | None) -> str:
    """Format regime consensus check results for the decision LLM."""
    if not consensus_result:
        return "Regime consensus check not available."
    
    if consensus_result.get("error"):
        return f"Regime consensus: {consensus_result['error']}"
    
    agreement = consensus_result.get("agreement")
    confidence_level = consensus_result.get("confidence_level", "UNKNOWN")
    wass_regime = consensus_result.get("wasserstein_regime", "N/A")
    hmm_regime = consensus_result.get("hmm_regime", "N/A")
    recommendation = consensus_result.get("recommendation", "")
    
    lines = ["Regime Consensus Check:"]
    lines.append(f"  Wasserstein says: {wass_regime}")
    lines.append(f"  HMM says: {hmm_regime}")
    lines.append(f"  Agreement: {'YES' if agreement else 'NO'}")
    lines.append(f"  Confidence Level: {confidence_level}")
    lines.append(f"  Recommendation: {recommendation}")
    
    return "\n".join(lines)


def analyze_trade_agent(
    trading_idea: str,
    symbol: str | None,
    vectordb: Any | None = None,
    k_main: int = 6,
    k_rules: int = 6,
) -> str:
    """
    Full agent:
      - RAG over trading books
      - Planner chooses tools (price, technicals, fundamentals, news)
      - Tools executed
      - Final decision based on books + tools
    """
    if vectordb is None:
        vectordb = load_vectorstore()

    # 1) RAG: book excerpts
    idea_query = trading_idea if not symbol else f"{symbol} {trading_idea}"
    idea_docs = vectordb.similarity_search(idea_query, k=k_main)
    rules_docs = vectordb.similarity_search(
        "risk management position sizing stop loss max risk per trade drawdown "
        "trading rules trading plan psychology discipline emotions losing streak",
        k=k_rules,
    )
    # New: ML/modeling suggestions query
    ml_docs = vectordb.similarity_search(
        "machine learning statistical models predictive models feature engineering "
        "random forest neural networks LSTM regression classification ensemble methods "
        "hidden markov models gaussian processes time series forecasting regime detection "
        "factor models alpha generation signal extraction backtesting validation",
        k=k_rules,  # Same k as rules (6 chunks)
    )

    idea_context = _format_docs(idea_docs)
    rules_context = _format_docs(rules_docs)
    ml_context = _format_docs(ml_docs)

    # 2) Planner: decide which tools to use
    tool_calls = plan_tools(trading_idea, symbol=symbol)
    
    # DEBUG: Show which tools were selected
    tool_names = [tc.get("tool_name", tc.get("name", "unknown")) for tc in tool_calls]
    print(f"\n[DEBUG] Planner selected tools: {', '.join(tool_names)}\n")

    # 3) Tool hub: execute tools
    state: Dict[str, Any] = {"messages": [], "tool_results": {}}
    state = run_tools(state, tool_calls)

    tool_results = state["tool_results"]
    
    # DEBUG: Show which tools returned results
    result_keys = list(tool_results.keys())
    print(f"[DEBUG] Tool results available: {', '.join(result_keys)}")
    if "ml_prediction" in tool_results:
        ml_res = tool_results["ml_prediction"]
        if ml_res.get("error"):
            print(f"[DEBUG] ML prediction error: {ml_res.get('error')}")
        else:
            print(f"[DEBUG] ML prediction SUCCESS for {ml_res.get('symbol')} ({ml_res.get('horizon')}d)")
    else:
        print("[DEBUG] ml_prediction was NOT called or returned no results")
    print()

    price_summary_text = _format_price_summary(
        tool_results.get("polygon_price_data")
    )
    news_summary_text = _format_news_summary(
        tool_results.get("news_sentiment_finviz_finbert")
    )
    tech_summary_text = _format_technical_summary(tool_results)
    ticker_details_text = _format_ticker_details(
        tool_results.get("polygon_ticker_details")
    )
    snapshot_text = _format_snapshot(
        tool_results.get("polygon_snapshot")
    )
    macro_summary_text = _format_macro_summary(
        tool_results.get("fred_macro_indicators")
    )
    fundamentals_text = _format_fundamentals(
        tool_results.get("fetch_fundamentals")
    )
    earnings_text = _format_earnings(
        tool_results.get("polygon_earnings")
    )
    # Only format regime data if available (not errors)
    regime_wasserstein_result = tool_results.get("regime_wasserstein")
    regime_hmm_result = tool_results.get("regime_hmm")
    regime_consensus_result = tool_results.get("regime_consensus")
    
    has_wasserstein = regime_wasserstein_result and not regime_wasserstein_result.get("error")
    has_hmm = regime_hmm_result and not regime_hmm_result.get("error")
    has_consensus = regime_consensus_result and not regime_consensus_result.get("error")
    
    regime_wasserstein_text = _format_regime_wasserstein(regime_wasserstein_result) if has_wasserstein else ""
    regime_hmm_text = _format_regime_hmm(regime_hmm_result) if has_hmm else ""
    regime_consensus_text = _format_regime_consensus(regime_consensus_result) if has_consensus else ""
    
    # Format ML predictions if available
    ml_result = tool_results.get("ml_prediction")
    has_ml = ml_result and not ml_result.get("error")
    ml_predictions_text = _format_ml_predictions(ml_result) if has_ml else ""

    # 4) Final decision LLM (using OpenAI SDK directly)
    symbol_str = symbol or "N/A"

    system_prompt = """
You are a trading coach and decision support assistant.

You MUST base all conclusions on:
- the provided trading book excerpts,
- the provided price/market data summaries,
- the provided technical indicator summaries (RSI, MACD, SMA, EMA, Bollinger Bands),
- the provided company details (market cap, sector, description),
- the provided real-time snapshot (current price, day's range),
- the provided news + FinBERT sentiment summary (if available),
- the provided REGIME DETECTION results (Wasserstein volatility-based and/or HMM trend-based),
- and the provided ML MODEL PREDICTIONS (if available).

Use the books to justify how you interpret:
- price and trend,
- technical indicators,
- company context,
- sentiment and narrative,
- REGIME CLASSIFICATION (volatility regimes and trend regimes),
- ML MODEL PREDICTIONS (consensus and confidence levels),
- and how you construct risk management and sizing.

REGIME DETECTION GUIDANCE:
- Wasserstein detects VOLATILITY regimes (Low/Med/High Vol). Use for position sizing and stop placement.
- HMM detects TREND regimes (Bearish/Sideways/Bullish). Use for directional bias and transition risks.
- When both methods AGREE, confidence is HIGH. When they DISAGREE, it signals UNCERTAINTY.
- In high volatility regimes: reduce position size 30-50%, widen stops (1.5-2x ATR).
- In uncertain regimes (disagreement): reduce size further or wait for clarity per Van Tharp.
- Check persistence probability for HMM: >0.85 is stable, <0.70 warns of potential transition.

ML MODEL PREDICTIONS GUIDANCE:
- ML predictions come from trained models (Sharpe 1.34-1.52) on 25 stocks, 2020-2025 data.
- CONSENSUS INTERPRETATION:
  * 75-100% agreement = STRONG signal (high confidence, models aligned)
  * 50-74% agreement = WEAK signal (moderate confidence, some disagreement)
  * <50% agreement = UNCLEAR signal (low confidence, significant model disagreement)
- NEVER USE ML ALONE: Always combine with price action, technicals, regime, and sentiment.
- ML provides directional bias but does NOT incorporate recent news (feature-driven, not news-reactive).
- Best for swing trades (5-10 day horizons). Performance degrades in novel market regimes.
- Use consensus strength to size positions: STRONG = normal size, WEAK = reduce 30%, UNCLEAR = skip.

Never guarantee profits. Always highlight risk and uncertainty.
Be explicit when the data is incomplete and what additional information would be needed.
""".strip()

    # Build user prompt with conditional regime sections
    user_prompt_parts = [f"""
TRADING IDEA
------------
Symbol: {symbol_str}
Idea: {trading_idea}

RECENT MARKET DATA (Polygon.io)
--------------------------------
{price_summary_text}

COMPANY DETAILS
---------------
{ticker_details_text}

COMPANY FUNDAMENTALS
--------------------
{fundamentals_text}

QUARTERLY EARNINGS & GROWTH
---------------------------
{earnings_text}

REAL-TIME SNAPSHOT
------------------
{snapshot_text}

TECHNICAL INDICATORS (RSI / MACD / SMA / EMA / Bollinger Bands)
----------------------------------------------------------------
{tech_summary_text}

RECENT NEWS + FINVIZ + FINBERT SENTIMENT
----------------------------------------
{news_summary_text}

MACRO & LIQUIDITY INDICATORS
----------------------------
{macro_summary_text}
"""]
    
    # Only add regime sections if data is available
    if has_wasserstein or has_hmm or has_consensus:
        regime_section = "\nREGIME DETECTION RESULTS\n------------------------\n"
        if has_wasserstein:
            regime_section += f"\nWasserstein (Volatility-Based):\n{regime_wasserstein_text}\n"
        if has_hmm:
            regime_section += f"\nHMM (Trend-Based):\n{regime_hmm_text}\n"
        if has_consensus:
            regime_section += f"\nConsensus Check:\n{regime_consensus_text}\n"
        user_prompt_parts.append(regime_section)
    
    # Add ML predictions section if available
    if has_ml:
        ml_section = f"\nML MODEL PREDICTIONS\n--------------------\n{ml_predictions_text}\n"
        user_prompt_parts.append(ml_section)
    
    user_prompt_parts.append(f"""
BOOK EXCERPTS – IDEA-RELATED
-----------------------------
{idea_context}

BOOK EXCERPTS – RISK MANAGEMENT & PSYCHOLOGY
--------------------------------------------
{rules_context}

BOOK EXCERPTS – ML/STATISTICAL MODELING INSIGHTS
------------------------------------------------
{ml_context}

TASK
----
Using ONLY the information above:
""")
    
    # Join all parts to create final user prompt
    user_prompt = "".join(user_prompt_parts)
    
    # Continue with rest of task section
    user_prompt += """

1. PRICE & TREND:
   - Discuss the recent price action and context (trend, volatility, any obvious regimes).
   - If price data is missing or incomplete, clearly state that and do not guess.

2. TECHNICALS:
   - Interpret RSI (momentum / overbought / oversold) and MACD (trend strength, crossovers).
   - Use ATR to discuss volatility and implications for stop distances and position sizing.
   - Explicitly connect your interpretations to the principles in the books.

3. FUNDAMENTALS:
   - Summarize valuation (P/E, PEG), profitability (margins, ROE), and growth trends.
   - Use the quarterly earnings data to assess revenue and EPS growth rates (QoQ and YoY).
   - Identify whether growth is accelerating, decelerating, or stable.
   - Explain whether fundamentals support or conflict with the trade idea, referencing the books' guidance
     on valuation and quality where relevant.
"""
    
    # Add regime analysis section only if regime data is available
    next_section = 4
    if has_wasserstein or has_hmm or has_consensus:
        user_prompt += f"""
{next_section}. REGIME ANALYSIS:
   - Analyze the REGIME DETECTION results (Wasserstein volatility and/or HMM trend).
   - If both models were called, check if they AGREE or DISAGREE:
     * Agreement = HIGH confidence in the regime classification
     * Disagreement = UNCERTAINTY signal, often precedes regime transitions
   - Interpret HMM persistence: >0.85 (stable), 0.70-0.85 (moderate), <0.70 (unstable/transition risk)
   - Use regime info to inform position sizing and stop placement:
     * High volatility regime → reduce size 30-50%, widen stops 1.5-2x ATR
     * Uncertain regime (disagreement) → reduce size further or wait for clarity
   - Reference the trading books' guidance on regime-based risk management
"""
        next_section += 1
    
    user_prompt += f"""
{next_section}. NEWS & SENTIMENT:
   - Summarize recent news flow and FinBERT sentiment.
   - State whether YOU agree or disagree with the aggregate sentiment label and why.
"""
    next_section += 1
    
    # Add ML predictions analysis section if available
    if has_ml:
        user_prompt += f"""
{next_section}. ML MODEL PREDICTIONS (MANDATORY DEDICATED SECTION):
   ⚠️  IMPORTANT: Create a FULL, DETAILED section analyzing the ML predictions. This is NOT optional.
   
   Your analysis MUST include:
   
   a) INDIVIDUAL MODEL REVIEW:
      - Discuss EACH model's prediction (Random Forest, XGBoost, Logistic Regression, Decision Tree)
      - Note their confidence levels and probability breakdowns
      - Highlight any models that disagree with the consensus
   
   b) CONSENSUS INTERPRETATION:
      - State the consensus direction clearly (STRONG UP, WEAK DOWN, etc.)
      - Explain the agreement level percentage (e.g., "100% agreement = all 4 models agree")
      - Interpret the strength (STRONG ≥75%, MODERATE 50-74%, WEAK <50%):
        * STRONG consensus = High confidence signal, models aligned
        * MODERATE consensus = Some disagreement, proceed with caution
        * WEAK consensus = High disagreement, consider skipping trade
   
   c) CROSS-VALIDATION WITH OTHER SIGNALS:
      - Compare ML predictions with technical indicators:
        * Do RSI, MACD, Bollinger Bands confirm or conflict with ML?
      - Compare with regime analysis:
        * Is the trend regime (HMM) aligned with ML direction?
        * Does volatility regime (Wasserstein) support or oppose the trade?
      - Compare with sentiment:
        * Does FinBERT sentiment align with ML predictions?
        * Are there recent news catalysts that ML might have missed?
   
   d) POSITION SIZING GUIDANCE (Based on ML Consensus):
      - STRONG consensus (≥75%) → Normal position size (2-3% of account)
      - MODERATE consensus (50-74%) → Reduced size by 30-50% (1-1.5% of account)
      - WEAK consensus (<50%) → Skip trade or minimal exposure (<1% of account)
   
   e) ML LIMITATIONS & CONTEXT:
      - Note: ML is FEATURE-DRIVEN (technical/fundamental features), NOT news-reactive
      - ML may MISS recent catalysts, breaking news, or sudden market shifts
      - Models trained on 2020-2025 data (includes COVID, bull market, 2022 bear)
      - Historical Sharpe ratios: 0.78 to 1.52 (solid but not perfect)
      - Short-term tactical signals (3-10 day horizons), NOT long-term forecasts
   
   f) FINAL ML VERDICT:
      - Summarize: Do ML predictions support or oppose the trade idea?
      - How does ML consensus affect your overall confidence in this trade?
      - Should ML predictions be weighted heavily here or taken with caution?
"""
        next_section += 1
    
    ml_guidance = ", ML predictions," if has_ml else ""
    regime_guidance = " and REGIME CLASSIFICATION" if (has_wasserstein or has_hmm) else ""
    regime_risk_guidance = " ADJUSTED FOR REGIME" if (has_wasserstein or has_hmm) else ""
    regime_risk_details = "\n   - If regime models disagree, recommend reducing position size or waiting for consensus." if has_consensus else ""
    regime_verdict_note = " (especially if regime models disagree)" if (has_wasserstein or has_hmm) else ""
    
    user_prompt += f"""
{next_section}. EDGE ASSESSMENT:
   - Evaluate whether the proposed trade has a reasonable edge according to the books, given:
     - price/technicals,
     - fundamentals,
     - sentiment{ml_guidance}{regime_guidance}.
   - If data is missing (e.g., no fundamentals or no news), explicitly factor that into your conclusion.
   - Synthesize ALL inputs: Do price, technicals, ML, regime, and sentiment ALIGN or CONFLICT?
   - Strongest edge when multiple signals confirm each other.
"""
    next_section += 1
    
    ml_risk_guidance = " + ML CONSENSUS" if has_ml else ""
    user_prompt += f"""
{next_section}. RISK MANAGEMENT{regime_risk_guidance}{ml_risk_guidance}:
   - Provide detailed risk management guidance:
     - reasonable position size as a % of account (a range){'adjusted for regime volatility/uncertainty' if (has_wasserstein or has_hmm) else ''}{'and ML consensus strength' if has_ml else ''}
     - stop loss ideas using price + ATR{'adjusted for regime (wider in high-vol regimes)' if (has_wasserstein or has_hmm) else ''}
     - risk/reward considerations (aimed R:R ratio, e.g., 2:1).{regime_risk_details}
     {'- Adjust size based on ML consensus: STRONG = normal, WEAK = reduce 30-50%, UNCLEAR = skip or minimal' if has_ml else ''}
"""
    next_section += 1
    
    user_prompt += f"""
{next_section}. VERDICT:
   - Provide a verdict in EXACTLY one of these forms (put this on a separate line prefixed with 'VERDICT:'):
     - VERDICT: NOT ATTRACTIVE based on the books and current data
     - VERDICT: UNCLEAR / NEEDS MORE INFORMATION{regime_verdict_note}
     - VERDICT: ATTRACTIVE IF STRICT RULES ARE FOLLOWED{f'- Your verdict MUST consider the regime analysis - uncertain regimes should lean toward UNCLEAR or NOT ATTRACTIVE.' if (has_wasserstein or has_hmm) else ''}
"""
    next_section += 1
    
    user_prompt += f"""
{next_section}. MODEL IMPROVEMENT SUGGESTIONS:
   - Based on the ML/statistical modeling excerpts from the books, suggest 2-4 specific quantitative models 
     or analysis techniques that could improve this trading system.
   - For each suggestion, explain:
     * What model/technique from the books (name the book and concept)
     * What problem it would solve (e.g., better entry timing, regime detection, risk estimation)
     * What data it would need (price, volume, sentiment, fundamentals, alternative data)
     * What advanced features (indicators, engineered features) it might use (e.g., wavelets, acceleration, reversion speeds, etc.)
     * Estimated implementation complexity (low/medium/high)
   - Focus on actionable, well-documented approaches from the provided book excerpts.
"""
    next_section += 1
    
    user_prompt += f"""
{next_section}. Additional Data Needed:
   - If your verdict is "UNCLEAR / NEEDS MORE INFORMATION", specify exactly what additional data
     would be required to make a confident decision (e.g., more price history, fundamental reports,
     alternative data sources, expert analysis). 

10. CHECKLIST:
   - Provide a short, actionable checklist the trader must confirm before taking the trade.

If the data is insufficient, explicitly say so and lean toward "UNCLEAR".
""".strip()

    # Call OpenAI directly (no LangChain)
    def _call_openai(system: str, user: str) -> str:
        """Call OpenAI chat completion API with timeout handling."""
        start_ts = time.time()
        logger.info("LLM call starting (model=%s)", DECISION_MODEL)
        
        client = OpenAIClient()
        timeout_sec = int(os.getenv('LLM_CALL_TIMEOUT_SEC', '120'))
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    lambda: client.chat.completions.create(
                        model=DECISION_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        timeout=timeout_sec,
                    )
                )
                try:
                    response = fut.result(timeout=timeout_sec + 10)
                except concurrent.futures.TimeoutError:
                    fut.cancel()
                    raise TimeoutError(f"LLM call timed out after {timeout_sec} seconds")
            
            content = response.choices[0].message.content
            logger.info("LLM call succeeded in %.2fs", time.time() - start_ts)
            return content
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise

    return _call_openai(system_prompt, user_prompt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run analyze_trade_agent in interactive or non-interactive mode.")
    parser.add_argument("--idea", type=str, help="Trading idea text (if provided runs non-interactively)")
    parser.add_argument("--symbol", type=str, help="Ticker symbol (e.g., NVDA)")
    parser.add_argument("--out", type=str, default=None, help="Optional output path to save analysis")
    args = parser.parse_args()

    vectordb = load_vectorstore()

    # Non-interactive mode: idea provided via CLI
    if args.idea:
        print("Running non-interactive analyze_trade_agent...")
        analysis = analyze_trade_agent(trading_idea=args.idea, symbol=args.symbol, vectordb=vectordb)
        if args.out:
            out_path = args.out
        else:
            out_path = f"research_reports/agent_analysis_{args.symbol or 'NA'}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.md"
            import os
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(f"Saved analysis to {out_path}")
    else:
        print("[Trading Agent] Ctrl+C or type 'q' to exit.")
        while True:
            idea = input("\nDescribe your trading idea (or 'q' to quit): ").strip()
            if idea.lower() in {"q", "quit", "exit"}:
                break
            symbol = input("Symbol (e.g., AAPL): ").strip() or None

            print("\n[Analyzing...]\n")
            analysis = analyze_trade_agent(
                trading_idea=idea,
                symbol=symbol,
                vectordb=vectordb,
            )
            print("=== ANALYSIS ===\n")
            print(analysis)
