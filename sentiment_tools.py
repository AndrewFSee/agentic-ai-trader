# sentiment_tools.py
# Finviz News Scraping + FinBERT Sentiment Analysis + Topic Sentiment
# Part of the Agentic AI Trader migration to Polygon.io

import os
import sqlite3
import datetime as dt
from typing import Callable, Any, Dict, TypedDict, List

from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

# Optional ML imports for FinBERT sentiment
_TORCH_WARNING_SHOWN = False
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    # OSError can occur with DLL loading issues on Windows
    # Only show warning once instead of on every import
    if not _TORCH_WARNING_SHOWN:
        logger.info(f"Note: torch/transformers not available ({type(e).__name__}). FinBERT sentiment will use basic classification.")
        _TORCH_WARNING_SHOWN = True

# =============================================================================
# Tool Registry (uses unified ToolSpec)
# =============================================================================

from tool_registry import ToolSpec

TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(tool: ToolSpec):
    TOOL_REGISTRY[tool["name"]] = tool


def get_all_tools() -> List[dict]:
    """
    Returns all registered tools in OpenAI function calling format.
    Used by the planner to understand available sentiment analysis tools.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in TOOL_REGISTRY.values()
    ]


def get_tool_function(name: str) -> Callable[[dict, dict], dict] | None:
    """Returns the tool function by name, or None if not found."""
    spec = TOOL_REGISTRY.get(name)
    return spec["fn"] if spec else None


# =============================================================================
# Topic Sentiment Model (keyword-based, no ML dependency)
# =============================================================================

from topic_sentiment_model import (
    TopicSentimentModel,
    EarningsTopicModel,
    HIGH_SIGNAL_TOPICS,
)

# Lazy-initialized singleton instances
_topic_model: TopicSentimentModel | None = None
_earnings_model: EarningsTopicModel | None = None

# News database path (from auto_researcher project)
NEWS_DB_PATH = os.getenv(
    "NEWS_DB_PATH",
    r"C:\Users\Andrew\projects\auto_researcher\data\news.db",
)


def _get_topic_model() -> TopicSentimentModel:
    """Lazy singleton for TopicSentimentModel."""
    global _topic_model
    if _topic_model is None:
        _topic_model = TopicSentimentModel()
    return _topic_model


def _get_earnings_model() -> EarningsTopicModel:
    """Lazy singleton for EarningsTopicModel."""
    global _earnings_model
    if _earnings_model is None:
        _earnings_model = EarningsTopicModel()
    return _earnings_model


# =============================================================================
# FinBERT Sentiment Model (Lazy Loading)
# =============================================================================

_FINBERT_MODEL_NAME = "ProsusAI/finbert"
_finbert_tokenizer = None
_finbert_model = None
_FINBERT_AVAILABLE = _TORCH_AVAILABLE  # only available if torch is installed


def _ensure_finbert_loaded() -> bool:
    """
    Lazily load FinBERT model/tokenizer the first time we need it.
    Downloads from HuggingFace on first run and caches locally.
    Requires torch and transformers to be installed.
    """
    global _finbert_tokenizer, _finbert_model, _FINBERT_AVAILABLE

    if not _TORCH_AVAILABLE:
        return False
    
    if not _FINBERT_AVAILABLE:
        return False

    if _finbert_model is not None and _finbert_tokenizer is not None:
        return True

    try:
        logger.info("Downloading FinBERT model '%s' from HuggingFace (first time only)...", _FINBERT_MODEL_NAME)
        _finbert_tokenizer = AutoTokenizer.from_pretrained(
            _FINBERT_MODEL_NAME,
            force_download=False,  # use cache if available
            resume_download=True    # resume partial downloads
        )
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(
            _FINBERT_MODEL_NAME,
            force_download=False,
            resume_download=True
        )
        logger.info("FinBERT model loaded successfully")
        return True
    except Exception as e:
        logger.warning("FinBERT model could not be loaded: %s", e)
        logger.warning("Falling back to 'unknown' sentiment for news analysis")
        _FINBERT_AVAILABLE = False
        return False


def _finbert_sentiment(text: str) -> dict:
    """
    Run FinBERT sentiment on a short text. If FinBERT is unavailable,
    return a neutral/unknown sentiment instead of crashing.
    """
    if not _ensure_finbert_loaded():
        return {
            "label": "unknown",
            "probs": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
        }

    inputs = _finbert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    labels = ["negative", "neutral", "positive"]
    label_idx = int(torch.tensor(probs).argmax())
    label = labels[label_idx]
    return {
        "label": label,
        "probs": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        },
    }


# =============================================================================
# Finviz News Scraping
# =============================================================================

def _get_recent_news_from_finviz(
    symbol: str, window_days: int = 3, max_articles: int = 20
) -> List[dict]:
    """
    Scrape recent news for a symbol from finviz.com.
    Returns a list of:
      {"headline": ..., "summary": "", "url": ..., "published_at": iso_timestamp}
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    news_table = soup.find("table", id="news-table")
    if not news_table:
        return []

    now = dt.datetime.utcnow()
    cutoff = now - dt.timedelta(days=window_days)

    articles: List[dict] = []
    last_date_str = None

    for row in news_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        date_time_str = cols[0].get_text(strip=True)
        headline_link = cols[1].find("a")
        headline = (
            headline_link.get_text(strip=True)
            if headline_link
            else cols[1].get_text(strip=True)
        )
        href = headline_link["href"] if headline_link else None

        # finviz formats date/time as "Dec-12-24 08:35AM" or "08:35AM"
        if " " in date_time_str:
            date_part, time_part = date_time_str.split(" ", 1)
            last_date_str = date_part
        else:
            time_part = date_time_str  # reuse last_date_str

        if last_date_str is None:
            continue

        try:
            dt_str = f"{last_date_str} {time_part}"
            pub_dt = dt.datetime.strptime(dt_str, "%b-%d-%y %I:%M%p")
        except ValueError:
            continue

        if pub_dt < cutoff:
            # Older than window; Finviz is reverse-chronological
            break

        articles.append(
            {
                "headline": headline,
                "summary": "",
                "url": href,
                "published_at": pub_dt.isoformat() + "Z",
            }
        )
        if len(articles) >= max_articles:
            break

    return articles


# =============================================================================
# News Sentiment Tool (Finviz + FinBERT)
# =============================================================================

def news_sentiment_finviz_finbert_tool_fn(state: dict, args: dict) -> dict:
    """
    Scrape recent news for a symbol from Finviz, run FinBERT sentiment, and store summary in
    state["tool_results"]["news_sentiment_finviz_finbert"].
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args["symbol"]
    window_days = int(args.get("window_days", 3))
    max_articles = int(args.get("max_articles", 20))

    articles = _get_recent_news_from_finviz(
        symbol, window_days=window_days, max_articles=max_articles
    )

    results = []
    pos_scores: List[float] = []

    # Topic model for classification
    topic_model = _get_topic_model()

    for art in articles:
        headline = art["headline"] or ""
        text = headline
        if not text.strip():
            continue
        s = _finbert_sentiment(text)

        # Topic classification (instant, no ML)
        topic_result = topic_model.analyze_article(text)

        results.append(
            {
                "headline": headline,
                "summary": "",
                "url": art["url"],
                "published_at": art["published_at"],
                "sentiment": s["label"],
                "probs": s["probs"],
                "topic": topic_result.topic.primary_topic,
                "topic_adjusted_sentiment": round(topic_result.topic_adjusted_sentiment, 3),
                "topic_signal": topic_result.trading_signal,
            }
        )
        pos_scores.append(s["probs"]["positive"])

    if pos_scores:
        avg_pos = sum(pos_scores) / len(pos_scores)
    else:
        avg_pos = None

    if avg_pos is None:
        agg_label = "unknown"
    elif avg_pos > 0.6:
        agg_label = "bullish"
    elif avg_pos < 0.4:
        agg_label = "bearish"
    else:
        agg_label = "mixed"

    # Topic aggregation for the scraped headlines
    topic_agg = topic_model.analyze_articles(
        [{"title": r["headline"], "sentiment_score": r["probs"].get("positive", 0.5) * 2 - 1} for r in results],
        symbol,
    )

    state["tool_results"]["news_sentiment_finviz_finbert"] = {
        "symbol": symbol.upper(),
        "window_days": window_days,
        "num_articles": len(results),
        "avg_positive_prob": avg_pos,
        "aggregate_label": agg_label,
        "topic_composite_score": round(topic_agg.composite_score, 3) if topic_agg else None,
        "topic_signal": topic_agg.composite_signal if topic_agg else None,
        "topic_confidence": round(topic_agg.confidence, 2) if topic_agg else None,
        "topic_breakdown": {
            t: {"count": topic_agg.topic_counts.get(t, 0), "avg_sentiment": round(topic_agg.topic_sentiment.get(t, 0.0), 3)}
            for t in topic_agg.topic_counts
        } if topic_agg else {},
        "high_signal_alerts": [
            a for a in [
                "LITIGATION ALERT" if topic_agg and topic_agg.litigation_alert else None,
                "MANAGEMENT ALERT" if topic_agg and topic_agg.management_alert else None,
                "EARNINGS SURPRISE" if topic_agg and topic_agg.earnings_surprise_detected else None,
            ] if a
        ] if topic_agg else [],
        "articles": results[:10],
    }
    return state


# Register the sentiment tool
register_tool(
    {
        "name": "news_sentiment_finviz_finbert",
        "description": (
            "Scrape recent news for a symbol from finviz.com and run FinBERT sentiment analysis "
            "on the headlines. Useful for understanding near-term news sentiment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "window_days": {"type": "integer", "default": 3},
                "max_articles": {"type": "integer", "default": 20},
            },
            "required": ["symbol"],
        },
        "fn": news_sentiment_finviz_finbert_tool_fn,
    }
)


# =============================================================================
# Topic Sentiment from News Database (deep history)
# =============================================================================

def topic_sentiment_newsdb_tool_fn(state: dict, args: dict) -> dict:
    """Query the local news database for recent articles on *symbol*, classify
    each by topic, compute topic-adjusted sentiment, and return an aggregated
    signal with dual interpretation (event-driven + contrarian).

    Stores results in state["tool_results"]["topic_sentiment_newsdb"].
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args["symbol"].upper()
    window_days = int(args.get("window_days", 7))

    db_path = NEWS_DB_PATH
    if not os.path.exists(db_path):
        state["tool_results"]["topic_sentiment_newsdb"] = {
            "symbol": symbol,
            "error": f"News database not found at {db_path}. Set NEWS_DB_PATH env var.",
        }
        return state

    cutoff = (dt.datetime.now() - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT title, snippet, published_date, source, url,
                   sentiment_score, sentiment_label,
                   fulltext_sentiment_score, fulltext_sentiment_label
            FROM articles
            WHERE UPPER(ticker) = ?
              AND published_date >= ?
            ORDER BY published_date DESC
            LIMIT 200
            """,
            (symbol, cutoff),
        ).fetchall()
        conn.close()
    except Exception as exc:
        state["tool_results"]["topic_sentiment_newsdb"] = {
            "symbol": symbol,
            "error": f"Database query failed: {exc}",
        }
        return state

    if not rows:
        state["tool_results"]["topic_sentiment_newsdb"] = {
            "symbol": symbol,
            "window_days": window_days,
            "num_articles": 0,
            "note": "No articles found in the database for this ticker/window.",
        }
        return state

    # Build article dicts for the topic model
    articles_for_model = []
    for r in rows:
        # prefer fulltext sentiment if available, else headline sentiment
        fb_score = r["fulltext_sentiment_score"] if r["fulltext_sentiment_score"] is not None else r["sentiment_score"]
        articles_for_model.append({
            "title": r["title"] or "",
            "snippet": r["snippet"] or "",
            "published_date": r["published_date"],
            "source": r["source"],
            "url": r["url"],
            "sentiment_score": fb_score if fb_score is not None else 0.0,
        })

    topic_model = _get_topic_model()
    agg = topic_model.analyze_articles(articles_for_model, symbol)

    # Get per-article topic details for the most recent 10
    recent_details = []
    for art_dict in articles_for_model[:10]:
        tr = topic_model.analyze_article(art_dict["title"])
        recent_details.append({
            "title": art_dict["title"][:120],
            "date": art_dict["published_date"],
            "source": art_dict["source"],
            "topic": tr.topic.primary_topic,
            "topic_adjusted_sentiment": round(tr.topic_adjusted_sentiment, 3),
            "signal": tr.trading_signal,
        })

    # Build topic breakdown dict from AggregatedTopicSignal fields
    topic_breakdown = {}
    if agg:
        for topic in agg.topic_counts:
            topic_breakdown[topic] = {
                "count": agg.topic_counts.get(topic, 0),
                "avg_sentiment": round(agg.topic_sentiment.get(topic, 0.0), 3),
                "signal": agg.topic_signals.get(topic, "neutral"),
                "is_high_signal": topic in HIGH_SIGNAL_TOPICS,
            }

    # Build alerts list from boolean flags
    alerts = []
    if agg:
        if agg.litigation_alert:
            alerts.append("LITIGATION ALERT: Multiple negative litigation articles detected")
        if agg.management_alert:
            alerts.append("MANAGEMENT ALERT: Negative management/leadership news detected")
        if agg.earnings_surprise_detected:
            alerts.append("EARNINGS SURPRISE: Strong earnings sentiment detected (positive or negative)")

    result = {
        "symbol": symbol,
        "window_days": window_days,
        "num_articles": len(articles_for_model),
        "composite_score": round(agg.composite_score, 3) if agg else None,
        "signal": agg.composite_signal if agg else None,
        "confidence": round(agg.confidence, 2) if agg else None,
        "topic_breakdown": topic_breakdown,
        "alerts": alerts,
        "interpretation": {
            "event_driven": "Fresh high-signal news (litigation, earnings, M&A, management) tends to have momentum — act with the signal for 3-7 days.",
            "contrarian": "Accumulated positive/negative sentiment over longer windows (>14d) tends to mean-revert — consider fading extreme readings.",
        },
        "recent_articles": recent_details,
    }

    state["tool_results"]["topic_sentiment_newsdb"] = result
    return state


register_tool(
    {
        "name": "topic_sentiment_newsdb",
        "description": (
            "Query the local news database (210k+ S&P500 articles, 2020-2026) for recent articles "
            "on a symbol, classify each headline by topic (earnings, M&A, litigation, etc.), and "
            "compute topic-adjusted sentiment. Returns aggregated signal with topic breakdown and "
            "high-signal alerts. Best for deep sentiment analysis with historical context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. AAPL)"},
                "window_days": {
                    "type": "integer",
                    "default": 7,
                    "description": "Lookback window in days (7=event-driven, 20=contrarian)",
                },
            },
            "required": ["symbol"],
        },
        "fn": topic_sentiment_newsdb_tool_fn,
    }
)


# =============================================================================
# Earnings Topic Signal (highest alpha)
# =============================================================================

def earnings_topic_signal_tool_fn(state: dict, args: dict) -> dict:
    """Filter news to earnings/analyst topics and check FinBERT agreement to
    produce a high-conviction tradeable signal. This is the highest-alpha
    sentiment feature (IC ≈ +0.021, 52× raw FinBERT).

    Stores results in state["tool_results"]["earnings_topic_signal"].
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args["symbol"].upper()
    window_days = int(args.get("window_days", 14))

    db_path = NEWS_DB_PATH
    if not os.path.exists(db_path):
        state["tool_results"]["earnings_topic_signal"] = {
            "symbol": symbol,
            "error": f"News database not found at {db_path}. Set NEWS_DB_PATH env var.",
        }
        return state

    cutoff = (dt.datetime.now() - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT title, snippet, published_date, source, url,
                   sentiment_score, sentiment_label,
                   fulltext_sentiment_score, fulltext_sentiment_label
            FROM articles
            WHERE UPPER(ticker) = ?
              AND published_date >= ?
            ORDER BY published_date DESC
            LIMIT 200
            """,
            (symbol, cutoff),
        ).fetchall()
        conn.close()
    except Exception as exc:
        state["tool_results"]["earnings_topic_signal"] = {
            "symbol": symbol,
            "error": f"Database query failed: {exc}",
        }
        return state

    if not rows:
        state["tool_results"]["earnings_topic_signal"] = {
            "symbol": symbol,
            "window_days": window_days,
            "num_articles": 0,
            "is_tradeable": False,
            "note": "No articles found for this ticker/window.",
        }
        return state

    # Build article dicts with FinBERT scores from the database
    articles_for_model = []
    for r in rows:
        fb_score = r["fulltext_sentiment_score"] if r["fulltext_sentiment_score"] is not None else r["sentiment_score"]
        # Convert FinBERT probability-style score to label for agreement check
        if fb_score is not None:
            if fb_score > 0.1:
                fb_label = "positive"
            elif fb_score < -0.1:
                fb_label = "negative"
            else:
                fb_label = "neutral"
        else:
            fb_label = "neutral"

        articles_for_model.append({
            "title": r["title"] or "",
            "snippet": r["snippet"] or "",
            "published_date": r["published_date"],
            "source": r["source"],
            "url": r["url"],
            "sentiment_score": fb_score if fb_score is not None else 0.0,
            "sentiment_label": fb_label,
        })

    earnings_model = _get_earnings_model()
    signal = earnings_model.analyze_news(articles_for_model, symbol)

    if signal.earnings_articles == 0:
        state["tool_results"]["earnings_topic_signal"] = {
            "symbol": symbol,
            "window_days": window_days,
            "num_articles": len(articles_for_model),
            "earnings_articles_found": 0,
            "is_tradeable": False,
            "note": "No earnings/analyst-related articles found in this window.",
        }
        return state

    result = {
        "symbol": symbol,
        "window_days": window_days,
        "num_articles_total": len(articles_for_model),
        "earnings_articles_found": signal.earnings_articles,
        "direction": signal.direction,
        "confidence": round(signal.confidence, 2),
        "is_tradeable": signal.is_tradeable,
        "expected_alpha_bps": round(signal.expected_alpha * 10000) if signal.expected_alpha else None,
        "high_conviction": signal.high_conviction,
        "topic_sentiment": round(signal.topic_sentiment, 3),
        "finbert_sentiment": round(signal.finbert_sentiment, 3) if signal.finbert_sentiment is not None else None,
        "models_agree": signal.models_agree,
        "earnings_surprise": signal.earnings_surprise,
        "note": (
            "This is the highest-alpha sentiment signal (IC≈+0.021). "
            "High-conviction signals historically show ~2.1% 5-day alpha. "
            "Use as a confirming factor alongside price/technical analysis."
        ),
    }

    state["tool_results"]["earnings_topic_signal"] = result
    return state


register_tool(
    {
        "name": "earnings_topic_signal",
        "description": (
            "Analyze earnings and analyst-related news with FinBERT agreement checking "
            "to produce a high-conviction tradeable signal. This is the highest-alpha "
            "sentiment feature (IC ≈ +0.021, 52× raw FinBERT). Returns direction, "
            "confidence, expected alpha, and whether the signal is tradeable. "
            "Requires local news database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. AAPL)"},
                "window_days": {
                    "type": "integer",
                    "default": 14,
                    "description": "Lookback window in days (default 14 for earnings cycle)",
                },
            },
            "required": ["symbol"],
        },
        "fn": earnings_topic_signal_tool_fn,
    }
)
