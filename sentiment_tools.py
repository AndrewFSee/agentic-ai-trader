# sentiment_tools.py
# Finviz News Scraping + FinBERT Sentiment Analysis
# Part of the Agentic AI Trader migration to Polygon.io

import os
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
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Only show warning once instead of on every import
    if not _TORCH_WARNING_SHOWN:
        logger.info("Note: torch/transformers not installed. FinBERT sentiment will use basic classification. Install with: pip install torch transformers")
        _TORCH_WARNING_SHOWN = True

# =============================================================================
# Generic Tool Registry
# =============================================================================

class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: dict  # OpenAI function calling schema
    fn: Callable[[dict, dict], dict]


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

    for art in articles:
        headline = art["headline"] or ""
        text = headline
        if not text.strip():
            continue
        s = _finbert_sentiment(text)
        results.append(
            {
                "headline": headline,
                "summary": "",
                "url": art["url"],
                "published_at": art["published_at"],
                "sentiment": s["label"],
                "probs": s["probs"],
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

    state["tool_results"]["news_sentiment_finviz_finbert"] = {
        "symbol": symbol.upper(),
        "window_days": window_days,
        "num_articles": len(results),
        "avg_positive_prob": avg_pos,
        "aggregate_label": agg_label,
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
