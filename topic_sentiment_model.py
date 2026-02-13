# topic_sentiment_model.py
"""
Topic-Based Sentiment Model (ported from auto_researcher).

Classifies news into financial topics and computes topic-specific sentiment.
Research shows certain topics (litigation, M&A, earnings) have stronger
predictive power than generic sentiment.

Key finding from backtest:
    Generic FinBERT IC ≈ +0.0004 (noise)
    Earnings-topic sentiment IC = +0.021 (52× improvement)

Academic basis:
    Garcia (2013): Media sentiment predicts returns, especially during recessions
    Boudoukh et al. (2019): News topics have different return predictability
    Tetlock et al. (2008): Negative words in firm-specific news predict earnings
    Loughran & McDonald (2011): Finance-specific sentiment dictionaries
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# TOPIC DEFINITIONS
# =============================================================================

TOPIC_DEFINITIONS = {
    "litigation": {
        "keywords": [
            "lawsuit", "sued", "sues", "suing", "litigation", "litigate",
            "plaintiff", "defendant", "settlement", "settles", "settled",
            "verdict", "jury", "court", "judge", "ruling", "appeal",
            "sec", "ftc", "doj", "antitrust", "investigation", "investigated",
            "probe", "probing", "subpoena", "indictment", "charged", "charges",
            "violation", "penalty", "fine", "fined", "sanctions", "compliance",
            "regulatory", "regulators", "enforcement", "consent decree",
            "fraud", "fraudulent", "scandal", "misconduct", "whistleblower",
            "accounting irregularities", "restatement", "restated",
        ],
        "sentiment_multiplier": 1.5,   # Negative sentiment hits 1.5× harder
        "base_impact": -0.02,          # Inherent negative bias
        "decay_days": 30,
    },
    "earnings": {
        "keywords": [
            "earnings", "revenue", "profit", "loss", "eps", "quarterly results",
            "beat", "beats", "miss", "missed", "guidance", "outlook", "forecast",
            "revenue growth", "margin", "margins", "gross margin", "operating margin",
            "same-store sales", "comparable sales", "comps", "revenue decline",
            "profit warning", "lowered guidance", "raised guidance", "preannounce",
            "fiscal quarter", "fiscal year", "annual report", "quarterly report",
        ],
        "sentiment_multiplier": 1.2,
        "base_impact": 0.0,
        "decay_days": 5,
    },
    "mna": {
        "keywords": [
            "merger", "acquisition", "acquire", "acquired", "acquires", "acquiring",
            "takeover", "bid", "bidding", "bidder", "buyout", "lbo",
            "target", "deal", "transaction", "combine", "combination",
            "spinoff", "spin-off", "divestiture", "divest", "divesting",
            "ipo", "spac", "going private", "take private", "strategic review",
            "activist investor", "proxy fight", "board seat",
        ],
        "sentiment_multiplier": 1.0,
        "base_impact": 0.0,
        "decay_days": 10,
    },
    "management": {
        "keywords": [
            "ceo", "chief executive", "cfo", "chief financial",
            "coo", "chief operating", "cto", "chief technology",
            "chairman", "board of directors", "director", "executive",
            "resign", "resigned", "resignation", "steps down", "stepping down",
            "retire", "retired", "retirement", "depart", "departure",
            "appoint", "appointed", "appointment", "hire", "hired", "hiring",
            "promoted", "promotion", "succession", "successor",
            "founder", "leadership", "management change", "restructuring",
        ],
        "sentiment_multiplier": 1.3,
        "base_impact": -0.005,
        "decay_days": 20,
    },
    "product": {
        "keywords": [
            "product", "launch", "launches", "launched", "announce", "announced",
            "unveil", "unveiled", "reveal", "revealed", "introduce", "introduced",
            "release", "released", "rollout", "new model", "new version",
            "innovation", "innovative", "patent", "patented", "technology",
            "breakthrough", "disruption", "disruptive", "fda approval", "approved",
            "recall", "recalled", "defect", "defective", "safety issue",
            "clinical trial", "trial results", "phase 3", "phase 2",
        ],
        "sentiment_multiplier": 0.8,
        "base_impact": 0.01,
        "decay_days": 60,
    },
    "analyst": {
        "keywords": [
            "upgrade", "upgraded", "downgrade", "downgraded", "rating",
            "buy rating", "sell rating", "hold rating", "outperform", "underperform",
            "price target", "target price", "analyst", "analysts",
            "wall street", "initiate", "initiated", "coverage",
            "overweight", "underweight", "equal weight", "neutral",
            "recommendation", "maintains", "reiterate", "reiterates",
        ],
        "sentiment_multiplier": 0.7,
        "base_impact": 0.0,
        "decay_days": 3,
    },
    "macro": {
        "keywords": [
            "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
            "inflation", "cpi", "ppi", "gdp", "unemployment", "jobs report",
            "tariff", "tariffs", "trade war", "trade deal", "sanctions",
            "recession", "recovery", "stimulus", "fiscal policy", "monetary policy",
            "treasury", "bond yields", "yield curve", "inversion",
            "oil prices", "commodity", "commodities", "supply chain",
        ],
        "sentiment_multiplier": 0.5,
        "base_impact": 0.0,
        "decay_days": 5,
    },
    "labor": {
        "keywords": [
            "layoff", "layoffs", "laid off", "job cuts", "workforce reduction",
            "restructuring", "downsizing", "furlough", "furloughed",
            "strike", "strikes", "union", "labor dispute", "workers",
            "hiring freeze", "cost cutting", "efficiency", "headcount",
        ],
        "sentiment_multiplier": 1.1,
        "base_impact": -0.01,
        "decay_days": 10,
    },
    "partnership": {
        "keywords": [
            "partnership", "partner", "partners", "partnered", "alliance",
            "collaboration", "collaborate", "joint venture", "jv",
            "licensing", "license", "licensed", "agreement", "contract",
            "deal", "signed", "multi-year", "exclusive", "strategic",
            "supplier", "customer", "client", "wins contract",
        ],
        "sentiment_multiplier": 0.9,
        "base_impact": 0.005,
        "decay_days": 15,
    },
    "competitive": {
        "keywords": [
            "competitor", "competition", "market share", "pricing pressure",
            "price war", "undercutting", "rival", "rivals", "losing share",
            "gaining share", "disrupted", "threat", "threatens", "challenged",
            "dominant", "dominance", "moat", "advantage", "disadvantage",
        ],
        "sentiment_multiplier": 1.0,
        "base_impact": 0.0,
        "decay_days": 20,
    },
}

# High-priority topics (strongest signal from literature + backtest)
HIGH_SIGNAL_TOPICS = {"litigation", "earnings", "management", "mna"}


# =============================================================================
# SENTIMENT DICTIONARIES (Loughran-McDonald inspired)
# =============================================================================

POSITIVE_WORDS = {
    "beat", "beats", "beating", "exceeded", "exceeds", "surpass", "surpassed",
    "record", "records", "growth", "growing", "grew", "strong", "stronger",
    "profit", "profitable", "gains", "gained", "rally", "rallied", "surge",
    "surged", "soar", "soared", "bullish", "optimistic", "positive", "upbeat",
    "outperform", "outperformed", "upgrade", "upgraded", "success", "successful",
    "innovation", "innovative", "breakthrough", "expand", "expansion", "expanded",
    "robust", "momentum", "accelerate", "accelerated", "improve", "improved",
    "improvement", "better", "best", "exceed", "excellent", "exceptional",
    "win", "wins", "won", "winning", "approval", "approved", "launch", "launched",
    "partnership", "deal", "agreement", "contract", "milestone", "achievement",
}

NEGATIVE_WORDS = {
    "miss", "missed", "misses", "fell", "fall", "falls", "decline", "declined",
    "declining", "drop", "dropped", "dropping", "plunge", "plunged", "crash",
    "crashed", "loss", "losses", "losing", "weak", "weaker", "weakness",
    "bearish", "pessimistic", "negative", "downgrade", "downgraded",
    "underperform", "underperformed", "warning", "concern", "concerns", "worried",
    "risk", "risks", "risky", "threat", "threatens", "challenged", "struggle",
    "struggling", "disappointing", "disappointed", "disappointment", "trouble",
    "layoff", "layoffs", "lawsuit", "sued", "investigation", "probe", "fraud",
    "scandal", "recall", "recalled", "defect", "bankruptcy", "default", "debt",
    "downside", "lower", "lowered", "cut", "cuts", "slashed", "slashing",
    "failed", "failure", "terminate", "terminated", "delay", "delayed",
    "slowing", "slowdown", "slow",
}

UNCERTAINTY_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "uncertain", "uncertainty",
    "unclear", "unknown", "question", "questions", "questioning", "doubt",
    "doubts", "volatile", "volatility", "unpredictable", "speculation",
    "speculate", "rumor", "rumors", "unconfirmed", "pending", "awaiting",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TopicClassification:
    """Classification of an article into topic(s)."""
    primary_topic: str
    secondary_topics: List[str]
    topic_scores: Dict[str, float]
    keywords_matched: Dict[str, List[str]]


@dataclass
class TopicSentimentResult:
    """Sentiment analysis result for a single article with topic context."""
    text: str
    topic: TopicClassification

    # Raw sentiment
    positive_count: int
    negative_count: int
    uncertainty_count: int
    word_count: int

    # Computed scores
    raw_sentiment: float           # -1 to +1
    topic_adjusted_sentiment: float  # Adjusted by topic multiplier
    uncertainty_score: float       # 0 to 1

    # Trading signal
    trading_signal: str            # "bullish", "bearish", "neutral"
    signal_strength: float         # 0 to 1
    expected_return: float         # Expected return contribution

    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedTopicSignal:
    """Aggregated topic signals for a ticker over a time period."""
    ticker: str
    start_date: datetime
    end_date: datetime

    # Per-topic aggregates
    topic_sentiment: Dict[str, float]    # Average sentiment by topic
    topic_counts: Dict[str, int]         # Article count by topic
    topic_signals: Dict[str, str]        # Signal by topic

    # Overall
    composite_signal: str                # "bullish", "bearish", "neutral"
    composite_score: float               # -1 to +1
    confidence: float                    # 0 to 1

    # Alerts
    litigation_alert: bool = False
    management_alert: bool = False
    earnings_surprise_detected: bool = False


# =============================================================================
# TOPIC SENTIMENT MODEL
# =============================================================================

class TopicSentimentModel:
    """
    Analyzes news sentiment with topic-specific adjustments.

    Key features:
    1. Classifies news into financial topics via keyword matching
    2. Applies topic-specific sentiment multipliers (asymmetric: negative amplified)
    3. Weights high-signal topics more heavily
    4. Detects topic-specific alerts (litigation, management changes)
    """

    def __init__(
        self,
        topic_definitions: Optional[Dict] = None,
        use_high_signal_weighting: bool = True,
    ):
        self.topics = topic_definitions or TOPIC_DEFINITIONS
        self.use_high_signal_weighting = use_high_signal_weighting
        self._compile_patterns()
        logger.debug("TopicSentimentModel initialized with %d topics", len(self.topics))

    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        self.topic_patterns = {}
        for topic, defn in self.topics.items():
            keywords = defn["keywords"]
            # Sort by length (longest first) to match multi-word patterns first
            keywords_sorted = sorted(keywords, key=len, reverse=True)
            pattern = "|".join(re.escape(kw) for kw in keywords_sorted)
            self.topic_patterns[topic] = re.compile(pattern, re.IGNORECASE)

    def classify_topic(self, text: str) -> TopicClassification:
        """Classify text into financial topics based on keyword matching."""
        text_lower = text.lower()
        topic_scores: Dict[str, float] = {}
        keywords_matched: Dict[str, List[str]] = {}

        for topic, pattern in self.topic_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                keywords_matched[topic] = matches
                unique_matches = len(set(matches))
                total_matches = len(matches)
                topic_scores[topic] = unique_matches + (total_matches - unique_matches) * 0.5

        if not topic_scores:
            return TopicClassification(
                primary_topic="general",
                secondary_topics=[],
                topic_scores={"general": 1.0},
                keywords_matched={},
            )

        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        primary = sorted_topics[0][0]
        secondary = [t for t, s in sorted_topics[1:] if s >= 1.0]

        return TopicClassification(
            primary_topic=primary,
            secondary_topics=secondary,
            topic_scores=topic_scores,
            keywords_matched=keywords_matched,
        )

    def compute_sentiment(self, text: str) -> Tuple[float, int, int, int, int]:
        """
        Compute raw sentiment from text using Loughran-McDonald dictionary.

        Returns:
            (sentiment_score, positive_count, negative_count, uncertainty_count, word_count)
        """
        words = text.lower().split()
        word_count = len(words)

        positive_count = sum(1 for w in words if w.strip(".,!?;:") in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w.strip(".,!?;:") in NEGATIVE_WORDS)
        uncertainty_count = sum(1 for w in words if w.strip(".,!?;:") in UNCERTAINTY_WORDS)

        if word_count > 0:
            sentiment = (positive_count - negative_count) / np.sqrt(word_count)
            sentiment = float(np.clip(sentiment, -1, 1))
        else:
            sentiment = 0.0

        return sentiment, positive_count, negative_count, uncertainty_count, word_count

    def analyze_article(self, text: str) -> TopicSentimentResult:
        """
        Analyze a single article for topic and sentiment.

        Applies asymmetric topic adjustment:
        - Negative sentiment amplified by multiplier (e.g., litigation 1.5×)
        - Positive sentiment dampened by (2 - multiplier)
        """
        topic = self.classify_topic(text)
        raw_sentiment, pos, neg, unc, wc = self.compute_sentiment(text)

        topic_config = self.topics.get(topic.primary_topic, {
            "sentiment_multiplier": 1.0,
            "base_impact": 0.0,
        })
        multiplier = topic_config.get("sentiment_multiplier", 1.0)
        base_impact = topic_config.get("base_impact", 0.0)

        # Asymmetric adjustment: negative amplified, positive dampened
        if raw_sentiment < 0:
            adjusted = raw_sentiment * multiplier + base_impact
        else:
            adjusted = raw_sentiment * (2 - multiplier) + base_impact
        adjusted = float(np.clip(adjusted, -1, 1))

        uncertainty = unc / np.sqrt(wc) if wc > 0 else 0.0
        uncertainty = min(uncertainty, 1.0)

        if adjusted > 0.15:
            signal = "bullish"
            strength = min(adjusted, 1.0)
        elif adjusted < -0.15:
            signal = "bearish"
            strength = min(abs(adjusted), 1.0)
        else:
            signal = "neutral"
            strength = 0.0

        expected_return = adjusted * 0.02  # 2% max for extreme sentiment

        return TopicSentimentResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            topic=topic,
            positive_count=pos,
            negative_count=neg,
            uncertainty_count=unc,
            word_count=wc,
            raw_sentiment=raw_sentiment,
            topic_adjusted_sentiment=adjusted,
            uncertainty_score=uncertainty,
            trading_signal=signal,
            signal_strength=strength,
            expected_return=expected_return,
        )

    def analyze_articles(
        self,
        articles: List[Dict],
        ticker: str,
        text_field: str = "title",
    ) -> AggregatedTopicSignal:
        """
        Analyze multiple articles and aggregate by topic.

        Returns AggregatedTopicSignal with topic-level and overall signals.
        """
        if not articles:
            return AggregatedTopicSignal(
                ticker=ticker,
                start_date=datetime.now(),
                end_date=datetime.now(),
                topic_sentiment={},
                topic_counts={},
                topic_signals={},
                composite_signal="neutral",
                composite_score=0.0,
                confidence=0.0,
            )

        results: List[TopicSentimentResult] = []
        dates: List[datetime] = []

        for article in articles:
            text = article.get(text_field, "") or ""
            if not text:
                continue
            result = self.analyze_article(text)
            results.append(result)

            # Track dates
            d = article.get("published_date") or article.get("published_at")
            if d:
                if isinstance(d, str):
                    try:
                        d = datetime.fromisoformat(d.replace("Z", "+00:00"))
                    except Exception:
                        d = datetime.now()
                if hasattr(d, "tzinfo") and d.tzinfo is not None:
                    d = d.replace(tzinfo=None)
                dates.append(d)

        if not results:
            return AggregatedTopicSignal(
                ticker=ticker,
                start_date=datetime.now(),
                end_date=datetime.now(),
                topic_sentiment={},
                topic_counts={},
                topic_signals={},
                composite_signal="neutral",
                composite_score=0.0,
                confidence=0.0,
            )

        start_date = min(dates) if dates else datetime.now()
        end_date = max(dates) if dates else datetime.now()

        # Aggregate by topic
        topic_sentiments: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            topic = r.topic.primary_topic
            topic_sentiments[topic].append(r.topic_adjusted_sentiment)
            for sec_topic in r.topic.secondary_topics:
                topic_sentiments[sec_topic].append(r.topic_adjusted_sentiment * 0.5)

        topic_sentiment = {t: float(np.mean(s)) for t, s in topic_sentiments.items()}
        topic_counts = {t: len(s) for t, s in topic_sentiments.items()}

        # Per-topic signals
        topic_signals = {}
        for topic, sent in topic_sentiment.items():
            if sent > 0.15:
                topic_signals[topic] = "bullish"
            elif sent < -0.15:
                topic_signals[topic] = "bearish"
            else:
                topic_signals[topic] = "neutral"

        # Composite signal with weighting
        total_weight = 0.0
        weighted_sentiment = 0.0
        for topic, sent in topic_sentiment.items():
            count = topic_counts.get(topic, 1)
            weight = np.sqrt(count)
            if self.use_high_signal_weighting and topic in HIGH_SIGNAL_TOPICS:
                weight *= 1.5
            weighted_sentiment += sent * weight
            total_weight += weight

        composite_score = float(weighted_sentiment / total_weight) if total_weight > 0 else 0.0

        if composite_score > 0.1:
            composite_signal = "bullish"
        elif composite_score < -0.1:
            composite_signal = "bearish"
        else:
            composite_signal = "neutral"

        confidence = min(len(results) / 10, 1.0)

        # Alerts
        litigation_alert = (
            topic_counts.get("litigation", 0) >= 2
            and topic_sentiment.get("litigation", 0) < -0.2
        )
        management_alert = (
            topic_counts.get("management", 0) >= 1
            and topic_sentiment.get("management", 0) < -0.1
        )
        earnings_surprise = (
            topic_counts.get("earnings", 0) >= 1
            and abs(topic_sentiment.get("earnings", 0)) > 0.3
        )

        return AggregatedTopicSignal(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            topic_sentiment=topic_sentiment,
            topic_counts=topic_counts,
            topic_signals=topic_signals,
            composite_signal=composite_signal,
            composite_score=composite_score,
            confidence=confidence,
            litigation_alert=litigation_alert,
            management_alert=management_alert,
            earnings_surprise_detected=earnings_surprise,
        )

    def get_topic_summary(self, signal: AggregatedTopicSignal) -> str:
        """Human-readable summary of topic signals."""
        lines = [f"Topic Sentiment Analysis for {signal.ticker}"]
        lines.append("=" * 50)

        if not signal.topic_counts:
            lines.append("No news articles analyzed.")
            return "\n".join(lines)

        lines.append(f"Period: {signal.start_date.date()} to {signal.end_date.date()}")
        lines.append(f"Total Topics Detected: {len(signal.topic_counts)}")
        lines.append("")

        for topic, count in sorted(signal.topic_counts.items(), key=lambda x: -x[1]):
            sent = signal.topic_sentiment.get(topic, 0)
            sig = signal.topic_signals.get(topic, "neutral")
            high = " [HIGH-SIGNAL]" if topic in HIGH_SIGNAL_TOPICS else ""
            lines.append(f"  {topic.upper()}{high}: {count} articles, sentiment={sent:.2f} ({sig})")

        lines.append("")
        lines.append(f"COMPOSITE: {signal.composite_signal.upper()} (score={signal.composite_score:.2f})")
        lines.append(f"Confidence: {signal.confidence:.0%}")

        if signal.litigation_alert:
            lines.append("!! LITIGATION ALERT: Multiple negative litigation articles")
        if signal.management_alert:
            lines.append("!! MANAGEMENT ALERT: Negative management news")
        if signal.earnings_surprise_detected:
            lines.append("!! EARNINGS SURPRISE: Strong sentiment on earnings news")

        return "\n".join(lines)


# =============================================================================
# EARNINGS TOPIC MODEL (strongest signal from backtest: IC = +0.021)
# =============================================================================

EARNINGS_MODEL_CONFIG = {
    "earnings_topics": ["earnings", "analyst"],
    "strong_positive_threshold": 0.2,
    "strong_negative_threshold": -0.2,
    "min_articles_for_signal": 1,
    "high_conviction_articles": 2,
    "agreement_ic_multiplier": 1.4,
    "signal_half_life_days": 5,
}


@dataclass
class EarningsTopicSignal:
    """Signal from earnings-focused topic sentiment analysis."""
    ticker: str
    total_articles: int
    earnings_articles: int
    topic_sentiment: float  # -1 to +1

    # FinBERT agreement (when db has pre-computed scores)
    finbert_sentiment: Optional[float] = None
    models_agree: Optional[bool] = None
    agreement_confidence_boost: float = 1.0

    # Signal
    is_tradeable: bool = False
    direction: str = "neutral"
    raw_score: float = 0.0
    confidence: float = 0.0

    # Expected outcome
    expected_5d_return: float = 0.0
    expected_alpha: float = 0.0

    # Context
    high_conviction: bool = False
    earnings_surprise: bool = False


class EarningsTopicModel:
    """
    Earnings-focused topic sentiment model.

    Based on empirical finding that earnings-related news sentiment
    has 3-4× the predictive power of generic sentiment.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or EARNINGS_MODEL_CONFIG
        self._topic_model = TopicSentimentModel()

    def analyze_article(self, text: str) -> Dict:
        """Classify a single article and check if earnings-related."""
        result = self._topic_model.analyze_article(text)
        is_earnings = result.topic.primary_topic in self.config["earnings_topics"]
        return {
            "is_earnings_related": is_earnings,
            "topic": result.topic.primary_topic,
            "topic_sentiment": result.topic_adjusted_sentiment,
            "raw_sentiment": result.raw_sentiment,
        }

    def analyze_news(
        self,
        articles: List[Dict],
        ticker: str,
        text_field: str = "title",
        finbert_field: Optional[str] = "sentiment_score",
    ) -> EarningsTopicSignal:
        """
        Analyze articles for earnings-specific sentiment.

        Filters to earnings/analyst topics, aggregates sentiment,
        checks for FinBERT agreement, and returns tradeable signal.
        """
        if not articles:
            return EarningsTopicSignal(
                ticker=ticker, total_articles=0,
                earnings_articles=0, topic_sentiment=0.0,
            )

        earnings_sentiments = []
        finbert_scores = []

        for article in articles:
            text = article.get(text_field, "") or ""
            if not text:
                continue
            result = self.analyze_article(text)

            if result["is_earnings_related"]:
                earnings_sentiments.append(result["topic_sentiment"])
                if finbert_field and finbert_field in article:
                    fb = article[finbert_field]
                    if fb is not None:
                        finbert_scores.append(fb)

        n_earnings = len(earnings_sentiments)

        if n_earnings == 0:
            return EarningsTopicSignal(
                ticker=ticker, total_articles=len(articles),
                earnings_articles=0, topic_sentiment=0.0,
            )

        topic_sentiment = float(np.mean(earnings_sentiments))

        # FinBERT agreement
        finbert_sentiment = float(np.mean(finbert_scores)) if finbert_scores else None
        models_agree = None
        agreement_boost = 1.0
        if finbert_sentiment is not None:
            same_sign = (topic_sentiment > 0 and finbert_sentiment > 0) or \
                        (topic_sentiment < 0 and finbert_sentiment < 0)
            models_agree = same_sign
            if same_sign:
                agreement_boost = self.config["agreement_ic_multiplier"]

        # Tradeable signal
        is_tradeable = n_earnings >= self.config["min_articles_for_signal"]
        high_conviction = n_earnings >= self.config["high_conviction_articles"]

        if topic_sentiment > self.config["strong_positive_threshold"]:
            direction = "bullish"
        elif topic_sentiment < self.config["strong_negative_threshold"]:
            direction = "bearish"
        else:
            direction = "neutral"

        # Confidence
        confidence = min(n_earnings / 5, 1.0) * (1.0 if not models_agree else agreement_boost)
        confidence = min(confidence, 1.0)

        # Earnings surprise detection
        earnings_surprise = abs(topic_sentiment) > 0.3

        # Expected alpha (from backtest calibration)
        if direction == "bullish":
            expected_5d = 0.0068
        elif direction == "bearish":
            expected_5d = 0.0003
        else:
            expected_5d = 0.0038  # base rate
        expected_alpha = expected_5d - 0.0038  # vs base rate

        raw_score = topic_sentiment * np.sqrt(n_earnings) * agreement_boost

        return EarningsTopicSignal(
            ticker=ticker,
            total_articles=len(articles),
            earnings_articles=n_earnings,
            topic_sentiment=topic_sentiment,
            finbert_sentiment=finbert_sentiment,
            models_agree=models_agree,
            agreement_confidence_boost=agreement_boost,
            is_tradeable=is_tradeable,
            direction=direction,
            raw_score=float(raw_score),
            confidence=float(confidence),
            expected_5d_return=expected_5d,
            expected_alpha=expected_alpha,
            high_conviction=high_conviction,
            earnings_surprise=earnings_surprise,
        )

    def get_summary(self, signal: EarningsTopicSignal) -> str:
        """Human-readable summary of earnings signal."""
        lines = [f"Earnings Topic Signal for {signal.ticker}"]
        lines.append(f"  Articles: {signal.total_articles} total, {signal.earnings_articles} earnings-related")
        lines.append(f"  Topic Sentiment: {signal.topic_sentiment:+.3f}")
        if signal.finbert_sentiment is not None:
            lines.append(f"  FinBERT Sentiment: {signal.finbert_sentiment:+.3f}")
            lines.append(f"  Models Agree: {signal.models_agree}")
        lines.append(f"  Direction: {signal.direction.upper()}")
        lines.append(f"  Tradeable: {signal.is_tradeable}")
        lines.append(f"  Confidence: {signal.confidence:.0%}")
        if signal.high_conviction:
            lines.append("  ** HIGH CONVICTION (2+ earnings articles)")
        if signal.earnings_surprise:
            lines.append("  ** EARNINGS SURPRISE DETECTED")
        return "\n".join(lines)
