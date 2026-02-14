"""
High-level BOCPD regime detector for use as an agent tool.

Downloads market data, runs BOCPD, labels the regime, and returns
a structured result dict that the trading agent can consume.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any

from models.bocpd_regime.config import BOCPDConfig, RegimeLabelConfig
from models.bocpd_regime.bocpd import run_bocpd
from models.bocpd_regime.labeling import label_regimes


# Regime → actionable interpretation for the trading agent
REGIME_INTERPRETATIONS = {
    "bull": (
        "Sustained uptrend with low volatility. Trend-following and "
        "breakout strategies favoured. Normal or slightly larger position sizing."
    ),
    "bear": (
        "Sustained downtrend. Mean-reversion longs are dangerous. "
        "Consider defensive positioning, tighter stops, or sitting out. "
        "Short setups or hedging may be appropriate."
    ),
    "bull_transition": (
        "Recovery or volatile rally — trend is turning positive but conditions "
        "are still unstable. Smaller positions with wider stops. "
        "Confirm with volume and momentum before committing."
    ),
    "bear_transition": (
        "Early weakness — trend is turning negative or instability is rising. "
        "Reduce exposure, tighten stops, avoid adding to longs. "
        "Watch for confirmation of a full bear regime."
    ),
    "consolidation": (
        "Range-bound / low conviction. Neither trend nor volatility gives a "
        "clear signal. Smaller positions, mean-reversion setups may work "
        "but with tight risk limits."
    ),
    "crisis": (
        "Extreme volatility with sharp decline. Capital preservation is the "
        "priority. Minimal or zero equity exposure. Wait for volatility to "
        "subside before re-entering."
    ),
}


def detect_current_regime(
    symbol: str = "SPY",
    lookback_years: int = 3,
) -> Dict[str, Any]:
    """
    Run BOCPD regime detection end-to-end and return current regime state.

    Fetches daily data via yfinance, computes features, runs BOCPD,
    labels regimes, and returns a structured dict for the agent.

    Args:
        symbol: Ticker to analyse (default SPY for broad market regime).
        lookback_years: Years of history for BOCPD warm-up.

    Returns:
        Dict with current_regime, risk_score, trends, ERL, interpretation, etc.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed — required for BOCPD regime detection"}

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            symbol, period=f"{lookback_years}y", progress=False, auto_adjust=True
        )

    if df is None or len(df) < 126:
        return {"error": f"Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)"}

    # Flatten multi-level columns if needed (yfinance >= 0.2.31)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()

    # ------------------------------------------------------------------
    # 2. Compute features
    # ------------------------------------------------------------------
    returns = np.log(close / close.shift(1))        # daily log returns
    ret_21 = returns.rolling(21).sum()               # 21-day rolling return
    vol_21 = returns.rolling(21).std() * np.sqrt(252)  # 21-day annualised vol

    # ------------------------------------------------------------------
    # 3. Run BOCPD on 21-day returns
    # ------------------------------------------------------------------
    bocpd_config = BOCPDConfig()
    label_config = RegimeLabelConfig()

    bocpd_df = run_bocpd(ret_21, bocpd_config, max_run_length=500)

    # ------------------------------------------------------------------
    # 4. Label regimes
    # ------------------------------------------------------------------
    labeled = label_regimes(
        bocpd_df=bocpd_df,
        returns=returns,
        volatility=vol_21,
        bocpd_config=bocpd_config,
        label_config=label_config,
    )

    # ------------------------------------------------------------------
    # 5. Extract current state
    # ------------------------------------------------------------------
    # Use the last valid (non-NaN) row
    valid = labeled.dropna(subset=["regime_smoothed"])
    if len(valid) == 0:
        return {"error": "BOCPD labeling produced no valid regime labels"}

    latest = valid.iloc[-1]
    current_regime = latest["regime_smoothed"]
    risk_score = float(latest["risk_score"])
    trend_21_val = float(latest["trend_21"])
    trend_63_val = float(latest["trend_63"])
    volatility_val = float(latest["volatility"])
    latest_date = str(valid.index[-1].date())

    # ERL from BOCPD output
    erl_val = float(bocpd_df.loc[valid.index[-1], "erl"]) if valid.index[-1] in bocpd_df.index else None

    # Recent regime history (last 5 distinct regimes)
    regime_history = []
    prev = None
    for idx, row in valid.tail(60).iterrows():
        r = row["regime_smoothed"]
        if r != prev:
            regime_history.append({"date": str(idx.date()), "regime": r})
            prev = r
    regime_history = regime_history[-5:]

    # Risk level label
    if risk_score >= 0.70:
        risk_level = "HIGH"
    elif risk_score >= 0.45:
        risk_level = "ELEVATED"
    elif risk_score >= 0.25:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "symbol": symbol,
        "date": latest_date,
        "current_regime": current_regime,
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "expected_run_length": round(erl_val, 1) if erl_val is not None else None,
        "trend_21d": round(trend_21_val, 4),
        "trend_63d": round(trend_63_val, 4),
        "volatility_ann": round(volatility_val, 4),
        "interpretation": REGIME_INTERPRETATIONS.get(current_regime, "Unknown regime."),
        "recent_regime_changes": regime_history,
    }
