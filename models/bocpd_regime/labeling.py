"""
Regime labeling from BOCPD outputs.

Converts continuous BOCPD metrics (cp_prob, ERL) into discrete regime labels:
bull, bear, bull_transition, bear_transition, consolidation, crisis.

Adapted from regime_aware_portfolio_allocator — standalone.
"""

import numpy as np
import pandas as pd
from typing import Optional

from models.bocpd_regime.config import BOCPDConfig, RegimeLabelConfig, REGIME_NAMES
from models.bocpd_regime.utils import (
    map_to_range,
    compute_erl_instability,
    apply_run_length_filter,
)


def compute_risk_score(
    bocpd_df: pd.DataFrame,
    bocpd_config: BOCPDConfig,
    label_config: RegimeLabelConfig,
) -> pd.Series:
    """Compute continuous risk score from BOCPD outputs. Returns [0, 1]."""
    cp_risk = map_to_range(
        bocpd_df["cp_prob"],
        bocpd_config.cp_prob_lo,
        bocpd_config.cp_prob_hi,
    )
    erl_risk = compute_erl_instability(
        bocpd_df["erl"],
        erl_floor=bocpd_config.erl_floor,
        erl_stable=bocpd_config.erl_exit,
    )
    risk_score = (
        label_config.risk_weight_cp * cp_risk
        + label_config.risk_weight_erl * erl_risk
    )
    total_weight = label_config.risk_weight_cp + label_config.risk_weight_erl
    if total_weight > 0:
        risk_score = risk_score / total_weight
    risk_score = risk_score.clip(0, 1)
    risk_score.name = "risk_score"
    return risk_score


def compute_trend(
    price_or_returns: pd.Series,
    window: int = 63,
    is_returns: bool = False,
) -> pd.Series:
    """Compute rolling trend (cumulative return) over a window."""
    if is_returns:
        trend = price_or_returns.rolling(window=window).sum()
    else:
        trend = price_or_returns.pct_change(periods=window)
    trend.name = "trend"
    return trend


def classify_regime_v2(
    risk_score: float,
    trend_21: float,
    trend_63: float,
    volatility: float,
    prev_regime: Optional[str] = None,
) -> str:
    """
    Multi-timeframe regime classification.

    Uses both 21-day and 63-day trends to capture:
    - Grinding bear markets with low volatility but negative trend
    - Volatile bear markets with high risk + downtrend
    - V-shaped recoveries with high vol + uptrend
    - Steady bull markets with low risk + uptrend
    """
    if pd.isna(risk_score) or pd.isna(trend_21) or pd.isna(trend_63):
        return "consolidation"

    # Risk thresholds
    VERY_HIGH_RISK = 0.80
    HIGH_RISK = 0.45
    MODERATE_RISK = 0.30
    LOW_RISK = 0.15

    # Volatility thresholds (annualised)
    HIGH_VOL = 0.20
    LOW_VOL = 0.14

    # Medium-term trend (63-day)
    STRONG_BEAR_TREND = -0.08
    BEAR_TREND = -0.03
    STRONG_BULL_TREND = 0.06
    BULL_TREND = 0.02

    # Short-term trend (21-day)
    STRONG_DOWN_21 = -0.05
    DOWN_21 = -0.01
    UP_21 = 0.01
    STRONG_UP_21 = 0.04

    is_very_high_risk = risk_score >= VERY_HIGH_RISK
    is_high_risk = risk_score >= HIGH_RISK
    is_moderate_risk = risk_score >= MODERATE_RISK
    is_low_risk = risk_score <= LOW_RISK

    is_high_vol = volatility >= HIGH_VOL
    is_low_vol = volatility <= LOW_VOL

    is_strong_bear_63 = trend_63 <= STRONG_BEAR_TREND
    is_bear_63 = trend_63 <= BEAR_TREND
    is_strong_bull_63 = trend_63 >= STRONG_BULL_TREND
    is_bull_63 = trend_63 >= BULL_TREND
    is_flat_63 = not is_bear_63 and not is_bull_63

    is_strong_down_21 = trend_21 <= STRONG_DOWN_21
    is_down_21 = trend_21 <= DOWN_21
    is_up_21 = trend_21 >= UP_21
    is_strong_up_21 = trend_21 >= STRONG_UP_21
    is_flat_21 = not is_down_21 and not is_up_21

    was_bearish = prev_regime in ("bear", "crisis", "bear_transition") if prev_regime else False

    # 1. CRISIS
    if volatility >= 0.40 and trend_21 <= 0.05:
        return "crisis"
    if volatility >= 0.35 and is_very_high_risk and trend_21 <= 0.03:
        return "crisis"
    if trend_21 <= -0.20:
        return "crisis"
    if trend_21 <= -0.12 and volatility >= 0.30:
        return "crisis"

    # 2. BULL_TRANSITION (recovery from bear)
    if was_bearish and is_up_21:
        return "bull_transition"
    if was_bearish and is_strong_up_21:
        return "bull_transition"

    # 3. BEAR
    if is_strong_bear_63:
        return "bear"
    if is_bear_63 and is_down_21:
        return "bear"
    if is_bear_63 and is_high_risk:
        return "bear"
    if is_high_risk and is_strong_down_21:
        return "bear"
    if is_high_vol and is_down_21:
        return "bear"
    if is_moderate_risk and trend_21 < -0.02 and trend_63 < 0:
        return "bear"
    if is_high_risk and is_down_21:
        return "bear"

    # 4. BULL_TRANSITION (other recovery/volatile rally)
    if is_high_vol and is_up_21 and not is_bear_63:
        return "bull_transition"
    if is_high_risk and is_up_21 and not is_strong_bear_63:
        return "bull_transition"
    if is_moderate_risk and is_up_21:
        return "bull_transition"
    if is_flat_63 and is_strong_up_21:
        return "bull_transition"

    # 5. BULL
    if is_bull_63 and is_low_vol and is_low_risk:
        return "bull"
    if is_bull_63 and is_up_21 and not is_high_risk and not is_high_vol:
        return "bull"
    if is_bull_63 and is_low_risk:
        return "bull"
    if is_strong_bull_63 and not is_high_vol:
        return "bull"
    if is_low_risk and is_up_21 and trend_63 >= 0:
        return "bull"
    if is_low_vol and is_up_21 and trend_63 >= 0 and not is_moderate_risk:
        return "bull"

    # 6. BEAR_TRANSITION
    if is_high_risk and is_flat_21 and not is_bear_63:
        return "bear_transition"
    if is_down_21 and is_flat_63:
        return "bear_transition"
    if is_moderate_risk and is_down_21 and not is_bear_63:
        return "bear_transition"

    # 7. CONSOLIDATION
    return "consolidation"


def smooth_regime_labels(
    regimes: pd.Series,
    min_duration: int = 4,
) -> pd.Series:
    """Smooth regime labels to enforce minimum duration."""
    return apply_run_length_filter(regimes, min_duration)


def label_regimes(
    bocpd_df: pd.DataFrame,
    returns: pd.Series,
    volatility: pd.Series,
    bocpd_config: BOCPDConfig,
    label_config: RegimeLabelConfig,
) -> pd.DataFrame:
    """
    Convert BOCPD outputs into discrete regime labels.

    This is the main labeling entry point for the tool.

    Args:
        bocpd_df: DataFrame with 'cp_prob' and 'erl' from run_bocpd
        returns: Daily log returns series
        volatility: Annualised rolling volatility series (e.g., 21-day)
        bocpd_config: BOCPDConfig
        label_config: RegimeLabelConfig

    Returns:
        DataFrame with risk_score, trend_21, trend_63, volatility,
        regime, regime_smoothed
    """
    common_idx = bocpd_df.index.intersection(returns.index).intersection(volatility.index)
    result = pd.DataFrame(index=common_idx)

    # Risk score from BOCPD
    bocpd_aligned = bocpd_df.loc[common_idx]
    base_risk = compute_risk_score(bocpd_aligned, bocpd_config, label_config)

    # Blend with volatility risk
    vol_series = volatility.loc[common_idx]
    vol_risk = map_to_range(vol_series, 0.12, 0.25)
    result["risk_score"] = (0.7 * base_risk + 0.3 * vol_risk).clip(0, 1)
    result["volatility"] = vol_series

    # Trends
    returns_aligned = returns.loc[common_idx]
    trend_21 = compute_trend(returns_aligned, window=21, is_returns=True)
    trend_63 = compute_trend(returns_aligned, window=63, is_returns=True)
    result["trend_21"] = trend_21
    result["trend_63"] = trend_63

    # Classify
    regimes = []
    prev_regime = None
    for idx in common_idx:
        if pd.isna(result.loc[idx, "trend_63"]):
            regime = "consolidation"
        else:
            regime = classify_regime_v2(
                result.loc[idx, "risk_score"],
                result.loc[idx, "trend_21"],
                result.loc[idx, "trend_63"],
                result.loc[idx, "volatility"],
                prev_regime=prev_regime,
            )
        regimes.append(regime)
        prev_regime = regime

    result["regime"] = regimes
    result["regime_smoothed"] = smooth_regime_labels(
        result["regime"],
        min_duration=label_config.min_regime_duration,
    )
    return result
