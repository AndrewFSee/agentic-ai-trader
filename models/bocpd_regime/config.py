"""
Configuration dataclasses for BOCPD regime detection.

Adapted from regime_aware_portfolio_allocator — standalone, no external dependencies.
"""

from dataclasses import dataclass


# Regime names
REGIME_NAMES = [
    "bull", "bear", "bull_transition", "bear_transition",
    "consolidation", "crisis",
]
REGIME_TO_IDX = {name: idx for idx, name in enumerate(REGIME_NAMES)}
IDX_TO_REGIME = {idx: name for idx, name in enumerate(REGIME_NAMES)}


@dataclass
class BOCPDConfig:
    """Configuration for Bayesian Online Changepoint Detection.

    The standard BOCPD algorithm with constant hazard always produces
    cp_prob ~ hazard_rate.  Changepoints are detected via drops in ERL
    (expected run length).  The ERL-based instability score is the primary
    signal for regime detection.
    """

    # Hazard rate: 1/H is the expected run length
    hazard: float = 1.0 / 126.0  # ~6 months

    # Dynamic hazard (volatility-scaled)
    use_dynamic_hazard: bool = True
    hazard_min: float = 1.0 / 252.0   # ~1 year
    hazard_max: float = 1.0 / 42.0    # ~2 months
    hazard_vol_window: int = 21
    hazard_vol_z_window: int = 126
    hazard_vol_scale: float = 1.0
    hazard_mapping: str = "sigmoid"    # "sigmoid", "linear", or "tanh"

    # Volatility-scaled input
    use_volatility_scaling: bool = True
    vol_scale_window: int = 21
    vol_scale_floor: float = 1e-6

    # Normal-Inverse-Gamma prior parameters
    # After vol-scaling, data has variance ~2-5; beta0/(alpha0-1) should match
    mu0: float = 0.0
    kappa0: float = 0.1
    alpha0: float = 2.0
    beta0: float = 2.0

    # Changepoint probability thresholds for risk scoring
    cp_prob_lo: float = 0.05
    cp_prob_hi: float = 0.5

    # Expected run length thresholds
    # With vol-scaling, ERL maxes ~50 (not 600+ like raw data)
    erl_floor: int = 3
    erl_enter: int = 10
    erl_exit: int = 30
    erl_spike_thr: int = 5


@dataclass
class RegimeLabelConfig:
    """Configuration for regime labeling from BOCPD outputs."""

    # Risk score weights (ERL instability is the primary signal)
    risk_weight_cp: float = 0.0
    risk_weight_erl: float = 1.0

    # Risk score thresholds — tuned via sweep
    risk_low: float = 0.20
    risk_high: float = 0.50

    # Trend calculation
    trend_window: int = 63
    trend_up: float = 0.0075
    trend_down: float = -0.0037

    # Regime smoothing
    min_regime_duration: int = 4

    # Confirmation delay (disabled for fast signals)
    use_adaptive_delay: bool = False
    confirmation_days: int = 1
