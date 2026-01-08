"""
Alpha Regime Detector v2 - Bull-Friendly Risk Overlay

Production-quality regime detection for identifying distribution shifts in 
return-generating processes. This version fixes the BOCPD sufficient statistics
bug and implements a bull-friendly risk overlay that reduces opportunity cost
while maintaining crisis protection.

Key Changes from v1:
    1. BOCPD Bug Fix: r=0 hypothesis now correctly includes x_t as first observation
    2. Bull-Friendly Overlay: Position scalar [0,1] instead of binary exit
    3. Transition-Risk Gating: Confirmation logic to avoid whipsaws
    4. Weighted Noisy-OR: Emphasis on volatility channels for crisis detection

Signal Timing (NO LOOKAHEAD):
    At end of day t-1, we have OHLCV[0:t-1] and can compute:
    - features[t-1], change_prob[t-1], regime_state[t-1]
    Signal for day t's position is computed using ONLY data through t-1.
    
Based on Adams & MacKay (2007): "Bayesian Online Changepoint Detection"
https://arxiv.org/abs/0710.3742

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings


# =============================================================================
# CONFIGURATION & DATA STRUCTURES
# =============================================================================

class VolatilityEstimator(Enum):
    """Volatility estimation method."""
    EWMA = "ewma"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"


class IntradayMeasure(Enum):
    """Intraday pressure measurement."""
    Z_DAY = "z_day"
    CLV = "clv"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    vol_estimator: VolatilityEstimator = VolatilityEstimator.EWMA
    ewma_span: int = 30
    vol_floor: float = 1e-8
    intraday_measure: IntradayMeasure = IntradayMeasure.CLV
    use_log_sigma: bool = True
    use_range_vol: bool = True  # Add Parkinson range-based vol channel
    rolling_window: int = 20


@dataclass
class BOCPDConfig:
    """Configuration for BOCPD model."""
    hazard_rate: float = 1/200  # Bull-friendly: expect ~200 day regimes
    mu0: float = 0.0
    kappa0: float = 0.1
    alpha0: float = 2.0
    beta0: float = 1.0
    max_run_length: int = 500


@dataclass 
class RegimeConfig:
    """Configuration for regime detection."""
    mode: Literal["univariate", "multivariate"] = "univariate"
    
    # Channel selection
    use_z_ret: bool = True
    use_z_gap: bool = True
    use_intraday: bool = True
    use_log_sigma: bool = True
    use_range_vol: bool = True  # Parkinson volatility channel
    
    # Channel weights for weighted noisy-OR (bias toward volatility)
    channel_weights: Dict[str, float] = field(default_factory=lambda: {
        'z_ret': 0.6,        # Lower weight - minor return shifts
        'z_gap': 0.5,        # Lower weight - gap noise
        'intraday': 0.5,     # Lower weight
        'log_sigma': 1.2,    # Higher weight - vol regime shifts
        'range_vol': 1.5,    # Highest weight - crisis indicator
    })
    
    # Detection settings
    detection_method: Literal["prob", "map_drop", "confirmed"] = "confirmed"
    cp_threshold: float = 0.3
    min_spacing: int = 15
    map_drop_from: int = 10
    map_drop_to: int = 3
    
    # Confirmation window (anti-whipsaw)
    confirmation_window: int = 7   # M: look back M days
    confirmation_count: int = 2    # K: need K confirmations
    cooldown_days: int = 10        # After detection, wait before next


@dataclass
class OverlayConfig:
    """Configuration for bull-friendly risk overlay with continuous risk scoring."""
    
    # Position scalars (bull-friendly: reduced penalty for transition)
    normal_position: float = 1.0      # Default: fully invested
    transition_position: float = 0.75 # Mild reduction during transition risk
    severe_position: float = 0.25     # Major reduction during severe risk
    
    # Continuous risk scoring: P(changepoint) linear ramp
    # Note: With BOCPD bug fix, change_prob tends toward hazard rate
    # So we use lower thresholds here
    cp_lo: float = 0.02               # risk_cp = 0 at/below hazard rate
    cp_hi: float = 0.10               # risk_cp = 1 above this (unusual)
    
    # Continuous risk scoring: expected run length linear ramp
    # ERL is the PRIMARY signal for regime changes after the bug fix
    erl_enter: int = 20               # Enter transition mode when ERL < this
    erl_floor: int = 5                # Maximum risk when ERL <= this
    erl_exit: int = 40                # Exit transition mode when ERL > this
    
    # Fast-in thresholds (immediate transition triggers)
    cp_spike_thr: float = 0.10        # P(CP) spike threshold (rare after fix)
    vol_spike_thr: float = 0.05       # Volatility channel CP threshold
    erl_spike_thr: int = 10           # ERL < this triggers severe mode
    
    # Slow-out thresholds (maintain transition while these hold)
    cp_hold_thr: float = 0.03         # EWMA(cp) threshold to maintain transition
    ema_span: int = 5                 # EMA span for smoothing
    
    # Calm-based recovery (not fixed days)
    calm_risk_thr: float = 0.15       # Risk must be below this to count as calm
    calm_days: int = 5                # Consecutive calm days before recovery
    
    # Severe mode: volatility channel AND low ERL
    vol_severe_thr: float = 0.10      # vol_channel_cp > this + low ERL = severe


@dataclass
class FeatureVector:
    """Container for computed features at a single time step."""
    timestamp: pd.Timestamp
    z_ret: float
    z_gap: float
    intraday: float
    log_sigma: float
    sigma: float
    range_vol: Optional[float] = None  # Parkinson volatility
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'z_ret': self.z_ret,
            'z_gap': self.z_gap,
            'intraday': self.intraday,
            'log_sigma': self.log_sigma,
            'range_vol': self.range_vol if self.range_vol is not None else 0.0
        }


@dataclass
class RegimeState:
    """Current regime state for a ticker."""
    ticker: str
    timestamp: pd.Timestamp
    change_prob: float
    individual_probs: Dict[str, float]
    run_length_map: int
    regime_id: int
    days_in_regime: int
    position_scalar: float = 1.0       # Risk overlay output
    expected_run_length: float = 0.0   # E[run length] from BOCPD
    in_transition: bool = False        # Whether in transition mode
    risk_cp: float = 0.0               # Continuous risk from P(CP)
    risk_rl: float = 0.0               # Continuous risk from E[RL]


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngine:
    """
    Computes scale-free, cross-ticker stable features from OHLCV data.
    All computations are strictly online (no lookahead).
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.reset()
    
    def reset(self):
        self._prev_close: Optional[float] = None
        self._ewma_var: Optional[float] = None
        self._ewma_alpha: float = 2.0 / (self.config.ewma_span + 1)
        self._z_ret_history: List[float] = []
        self._range_vol_history: List[float] = []
    
    def _compute_ewma_volatility(self, return_sq: float) -> float:
        if self._ewma_var is None:
            self._ewma_var = return_sq
        else:
            self._ewma_var = (
                self._ewma_alpha * return_sq + 
                (1 - self._ewma_alpha) * self._ewma_var
            )
        return np.sqrt(max(self._ewma_var, self.config.vol_floor))
    
    def _compute_parkinson_volatility(self, high: float, low: float) -> float:
        """Parkinson (1980) range-based volatility estimator."""
        if high <= 0 or low <= 0 or high <= low:
            return np.sqrt(self.config.vol_floor)
        log_hl = np.log(high / low)
        variance = log_hl**2 / (4 * np.log(2))
        return np.sqrt(max(variance, self.config.vol_floor))
    
    def _compute_clv(self, high: float, low: float, close: float) -> float:
        """Close Location Value: where close falls in day's range."""
        range_ = high - low
        if range_ < self.config.vol_floor:
            return 0.0
        return (2 * close - high - low) / range_
    
    def update(
        self,
        timestamp: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None
    ) -> Optional[FeatureVector]:
        """Process new OHLCV bar and compute features (strictly online)."""
        if self._prev_close is None:
            self._prev_close = close
            return None
        
        log_return = np.log(close / self._prev_close) if self._prev_close > 0 else 0.0
        overnight_gap = np.log(open_ / self._prev_close) if self._prev_close > 0 else 0.0
        
        # EWMA volatility (primary)
        sigma = self._compute_ewma_volatility(log_return**2)
        sigma = max(sigma, self.config.vol_floor)
        
        # Parkinson range volatility (for crisis detection)
        range_vol = self._compute_parkinson_volatility(high, low)
        
        # Standardized features
        z_ret = log_return / sigma
        z_gap = overnight_gap / sigma
        
        # Intraday measure
        if self.config.intraday_measure == IntradayMeasure.CLV:
            intraday = self._compute_clv(high, low, close)
        else:
            intraday_return = np.log(close / open_) if open_ > 0 else 0.0
            intraday = intraday_return / sigma
        
        log_sigma = np.log(sigma)
        
        # Standardize range_vol using recent history
        self._range_vol_history.append(range_vol)
        if len(self._range_vol_history) > self.config.rolling_window * 2:
            self._range_vol_history = self._range_vol_history[-self.config.rolling_window * 2:]
        
        if len(self._range_vol_history) >= self.config.rolling_window:
            rv_mean = np.mean(self._range_vol_history[-self.config.rolling_window:])
            rv_std = np.std(self._range_vol_history[-self.config.rolling_window:])
            z_range_vol = (range_vol - rv_mean) / max(rv_std, 1e-8)
        else:
            z_range_vol = 0.0
        
        self._z_ret_history.append(z_ret)
        if len(self._z_ret_history) > self.config.rolling_window * 2:
            self._z_ret_history = self._z_ret_history[-self.config.rolling_window * 2:]
        
        self._prev_close = close
        
        return FeatureVector(
            timestamp=timestamp,
            z_ret=z_ret,
            z_gap=z_gap,
            intraday=intraday,
            log_sigma=log_sigma,
            sigma=sigma,
            range_vol=z_range_vol
        )
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire DataFrame of OHLCV data."""
        self.reset()
        features = []
        
        for idx, row in df.iterrows():
            timestamp = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
            feat = self.update(
                timestamp=timestamp,
                open_=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', None)
            )
            if feat is not None:
                features.append({
                    'timestamp': feat.timestamp,
                    'z_ret': feat.z_ret,
                    'z_gap': feat.z_gap,
                    'intraday': feat.intraday,
                    'log_sigma': feat.log_sigma,
                    'sigma': feat.sigma,
                    'range_vol': feat.range_vol
                })
        
        result_df = pd.DataFrame(features)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        return result_df


# =============================================================================
# BOCPD IMPLEMENTATION (BUG FIXED)
# =============================================================================

class BOCPD:
    """
    Bayesian Online Change Point Detection (Adams & MacKay, 2007).
    
    BUG FIX (v2): When r_t = 0 (changepoint), the new regime's sufficient
    statistics now correctly include x_t as the FIRST observation:
        n[0] = 1, sum_x[0] = x, sum_x2[0] = x^2
    
    This makes the posterior for the new regime immediately data-informed,
    reducing spurious CP probability spikes.
    """
    
    def __init__(self, config: Optional[BOCPDConfig] = None):
        self.config = config or BOCPDConfig()
        self._log_H = np.log(self.config.hazard_rate)
        self._log_1_minus_H = np.log(1 - self.config.hazard_rate)
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.t = 0
        self._log_R = np.array([0.0])
        self._n = np.array([0.0])
        self._sum_x = np.array([0.0])
        self._sum_x2 = np.array([0.0])
        self._map_run_lengths: List[int] = []
        self._change_probs: List[float] = []
    
    def _get_posterior_params(
        self,
        n: np.ndarray,
        sum_x: np.ndarray,
        sum_x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Normal-Inverse-Gamma posterior parameters."""
        kappa_n = self.config.kappa0 + n
        mu_n = (self.config.kappa0 * self.config.mu0 + sum_x) / kappa_n
        alpha_n = self.config.alpha0 + n / 2.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            xbar = np.where(n > 0, sum_x / n, 0.0)
        
        ss = np.maximum(sum_x2 - n * xbar**2, 0.0)
        coupling = self.config.kappa0 * n * (xbar - self.config.mu0)**2 / kappa_n
        beta_n = self.config.beta0 + 0.5 * ss + 0.5 * coupling
        
        return mu_n, kappa_n, alpha_n, beta_n
    
    def _predictive_log_likelihood(
        self,
        x: float,
        n: np.ndarray,
        sum_x: np.ndarray,
        sum_x2: np.ndarray
    ) -> np.ndarray:
        """Compute log-likelihood of x under Student-t predictive."""
        mu_n, kappa_n, alpha_n, beta_n = self._get_posterior_params(n, sum_x, sum_x2)
        
        df = 2.0 * alpha_n
        scale = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        scale = np.maximum(scale, 1e-10)
        
        z = (x - mu_n)**2 / (df * scale)
        
        log_pdf = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi * scale)
            - ((df + 1) / 2) * np.log1p(z)
        )
        
        return log_pdf
    
    def update(self, x: float) -> float:
        """
        Process new observation and update run-length posterior.
        
        BUG FIX: The r=0 (changepoint) hypothesis now correctly includes
        x as the first observation of the new regime, so:
            n[0] = 1, sum_x[0] = x, sum_x2[0] = x^2
        
        Args:
            x: New observation (should be standardized for stability)
            
        Returns:
            P(r_t = 0): Probability that this is a change point
        """
        self.t += 1
        
        # Predictive likelihood for each current run length
        log_pred = self._predictive_log_likelihood(
            x, self._n, self._sum_x, self._sum_x2
        )
        
        # Growth probabilities: P(r_t = r_{t-1} + 1 | x_{1:t})
        log_growth = self._log_R + log_pred + self._log_1_minus_H
        
        # Changepoint probability: P(r_t = 0 | x_{1:t})
        log_cp = logsumexp(self._log_R + log_pred) + self._log_H
        
        # New run length distribution
        log_R_new = np.concatenate([[log_cp], log_growth])
        
        # Normalize
        log_evidence = logsumexp(log_R_new)
        log_R_new = log_R_new - log_evidence
        
        # =====================================================================
        # BUG FIX: Update sufficient statistics
        # The r=0 hypothesis (new regime) should include x as FIRST observation
        # OLD (buggy): n[0]=0, sum_x[0]=0, sum_x2[0]=0
        # NEW (fixed): n[0]=1, sum_x[0]=x, sum_x2[0]=x^2
        # =====================================================================
        self._n = np.concatenate([[1.0], self._n + 1])
        self._sum_x = np.concatenate([[x], self._sum_x + x])
        self._sum_x2 = np.concatenate([[x**2], self._sum_x2 + x**2])
        
        # Truncate for efficiency
        if len(log_R_new) > self.config.max_run_length:
            log_R_new = log_R_new[:self.config.max_run_length]
            self._n = self._n[:self.config.max_run_length]
            self._sum_x = self._sum_x[:self.config.max_run_length]
            self._sum_x2 = self._sum_x2[:self.config.max_run_length]
        
        self._log_R = log_R_new
        
        R = np.exp(log_R_new)
        change_prob = float(R[0])
        map_run_length = int(np.argmax(R))
        
        self._map_run_lengths.append(map_run_length)
        self._change_probs.append(change_prob)
        
        return change_prob
    
    def get_change_prob(self) -> float:
        """Get current P(r = 0)."""
        return float(np.exp(self._log_R[0]))
    
    def detect_map_drop(self, from_threshold: int = 5, to_threshold: int = 2) -> bool:
        """Detect if MAP run length dropped significantly."""
        if len(self._map_run_lengths) < 2:
            return False
        prev_map = self._map_run_lengths[-2]
        curr_map = self._map_run_lengths[-1]
        return prev_map > from_threshold and curr_map <= to_threshold
    
    def get_run_length_distribution(self) -> np.ndarray:
        return np.exp(self._log_R)
    
    def get_map_run_length(self) -> int:
        return int(np.argmax(np.exp(self._log_R)))
    
    def get_expected_run_length(self) -> float:
        R = np.exp(self._log_R)
        return float(np.sum(np.arange(len(R)) * R))
    
    def get_posterior_mean_at_r0(self) -> float:
        """Get posterior mean for r=0 hypothesis (for testing bug fix)."""
        if len(self._n) == 0:
            return self.config.mu0
        mu_n, _, _, _ = self._get_posterior_params(
            self._n[:1], self._sum_x[:1], self._sum_x2[:1]
        )
        return float(mu_n[0])
    
    @property
    def change_prob_history(self) -> np.ndarray:
        return np.array(self._change_probs)
    
    @property
    def map_run_length_history(self) -> np.ndarray:
        return np.array(self._map_run_lengths)


# =============================================================================
# FAST-IN / SLOW-OUT TRANSITION GATE
# =============================================================================

class FastInSlowOutGate:
    """
    Fast-in / slow-out logic to reduce whipsaw while being responsive.
    
    Fast-In Triggers (immediate transition):
        - P(CP) > cp_spike_thr
        - Expected run length < erl_enter
        - Vol channel P(CP) > vol_spike_thr
        
    Slow-Out Conditions (maintain transition while any hold):
        - EWMA(P(CP)) > cp_hold_thr
        - Expected run length < erl_exit
        
    Calm-Based Recovery:
        - Exit transition only when continuous risk < calm_thr for calm_days
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        
        # State tracking
        self._in_transition: bool = False
        self._ema_cp: float = 0.0
        self._ema_alpha: float = 2.0 / (self.config.ema_span + 1)
        self._calm_streak: int = 0
        self._t: int = 0
    
    def reset(self):
        self._in_transition = False
        self._ema_cp = 0.0
        self._calm_streak = 0
        self._t = 0
    
    def update(
        self,
        combined_cp: float,
        expected_run_length: float,
        vol_channel_cp: float
    ) -> Tuple[bool, float, float]:
        """
        Update gate with new observation.
        
        Args:
            combined_cp: Combined P(changepoint) from noisy-OR
            expected_run_length: E[run length] from BOCPD
            vol_channel_cp: P(changepoint) from volatility channel specifically
            
        Returns:
            (in_transition, risk_cp, risk_rl)
            - in_transition: Whether we're in transition mode
            - risk_cp: Continuous risk score from change probability [0, 1]
            - risk_rl: Continuous risk score from expected run length [0, 1]
        """
        self._t += 1
        
        # Update EWMA of change probability
        self._ema_cp = self._ema_alpha * combined_cp + (1 - self._ema_alpha) * self._ema_cp
        
        # Compute continuous risk scores
        # risk_cp: linear ramp from cp_lo to cp_hi
        risk_cp = np.clip(
            (combined_cp - self.config.cp_lo) / (self.config.cp_hi - self.config.cp_lo),
            0.0, 1.0
        )
        
        # risk_rl: linear ramp from erl_enter to erl_floor
        # When ERL is low (young regime), risk is high
        if expected_run_length >= self.config.erl_enter:
            risk_rl = 0.0
        elif expected_run_length <= self.config.erl_floor:
            risk_rl = 1.0
        else:
            risk_rl = (self.config.erl_enter - expected_run_length) / (self.config.erl_enter - self.config.erl_floor)
        
        # Combined continuous risk (take maximum - worst case)
        continuous_risk = max(risk_cp, risk_rl)
        
        # Fast-in triggers (any one triggers transition)
        fast_in_triggered = (
            combined_cp > self.config.cp_spike_thr or
            expected_run_length < self.config.erl_enter or
            vol_channel_cp > self.config.vol_spike_thr
        )
        
        # Slow-out conditions (all must be false to consider exit)
        slow_out_hold = (
            self._ema_cp > self.config.cp_hold_thr or
            expected_run_length < self.config.erl_exit
        )
        
        # State machine update
        if not self._in_transition:
            # Currently in normal mode
            if fast_in_triggered:
                self._in_transition = True
                self._calm_streak = 0
        else:
            # Currently in transition mode
            if slow_out_hold:
                # Conditions still risky, maintain transition
                self._calm_streak = 0
            else:
                # Conditions calming down
                if continuous_risk < self.config.calm_risk_thr:
                    self._calm_streak += 1
                else:
                    self._calm_streak = 0
                
                # Exit transition only after sustained calm period
                if self._calm_streak >= self.config.calm_days:
                    self._in_transition = False
                    self._calm_streak = 0
        
        return self._in_transition, float(risk_cp), float(risk_rl)
    
    @property
    def in_transition(self) -> bool:
        return self._in_transition
    
    @property
    def calm_streak(self) -> int:
        return self._calm_streak
    
    @property
    def ema_cp(self) -> float:
        return self._ema_cp


# Legacy alias for backward compatibility
TransitionRiskGate = FastInSlowOutGate


# =============================================================================
# BULL-FRIENDLY RISK OVERLAY WITH CONTINUOUS RISK SCORING
# =============================================================================

class RiskOverlay:
    """
    Bull-friendly risk overlay with continuous risk scoring.
    
    Signal Timing (NO LOOKAHEAD):
        The position scalar for day t is computed at end of day t-1,
        using only OHLCV[0:t-1] and BOCPD state through t-1.
        
    Continuous Risk Scoring:
        - risk_cp: Linear ramp based on P(changepoint)
        - risk_rl: Linear ramp based on expected run length
        - combined_risk: max(risk_cp, risk_rl)
        
    Position Mapping (smooth):
        - Normal mode: pos = 1.0
        - Transition mode: pos = 1 - risk * (1 - transition_position)
        - Severe mode: pos = severe_position
        
    Recovery:
        Conditional on calm (risk < threshold for N consecutive days),
        NOT fixed days.
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        self._ema_position: Optional[float] = None
        self._ema_alpha: float = 2.0 / (self.config.ema_span + 1)
    
    def reset(self):
        self._ema_position = None
    
    def compute_position(
        self,
        in_transition: bool,
        risk_cp: float,
        risk_rl: float,
        vol_channel_cp: float,
        expected_run_length: float = 100.0
    ) -> float:
        """
        Compute position scalar [0, 1] for next day.
        
        Args:
            in_transition: Whether we're in transition mode (from gate)
            risk_cp: Continuous risk score from P(changepoint) [0, 1]
            risk_rl: Continuous risk score from expected run length [0, 1]
            vol_channel_cp: P(changepoint) from volatility channel
            expected_run_length: E[run length] from BOCPD (for severe mode)
            
        Returns:
            Position scalar in [0, 1]
        """
        # Check for severe mode:
        # 1. Volatility channel spike, OR
        # 2. Very low expected run length (young regime with high uncertainty)
        is_severe = (
            vol_channel_cp > self.config.vol_severe_thr or
            expected_run_length < self.config.erl_spike_thr or
            (risk_rl > 0.9 and vol_channel_cp > self.config.vol_spike_thr)  # ERL low + vol elevated
        )
        
        if is_severe:
            raw_position = self.config.severe_position
        elif in_transition:
            # Continuous risk scoring: take worst case
            combined_risk = max(risk_cp, risk_rl)
            
            # Smooth position mapping:
            # pos = 1 - risk * (1 - transition_position)
            # When risk=0: pos=1.0, when risk=1: pos=transition_position
            raw_position = 1.0 - combined_risk * (1.0 - self.config.transition_position)
        else:
            # Normal mode - fully invested
            raw_position = self.config.normal_position
        
        # EMA smoothing to reduce whipsaws
        if self._ema_position is None:
            self._ema_position = raw_position
        else:
            self._ema_position = (
                self._ema_alpha * raw_position + 
                (1 - self._ema_alpha) * self._ema_position
            )
        
        return float(self._ema_position)


# =============================================================================
# WEIGHTED NOISY-OR COMBINATION
# =============================================================================

def weighted_noisy_or(probs: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Combine change probabilities using weighted noisy-OR rule.
    
    P(CP) = 1 - prod(1 - w_i * p_i)
    
    Weights > 1 amplify the channel's influence (for volatility channels).
    Weights < 1 dampen the channel's influence (for noisy channels).
    
    Args:
        probs: Dict of channel_name -> P(changepoint)
        weights: Dict of channel_name -> weight
        
    Returns:
        Combined P(changepoint)
    """
    prod_no_change = 1.0
    
    for name, p in probs.items():
        w = weights.get(name, 1.0)
        # Clamp weighted probability to [0, 1]
        wp = min(1.0, max(0.0, w * p))
        prod_no_change *= (1 - wp)
    
    return 1 - prod_no_change


# =============================================================================
# ALPHA REGIME DETECTOR v2
# =============================================================================

class AlphaRegimeDetectorV2:
    """
    High-level regime detector with bull-friendly risk overlay.
    
    Key Improvements:
        1. BOCPD bug fix (r=0 includes x_t)
        2. Weighted noisy-OR emphasizing volatility channels
        3. Transition risk gating (anti-whipsaw)
        4. Position scalar output instead of binary exit
        
    Signal Timing (CRITICAL - NO LOOKAHEAD):
        update(t) processes OHLCV[t] and returns RegimeState for end of day t.
        The position_scalar in RegimeState should be applied to day t+1's returns.
        This is because we don't know day t's OHLCV until end of day t.
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        bocpd_config: Optional[BOCPDConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        overlay_config: Optional[OverlayConfig] = None
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.bocpd_config = bocpd_config or BOCPDConfig()
        self.regime_config = regime_config or RegimeConfig()
        self.overlay_config = overlay_config or OverlayConfig()
        
        self._tickers: Dict[str, Dict] = {}
    
    def _init_ticker(self, ticker: str):
        """Initialize state for a new ticker."""
        feature_engine = FeatureEngine(self.feature_config)
        
        bocpds = {}
        if self.regime_config.use_z_ret:
            bocpds['z_ret'] = BOCPD(self.bocpd_config)
        if self.regime_config.use_z_gap:
            bocpds['z_gap'] = BOCPD(self.bocpd_config)
        if self.regime_config.use_intraday:
            bocpds['intraday'] = BOCPD(self.bocpd_config)
        if self.regime_config.use_log_sigma:
            bocpds['log_sigma'] = BOCPD(self.bocpd_config)
        if self.regime_config.use_range_vol:
            bocpds['range_vol'] = BOCPD(self.bocpd_config)
        
        self._tickers[ticker] = {
            'feature_engine': feature_engine,
            'bocpds': bocpds,
            'transition_gate': FastInSlowOutGate(self.overlay_config),
            'risk_overlay': RiskOverlay(self.overlay_config),
            'regime_id': 0,
            'days_in_regime': 0,
            'change_prob_history': [],
            'detected_cps': [],
            'last_cp_time': 0
        }
    
    def update(
        self,
        ticker: str,
        timestamp: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None
    ) -> Optional[RegimeState]:
        """
        Process new OHLCV bar for a ticker.
        
        Signal Timing:
            This method is called at end of day t with OHLCV[t].
            The returned position_scalar should be applied to day t+1.
        """
        if ticker not in self._tickers:
            self._init_ticker(ticker)
        
        state = self._tickers[ticker]
        
        # Compute features
        feat = state['feature_engine'].update(
            timestamp, open_, high, low, close, volume
        )
        
        if feat is None:
            return None
        
        # Update each BOCPD channel
        individual_probs = {}
        map_drop_any = False
        
        for name, bocpd in state['bocpds'].items():
            if name == 'z_ret':
                p = bocpd.update(feat.z_ret)
            elif name == 'z_gap':
                p = bocpd.update(feat.z_gap)
            elif name == 'intraday':
                p = bocpd.update(feat.intraday)
            elif name == 'log_sigma':
                p = bocpd.update(feat.log_sigma)
            elif name == 'range_vol' and feat.range_vol is not None:
                p = bocpd.update(feat.range_vol)
            else:
                continue
            
            individual_probs[name] = p
            
            if bocpd.detect_map_drop(
                from_threshold=self.regime_config.map_drop_from,
                to_threshold=self.regime_config.map_drop_to
            ):
                map_drop_any = True
        
        # Weighted noisy-OR combination
        change_prob = weighted_noisy_or(
            individual_probs, 
            self.regime_config.channel_weights
        )
        
        # Get MAP run length from primary channel
        if 'z_ret' in state['bocpds']:
            map_rl = state['bocpds']['z_ret'].get_map_run_length()
        else:
            map_rl = list(state['bocpds'].values())[0].get_map_run_length()
        
        state['change_prob_history'].append(change_prob)
        state['days_in_regime'] += 1
        t = len(state['change_prob_history'])
        
        # Get expected run length from primary channel (for continuous risk scoring)
        if 'z_ret' in state['bocpds']:
            expected_rl = state['bocpds']['z_ret'].get_expected_run_length()
        else:
            expected_rl = list(state['bocpds'].values())[0].get_expected_run_length()
        
        # Get volatility channel change prob (for severe mode detection)
        vol_channel_cp = 0.0
        if 'range_vol' in individual_probs:
            vol_channel_cp = max(vol_channel_cp, individual_probs['range_vol'])
        if 'log_sigma' in individual_probs:
            vol_channel_cp = max(vol_channel_cp, individual_probs['log_sigma'])
        
        # Fast-in / slow-out transition gating
        in_transition, risk_cp, risk_rl = state['transition_gate'].update(
            combined_cp=change_prob,
            expected_run_length=expected_rl,
            vol_channel_cp=vol_channel_cp
        )
        
        # Detect change point (for regime_id tracking)
        time_since_last = t - state['last_cp_time']
        if in_transition and time_since_last >= self.regime_config.min_spacing:
            # Only increment regime_id if we see a sustained high-risk period
            if risk_cp > 0.5 or risk_rl > 0.5:
                state['detected_cps'].append(t)
                state['regime_id'] += 1
                state['days_in_regime'] = 0
                state['last_cp_time'] = t
        
        # Risk overlay - compute position scalar using continuous risk
        position_scalar = state['risk_overlay'].compute_position(
            in_transition=in_transition,
            risk_cp=risk_cp,
            risk_rl=risk_rl,
            vol_channel_cp=vol_channel_cp,
            expected_run_length=expected_rl
        )
        
        return RegimeState(
            ticker=ticker,
            timestamp=timestamp,
            change_prob=change_prob,
            individual_probs=individual_probs,
            run_length_map=map_rl,
            regime_id=state['regime_id'],
            days_in_regime=state['days_in_regime'],
            position_scalar=position_scalar,
            expected_run_length=expected_rl,
            in_transition=in_transition,
            risk_cp=risk_cp,
            risk_rl=risk_rl
        )
    
    def process_dataframe(
        self,
        ticker: str,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process entire DataFrame of OHLCV data for a ticker."""
        results = []
        
        for idx, row in df.iterrows():
            timestamp = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
            
            state = self.update(
                ticker=ticker,
                timestamp=timestamp,
                open_=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row.get('Volume', None)
            )
            
            if state is not None:
                result = {
                    'timestamp': state.timestamp,
                    'change_prob': state.change_prob,
                    'run_length_map': state.run_length_map,
                    'regime_id': state.regime_id,
                    'days_in_regime': state.days_in_regime,
                    'position_scalar': state.position_scalar,
                    'expected_run_length': state.expected_run_length,
                    'in_transition': state.in_transition,
                    'risk_cp': state.risk_cp,
                    'risk_rl': state.risk_rl
                }
                for name, prob in state.individual_probs.items():
                    result[f'cp_{name}'] = prob
                results.append(result)
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        return result_df
    
    def get_detected_changepoints(self, ticker: str) -> List[int]:
        if ticker not in self._tickers:
            return []
        return self._tickers[ticker]['detected_cps']
    
    def reset(self, ticker: Optional[str] = None):
        if ticker is not None:
            if ticker in self._tickers:
                del self._tickers[ticker]
        else:
            self._tickers.clear()


# =============================================================================
# UNBIASED BACKTEST STRATEGY
# =============================================================================

class UnbiasedRegimeStrategy:
    """
    Backtest-ready strategy with proper signal lagging.
    
    Signal Timing (NO LOOKAHEAD):
        signals[t] is computed at end of day t-1, using OHLCV[0:t-1].
        signals[t] is applied to return[t] = close[t]/close[t-1] - 1.
        
    This class handles the 1-day lag automatically.
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        bocpd_config: Optional[BOCPDConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        overlay_config: Optional[OverlayConfig] = None
    ):
        self.detector = AlphaRegimeDetectorV2(
            feature_config, bocpd_config, regime_config, overlay_config
        )
        self._position_history: List[float] = []
    
    def get_signals(
        self,
        ticker: str,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Generate properly lagged position signals.
        
        Returns:
            Series indexed by date, with position scalar for that day.
            signals[t] was computed using data available at end of t-1.
        """
        results = self.detector.process_dataframe(ticker, df)
        
        if results.empty:
            return pd.Series(dtype=float)
        
        # The position_scalar at index t is computed using data through t,
        # so it should be applied to day t+1. Shift by 1.
        signals = results['position_scalar'].shift(1).fillna(1.0)
        
        return signals
    
    def backtest(
        self,
        ticker: str,
        df: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Run backtest with proper signal lagging.
        
        Returns:
            Dict with strategy and benchmark metrics
        """
        signals = self.get_signals(ticker, df)
        
        if signals.empty:
            return {'error': 'No signals generated'}
        
        # Calculate returns
        close = df['Close']
        returns = close.pct_change()
        
        # Align signals with returns
        aligned = pd.DataFrame({
            'return': returns,
            'signal': signals
        }).dropna()
        
        if len(aligned) == 0:
            return {'error': 'No aligned data'}
        
        # Strategy returns
        strat_returns = aligned['signal'] * aligned['return']
        
        # Transaction costs
        position_changes = aligned['signal'].diff().abs().fillna(0)
        tc = position_changes * transaction_cost
        strat_returns_net = strat_returns - tc
        
        # Metrics
        def calc_metrics(rets: pd.Series) -> Dict:
            total = (1 + rets).prod() - 1
            years = len(rets) / 252
            ann_ret = (1 + total) ** (1/max(years, 0.01)) - 1
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cum = (1 + rets).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / peak
            max_dd = dd.min()
            return {
                'total_return': total * 100,
                'annual_return': ann_ret * 100,
                'annual_vol': ann_vol * 100,
                'sharpe': sharpe,
                'max_drawdown': max_dd * 100
            }
        
        strat_m = calc_metrics(strat_returns_net)
        bh_m = calc_metrics(aligned['return'])
        
        # Additional stats
        avg_position = aligned['signal'].mean()
        time_reduced = (aligned['signal'] < 1.0).mean()
        trades = (position_changes > 0.1).sum()
        
        return {
            'strategy': strat_m,
            'benchmark': bh_m,
            'avg_position': avg_position,
            'time_reduced': time_reduced * 100,
            'trades': trades,
            'days': len(aligned)
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate the bull-friendly regime detector."""
    print("=" * 70)
    print("ALPHA REGIME DETECTOR V2 - Bull-Friendly Risk Overlay")
    print("=" * 70)
    
    try:
        import yfinance as yf
        
        print("\nDownloading SPY data...")
        spy = yf.download("SPY", start="2020-01-01", end="2025-01-01", progress=False)
        
        if spy.empty:
            print("Failed to download data")
            return
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        print(f"Data: {len(spy)} bars")
        
        # Initialize with bull-friendly settings (continuous risk scoring)
        strategy = UnbiasedRegimeStrategy(
            feature_config=FeatureConfig(ewma_span=30, use_range_vol=True),
            bocpd_config=BOCPDConfig(hazard_rate=1/200),  # Less frequent CPs
            regime_config=RegimeConfig(
                use_range_vol=True,
                detection_method="confirmed",
                min_spacing=15
            ),
            overlay_config=OverlayConfig(
                normal_position=1.0,
                transition_position=0.75,  # Bull-friendly: only 25% reduction
                severe_position=0.25,
                cp_lo=0.15,
                cp_hi=0.40,
                erl_enter=15,
                erl_floor=5,
                erl_exit=35,
                calm_days=5
            )
        )
        
        # Backtest
        results = strategy.backtest("SPY", spy, transaction_cost=0.001)
        
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS (SPY 2020-2024)")
        print("=" * 50)
        
        print(f"\nStrategy:")
        print(f"  Total Return:  {results['strategy']['total_return']:+.1f}%")
        print(f"  Sharpe Ratio:  {results['strategy']['sharpe']:.2f}")
        print(f"  Max Drawdown:  {results['strategy']['max_drawdown']:.1f}%")
        
        print(f"\nBuy & Hold:")
        print(f"  Total Return:  {results['benchmark']['total_return']:+.1f}%")
        print(f"  Sharpe Ratio:  {results['benchmark']['sharpe']:.2f}")
        print(f"  Max Drawdown:  {results['benchmark']['max_drawdown']:.1f}%")
        
        print(f"\nStrategy Stats:")
        print(f"  Avg Position:  {results['avg_position']:.2f}")
        print(f"  Time Reduced:  {results['time_reduced']:.1f}%")
        print(f"  Trade Count:   {results['trades']}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    example_usage()
