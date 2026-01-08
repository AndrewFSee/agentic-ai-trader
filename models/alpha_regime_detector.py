"""
Alpha Regime Detector using Bayesian Online Change Point Detection

Production-quality regime detection for identifying distribution shifts in 
return-generating processes across multiple tickers using OHLCV data.

Based on Adams & MacKay (2007): "Bayesian Online Changepoint Detection"
https://arxiv.org/abs/0710.3742

Architecture:
    OHLCV Data → Feature Engineering → BOCPD → Regime Detection → Trading Signals

Key Features:
    - Scale-free, cross-ticker stable features
    - No lookahead bias (strictly online)
    - Numerically stable log-space calculations
    - Multi-ticker support with shared hyperparameters
    - Noisy-OR combination of multiple BOCPD channels

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import warnings

# =============================================================================
# CONFIGURATION & DATA STRUCTURES
# =============================================================================

class VolatilityEstimator(Enum):
    """Volatility estimation method."""
    EWMA = "ewma"                    # Exponentially weighted moving average of squared returns
    PARKINSON = "parkinson"          # High-Low range based (Parkinson 1980)
    GARMAN_KLASS = "garman_klass"    # OHLC-based (Garman & Klass 1980)


class IntradayMeasure(Enum):
    """Intraday pressure measurement."""
    Z_DAY = "z_day"                  # Standardized close-to-open return
    CLV = "clv"                      # Close Location Value


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    
    # Volatility estimation
    vol_estimator: VolatilityEstimator = VolatilityEstimator.EWMA
    ewma_span: int = 30              # EWMA span for volatility
    vol_floor: float = 1e-8          # Minimum volatility to prevent division by zero
    
    # Intraday measure
    intraday_measure: IntradayMeasure = IntradayMeasure.CLV
    
    # Optional features (disabled by default)
    use_log_sigma: bool = True       # Include log volatility as feature
    use_tail_rate: bool = False      # Rolling tail exceedance rate
    use_downside_freq: bool = False  # Rolling downside frequency
    use_volume_cond: bool = False    # Volume-conditioned returns
    
    # Rolling window for optional features
    rolling_window: int = 20
    tail_threshold: float = 2.0      # |z| > threshold for tail events


@dataclass
class BOCPDConfig:
    """Configuration for BOCPD model."""
    
    hazard_rate: float = 0.05        # Prior prob of changepoint (1/expected_run_length)
    
    # Normal-Inverse-Gamma prior parameters
    mu0: float = 0.0                 # Prior mean
    kappa0: float = 0.1              # Prior precision weight
    alpha0: float = 2.0              # Prior shape (> 1 for proper prior)
    beta0: float = 1.0               # Prior rate
    
    max_run_length: int = 500        # Truncation for efficiency


@dataclass 
class RegimeConfig:
    """Configuration for regime detection."""
    
    # Detection mode
    mode: Literal["univariate", "multivariate"] = "univariate"
    
    # Features to use in BOCPD (for univariate mode)
    use_z_ret: bool = True
    use_z_gap: bool = True
    use_intraday: bool = True
    use_log_sigma: bool = False      # Often noisy, disabled by default
    
    # Change point detection method
    detection_method: Literal["prob", "map_drop"] = "map_drop"
    
    # Thresholds
    cp_threshold: float = 0.5        # P(change) threshold (for prob method)
    min_spacing: int = 5             # Minimum days between detected CPs
    
    # MAP drop parameters (for map_drop method)
    map_drop_from: int = 5           # MAP must drop from > this
    map_drop_to: int = 2             # MAP must drop to <= this
    
    # False positive control
    use_quantile_threshold: bool = False
    quantile_level: float = 0.95     # If using quantile-based threshold


@dataclass
class FeatureVector:
    """Container for computed features at a single time step."""
    timestamp: pd.Timestamp
    z_ret: float                     # Standardized return
    z_gap: float                     # Standardized overnight gap
    intraday: float                  # CLV or z_day
    log_sigma: float                 # Log volatility
    sigma: float                     # Raw volatility estimate
    
    # Optional features
    tail_rate: Optional[float] = None
    downside_freq: Optional[float] = None
    volume_cond_ret: Optional[float] = None
    
    def to_array(self, include_optional: bool = False) -> np.ndarray:
        """Convert to numpy array for BOCPD input."""
        features = [self.z_ret, self.z_gap, self.intraday, self.log_sigma]
        if include_optional:
            if self.tail_rate is not None:
                features.append(self.tail_rate)
            if self.downside_freq is not None:
                features.append(self.downside_freq)
            if self.volume_cond_ret is not None:
                features.append(self.volume_cond_ret)
        return np.array(features)


@dataclass
class RegimeState:
    """Current regime state for a ticker."""
    ticker: str
    timestamp: pd.Timestamp
    change_prob: float               # Combined P(changepoint)
    individual_probs: Dict[str, float]  # Per-feature change probs
    run_length_map: int              # MAP estimate of current run length
    regime_id: int                   # Current regime identifier
    days_in_regime: int              # Days since last detected change


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngine:
    """
    Computes scale-free, cross-ticker stable features from OHLCV data.
    
    All computations are strictly online (no lookahead).
    
    Features:
        z_ret: Standardized log return
        z_gap: Standardized overnight gap
        intraday: Close Location Value or standardized intraday return
        log_sigma: Log of volatility estimate
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature engine with configuration."""
        self.config = config or FeatureConfig()
        self.reset()
    
    def reset(self):
        """Reset internal state for new ticker."""
        self._prev_close: Optional[float] = None
        self._ewma_var: Optional[float] = None
        self._ewma_alpha: float = 2.0 / (self.config.ewma_span + 1)
        
        # For optional rolling features
        self._z_ret_history: List[float] = []
        self._log_volume_history: List[float] = []
    
    def _compute_ewma_volatility(self, return_sq: float) -> float:
        """Update EWMA volatility estimate."""
        if self._ewma_var is None:
            self._ewma_var = return_sq
        else:
            self._ewma_var = (
                self._ewma_alpha * return_sq + 
                (1 - self._ewma_alpha) * self._ewma_var
            )
        return np.sqrt(max(self._ewma_var, self.config.vol_floor))
    
    def _compute_parkinson_volatility(
        self, high: float, low: float
    ) -> float:
        """
        Parkinson (1980) volatility estimator using high-low range.
        
        σ² = (1/4ln(2)) * (ln(H/L))²
        """
        if high <= 0 or low <= 0 or high <= low:
            return np.sqrt(self.config.vol_floor)
        
        log_hl = np.log(high / low)
        variance = log_hl**2 / (4 * np.log(2))
        return np.sqrt(max(variance, self.config.vol_floor))
    
    def _compute_garman_klass_volatility(
        self, 
        open_: float, 
        high: float, 
        low: float, 
        close: float
    ) -> float:
        """
        Garman-Klass (1980) volatility estimator using OHLC.
        
        σ² = 0.5*(ln(H/L))² - (2ln(2)-1)*(ln(C/O))²
        """
        if high <= 0 or low <= 0 or open_ <= 0 or close <= 0:
            return np.sqrt(self.config.vol_floor)
        if high <= low:
            return np.sqrt(self.config.vol_floor)
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return np.sqrt(max(variance, self.config.vol_floor))
    
    def _compute_clv(
        self, 
        high: float, 
        low: float, 
        close: float
    ) -> float:
        """
        Close Location Value: where close falls in day's range.
        
        CLV = ((C-L) - (H-C)) / (H-L)
             = (2C - H - L) / (H - L)
        
        Returns value in [-1, 1]:
            +1: Closed at high
            -1: Closed at low
             0: Closed at midpoint
        """
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
        """
        Process new OHLCV bar and compute features.
        
        Args:
            timestamp: Bar timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume (optional, for volume-conditioned features)
            
        Returns:
            FeatureVector if enough history, else None
        """
        # Need previous close for returns
        if self._prev_close is None:
            self._prev_close = close
            return None
        
        # Compute log return
        log_return = np.log(close / self._prev_close) if self._prev_close > 0 else 0.0
        
        # Compute overnight gap
        overnight_gap = np.log(open_ / self._prev_close) if self._prev_close > 0 else 0.0
        
        # Compute intraday return
        intraday_return = np.log(close / open_) if open_ > 0 else 0.0
        
        # Compute volatility estimate
        if self.config.vol_estimator == VolatilityEstimator.EWMA:
            sigma = self._compute_ewma_volatility(log_return**2)
        elif self.config.vol_estimator == VolatilityEstimator.PARKINSON:
            sigma = self._compute_parkinson_volatility(high, low)
        elif self.config.vol_estimator == VolatilityEstimator.GARMAN_KLASS:
            sigma = self._compute_garman_klass_volatility(open_, high, low, close)
        else:
            sigma = self._compute_ewma_volatility(log_return**2)
        
        # Ensure sigma is positive
        sigma = max(sigma, self.config.vol_floor)
        
        # Compute standardized features
        z_ret = log_return / sigma
        z_gap = overnight_gap / sigma
        
        # Compute intraday measure
        if self.config.intraday_measure == IntradayMeasure.CLV:
            intraday = self._compute_clv(high, low, close)
        else:  # Z_DAY
            intraday = intraday_return / sigma
        
        # Log volatility
        log_sigma = np.log(sigma)
        
        # Update history for optional features
        self._z_ret_history.append(z_ret)
        if volume is not None and volume > 0:
            self._log_volume_history.append(np.log(volume))
        
        # Compute optional features
        tail_rate = None
        downside_freq = None
        volume_cond_ret = None
        
        if len(self._z_ret_history) >= self.config.rolling_window:
            recent_z = self._z_ret_history[-self.config.rolling_window:]
            
            if self.config.use_tail_rate:
                tail_rate = np.mean([1 if abs(z) > self.config.tail_threshold else 0 
                                     for z in recent_z])
            
            if self.config.use_downside_freq:
                downside_freq = np.mean([1 if z < 0 else 0 for z in recent_z])
            
            if self.config.use_volume_cond and len(self._log_volume_history) >= self.config.rolling_window:
                recent_vol = self._log_volume_history[-self.config.rolling_window:]
                vol_mean = np.mean(recent_vol)
                vol_std = np.std(recent_vol)
                if vol_std > 1e-10:
                    vol_zscore = (self._log_volume_history[-1] - vol_mean) / vol_std
                    volume_cond_ret = z_ret * vol_zscore
        
        # Trim history to prevent memory growth
        max_history = self.config.rolling_window * 2
        if len(self._z_ret_history) > max_history:
            self._z_ret_history = self._z_ret_history[-max_history:]
        if len(self._log_volume_history) > max_history:
            self._log_volume_history = self._log_volume_history[-max_history:]
        
        # Update previous close
        self._prev_close = close
        
        return FeatureVector(
            timestamp=timestamp,
            z_ret=z_ret,
            z_gap=z_gap,
            intraday=intraday,
            log_sigma=log_sigma,
            sigma=sigma,
            tail_rate=tail_rate,
            downside_freq=downside_freq,
            volume_cond_ret=volume_cond_ret
        )
    
    def process_dataframe(
        self, 
        df: pd.DataFrame,
        ohlcv_cols: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Process entire DataFrame of OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            ohlcv_cols: Optional mapping of column names
                        Default: {'open': 'Open', 'high': 'High', ...}
        
        Returns:
            DataFrame with computed features
        """
        self.reset()
        
        # Default column mapping
        cols = ohlcv_cols or {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        features = []
        
        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            
            volume = row.get(cols.get('volume'), None)
            if pd.isna(volume):
                volume = None
            
            feat = self.update(
                timestamp=timestamp,
                open_=row[cols['open']],
                high=row[cols['high']],
                low=row[cols['low']],
                close=row[cols['close']],
                volume=volume
            )
            
            if feat is not None:
                features.append({
                    'timestamp': feat.timestamp,
                    'z_ret': feat.z_ret,
                    'z_gap': feat.z_gap,
                    'intraday': feat.intraday,
                    'log_sigma': feat.log_sigma,
                    'sigma': feat.sigma,
                    'tail_rate': feat.tail_rate,
                    'downside_freq': feat.downside_freq,
                    'volume_cond_ret': feat.volume_cond_ret
                })
        
        result_df = pd.DataFrame(features)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        
        return result_df


# =============================================================================
# BOCPD IMPLEMENTATION
# =============================================================================

class BOCPD:
    """
    Bayesian Online Change Point Detection (Adams & MacKay, 2007).
    
    Maintains posterior distribution over run lengths (time since last
    change point) and updates online with each new observation.
    
    Uses Normal-Inverse-Gamma conjugate prior for Gaussian likelihood
    with unknown mean and variance, resulting in Student-t predictive.
    
    BUG FIX (Jan 2026): When r_t = 0 (changepoint), the new regime's
    sufficient statistics now correctly include x_t as the FIRST observation:
        n[0] = 1, sum_x[0] = x, sum_x2[0] = x^2
    """
    
    def __init__(self, config: Optional[BOCPDConfig] = None):
        """Initialize BOCPD with configuration."""
        self.config = config or BOCPDConfig()
        
        # Precompute log hazards
        self._log_H = np.log(self.config.hazard_rate)
        self._log_1_minus_H = np.log(1 - self.config.hazard_rate)
        
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.t = 0
        
        # Run length posterior in log space
        # Initially P(r=0) = 1
        self._log_R = np.array([0.0])
        
        # Sufficient statistics for each run length hypothesis
        self._n = np.array([0.0])         # Observation count
        self._sum_x = np.array([0.0])     # Sum of observations
        self._sum_x2 = np.array([0.0])    # Sum of squared observations
        
        # History for diagnostics
        self._map_run_lengths: List[int] = []
        self._change_probs: List[float] = []
    
    def _get_posterior_params(
        self,
        n: np.ndarray,
        sum_x: np.ndarray,
        sum_x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Normal-Inverse-Gamma posterior parameters.
        
        Returns:
            (mu_n, kappa_n, alpha_n, beta_n) posterior parameters
        """
        kappa_n = self.config.kappa0 + n
        mu_n = (self.config.kappa0 * self.config.mu0 + sum_x) / kappa_n
        alpha_n = self.config.alpha0 + n / 2.0
        
        # Sample variance (with protection against division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            xbar = np.where(n > 0, sum_x / n, 0.0)
        
        ss = np.maximum(sum_x2 - n * xbar**2, 0.0)
        
        # Coupling term from prior-data disagreement
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
        """
        Compute log-likelihood of x under Student-t predictive distribution.
        
        The predictive is Student-t with:
            - df = 2 * alpha_n
            - location = mu_n
            - scale = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
        """
        mu_n, kappa_n, alpha_n, beta_n = self._get_posterior_params(n, sum_x, sum_x2)
        
        df = 2.0 * alpha_n
        scale = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        scale = np.maximum(scale, 1e-10)  # Numerical stability
        
        # Standardized deviation
        z = (x - mu_n)**2 / (df * scale)
        
        # Student-t log PDF
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
        
        # Update sufficient statistics
        # BUG FIX: r=0 hypothesis includes x as first observation
        # OLD (buggy): n[0]=0, sum_x[0]=0, sum_x2[0]=0
        # NEW (fixed): n[0]=1, sum_x[0]=x, sum_x2[0]=x^2
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
        
        # Compute change probability P(r=0)
        R = np.exp(log_R_new)
        change_prob = float(R[0])
        
        # Track history
        map_run_length = int(np.argmax(R))
        self._map_run_lengths.append(map_run_length)
        self._change_probs.append(change_prob)
        
        return change_prob
    
    def get_change_prob(self) -> float:
        """Get current P(r = 0) - probability this is a change point."""
        R = np.exp(self._log_R)
        return float(R[0])
    
    def detect_map_drop(self, from_threshold: int = 5, to_threshold: int = 2) -> bool:
        """
        Detect if MAP run length dropped significantly.
        
        This is the recommended detection method from Adams & MacKay.
        A change is detected when MAP drops from >from_threshold to <=to_threshold.
        
        Args:
            from_threshold: MAP must have been > this value
            to_threshold: MAP must drop to <= this value
            
        Returns:
            True if a change was just detected
        """
        if len(self._map_run_lengths) < 2:
            return False
        
        prev_map = self._map_run_lengths[-2]
        curr_map = self._map_run_lengths[-1]
        
        return prev_map > from_threshold and curr_map <= to_threshold
    
    def get_run_length_distribution(self) -> np.ndarray:
        """Get full run length posterior distribution."""
        return np.exp(self._log_R)
    
    def get_map_run_length(self) -> int:
        """Get MAP estimate of current run length."""
        R = np.exp(self._log_R)
        return int(np.argmax(R))
    
    def get_expected_run_length(self) -> float:
        """Get expected run length under posterior."""
        R = np.exp(self._log_R)
        return float(np.sum(np.arange(len(R)) * R))
    
    @property
    def change_prob_history(self) -> np.ndarray:
        """Get history of change probabilities."""
        return np.array(self._change_probs)
    
    @property
    def map_run_length_history(self) -> np.ndarray:
        """Get history of MAP run lengths."""
        return np.array(self._map_run_lengths)


class MultivariateBOCPD:
    """
    Multivariate BOCPD for joint feature modeling.
    
    Uses Normal-Inverse-Wishart prior for multivariate Gaussian.
    Predictive is multivariate Student-t.
    """
    
    def __init__(
        self, 
        dim: int,
        config: Optional[BOCPDConfig] = None
    ):
        """
        Initialize multivariate BOCPD.
        
        Args:
            dim: Dimensionality of feature vector
            config: BOCPD configuration
        """
        self.dim = dim
        self.config = config or BOCPDConfig()
        
        self._log_H = np.log(self.config.hazard_rate)
        self._log_1_minus_H = np.log(1 - self.config.hazard_rate)
        
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.t = 0
        self._log_R = np.array([0.0])
        
        # Sufficient statistics: list of dicts for each run length
        self._stats = [{
            'n': 0,
            'sum_x': np.zeros(self.dim),
            'sum_xx': np.zeros((self.dim, self.dim))
        }]
        
        self._change_probs: List[float] = []
        self._map_run_lengths: List[int] = []
    
    def _compute_predictive_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute predictive log-likelihood for each run length.
        
        Simplified: Use product of independent Student-t marginals.
        (Full Wishart approach is complex; this is a reasonable approximation)
        """
        log_liks = np.zeros(len(self._stats))
        
        for i, stats in enumerate(self._stats):
            n = stats['n']
            if n < 2:
                # Use prior
                log_liks[i] = -0.5 * self.dim * np.log(2 * np.pi * self.config.beta0)
                log_liks[i] -= 0.5 * np.sum(x**2) / self.config.beta0
            else:
                # Empirical mean and variance
                mean = stats['sum_x'] / n
                var = np.diag(stats['sum_xx']) / n - mean**2
                var = np.maximum(var, 1e-6)
                
                # Simple Gaussian likelihood (approximation)
                log_liks[i] = -0.5 * np.sum(np.log(2 * np.pi * var))
                log_liks[i] -= 0.5 * np.sum((x - mean)**2 / var)
        
        return log_liks
    
    def update(self, x: np.ndarray) -> float:
        """
        Update with new multivariate observation.
        
        Args:
            x: Feature vector of shape (dim,)
            
        Returns:
            P(r_t = 0): Change probability
        """
        self.t += 1
        x = np.asarray(x)
        
        log_pred = self._compute_predictive_log_likelihood(x)
        
        log_growth = self._log_R + log_pred + self._log_1_minus_H
        log_cp = logsumexp(self._log_R + log_pred) + self._log_H
        
        log_R_new = np.concatenate([[log_cp], log_growth])
        log_R_new = log_R_new - logsumexp(log_R_new)
        
        # Update sufficient statistics
        new_stats = [{
            'n': 0,
            'sum_x': np.zeros(self.dim),
            'sum_xx': np.zeros((self.dim, self.dim))
        }]
        
        for stats in self._stats:
            new_stats.append({
                'n': stats['n'] + 1,
                'sum_x': stats['sum_x'] + x,
                'sum_xx': stats['sum_xx'] + np.outer(x, x)
            })
        
        # Truncate
        if len(log_R_new) > self.config.max_run_length:
            log_R_new = log_R_new[:self.config.max_run_length]
            new_stats = new_stats[:self.config.max_run_length]
        
        self._log_R = log_R_new
        self._stats = new_stats
        
        R = np.exp(log_R_new)
        change_prob = float(R[0])
        
        self._change_probs.append(change_prob)
        self._map_run_lengths.append(int(np.argmax(R)))
        
        return change_prob
    
    def get_change_prob(self) -> float:
        """Get current change probability."""
        return float(np.exp(self._log_R[0]))
    
    @property
    def change_prob_history(self) -> np.ndarray:
        return np.array(self._change_probs)


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class AlphaRegimeDetector:
    """
    High-level regime detector combining features and BOCPD.
    
    Supports:
        - Univariate mode: Separate BOCPD per feature, noisy-OR combination
        - Multivariate mode: Joint BOCPD on feature vector
        - Multi-ticker processing with shared hyperparameters
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        bocpd_config: Optional[BOCPDConfig] = None,
        regime_config: Optional[RegimeConfig] = None
    ):
        """
        Initialize regime detector.
        
        Args:
            feature_config: Configuration for feature engineering
            bocpd_config: Configuration for BOCPD model
            regime_config: Configuration for regime detection
        """
        self.feature_config = feature_config or FeatureConfig()
        self.bocpd_config = bocpd_config or BOCPDConfig()
        self.regime_config = regime_config or RegimeConfig()
        
        # Per-ticker state
        self._tickers: Dict[str, Dict] = {}
    
    def _init_ticker(self, ticker: str):
        """Initialize state for a new ticker."""
        feature_engine = FeatureEngine(self.feature_config)
        
        if self.regime_config.mode == "univariate":
            bocpds = {}
            if self.regime_config.use_z_ret:
                bocpds['z_ret'] = BOCPD(self.bocpd_config)
            if self.regime_config.use_z_gap:
                bocpds['z_gap'] = BOCPD(self.bocpd_config)
            if self.regime_config.use_intraday:
                bocpds['intraday'] = BOCPD(self.bocpd_config)
            if self.regime_config.use_log_sigma:
                bocpds['log_sigma'] = BOCPD(self.bocpd_config)
            
            self._tickers[ticker] = {
                'feature_engine': feature_engine,
                'bocpds': bocpds,
                'regime_id': 0,
                'days_in_regime': 0,
                'change_prob_history': [],
                'detected_cps': [],
                'last_cp_time': 0
            }
        else:
            # Multivariate mode
            dim = sum([
                self.regime_config.use_z_ret,
                self.regime_config.use_z_gap,
                self.regime_config.use_intraday,
                self.regime_config.use_log_sigma
            ])
            
            self._tickers[ticker] = {
                'feature_engine': feature_engine,
                'bocpd': MultivariateBOCPD(dim, self.bocpd_config),
                'regime_id': 0,
                'days_in_regime': 0,
                'change_prob_history': [],
                'detected_cps': [],
                'last_cp_time': 0
            }
    
    def _combine_probs_noisy_or(self, probs: Dict[str, float]) -> float:
        """
        Combine change probabilities using noisy-OR rule.
        
        P(CP) = 1 - prod(1 - p_i)
        
        This models: "a change occurred in at least one feature"
        """
        if not probs:
            return 0.0
        
        prod_no_change = 1.0
        for p in probs.values():
            prod_no_change *= (1 - p)
        
        return 1 - prod_no_change
    
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
        
        Args:
            ticker: Ticker symbol
            timestamp: Bar timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume (optional)
            
        Returns:
            RegimeState if features computed, else None
        """
        # Initialize ticker if new
        if ticker not in self._tickers:
            self._init_ticker(ticker)
        
        state = self._tickers[ticker]
        
        # Compute features
        feat = state['feature_engine'].update(
            timestamp, open_, high, low, close, volume
        )
        
        if feat is None:
            return None
        
        # Update BOCPD(s)
        individual_probs = {}
        
        if self.regime_config.mode == "univariate":
            # Update each BOCPD channel
            for name, bocpd in state['bocpds'].items():
                if name == 'z_ret':
                    p = bocpd.update(feat.z_ret)
                elif name == 'z_gap':
                    p = bocpd.update(feat.z_gap)
                elif name == 'intraday':
                    p = bocpd.update(feat.intraday)
                elif name == 'log_sigma':
                    p = bocpd.update(feat.log_sigma)
                else:
                    continue
                individual_probs[name] = p
            
            # Combine using noisy-OR
            change_prob = self._combine_probs_noisy_or(individual_probs)
            
            # Get MAP run length from primary channel (z_ret)
            if 'z_ret' in state['bocpds']:
                map_rl = state['bocpds']['z_ret'].get_map_run_length()
            else:
                map_rl = list(state['bocpds'].values())[0].get_map_run_length()
        
        else:
            # Multivariate mode
            feat_vec = []
            if self.regime_config.use_z_ret:
                feat_vec.append(feat.z_ret)
            if self.regime_config.use_z_gap:
                feat_vec.append(feat.z_gap)
            if self.regime_config.use_intraday:
                feat_vec.append(feat.intraday)
            if self.regime_config.use_log_sigma:
                feat_vec.append(feat.log_sigma)
            
            change_prob = state['bocpd'].update(np.array(feat_vec))
            individual_probs = {'joint': change_prob}
            map_rl = state['bocpd']._map_run_lengths[-1] if state['bocpd']._map_run_lengths else 0
        
        state['change_prob_history'].append(change_prob)
        
        # Detect change point
        state['days_in_regime'] += 1
        t = len(state['change_prob_history'])
        
        # Check if we should detect a change
        time_since_last = t - state['last_cp_time']
        detected = False
        
        if self.regime_config.detection_method == "map_drop":
            # Use MAP drop detection (recommended)
            if self.regime_config.mode == "univariate":
                # Detect if ANY channel shows MAP drop
                for name, bocpd in state['bocpds'].items():
                    if bocpd.detect_map_drop(
                        from_threshold=self.regime_config.map_drop_from,
                        to_threshold=self.regime_config.map_drop_to
                    ):
                        detected = True
                        break
            else:
                # Multivariate
                if len(state['bocpd']._map_run_lengths) >= 2:
                    prev_map = state['bocpd']._map_run_lengths[-2]
                    curr_map = state['bocpd']._map_run_lengths[-1]
                    detected = (prev_map > self.regime_config.map_drop_from and 
                               curr_map <= self.regime_config.map_drop_to)
        else:
            # Use probability threshold
            detected = change_prob > self.regime_config.cp_threshold
        
        if detected and time_since_last >= self.regime_config.min_spacing:
            state['detected_cps'].append(t)
            state['regime_id'] += 1
            state['days_in_regime'] = 0
            state['last_cp_time'] = t
        
        return RegimeState(
            ticker=ticker,
            timestamp=timestamp,
            change_prob=change_prob,
            individual_probs=individual_probs,
            run_length_map=map_rl,
            regime_id=state['regime_id'],
            days_in_regime=state['days_in_regime']
        )
    
    def process_dataframe(
        self,
        ticker: str,
        df: pd.DataFrame,
        ohlcv_cols: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Process entire DataFrame of OHLCV data for a ticker.
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV columns
            ohlcv_cols: Optional column name mapping
            
        Returns:
            DataFrame with regime detection results
        """
        cols = ohlcv_cols or {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        results = []
        
        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            
            volume = row.get(cols.get('volume'), None)
            if pd.isna(volume):
                volume = None
            
            state = self.update(
                ticker=ticker,
                timestamp=timestamp,
                open_=row[cols['open']],
                high=row[cols['high']],
                low=row[cols['low']],
                close=row[cols['close']],
                volume=volume
            )
            
            if state is not None:
                result = {
                    'timestamp': state.timestamp,
                    'change_prob': state.change_prob,
                    'run_length_map': state.run_length_map,
                    'regime_id': state.regime_id,
                    'days_in_regime': state.days_in_regime
                }
                # Add individual probs
                for name, prob in state.individual_probs.items():
                    result[f'cp_{name}'] = prob
                
                results.append(result)
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        
        return result_df
    
    def get_detected_changepoints(self, ticker: str) -> List[int]:
        """Get list of detected change point indices for a ticker."""
        if ticker not in self._tickers:
            return []
        return self._tickers[ticker]['detected_cps']
    
    def get_change_prob_history(self, ticker: str) -> np.ndarray:
        """Get change probability history for a ticker."""
        if ticker not in self._tickers:
            return np.array([])
        return np.array(self._tickers[ticker]['change_prob_history'])
    
    def reset(self, ticker: Optional[str] = None):
        """Reset state for a ticker or all tickers."""
        if ticker is not None:
            if ticker in self._tickers:
                del self._tickers[ticker]
        else:
            self._tickers.clear()


# =============================================================================
# MULTI-TICKER PROCESSOR
# =============================================================================

class MultiTickerRegimeDetector:
    """
    Process multiple tickers with shared hyperparameters.
    
    Provides batch processing and cross-ticker analysis.
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        bocpd_config: Optional[BOCPDConfig] = None,
        regime_config: Optional[RegimeConfig] = None
    ):
        """Initialize with shared configurations."""
        self.feature_config = feature_config or FeatureConfig()
        self.bocpd_config = bocpd_config or BOCPDConfig()
        self.regime_config = regime_config or RegimeConfig()
        
        self._detector = AlphaRegimeDetector(
            self.feature_config,
            self.bocpd_config,
            self.regime_config
        )
        
        self._results: Dict[str, pd.DataFrame] = {}
    
    def process_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        ohlcv_cols: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Process a single ticker.
        
        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame
            ohlcv_cols: Optional column mapping
            
        Returns:
            DataFrame with regime detection results
        """
        result = self._detector.process_dataframe(ticker, df, ohlcv_cols)
        self._results[ticker] = result
        return result
    
    def process_multiple(
        self,
        data: Dict[str, pd.DataFrame],
        ohlcv_cols: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple tickers.
        
        Args:
            data: Dict mapping ticker -> OHLCV DataFrame
            ohlcv_cols: Optional column mapping
            
        Returns:
            Dict mapping ticker -> results DataFrame
        """
        results = {}
        for ticker, df in data.items():
            results[ticker] = self.process_ticker(ticker, df, ohlcv_cols)
        return results
    
    def get_all_changepoints(self) -> Dict[str, List[int]]:
        """Get detected change points for all tickers."""
        return {
            ticker: self._detector.get_detected_changepoints(ticker)
            for ticker in self._results
        }
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of detected regimes across all tickers."""
        summaries = []
        
        for ticker, result_df in self._results.items():
            if result_df.empty:
                continue
            
            cps = self._detector.get_detected_changepoints(ticker)
            
            summaries.append({
                'ticker': ticker,
                'n_observations': len(result_df),
                'n_changepoints': len(cps),
                'n_regimes': result_df['regime_id'].max() + 1 if 'regime_id' in result_df else 1,
                'avg_change_prob': result_df['change_prob'].mean(),
                'max_change_prob': result_df['change_prob'].max()
            })
        
        return pd.DataFrame(summaries)


# =============================================================================
# DIAGNOSTICS & VISUALIZATION
# =============================================================================

class RegimeDiagnostics:
    """Diagnostic utilities for regime detection."""
    
    @staticmethod
    def compute_quantile_threshold(
        change_probs: np.ndarray,
        quantile: float = 0.95
    ) -> float:
        """Compute quantile-based threshold for change probabilities."""
        return float(np.quantile(change_probs, quantile))
    
    @staticmethod
    def detect_with_quantile(
        change_probs: np.ndarray,
        quantile: float = 0.95,
        min_spacing: int = 5
    ) -> List[int]:
        """
        Detect change points using quantile-based threshold.
        
        Args:
            change_probs: Array of change probabilities
            quantile: Quantile for threshold
            min_spacing: Minimum spacing between detections
            
        Returns:
            List of detected change point indices
        """
        threshold = np.quantile(change_probs, quantile)
        
        cps = []
        last_cp = -min_spacing
        
        for i, p in enumerate(change_probs):
            if p > threshold and (i - last_cp) >= min_spacing:
                cps.append(i)
                last_cp = i
        
        return cps
    
    @staticmethod
    def plot_change_probability(
        change_probs: np.ndarray,
        detected_cps: Optional[List[int]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        threshold: Optional[float] = None,
        title: str = "Change Probability"
    ):
        """
        Plot change probability over time.
        
        Args:
            change_probs: Array of change probabilities
            detected_cps: Optional list of detected change points
            dates: Optional datetime index for x-axis
            threshold: Optional threshold line
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        x = dates if dates is not None else np.arange(len(change_probs))
        
        ax.plot(x, change_probs, 'b-', alpha=0.7, label='P(changepoint)')
        
        if threshold is not None:
            ax.axhline(threshold, color='r', linestyle='--', 
                      alpha=0.7, label=f'Threshold = {threshold:.3f}')
        
        if detected_cps is not None:
            for cp in detected_cps:
                if cp < len(x):
                    ax.axvline(x[cp] if dates is not None else cp,
                              color='g', alpha=0.5, linestyle=':')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Change Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_returns_with_regimes(
        returns: np.ndarray,
        regime_ids: np.ndarray,
        detected_cps: Optional[List[int]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Returns with Detected Regimes"
    ):
        """
        Plot standardized returns with regime coloring.
        
        Args:
            returns: Array of (standardized) returns
            regime_ids: Array of regime identifiers
            detected_cps: Optional list of detected change points
            dates: Optional datetime index
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        x = dates if dates is not None else np.arange(len(returns))
        
        # Color by regime
        unique_regimes = np.unique(regime_ids)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            mask = regime_ids == regime
            ax.scatter(
                np.array(x)[mask], 
                returns[mask],
                c=[colors[i]], 
                alpha=0.6, 
                s=10,
                label=f'Regime {regime}'
            )
        
        if detected_cps is not None:
            for cp in detected_cps:
                if cp < len(x):
                    ax.axvline(x[cp] if dates is not None else cp,
                              color='red', alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Standardized Return')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate usage with synthetic and real data."""
    
    print("=" * 70)
    print("ALPHA REGIME DETECTOR - Example Usage")
    print("=" * 70)
    
    # Try to get real data
    try:
        import yfinance as yf
        
        print("\nDownloading SPY data...")
        spy = yf.download("SPY", start="2020-01-01", end="2025-01-01", progress=False)
        
        if spy.empty:
            print("Failed to download data")
            return
        
        # Handle multi-index columns
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        print(f"Data: {len(spy)} bars from {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
        
        # Initialize detector with configurations
        feature_config = FeatureConfig(
            vol_estimator=VolatilityEstimator.EWMA,
            ewma_span=30,
            intraday_measure=IntradayMeasure.CLV,
            use_log_sigma=True
        )
        
        bocpd_config = BOCPDConfig(
            hazard_rate=0.05,  # Expect ~20 day regimes
            mu0=0.0,
            kappa0=0.1,
            alpha0=2.0,
            beta0=1.0
        )
        
        regime_config = RegimeConfig(
            mode="univariate",
            use_z_ret=True,
            use_z_gap=True,
            use_intraday=True,
            use_log_sigma=False,
            detection_method="map_drop",
            map_drop_from=5,
            map_drop_to=2,
            min_spacing=10
        )
        
        detector = AlphaRegimeDetector(
            feature_config=feature_config,
            bocpd_config=bocpd_config,
            regime_config=regime_config
        )
        
        # Process data
        print("\nProcessing data...")
        results = detector.process_dataframe("SPY", spy)
        
        print(f"\nResults: {len(results)} observations")
        print(f"Detected change points: {len(detector.get_detected_changepoints('SPY'))}")
        print(f"Number of regimes: {results['regime_id'].max() + 1}")
        
        # Show change point dates
        cps = detector.get_detected_changepoints("SPY")
        print(f"\nChange point dates:")
        for i, cp in enumerate(cps[:15]):  # First 15
            if cp < len(results):
                date = results.index[cp]
                regime = results.iloc[cp]['regime_id']
                print(f"  {i+1}. {date.strftime('%Y-%m-%d')}: Regime {regime}")
        
        # Summary statistics
        print(f"\nRegime statistics:")
        regime_lengths = results.groupby('regime_id').size()
        print(f"  Average regime length: {regime_lengths.mean():.1f} days")
        print(f"  Min regime length: {regime_lengths.min()} days")
        print(f"  Max regime length: {regime_lengths.max()} days")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_multi_ticker():
    """Example of multi-ticker processing."""
    print("\n" + "=" * 70)
    print("MULTI-TICKER EXAMPLE")
    print("=" * 70)
    
    try:
        import yfinance as yf
        
        tickers = ['SPY', 'QQQ', 'IWM']
        data = {}
        
        for ticker in tickers:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        
        # Use multi-ticker detector
        detector = MultiTickerRegimeDetector(
            feature_config=FeatureConfig(ewma_span=20),
            bocpd_config=BOCPDConfig(hazard_rate=0.05),
            regime_config=RegimeConfig(
                detection_method="map_drop",
                min_spacing=10
            )
        )
        
        results = detector.process_multiple(data)
        
        print("\nRegime Summary:")
        print(detector.get_regime_summary().to_string())
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    results = example_usage()
    multi_results = example_multi_ticker()
