"""
Bayesian Online Change Point Detection (BOCPD)

Implementation following Adams & MacKay (2007):
"Bayesian Online Changepoint Detection"
https://arxiv.org/abs/0710.3742

Key insight: In BOCPD, P(r=0) ≈ H (hazard rate) by design. Change point
detection uses the MAP run length or P(r <= k) for small k.

A change point is detected when:
1. MAP run length suddenly drops to a small value (typically 1 or 2)
2. Or P(r <= k) becomes high for small k

Uses Normal-Inverse-Gamma conjugate prior for Gaussian likelihood
with unknown mean and variance.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BOCPDResult:
    """Container for BOCPD results at each time step."""
    time: int
    observation: float
    map_run_length: int
    run_length_probs: np.ndarray
    prob_short_run: float  # P(r <= short_threshold)
    predictive_mean: float
    predictive_std: float


class BOCPD:
    """
    Bayesian Online Change Point Detection following Adams & MacKay (2007).
    
    Detects structural breaks in time series by maintaining a posterior
    distribution over "run lengths" (time since last change point).
    
    Uses Normal-Inverse-Gamma conjugate prior for Gaussian likelihood
    with unknown mean and variance.
    
    BUG FIX (Jan 2026): When r_t = 0 (changepoint), the new regime's
    sufficient statistics now correctly include x_t as the FIRST observation:
        n[0] = 1, sum_x[0] = x, sum_x2[0] = x^2
    """
    
    def __init__(
        self,
        hazard_rate: float = 0.01,
        mu0: float = 0.0,
        kappa0: float = 0.1,
        alpha0: float = 2.0,
        beta0: float = 1.0,
        max_run_length: int = 500
    ):
        """
        Initialize BOCPD.
        
        Args:
            hazard_rate: Prior probability of change point at each step (1/expected_run_length)
            mu0: Prior mean for Gaussian observations
            kappa0: Prior precision weight (pseudo-observations for mean)
            alpha0: Prior shape for inverse-gamma (should be > 1 for proper prior)
            beta0: Prior rate for inverse-gamma (controls variance scale)
            max_run_length: Maximum run length to track
        """
        self.hazard_rate = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length
        
        # Precompute log hazards
        self.log_H = np.log(hazard_rate)
        self.log_1_minus_H = np.log(1 - hazard_rate)
        
        self.reset()
    
    def reset(self):
        """Reset detector to initial state."""
        self.t = 0
        
        # Run length posterior in log space
        self.log_R = np.array([0.0])  # P(r=0) = 1 initially
        
        # Sufficient statistics for each run length
        self.n = np.array([0.0])       # observation count
        self.sum_x = np.array([0.0])   # sum of observations
        self.sum_x2 = np.array([0.0])  # sum of squared observations
        
        # History
        self.observations: List[float] = []
        self.map_run_lengths: List[int] = []
        self.prob_short_runs: List[float] = []
        self.results: List[BOCPDResult] = []
    
    def _get_nig_params(
        self,
        n: np.ndarray,
        sum_x: np.ndarray,
        sum_x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute NIG posterior parameters from sufficient statistics."""
        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + sum_x) / kappa_n
        alpha_n = self.alpha0 + n / 2.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            xbar = np.where(n > 0, sum_x / n, 0.0)
        
        ss = np.maximum(sum_x2 - n * xbar**2, 0.0)
        coupling = self.kappa0 * n * (xbar - self.mu0)**2 / kappa_n
        beta_n = self.beta0 + 0.5 * ss + 0.5 * coupling
        
        return mu_n, kappa_n, alpha_n, beta_n
    
    def _predictive_log_pdf(
        self,
        x: float,
        n: np.ndarray,
        sum_x: np.ndarray,
        sum_x2: np.ndarray
    ) -> np.ndarray:
        """Compute predictive log-likelihood under NIG posterior (Student-t)."""
        mu_n, kappa_n, alpha_n, beta_n = self._get_nig_params(n, sum_x, sum_x2)
        
        df = 2.0 * alpha_n
        var = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        var = np.maximum(var, 1e-10)
        
        z = (x - mu_n)**2 / (df * var)
        
        log_pdf = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi * var)
            - ((df + 1) / 2) * np.log1p(z)
        )
        
        return log_pdf
    
    def update(self, x: float, short_threshold: int = 3) -> BOCPDResult:
        """
        Process new observation and update run-length posterior.
        
        BUG FIX: The r=0 (changepoint) hypothesis now correctly includes
        x as the first observation of the new regime:
            n[0] = 1, sum_x[0] = x, sum_x2[0] = x^2
        
        Args:
            x: New observation
            short_threshold: Threshold for "short run" probability calculation
            
        Returns:
            BOCPDResult with detection information
        """
        self.t += 1
        self.observations.append(x)
        
        # Predictive likelihood for each run length
        log_pred = self._predictive_log_pdf(x, self.n, self.sum_x, self.sum_x2)
        
        # Growth probabilities
        log_growth = self.log_R + log_pred + self.log_1_minus_H
        
        # Changepoint probability
        log_cp = logsumexp(self.log_R + log_pred) + self.log_H
        
        # New run length distribution
        log_R_new = np.concatenate([[log_cp], log_growth])
        
        # Normalize
        log_evidence = logsumexp(log_R_new)
        log_R_new = log_R_new - log_evidence
        
        # Update sufficient statistics
        # BUG FIX: r=0 hypothesis includes x as first observation
        self.n = np.concatenate([[1.0], self.n + 1])
        self.sum_x = np.concatenate([[x], self.sum_x + x])
        self.sum_x2 = np.concatenate([[x**2], self.sum_x2 + x**2])
        
        # Truncate if needed
        if len(log_R_new) > self.max_run_length:
            log_R_new = log_R_new[:self.max_run_length]
            self.n = self.n[:self.max_run_length]
            self.sum_x = self.sum_x[:self.max_run_length]
            self.sum_x2 = self.sum_x2[:self.max_run_length]
        
        self.log_R = log_R_new
        
        # Compute outputs
        R = np.exp(log_R_new)
        map_run_length = int(np.argmax(R))
        
        # P(r <= short_threshold) - probability of being in a "new" segment
        prob_short = float(np.sum(R[:min(short_threshold + 1, len(R))]))
        
        # Predictive moments
        mu_n, kappa_n, alpha_n, beta_n = self._get_nig_params(self.n, self.sum_x, self.sum_x2)
        pred_mean = float(np.sum(R * mu_n))
        pred_var = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
        pred_std = float(np.sqrt(np.sum(R * pred_var)))
        
        self.map_run_lengths.append(map_run_length)
        self.prob_short_runs.append(prob_short)
        
        result = BOCPDResult(
            time=self.t,
            observation=x,
            map_run_length=map_run_length,
            run_length_probs=R.copy(),
            prob_short_run=prob_short,
            predictive_mean=pred_mean,
            predictive_std=pred_std
        )
        self.results.append(result)
        
        return result
    
    def get_change_prob(self) -> float:
        """Get P(r <= 1) as change point indicator."""
        R = np.exp(self.log_R)
        return float(np.sum(R[:min(2, len(R))]))
    
    def detect_change_points(
        self,
        method: str = 'map_drop',
        threshold: float = 0.5,
        min_spacing: int = 5
    ) -> List[int]:
        """
        Detect change points from history.
        
        Args:
            method: Detection method
                - 'map_drop': Detect when MAP run length drops significantly
                - 'prob_short': Detect when P(r <= k) exceeds threshold
            threshold: Threshold for prob_short method (ignored for map_drop)
            min_spacing: Minimum spacing between detections
            
        Returns:
            List of time indices where change points detected
        """
        if not self.map_run_lengths:
            return []
        
        cps = []
        last_cp = -min_spacing
        
        if method == 'map_drop':
            # Detect when MAP suddenly drops
            for i in range(1, len(self.map_run_lengths)):
                prev_map = self.map_run_lengths[i-1]
                curr_map = self.map_run_lengths[i]
                
                # Change detected if MAP drops from >3 to <=2
                if prev_map > 3 and curr_map <= 2 and (i - last_cp) >= min_spacing:
                    cps.append(i + 1)  # 1-indexed
                    last_cp = i
                    
        elif method == 'prob_short':
            for i, prob in enumerate(self.prob_short_runs):
                if prob > threshold and (i - last_cp) >= min_spacing:
                    cps.append(i + 1)
                    last_cp = i
        
        return cps
    
    def get_regime_segments(
        self,
        method: str = 'map_drop',
        threshold: float = 0.5,
        min_spacing: int = 5
    ) -> List[Dict[str, Any]]:
        """Get regime segments based on detected change points."""
        cps = [0] + self.detect_change_points(method, threshold, min_spacing) + [self.t]
        
        segments = []
        for i in range(len(cps) - 1):
            start = cps[i]
            end = cps[i + 1]
            obs = self.observations[start:end]
            
            if obs:
                segments.append({
                    'start': start + 1,
                    'end': end,
                    'length': end - start,
                    'mean': float(np.mean(obs)),
                    'std': float(np.std(obs)) if len(obs) > 1 else 0.0
                })
        
        return segments


def test_synthetic():
    """Test on synthetic data with known change points."""
    print("=" * 70)
    print("SYNTHETIC DATA TEST - Mean Shifts")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Three segments with different means
    n1, n2, n3 = 50, 50, 50
    data = np.concatenate([
        np.random.randn(n1) * 0.5,          # Mean 0
        np.random.randn(n2) * 0.5 + 3.0,    # Mean 3
        np.random.randn(n3) * 0.5 - 2.0     # Mean -2
    ])
    
    true_cps = [n1, n1 + n2]
    print(f"True change points: {true_cps}")
    print(f"Segment means: [0, 3, -2], std: 0.5")
    
    detector = BOCPD(
        hazard_rate=0.02,  # Expect ~50 points per segment
        mu0=0.0,
        kappa0=0.1,  # Weak prior on mean
        alpha0=2.0,
        beta0=0.25   # Prior variance = beta0/(alpha0-1) = 0.25
    )
    
    for x in data:
        detector.update(x)
    
    # Analyze MAP run lengths around true change points
    maps = np.array(detector.map_run_lengths)
    probs = np.array(detector.prob_short_runs)
    
    print("\n[MAP Run Length Analysis]")
    for cp in true_cps:
        window = slice(max(0, cp-5), min(len(maps), cp+5))
        times = list(range(window.start + 1, window.stop + 1))
        print(f"\nAround t={cp}:")
        print(f"  Times:     {times}")
        print(f"  MAP r:     {list(maps[window])}")
        print(f"  P(r<=3):   {[f'{p:.3f}' for p in probs[window]]}")
    
    # Detect change points
    cps_map = detector.detect_change_points(method='map_drop', min_spacing=5)
    cps_prob = detector.detect_change_points(method='prob_short', threshold=0.9, min_spacing=5)
    
    print(f"\n[Detection Results]")
    print(f"  MAP drop method:   {cps_map}")
    print(f"  P(r<=3) > 0.9:     {cps_prob}")


def test_variance_shift():
    """Test on variance shift."""
    print("\n" + "=" * 70)
    print("VARIANCE SHIFT TEST")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Variance shift: low -> high -> low
    n1, n2, n3 = 50, 50, 50
    data = np.concatenate([
        np.random.randn(n1) * 0.3,   # Low vol
        np.random.randn(n2) * 2.0,   # High vol
        np.random.randn(n3) * 0.3    # Low vol
    ])
    
    true_cps = [n1, n1 + n2]
    print(f"True change points: {true_cps}")
    print(f"Segment std: [0.3, 2.0, 0.3]")
    
    detector = BOCPD(
        hazard_rate=0.02,
        mu0=0.0,
        kappa0=0.1,
        alpha0=2.0,
        beta0=0.1
    )
    
    for x in data:
        detector.update(x)
    
    cps_map = detector.detect_change_points(method='map_drop', min_spacing=5)
    print(f"\nDetected (MAP drop): {cps_map}")


def test_financial():
    """Test on real financial data."""
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("FINANCIAL DATA TEST (SPY)")
    print("=" * 70)
    
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    # Download SPY
    spy = yf.download("SPY", period="5y", progress=False)
    if spy.empty:
        print("Failed to download")
        return
    
    # Get returns - handle both old and new yfinance column formats
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']['SPY']
    else:
        close = spy['Close']
    
    returns = close.pct_change().dropna().values
    dates = close.index[1:]  # Dates for returns
    
    print(f"Data: {len(returns)} daily returns")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Standardize returns
    returns_std = (returns - np.mean(returns)) / np.std(returns)
    
    detector = BOCPD(
        hazard_rate=1/100,  # Expect regimes of ~100 days
        mu0=0.0,
        kappa0=0.1,
        alpha0=2.0,
        beta0=1.0
    )
    
    for x in returns_std:
        detector.update(x)
    
    cps = detector.detect_change_points(method='map_drop', min_spacing=20)
    
    print(f"\nDetected change points: {len(cps)}")
    if cps:
        print("\nFirst 15 change points with dates:")
        for i, cp in enumerate(cps[:15]):
            if cp < len(dates):
                print(f"  {i+1}. t={cp}: {dates[cp-1].strftime('%Y-%m-%d')}")
    
    # Show run length stats
    maps = np.array(detector.map_run_lengths)
    print(f"\nMAP run length stats:")
    print(f"  Mean: {maps.mean():.1f}")
    print(f"  Max: {maps.max()}")
    print(f"  Times MAP <= 2: {np.sum(maps <= 2)}")


if __name__ == "__main__":
    test_synthetic()
    test_variance_shift()
    test_financial()
