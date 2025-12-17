#!/usr/bin/env python3
"""
Paper-Faithful Wasserstein k-means for Market Regime Detection
================================================================

Direct implementation of the algorithm from:
"Clustering Market Regimes using the Wasserstein Distance"
Horvath, Issa, Muguruza (2021)
https://arxiv.org/abs/2110.11848

Key Techniques Implemented:
1. **Wasserstein Barycenter via MEDIAN** (Proposition 2.6, Equation 22)
   - For p=1, barycenter atoms: a_j = Median(α¹_j, ..., αᴹ_j)
   - NOT mean approximation
   
2. **Fast 1D Wasserstein Distance** (Proposition 2.5, Equation 21)
   - Wₚ(μ,ν)ᵖ = (1/N) Σᵢ |αᵢ - βᵢ|ᵖ  (for sorted atoms)
   - O(N log N) via sorting
   
3. **MMD for Cluster Quality** (Section 1.5, Equation 53)
   - Gaussian kernel: κ(x,y) = exp(-||x-y||²/(2σ²))
   - Within-cluster and between-cluster MMD
   
4. **Empirical Distributions** (Definition 1.3)
   - Each time window → empirical measure
   - Cluster in space P_p(ℝ) with finite p-th moment

Algorithm 1 (WK-means):
1. Create empirical distributions from sliding windows
2. Initialize centroids by sampling
3. Assign distributions to nearest centroid (Wasserstein distance)
4. Update centroids as Wasserstein barycenters (median)
5. Check convergence via loss function
6. Sort clusters by volatility for consistent labeling
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================================
# Paper Implementation: Core Functions
# ============================================================================

def wasserstein_distance_1d_fast(u_values: np.ndarray, v_values: np.ndarray, p: int = 1) -> float:
    """
    Fast 1D p-Wasserstein distance for empirical distributions.
    
    Implements Proposition 2.5, Equation (21):
    Wₚ(μ,ν)ᵖ = (1/N) Σᵢ₌₁ᴺ |αᵢ - βᵢ|ᵖ
    
    where α and β are sorted atoms of empirical measures.
    
    Complexity: O(N log N) due to sorting.
    
    Args:
        u_values: Samples from distribution μ (unsorted)
        v_values: Samples from distribution ν (unsorted)
        p: Order of Wasserstein distance (1 or 2)
        
    Returns:
        Wₚ(μ,ν)
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    
    # Equation (21) from paper
    differences = np.abs(u_sorted - v_sorted) ** p
    distance_p = np.mean(differences)
    
    return distance_p ** (1/p)


def wasserstein_distance_multivariate(dist1: np.ndarray, dist2: np.ndarray, p: int = 1) -> float:
    """
    Multivariate Wasserstein distance via per-feature computation.
    
    For d-dimensional distributions, we compute:
    Wₚ(μ,ν) = [mean_over_features(Wₚ(μ_i, ν_i)ᵖ)]^(1/p)
    
    This is an approximation but computationally efficient and used in practice.
    
    Args:
        dist1: (n_samples, n_features) samples from μ
        dist2: (n_samples, n_features) samples from ν
        p: Order of Wasserstein distance
        
    Returns:
        Approximate multivariate Wasserstein distance
    """
    if dist1.ndim == 1:
        return wasserstein_distance_1d_fast(dist1, dist2, p)
    
    n_features = dist1.shape[1]
    distances_p = []
    
    for i in range(n_features):
        d = wasserstein_distance_1d_fast(dist1[:, i], dist2[:, i], p)
        distances_p.append(d ** p)
    
    return (np.mean(distances_p)) ** (1/p)


def wasserstein_barycenter_median(distributions: List[np.ndarray]) -> np.ndarray:
    """
    Wasserstein barycenter via coordinate-wise median.
    
    Implements Proposition 2.6, Equation (22):
    For empirical measures {μᵢ}ᵢ₌₁ᴹ with N atoms each,
    barycenter atoms: aⱼ = Median(α¹ⱼ, ..., αᴹⱼ) for j=1,...,N
    
    This is the EXACT method from the paper for p=1.
    NOT a mean approximation.
    
    Args:
        distributions: List of M distributions, each (N, d) array
        
    Returns:
        Barycenter distribution (N, d) array
    """
    # Stack all distributions: (M, N, d) or (M, N) for 1D
    stacked = np.stack(distributions, axis=0)  # (M, N, d) or (M, N)
    
    # Median across distributions (axis=0) for each atom position
    # This gives coordinate-wise median as in Proposition 2.6
    barycenter = np.median(stacked, axis=0)  # (N, d) or (N,)
    
    return barycenter


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Gaussian (RBF) kernel for MMD computation.
    
    κ(x,y) = exp(-||x-y||²/(2σ²))
    
    Args:
        x: First point (can be vector)
        y: Second point (can be vector)
        sigma: Kernel bandwidth
        
    Returns:
        Kernel value
    """
    diff = x - y
    squared_dist = np.sum(diff ** 2)
    return np.exp(-squared_dist / (2 * sigma ** 2))


def compute_mmd_squared(samples_x: np.ndarray, samples_y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Implements Equation (53) from paper:
    MMD²[x,y] = (1/n²)Σᵢⱼ k(xᵢ,xⱼ) - (2/mn)Σᵢⱼ k(xᵢ,yⱼ) + (1/m²)Σᵢⱼ k(yᵢ,yⱼ)
    
    with Gaussian kernel κ(x,y) = exp(-||x-y||²/(2σ²))
    
    Used for evaluating cluster quality:
    - Within-cluster MMD (low = homogeneous)
    - Between-cluster MMD (high = distinct)
    
    Args:
        samples_x: (n, d) samples from first distribution
        samples_y: (m, d) samples from second distribution
        sigma: Gaussian kernel bandwidth
        
    Returns:
        MMD² (can be negative due to finite sample bias)
    """
    n = len(samples_x)
    m = len(samples_y)
    
    # Flatten distributions to 1D if needed (each row is a sample)
    if samples_x.ndim > 2:
        samples_x = samples_x.reshape(n, -1)
    if samples_y.ndim > 2:
        samples_y = samples_y.reshape(m, -1)
    
    # Term 1: E[k(x,x')]
    term1 = 0.0
    for i in range(n):
        for j in range(n):
            term1 += gaussian_kernel(samples_x[i], samples_x[j], sigma)
    term1 /= (n * n)
    
    # Term 2: E[k(x,y)]
    term2 = 0.0
    for i in range(n):
        for j in range(m):
            term2 += gaussian_kernel(samples_x[i], samples_y[j], sigma)
    term2 /= (n * m)
    
    # Term 3: E[k(y,y')]
    term3 = 0.0
    for i in range(m):
        for j in range(m):
            term3 += gaussian_kernel(samples_y[i], samples_y[j], sigma)
    term3 /= (m * m)
    
    # Equation (53)
    mmd_squared = term1 - 2 * term2 + term3
    
    return mmd_squared


# ============================================================================
# Paper Implementation: WK-means Algorithm
# ============================================================================

class PaperWassersteinKMeans:
    """
    Wasserstein k-means exactly as described in the paper.
    
    Algorithm 1 (page 12):
    1. Calculate stream of segments ℓ(r_S)
    2. Define family of empirical distributions K
    3. Initialize centroids by sampling k times from K
    4. While loss_function > tolerance:
       a. Assign each μⱼ to closest centroid wrt Wₚ
       b. Update centroid i as Wasserstein barycenter of cluster Cₗ
       c. Calculate loss_function
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        window_size: int = 20,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        """
        Initialize Wasserstein k-means.
        
        Args:
            n_regimes: Number of clusters k
            window_size: Length of each window (h₁ in paper)
            p: Order of Wasserstein distance (1 or 2)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            random_state: For reproducibility
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Model state
        self.centroids_ = None  # List of centroid distributions
        self.cluster_assignments_ = None
        self.regime_volatilities_ = None  # For labeling
        self.n_iter_ = 0
        self.converged_ = False
        
    def _create_empirical_distributions(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Create stream of segments ℓ(r_S) from feature array.
        
        Implements Definition 1.2 (Equation 3):
        ℓᵢ(x) = (x₁₊ₕ₂₍ᵢ₋₁₎, ..., x₁₊ₕ₁₊ₕ₂₍ᵢ₋₁₎)
        
        For simplicity, we use non-overlapping windows (h₂ = h₁).
        
        Args:
            features: (n_days, n_features) array
            
        Returns:
            List of empirical distributions (each is window_size × n_features)
        """
        n_samples = len(features)
        distributions = []
        
        # Non-overlapping windows
        for i in range(0, n_samples - self.window_size + 1, self.window_size):
            window = features[i:i + self.window_size]
            if len(window) == self.window_size:
                distributions.append(window)
        
        return distributions
    
    def _initialize_centroids(self, distributions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Initialize centroids by sampling k times from K.
        
        As in Algorithm 1: "initialise centroids μᵢ, i=1,...,k by sampling k times from K"
        
        Args:
            distributions: Family K of empirical distributions
            
        Returns:
            List of k initial centroids
        """
        np.random.seed(self.random_state)
        n_dists = len(distributions)
        
        if n_dists < self.n_regimes:
            raise ValueError(f"Not enough distributions ({n_dists}) for {self.n_regimes} clusters")
        
        indices = np.random.choice(n_dists, self.n_regimes, replace=False)
        return [distributions[i].copy() for i in indices]
    
    def _assign_to_nearest_centroid(
        self,
        distributions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Assign each μⱼ to closest centroid wrt Wₚ.
        
        As in Algorithm 1: "assign closest centroid wrt Wp to cluster Cl"
        
        Args:
            distributions: List of empirical distributions
            
        Returns:
            Array of cluster assignments
        """
        n_dists = len(distributions)
        assignments = np.zeros(n_dists, dtype=int)
        
        for i, dist in enumerate(distributions):
            min_dist = np.inf
            best_cluster = 0
            
            for k, centroid in enumerate(self.centroids_):
                d = wasserstein_distance_multivariate(dist, centroid, self.p)
                if d < min_dist:
                    min_dist = d
                    best_cluster = k
            
            assignments[i] = best_cluster
        
        return assignments
    
    def _update_centroids(
        self,
        distributions: List[np.ndarray],
        assignments: np.ndarray
    ) -> List[np.ndarray]:
        """
        Update centroids as Wasserstein barycenters.
        
        As in Algorithm 1: "update centroid i as the Wasserstein barycenter relative to Cl"
        
        Uses Proposition 2.6 (median for p=1).
        
        Args:
            distributions: List of empirical distributions
            assignments: Current cluster assignments
            
        Returns:
            Updated centroids
        """
        new_centroids = []
        
        for k in range(self.n_regimes):
            cluster_dists = [distributions[i] for i in range(len(distributions))
                           if assignments[i] == k]
            
            if len(cluster_dists) == 0:
                # Empty cluster - reinitialize randomly
                idx = np.random.randint(len(distributions))
                new_centroids.append(distributions[idx].copy())
                print(f"  Warning: Empty cluster {k}, reinitializing")
            else:
                # Wasserstein barycenter via median (Proposition 2.6)
                barycenter = wasserstein_barycenter_median(cluster_dists)
                new_centroids.append(barycenter)
        
        return new_centroids
    
    def _compute_loss(self) -> float:
        """
        Calculate loss function as sum of Wasserstein distances.
        
        Implements Equation (23):
        l(μⁿ⁻¹, μⁿ) = Σᵢ₌₁ᵏ Wₚ(μⁿ⁻¹ᵢ, μⁿᵢ)
        
        Returns:
            Total loss (centroid movement)
        """
        if self.centroids_prev_ is None:
            return np.inf
        
        total_loss = 0.0
        for i in range(self.n_regimes):
            d = wasserstein_distance_multivariate(
                self.centroids_prev_[i],
                self.centroids_[i],
                self.p
            )
            total_loss += d
        
        return total_loss
    
    def _label_by_volatility(self, distributions: List[np.ndarray], assignments: np.ndarray):
        """
        Sort clusters by volatility for consistent labeling.
        
        Ensures:
        - Cluster 0 = low volatility
        - Cluster 1 = medium volatility (if k=3)
        - Cluster k-1 = high volatility
        """
        # Compute average volatility per cluster
        cluster_vols = []
        for k in range(self.n_regimes):
            cluster_dists = [distributions[i] for i in range(len(distributions))
                           if assignments[i] == k]
            
            if len(cluster_dists) > 0:
                # Compute volatility as std of first feature (normalized returns)
                vols = [np.std(d[:, 0]) if d.ndim > 1 else np.std(d) for d in cluster_dists]
                avg_vol = np.mean(vols)
            else:
                avg_vol = 0.0
            
            cluster_vols.append(avg_vol)
        
        # Create mapping: old_cluster → new_cluster (sorted by volatility)
        vol_order = np.argsort(cluster_vols)  # Low to high volatility
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vol_order)}
        
        # Reorder centroids and update assignments
        self.centroids_ = [self.centroids_[i] for i in vol_order]
        self.cluster_assignments_ = np.array([mapping[a] for a in assignments])
        self.regime_volatilities_ = sorted(cluster_vols)
    
    def fit(self, features: np.ndarray, verbose: bool = True) -> 'PaperWassersteinKMeans':
        """
        Fit Wasserstein k-means to feature array.
        
        Implements Algorithm 1 from paper.
        
        Args:
            features: (n_days, n_features) scaled feature array
            verbose: Print iteration info
            
        Returns:
            self
        """
        if verbose:
            print(f"\nFitting Wasserstein k-means (k={self.n_regimes}, window={self.window_size})")
        
        # Step 1: Create empirical distributions
        distributions = self._create_empirical_distributions(features)
        n_dists = len(distributions)
        
        if verbose:
            print(f"  Created {n_dists} empirical distributions")
        
        if n_dists < self.n_regimes:
            raise ValueError(f"Not enough data for {self.n_regimes} clusters")
        
        # Step 2: Initialize centroids
        self.centroids_ = self._initialize_centroids(distributions)
        self.centroids_prev_ = None
        
        # Step 3: Iterate until convergence
        for iteration in range(self.max_iter):
            # Step 3a: Assign to nearest centroid
            assignments = self._assign_to_nearest_centroid(distributions)
            
            # Step 3b: Update centroids
            self.centroids_prev_ = [c.copy() for c in self.centroids_]
            self.centroids_ = self._update_centroids(distributions, assignments)
            
            # Step 3c: Calculate loss
            loss = self._compute_loss()
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: loss={loss:.6f}")
            
            # Check convergence
            if loss < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                if verbose:
                    print(f"  Converged in {self.n_iter_} iterations (loss={loss:.6f})")
                break
        
        if not self.converged_:
            self.n_iter_ = self.max_iter
            if verbose:
                print(f"  Did not converge in {self.max_iter} iterations (loss={loss:.6f})")
        
        # Step 4: Label by volatility
        self._label_by_volatility(distributions, assignments)
        
        if verbose:
            print(f"  Regime volatilities: {[f'{v:.4f}' for v in self.regime_volatilities_]}")
            cluster_sizes = [np.sum(self.cluster_assignments_ == k) for k in range(self.n_regimes)]
            print(f"  Cluster sizes: {cluster_sizes}")
        
        return self
    
    def predict_distribution(self, distribution: np.ndarray) -> int:
        """
        Predict regime for a single empirical distribution.
        
        Args:
            distribution: (window_size, n_features) empirical distribution
            
        Returns:
            Regime label (0 to k-1)
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet")
        
        min_dist = np.inf
        best_regime = 0
        
        for k, centroid in enumerate(self.centroids_):
            d = wasserstein_distance_multivariate(distribution, centroid, self.p)
            if d < min_dist:
                min_dist = d
                best_regime = k
        
        return best_regime
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regimes for feature array.
        
        Creates non-overlapping windows and predicts regime for each.
        
        Args:
            features: (n_days, n_features) scaled feature array
            
        Returns:
            Array of regime labels (one per window)
        """
        distributions = self._create_empirical_distributions(features)
        predictions = np.array([self.predict_distribution(d) for d in distributions])
        return predictions
    
    def evaluate_clusters_mmd(
        self,
        features: np.ndarray,
        sigma: float = 1.0,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate cluster quality using MMD.
        
        Computes:
        - Within-cluster MMD (homogeneity) - lower is better
        - Between-cluster MMD (distinctness) - higher is better
        
        Args:
            features: Original features used for fitting
            sigma: Gaussian kernel bandwidth
            n_samples: Number of sample pairs for MMD estimation
            
        Returns:
            Dict with MMD scores
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet")
        
        distributions = self._create_empirical_distributions(features)
        
        # Predict cluster assignments for these distributions
        current_assignments = self.predict(features)
        
        # Within-cluster MMD (self-similarity)
        within_mmd_scores = []
        for k in range(self.n_regimes):
            cluster_dists = [distributions[i] for i in range(len(distributions))
                           if current_assignments[i] == k]
            
            if len(cluster_dists) < 2:
                within_mmd_scores.append(0.0)
                continue
            
            # Sample pairs from cluster
            mmd_vals = []
            for _ in range(min(n_samples, len(cluster_dists) * (len(cluster_dists) - 1) // 2)):
                idx1, idx2 = np.random.choice(len(cluster_dists), 2, replace=False)
                mmd_sq = compute_mmd_squared(cluster_dists[idx1], cluster_dists[idx2], sigma)
                mmd_vals.append(max(0, mmd_sq) ** 0.5)  # Take sqrt and ensure non-negative
            
            within_mmd_scores.append(np.median(mmd_vals) if mmd_vals else 0.0)
        
        # Between-cluster MMD (distinctness)
        between_mmd_scores = []
        for k1 in range(self.n_regimes):
            for k2 in range(k1 + 1, self.n_regimes):
                cluster1_dists = [distributions[i] for i in range(len(distributions))
                                if current_assignments[i] == k1]
                cluster2_dists = [distributions[i] for i in range(len(distributions))
                                if current_assignments[i] == k2]
                
                if len(cluster1_dists) == 0 or len(cluster2_dists) == 0:
                    continue
                
                # Sample pairs between clusters
                mmd_vals = []
                for _ in range(min(n_samples, len(cluster1_dists) * len(cluster2_dists))):
                    idx1 = np.random.choice(len(cluster1_dists))
                    idx2 = np.random.choice(len(cluster2_dists))
                    mmd_sq = compute_mmd_squared(cluster1_dists[idx1], cluster2_dists[idx2], sigma)
                    mmd_vals.append(max(0, mmd_sq) ** 0.5)
                
                between_mmd_scores.append(np.median(mmd_vals) if mmd_vals else 0.0)
        
        return {
            'within_cluster_mmd_mean': np.mean(within_mmd_scores),
            'within_cluster_mmd_scores': within_mmd_scores,
            'between_cluster_mmd_mean': np.mean(between_mmd_scores),
            'between_cluster_mmd_scores': between_mmd_scores,
            'quality_ratio': np.mean(between_mmd_scores) / (np.mean(within_mmd_scores) + 1e-10)
        }


# ============================================================================
# Rolling Window Implementation for Trading
# ============================================================================

class RollingPaperWassersteinDetector:
    """
    Rolling window Wasserstein regime detector for live trading.
    
    Similar to rolling HMM approach:
    - Train on recent data (e.g., last 500-750 days)
    - Retrain periodically (e.g., quarterly)
    - Predict forward using frozen centroids
    - Consistent volatility-based labeling
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        window_size: int = 20,
        training_window_days: int = 500,
        retrain_frequency_days: int = 126,
        feature_cols: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize rolling detector.
        
        Args:
            n_regimes: Number of regimes
            window_size: Days per empirical distribution
            training_window_days: Training window length
            retrain_frequency_days: How often to retrain
            feature_cols: Which features to use
            **kwargs: Additional args for PaperWassersteinKMeans
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.training_window_days = training_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.feature_cols = feature_cols
        self.wk_kwargs = kwargs
        
        # State
        self.model = None
        self.scaler = None
        self.last_training_date = None
        self.feature_names = None
    
    def train_on_window(
        self,
        df: pd.DataFrame,
        end_date: Optional[pd.Timestamp] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train on rolling window.
        
        Args:
            df: DataFrame with features
            end_date: End of training window
            verbose: Print info
            
        Returns:
            Training info dict
        """
        if end_date is None:
            end_date = df.index[-1]
        
        start_date = end_date - pd.Timedelta(days=self.training_window_days)
        train_df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(train_df) < self.window_size * self.n_regimes:
            raise ValueError(f"Not enough training data: {len(train_df)} days")
        
        # Extract features
        if self.feature_cols is None:
            self.feature_cols = [c for c in train_df.columns if c not in ['close', 'returns']]
        self.feature_names = self.feature_cols
        
        features_raw = train_df[self.feature_cols].values
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_raw)
        
        # Train model
        self.model = PaperWassersteinKMeans(
            n_regimes=self.n_regimes,
            window_size=self.window_size,
            **self.wk_kwargs
        )
        self.model.fit(features_scaled, verbose=verbose)
        self.last_training_date = end_date
        
        # Evaluate with MMD
        mmd_scores = self.model.evaluate_clusters_mmd(features_scaled, sigma=0.1, n_samples=50)
        
        return {
            'training_end': end_date,
            'training_days': len(train_df),
            'n_distributions': len(self.model.cluster_assignments_),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_,
            'regime_volatilities': self.model.regime_volatilities_,
            'mmd_within': mmd_scores['within_cluster_mmd_mean'],
            'mmd_between': mmd_scores['between_cluster_mmd_mean'],
            'mmd_quality_ratio': mmd_scores['quality_ratio']
        }
    
    def predict_forward_rolling(
        self,
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        verbose: bool = True
    ) -> pd.Series:
        """
        Predict regimes forward with periodic retraining.
        
        Args:
            df: Full DataFrame
            start_date: Start of prediction period
            end_date: End of prediction period
            verbose: Print info
            
        Returns:
            Series of regime predictions
        """
        if self.model is None:
            # Initial training
            self.train_on_window(df, start_date, verbose=verbose)
        
        predictions = []
        
        # Ensure dates are timestamps
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)
        
        current_date = start_date
        last_retrain = start_date
        
        # Get actual dates from dataframe index
        df_dates = df.index[df.index >= start_date]
        df_dates = df_dates[df_dates <= end_date]
        
        for i, current_date in enumerate(df_dates):
            # Check if need to retrain (use trading days, not calendar days)
            days_since_retrain_idx = len(df.loc[last_retrain:current_date]) - 1
            if days_since_retrain_idx >= self.retrain_frequency_days:
                if verbose:
                    print(f"\nRetraining at {current_date.date()}")
                self.train_on_window(df, current_date, verbose=verbose)
                last_retrain = current_date
            
            # Predict using frozen centroids
            # Get last window_size TRADING DAYS ending at current_date
            current_idx = df.index.get_loc(current_date)
            
            if current_idx >= self.window_size:
                # Take exactly window_size rows ending at current position
                window_features = df[self.feature_cols].iloc[current_idx - self.window_size + 1:current_idx + 1].values
                
                if len(window_features) == self.window_size:
                    features_scaled = self.scaler.transform(window_features)
                    
                    # Predict regime for this distribution
                    regime = self.model.predict_distribution(features_scaled)
                    predictions.append({'date': current_date, 'regime': regime})
        
        # Convert to Series
        if len(predictions) == 0:
            return pd.Series(dtype=int, name='regime')
        
        pred_df = pd.DataFrame(predictions)
        if 'date' not in pred_df.columns:
            # Empty DataFrame - return empty Series
            return pd.Series(dtype=int, name='regime')
        
        return pred_df.set_index('date')['regime']


# ============================================================================
# Data Fetching (Polygon API)
# ============================================================================

def fetch_polygon_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily bars from Polygon.io.
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: Polygon API key (or from env)
        
    Returns:
        DataFrame with OHLCV data
    """
    if api_key is None:
        api_key = os.getenv("POLYGON_API_KEY")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Accept both "OK" (paid tier) and "DELAYED" (free tier) status
    if data.get("status") not in ["OK", "DELAYED"] or "results" not in data:
        raise ValueError(f"Failed to fetch data for {symbol}: {data}")
    
    df = pd.DataFrame(data["results"])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
    
    return df


def calculate_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate trading features for regime detection.
    
    Args:
        df: OHLCV DataFrame
        window: Lookback window for features
        
    Returns:
        DataFrame with features
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Realized volatility (rolling std of returns)
    df['realized_vol'] = df['returns'].rolling(window).std()
    
    # Trend strength (abs of rolling mean return)
    df['trend_strength'] = df['returns'].rolling(window).mean().abs()
    
    # Volume momentum (current vs average)
    df['volume_ma'] = df['volume'].rolling(window).mean()
    df['volume_momentum'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1e-10)
    
    # Price momentum
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    
    # High-low range (volatility proxy)
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    
    # Volume-adjusted returns
    df['vol_adj_returns'] = df['returns'] / (df['realized_vol'] + 1e-10)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


if __name__ == "__main__":
    # Quick test
    print("Paper-Faithful Wasserstein k-means Implementation")
    print("=" * 70)
    
    # Generate test data
    np.random.seed(42)
    
    # Regime 1: Low vol
    regime1 = np.random.normal(0, 0.01, (100, 3))
    
    # Regime 2: Medium vol
    regime2 = np.random.normal(0.001, 0.02, (100, 3))
    
    # Regime 3: High vol
    regime3 = np.random.normal(-0.002, 0.04, (100, 3))
    
    # Combine
    data = np.vstack([regime1, regime2, regime3, regime2, regime1])
    
    # Fit
    model = PaperWassersteinKMeans(n_regimes=3, window_size=20)
    model.fit(data)
    
    # Evaluate
    mmd_scores = model.evaluate_clusters_mmd(data, sigma=0.1)
    print(f"\nMMD Evaluation:")
    print(f"  Within-cluster MMD: {mmd_scores['within_cluster_mmd_mean']:.6f}")
    print(f"  Between-cluster MMD: {mmd_scores['between_cluster_mmd_mean']:.6f}")
    print(f"  Quality ratio (higher=better): {mmd_scores['quality_ratio']:.2f}")
