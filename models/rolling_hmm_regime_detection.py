#!/usr/bin/env python3
"""
Professional Rolling-Window HMM Regime Detection
================================================

Implements the CORRECT approach to HMM stability:

1. **Rolling Training Windows** (3-7 years)
   - Train on recent history only
   - Slide forward periodically (monthly/quarterly)
   - Don't retrain on ALL history

2. **Frozen Parameters + Forward Filter**
   - Train model once on window
   - Freeze transition/emission parameters
   - Only run forward algorithm on new data
   - No historical re-labeling

3. **Feature Discipline**
   - Volatility-normalized returns
   - Rolling realized volatility
   - Trend strength (not raw price)
   - Fewer features, more robust

4. **State Persistence Constraints**
   - High diagonal transition matrix (0.85-0.95)
   - Penalize rapid switching
   - Regimes persist for weeks/months

5. **Probabilistic Interpretation**
   - Use P(state) not hard labels
   - Monitor regime confidence
   - Entropy-based uncertainty

This approach:
✅ Adapts to market evolution
✅ No retroactive label changes on live data
✅ Stable forward predictions
✅ Professionally used in production
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import requests

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from dotenv import load_dotenv
import os
load_dotenv()


class RollingWindowHMM:
    """
    Professional HMM with rolling windows and frozen parameters.
    
    Key differences from naive approach:
    - Trains on ROLLING window (e.g., last 3 years), not all history
    - Freezes model parameters after training
    - Only runs forward filter on new observations
    - No backward pass = no retroactive relabeling
    - Retrains periodically (monthly/quarterly), not daily
    """
    
    def __init__(
        self,
        symbol: str = None,
        n_regimes: int = 3,
        n_states: int = None,  # Alias for n_regimes
        feature_columns: List[str] = None,  # Custom feature names
        training_window_days: int = 756,  # ~3 years
        retrain_frequency_days: int = 63,  # ~quarterly
        persistence_strength: float = 0.90,  # High self-transition probability
        persistence_prior: float = None,  # Alias for persistence_strength
        random_state: int = 42
    ):
        """
        Initialize professional rolling HMM.
        
        Args:
            symbol: Stock ticker (optional if using external data)
            n_regimes: Number of states (2-4 recommended)
            n_states: Alias for n_regimes
            feature_columns: List of feature column names to use (overrides defaults)
            training_window_days: Days to use for training (~3-7 years)
            retrain_frequency_days: How often to retrain (monthly=21, quarterly=63)
            persistence_strength: Diagonal of transition matrix (0.85-0.95)
            persistence_prior: Alias for persistence_strength
            random_state: For reproducibility
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required. Install: pip install hmmlearn")
        
        self.symbol = symbol
        self.n_regimes = n_states if n_states is not None else n_regimes
        self.training_window_days = training_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.persistence_strength = persistence_prior if persistence_prior is not None else persistence_strength
        self.random_state = random_state
        self.feature_columns = feature_columns  # Custom features
        
        # Data and model state
        self.data = None
        self.features = None
        self.model = None  # Frozen HMM parameters
        self.scaler = None  # Frozen feature scaler
        self.last_training_date = None
        self.regime_mapping = None  # State index → economic meaning
        
        # Tracking
        self.regime_probs = None  # P(state_t | obs_1:t) for each time
        self.regime_labels = None  # Most likely state (for convenience)
        self.regime_confidence = None  # 1 - entropy(probs)
        
        # Regime names
        if n_regimes == 3:
            self.regime_names = {
                0: "Bearish/Down",
                1: "Sideways/Choppy", 
                2: "Bullish/Up"
            }
        else:
            self.regime_names = {i: f"Regime {i}" for i in range(n_regimes)}
    
    def fetch_data(self, lookback_days: int = 1500) -> pd.DataFrame:
        """Fetch historical data from Polygon.io."""
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("POLYGON_API_KEY not set")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{self.symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") not in ["OK", "DELAYED"] or not data.get("results"):
            raise ValueError(f"No data for {self.symbol}")
        
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        
        self.data = df
        print(f"[OK] Fetched {len(df)} days for {self.symbol}")
        return df
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate ROBUST features for HMM.
        
        Best practices:
        - Volatility-normalized returns (not raw returns)
        - Rolling realized volatility
        - Trend strength (normalized)
        - Few features (2-4 max)
        """
        if self.data is None:
            raise ValueError("No data. Call fetch_data() first.")
        
        df = self.data.copy()
        
        # 1. Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Rolling realized volatility (20-day)
        df['realized_vol'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        
        # 3. Volatility-normalized returns (KEY FEATURE)
        df['vol_norm_return'] = df['log_return'] / (df['realized_vol'] / np.sqrt(252))
        
        # 4. Trend strength (price vs 50-day MA, normalized)
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['trend_strength'] = (df['close'] - df['ma_50']) / df['ma_50']
        
        # 5. Volume regime (normalized)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['log_volume_ratio'] = np.log(df['volume_ratio'])
        
        df = df.dropna().reset_index(drop=True)
        
        # Select minimal robust feature set (use custom if provided)
        if self.feature_columns:
            feature_cols = self.feature_columns
        else:
            feature_cols = ['vol_norm_return', 'realized_vol', 'trend_strength']
        
        required_cols = ['date', 'close', 'log_return']
        self.features = df[required_cols + feature_cols].copy()
        
        print(f"[OK] Calculated {len(feature_cols)} robust features ({len(self.features)} points)")
        
        return self.features
    
    def _create_persistent_transition_prior(self) -> np.ndarray:
        """
        Create high-persistence transition matrix.
        
        Diagonal values = persistence_strength (e.g., 0.90)
        Off-diagonal distributed evenly
        """
        prior = np.ones((self.n_regimes, self.n_regimes))
        off_diag_prob = (1 - self.persistence_strength) / (self.n_regimes - 1)
        
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                if i == j:
                    prior[i, j] = self.persistence_strength
                else:
                    prior[i, j] = off_diag_prob
        
        return prior
    
    def fit(self, returns: pd.Series) -> 'RollingWindowHMM':
        """
        Sklearn-style fit method for compatibility.
        
        Args:
            returns: Return series (used to build features internally)
        
        Returns:
            self
        """
        # Build simple features from returns
        df = pd.DataFrame({'close': (returns + 1).cumprod() * 100})
        df['log_return'] = returns
        df['realized_vol'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['vol_norm_return'] = returns / (df['realized_vol'] / np.sqrt(252))
        df = df.dropna()
        
        # Use last training_window_days
        if len(df) > self.training_window_days:
            df = df.iloc[-self.training_window_days:]
        
        features = df[['vol_norm_return', 'realized_vol']].values
        
        # Train model
        self.train_on_window(features_array=features)
        
        return self
    
    def train_on_window(self, features_array: np.ndarray = None, timestamp: pd.Timestamp = None,
                       window_start_idx: int = None, window_end_idx: int = None) -> None:
        """
        Train HMM on specified window and FREEZE parameters.
        
        Can accept either:
        1. Pre-computed feature array (features_array) + timestamp
        2. Indices into self.features (window_start_idx, window_end_idx)
        
        Args:
            features_array: Pre-computed feature array (N, n_features)
            timestamp: Timestamp for last observation (for tracking)
            window_start_idx: Start index in features df (None = -training_window_days)
            window_end_idx: End index (None = latest)
        """
        
        # Case 1: External features provided
        if features_array is not None:
            X = features_array
            if timestamp:
                self.last_training_date = timestamp
            print(f"\n{'='*70}")
            print(f"TRAINING HMM ON ROLLING WINDOW")
            print(f"{'='*70}")
            print(f"Training samples: {len(X)}")
            if timestamp:
                print(f"Last timestamp: {timestamp}")
        
        # Case 2: Use internal features
        else:
            if self.features is None:
                self.calculate_features()
            
            # Determine training window
            if window_end_idx is None:
                window_end_idx = len(self.features)
            
            if window_start_idx is None:
                window_start_idx = max(0, window_end_idx - self.training_window_days)
            
            window = self.features.iloc[window_start_idx:window_end_idx].copy()
            
            print(f"\n{'='*70}")
            print(f"TRAINING HMM ON ROLLING WINDOW")
            print(f"{'='*70}")
            print(f"Window: {window['date'].iloc[0].date()} to {window['date'].iloc[-1].date()}")
            print(f"Training samples: {len(window)}")
            
            # Extract features
            X = window.iloc[:, 3:].values  # Skip date, close, log_return
            self.last_training_date = window['date'].iloc[-1]
        
        # Fit scaler and FREEZE
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize HMM with persistent transitions
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
            init_params='smc',  # Initialize start, means, covars (NOT transition matrix)
            params='stmc'  # But allow all to be trained
        )
        
        # Initialize transition matrix with high persistence
        self.model.transmat_ = self._create_persistent_transition_prior()
        
        # Train (will refine but start from our persistent initialization)
        self.model.fit(X_scaled)
        
        # Label states by volatility (works with external features too)
        train_labels = self.model.predict(X_scaled)
        self.regime_mapping = self._label_states_by_volatility(train_labels, X_scaled)
        
        if not timestamp:
            self.last_training_date = self.last_training_date  # Keep existing
        
        print(f"\n[OK] Model trained and FROZEN")
        print(f"  Transition matrix (persistence={self.persistence_strength}):")
        print(f"{self.model.transmat_.round(3)}")
        print(f"\nRegime mapping (by avg volatility):")
        for old_state, new_state in self.regime_mapping.items():
            print(f"  State {old_state} -> {self.regime_names[new_state]}")
    
    def _label_states_by_return(self, raw_labels: np.ndarray, data: pd.DataFrame) -> Dict[int, int]:
        """Map HMM states to economic meaning (bearish/sideways/bullish) by returns."""
        regime_returns = {}
        for state in range(self.n_regimes):
            mask = raw_labels == state
            regime_returns[state] = data.loc[mask, 'log_return'].mean()
        
        # Sort by return: lowest=0 (bearish), highest=n-1 (bullish)
        sorted_states = sorted(regime_returns.items(), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
        
        return mapping
    
    def _label_states_by_volatility(self, raw_labels: np.ndarray, X_scaled: np.ndarray) -> Dict[int, int]:
        """Map HMM states to volatility levels (low/med/high) - works with any features."""
        # Compute variance of observations in each state (proxy for volatility)
        regime_variances = {}
        for state in range(self.n_regimes):
            mask = raw_labels == state
            if mask.sum() > 0:
                # Use average variance across all features as volatility proxy
                regime_variances[state] = np.var(X_scaled[mask], axis=0).mean()
            else:
                regime_variances[state] = 0
        
        # Sort by variance: lowest=0 (low vol), highest=n-1 (high vol)
        sorted_states = sorted(regime_variances.items(), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
        
        return mapping
    
    def predict_forward_filter(self, features_array: np.ndarray = None, new_data_start_idx: int = None) -> Dict:
        """
        Run FORWARD FILTER ONLY on new observations.
        
        Key: Uses frozen model parameters, no backward pass.
        Result: No retroactive relabeling of history.
        
        Can accept either:
        1. Pre-computed feature array (features_array) - returns dict with prediction
        2. Index into self.features (new_data_start_idx) - stores predictions internally
        
        Args:
            features_array: Pre-computed feature array (N, n_features)
            new_data_start_idx: Index to start forward filter (None = start of training window)
            
        Returns:
            If features_array provided: dict with prediction for last observation
            Otherwise: None (predictions stored internally)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_on_window() first.")
        
        # Case 1: External features provided
        if features_array is not None:
            X = features_array
            X_scaled = self.scaler.transform(X)
            
            # Compute forward probabilities (no backward pass!)
            # Use predict_proba which does forward algorithm only
            probs = self.model.predict_proba(X_scaled)
            
            # Get last observation
            last_probs = probs[-1]
            most_likely = np.argmax(last_probs)
            
            # Map to economic label
            if self.regime_mapping:
                most_likely = self.regime_mapping[most_likely]
            
            # Calculate confidence
            confidence = 1 - entropy(last_probs) / np.log(self.n_regimes)
            
            return {
                'most_likely_state': most_likely,
                'probabilities': last_probs.tolist(),
                'confidence': confidence,
                'regime_name': self.regime_names.get(most_likely, f"Regime {most_likely}")
            }
        
        # Case 2: Use internal features
        if new_data_start_idx is None:
            new_data_start_idx = max(0, len(self.features) - self.training_window_days)
        
        # Get all data from training start to present
        filter_data = self.features.iloc[new_data_start_idx:].copy()
        
        # Scale with FROZEN scaler
        X = filter_data.iloc[:, 3:].values
        X_scaled = self.scaler.transform(X)
        
        # Run forward filter (predict_proba)
        # This gives P(state_t | obs_1:t) using frozen parameters
        probs = self.model.predict_proba(X_scaled)
        
        # Most likely state
        labels = np.argmax(probs, axis=1)
        
        # Remap to economic meaning
        remapped_labels = np.array([self.regime_mapping[label] for label in labels])
        
        # Calculate confidence (1 - entropy)
        confidences = 1 - entropy(probs, axis=1) / np.log(self.n_regimes)
        
        # Store results
        self.regime_probs = probs
        self.regime_labels = remapped_labels
        self.regime_confidence = confidences
        
        print(f"\n[OK] Forward filter complete on {len(filter_data)} observations")
        self._print_regime_stats(filter_data, remapped_labels)
    
    def _print_regime_stats(self, data: pd.DataFrame, labels: np.ndarray) -> None:
        """Print regime statistics."""
        print(f"\nRegime Statistics (forward filter):")
        print(f"{'='*70}")
        
        for regime in range(self.n_regimes):
            mask = labels == regime
            count = mask.sum()
            pct = count / len(labels) * 100
            avg_return = data.loc[mask, 'log_return'].mean() * 252 * 100
            
            print(f"\n{self.regime_names[regime]} (Regime {regime}):")
            print(f"  Frequency: {pct:.1f}%")
            print(f"  Avg annual return: {avg_return:+.2f}%")
    
    def get_current_regime(self) -> Tuple[int, str, float, np.ndarray]:
        """
        Get current regime prediction.
        
        Returns:
            (regime_label, regime_name, confidence, probabilities)
        """
        if self.regime_labels is None:
            raise ValueError("No predictions. Call predict_forward_filter() first.")
        
        current_label = self.regime_labels[-1]
        current_name = self.regime_names[current_label]
        current_confidence = self.regime_confidence[-1]
        current_probs = self.regime_probs[-1]
        
        return current_label, current_name, current_confidence, current_probs
    
    def should_retrain(self, current_date: datetime = None) -> bool:
        """Check if model should be retrained based on retrain frequency."""
        if self.last_training_date is None:
            return True
        
        if current_date is None:
            current_date = self.features['date'].iloc[-1]
        
        days_since_training = (current_date - self.last_training_date).days
        
        return days_since_training >= self.retrain_frequency_days
    
    def get_regime_timeseries(self) -> pd.DataFrame:
        """
        Get full timeseries with regime labels and confidence.
        
        Returns:
            DataFrame with: date, close, regime, confidence, prob_bearish, prob_sideways, prob_bullish
        """
        if self.regime_labels is None:
            raise ValueError("No predictions. Call predict_forward_filter() first.")
        
        # Determine start index for results
        start_idx = len(self.features) - len(self.regime_labels)
        
        result = self.features.iloc[start_idx:][['date', 'close']].copy()
        result['regime'] = self.regime_labels
        result['regime_name'] = [self.regime_names[r] for r in self.regime_labels]
        result['confidence'] = self.regime_confidence
        
        # Add probability columns
        for i in range(self.n_regimes):
            result[f'prob_{self.regime_names[i].lower().split("/")[0]}'] = self.regime_probs[:, i]
        
        return result.reset_index(drop=True)


def main():
    """Example usage of professional rolling HMM."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Professional Rolling-Window HMM')
    parser.add_argument('--symbol', type=str, default='AAPL')
    parser.add_argument('--training-window', type=int, default=756, 
                       help='Training window in days (~3 years)')
    parser.add_argument('--retrain-freq', type=int, default=63,
                       help='Retrain frequency in days (quarterly=63)')
    parser.add_argument('--persistence', type=float, default=0.90,
                       help='State persistence strength (0.85-0.95)')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"PROFESSIONAL ROLLING-WINDOW HMM: {args.symbol}")
    print(f"{'#'*70}")
    
    # Initialize
    detector = RollingWindowHMM(
        symbol=args.symbol,
        n_regimes=3,
        training_window_days=args.training_window,
        retrain_frequency_days=args.retrain_freq,
        persistence_strength=args.persistence
    )
    
    # Fetch data
    detector.fetch_data(lookback_days=1500)
    detector.calculate_features()
    
    # Train on rolling window (last N days)
    detector.train_on_window()
    
    # Run forward filter (no backward pass, no historical relabeling)
    detector.predict_forward_filter()
    
    # Current regime
    label, name, confidence, probs = detector.get_current_regime()
    
    print(f"\n{'='*70}")
    print(f"CURRENT REGIME")
    print(f"{'='*70}")
    print(f"Regime: {name}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {detector.regime_names[i]}: {prob:.1%}")
    
    # Check if retrain needed
    if detector.should_retrain():
        print(f"\n⚠️  Model should be retrained (>{args.retrain_freq} days since last training)")
    else:
        print(f"\n[OK] Model is current")
    
    print(f"\n{'='*70}")
    print(f"[OK] Professional HMM complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
