"""
Leading Indicators Module for Enhanced Regime Detection

This module provides leading indicators that can signal regime changes
BEFORE they are reflected in price-based BOCPD.

Key Leading Indicators:
    1. VIX - Implied volatility tends to spike at start of (or before) crashes
    2. VIX Term Structure - Inversion signals stress
    3. Credit Spreads (HYG-LQD) - Widens before equity stress
    4. VIX Rate of Change - Fast rises are predictive

Theory:
    BOCPD on price data is REACTIVE - it needs to see the new regime
    before updating its posterior. Leading indicators can provide
    EARLIER warning because:
    - VIX reflects options market expectations (forward-looking)
    - Credit spreads reflect credit risk premium (leading indicator)
    - These often move BEFORE equity prices collapse

Signal Timing (NO LOOKAHEAD):
    All signals computed at end of day t use only data through day t.
    The signal is applied to day t+1's position.

Author: Agentic AI Trader
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class VIXRegime(Enum):
    """VIX regime classification."""
    LOW = "low"           # VIX < 15: Complacency
    NORMAL = "normal"     # VIX 15-25: Normal
    ELEVATED = "elevated" # VIX 25-35: Elevated stress
    HIGH = "high"         # VIX 35-50: High stress
    EXTREME = "extreme"   # VIX > 50: Crisis mode


@dataclass
class LeadingIndicatorConfig:
    """Configuration for leading indicators."""
    
    # VIX thresholds (absolute levels)
    vix_low: float = 15.0
    vix_normal: float = 25.0
    vix_elevated: float = 35.0
    vix_extreme: float = 50.0
    
    # VIX rate of change thresholds
    vix_roc_window: int = 5           # Days for rate of change
    vix_roc_warning: float = 0.20     # 20% rise in 5 days = warning
    vix_roc_danger: float = 0.40      # 40% rise in 5 days = danger
    
    # VIX EWMA for smoothing
    vix_ewma_span: int = 5
    
    # VIX term structure (VIX vs VIX3M)
    term_inversion_threshold: float = 1.05  # VIX/VIX3M > 1.05 = inverted
    
    # Credit spread thresholds (HYG-LQD or similar)
    spread_ewma_span: int = 20
    spread_zscore_warning: float = 1.5    # Z > 1.5 = warning
    spread_zscore_danger: float = 2.5     # Z > 2.5 = danger
    
    # Combined risk scoring weights
    weight_vix_level: float = 0.25
    weight_vix_roc: float = 0.35          # Rate of change most predictive
    weight_vix_term: float = 0.15
    weight_credit: float = 0.25


@dataclass
class LeadingIndicatorState:
    """Current state of leading indicators."""
    timestamp: pd.Timestamp
    
    # VIX signals
    vix_level: float = 0.0
    vix_regime: VIXRegime = VIXRegime.NORMAL
    vix_roc: float = 0.0                  # Rate of change (5-day)
    vix_ewma: float = 0.0
    vix_zscore: float = 0.0               # Standardized level
    
    # VIX term structure
    vix_term_ratio: float = 1.0           # VIX / VIX3M (if available)
    term_inverted: bool = False
    
    # Credit signals (if available)
    credit_spread: float = 0.0
    credit_spread_zscore: float = 0.0
    
    # Combined risk score [0, 1]
    leading_risk_score: float = 0.0
    
    # Individual risk components
    risk_vix_level: float = 0.0
    risk_vix_roc: float = 0.0
    risk_vix_term: float = 0.0
    risk_credit: float = 0.0


class LeadingIndicatorEngine:
    """
    Computes leading indicator signals for regime detection.
    
    Supports:
        - VIX level and regime classification
        - VIX rate of change (most predictive)
        - VIX term structure inversion
        - Credit spreads (HYG-LQD)
        
    Usage:
        engine = LeadingIndicatorEngine()
        
        # Update with VIX data
        state = engine.update(timestamp, vix_close)
        
        # Or with full data
        state = engine.update(
            timestamp, 
            vix_close,
            vix3m_close=vix3m,  # Optional
            hyg_close=hyg,      # Optional
            lqd_close=lqd       # Optional
        )
        
        # Use state.leading_risk_score as risk signal
    """
    
    def __init__(self, config: Optional[LeadingIndicatorConfig] = None):
        self.config = config or LeadingIndicatorConfig()
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        # VIX history
        self._vix_history: List[float] = []
        self._vix_ewma: float = 0.0
        self._vix_ewma_alpha: float = 2.0 / (self.config.vix_ewma_span + 1)
        
        # Credit spread history
        self._spread_history: List[float] = []
        self._spread_ewma: float = 0.0
        self._spread_ewma_alpha: float = 2.0 / (self.config.spread_ewma_span + 1)
        
        # Long-term VIX stats for z-scoring
        self._vix_sum: float = 0.0
        self._vix_sum_sq: float = 0.0
        self._vix_count: int = 0
    
    def _classify_vix_regime(self, vix: float) -> VIXRegime:
        """Classify VIX into regime buckets."""
        if vix < self.config.vix_low:
            return VIXRegime.LOW
        elif vix < self.config.vix_normal:
            return VIXRegime.NORMAL
        elif vix < self.config.vix_elevated:
            return VIXRegime.ELEVATED
        elif vix < self.config.vix_extreme:
            return VIXRegime.HIGH
        else:
            return VIXRegime.EXTREME
    
    def _compute_vix_zscore(self, vix: float) -> float:
        """Compute z-score of current VIX vs historical."""
        if self._vix_count < 20:
            return 0.0
        
        mean = self._vix_sum / self._vix_count
        var = (self._vix_sum_sq / self._vix_count) - mean**2
        std = np.sqrt(max(var, 1e-8))
        
        return (vix - mean) / std
    
    def _compute_risk_from_vix_level(self, vix: float) -> float:
        """
        Compute risk score from VIX level.
        
        Returns [0, 1]:
            VIX < 15: 0
            VIX 15-35: linear 0 to 0.5
            VIX 35-50: linear 0.5 to 0.8
            VIX > 50: 0.8 to 1.0
        """
        if vix < self.config.vix_low:
            return 0.0
        elif vix < self.config.vix_elevated:
            # Linear from 0 to 0.5 between vix_low and vix_elevated
            return 0.5 * (vix - self.config.vix_low) / (self.config.vix_elevated - self.config.vix_low)
        elif vix < self.config.vix_extreme:
            # Linear from 0.5 to 0.8 between elevated and extreme
            return 0.5 + 0.3 * (vix - self.config.vix_elevated) / (self.config.vix_extreme - self.config.vix_elevated)
        else:
            # VIX > 50: asymptotic to 1.0
            excess = vix - self.config.vix_extreme
            return 0.8 + 0.2 * (1 - np.exp(-excess / 20))
    
    def _compute_risk_from_vix_roc(self, roc: float) -> float:
        """
        Compute risk score from VIX rate of change.
        
        This is the MOST PREDICTIVE indicator.
        A rapidly rising VIX often precedes or coincides with crash onset.
        
        Returns [0, 1]:
            ROC < 0: 0 (VIX falling = calming)
            ROC 0-20%: linear 0 to 0.3
            ROC 20-40%: linear 0.3 to 0.7
            ROC > 40%: 0.7 to 1.0
        """
        if roc < 0:
            return 0.0
        elif roc < self.config.vix_roc_warning:
            return 0.3 * roc / self.config.vix_roc_warning
        elif roc < self.config.vix_roc_danger:
            return 0.3 + 0.4 * (roc - self.config.vix_roc_warning) / (self.config.vix_roc_danger - self.config.vix_roc_warning)
        else:
            # Asymptotic to 1.0
            excess = roc - self.config.vix_roc_danger
            return 0.7 + 0.3 * (1 - np.exp(-excess / 0.3))
    
    def _compute_risk_from_term(self, term_ratio: float) -> float:
        """
        Compute risk from VIX term structure.
        
        Normal: VIX < VIX3M (contango), term_ratio < 1
        Stressed: VIX > VIX3M (backwardation), term_ratio > 1
        
        Returns [0, 1]:
            ratio < 1.0: 0
            ratio 1.0-1.05: linear 0 to 0.5
            ratio > 1.05: linear 0.5 to 1.0
        """
        if term_ratio < 1.0:
            return 0.0
        elif term_ratio < self.config.term_inversion_threshold:
            return 0.5 * (term_ratio - 1.0) / (self.config.term_inversion_threshold - 1.0)
        else:
            excess = term_ratio - self.config.term_inversion_threshold
            return 0.5 + 0.5 * (1 - np.exp(-excess / 0.1))
    
    def _compute_risk_from_credit(self, zscore: float) -> float:
        """
        Compute risk from credit spread z-score.
        
        Returns [0, 1]:
            z < 0: 0 (spreads tightening)
            z 0-1.5: linear 0 to 0.3
            z 1.5-2.5: linear 0.3 to 0.7
            z > 2.5: 0.7 to 1.0
        """
        if zscore < 0:
            return 0.0
        elif zscore < self.config.spread_zscore_warning:
            return 0.3 * zscore / self.config.spread_zscore_warning
        elif zscore < self.config.spread_zscore_danger:
            return 0.3 + 0.4 * (zscore - self.config.spread_zscore_warning) / (self.config.spread_zscore_danger - self.config.spread_zscore_warning)
        else:
            excess = zscore - self.config.spread_zscore_danger
            return 0.7 + 0.3 * (1 - np.exp(-excess / 1.0))
    
    def update(
        self,
        timestamp: pd.Timestamp,
        vix_close: float,
        vix3m_close: Optional[float] = None,
        hyg_close: Optional[float] = None,
        lqd_close: Optional[float] = None
    ) -> LeadingIndicatorState:
        """
        Update with new data and compute risk signals.
        
        Args:
            timestamp: Current timestamp
            vix_close: VIX close price (required)
            vix3m_close: VIX3M close price (optional, for term structure)
            hyg_close: HYG ETF close (optional, for credit spreads)
            lqd_close: LQD ETF close (optional, for credit spreads)
            
        Returns:
            LeadingIndicatorState with all computed signals
        """
        # Update VIX history
        self._vix_history.append(vix_close)
        if len(self._vix_history) > 100:
            self._vix_history = self._vix_history[-100:]
        
        # Update VIX running stats
        self._vix_sum += vix_close
        self._vix_sum_sq += vix_close ** 2
        self._vix_count += 1
        
        # VIX EWMA
        if self._vix_ewma == 0:
            self._vix_ewma = vix_close
        else:
            self._vix_ewma = self._vix_ewma_alpha * vix_close + (1 - self._vix_ewma_alpha) * self._vix_ewma
        
        # VIX regime
        vix_regime = self._classify_vix_regime(vix_close)
        
        # VIX z-score
        vix_zscore = self._compute_vix_zscore(vix_close)
        
        # VIX rate of change
        if len(self._vix_history) > self.config.vix_roc_window:
            prev_vix = self._vix_history[-self.config.vix_roc_window - 1]
            vix_roc = (vix_close - prev_vix) / max(prev_vix, 1e-8)
        else:
            vix_roc = 0.0
        
        # VIX term structure
        if vix3m_close is not None and vix3m_close > 0:
            vix_term_ratio = vix_close / vix3m_close
            term_inverted = vix_term_ratio > self.config.term_inversion_threshold
        else:
            vix_term_ratio = 1.0
            term_inverted = False
        
        # Credit spread (HYG yield - LQD yield, approximated by price ratio)
        if hyg_close is not None and lqd_close is not None and lqd_close > 0:
            # When HYG underperforms LQD, spread is widening
            # Use inverse ratio so higher = more stress
            credit_spread = lqd_close / hyg_close
            
            self._spread_history.append(credit_spread)
            if len(self._spread_history) > 100:
                self._spread_history = self._spread_history[-100:]
            
            # Credit spread EWMA
            if self._spread_ewma == 0:
                self._spread_ewma = credit_spread
            else:
                self._spread_ewma = self._spread_ewma_alpha * credit_spread + (1 - self._spread_ewma_alpha) * self._spread_ewma
            
            # Credit spread z-score
            if len(self._spread_history) >= self.config.spread_ewma_span:
                spread_mean = np.mean(self._spread_history[-self.config.spread_ewma_span:])
                spread_std = np.std(self._spread_history[-self.config.spread_ewma_span:])
                credit_spread_zscore = (credit_spread - spread_mean) / max(spread_std, 1e-8)
            else:
                credit_spread_zscore = 0.0
        else:
            credit_spread = 0.0
            credit_spread_zscore = 0.0
        
        # Compute individual risk components
        risk_vix_level = self._compute_risk_from_vix_level(vix_close)
        risk_vix_roc = self._compute_risk_from_vix_roc(vix_roc)
        risk_vix_term = self._compute_risk_from_term(vix_term_ratio)
        risk_credit = self._compute_risk_from_credit(credit_spread_zscore)
        
        # Combined risk score (weighted average)
        leading_risk_score = (
            self.config.weight_vix_level * risk_vix_level +
            self.config.weight_vix_roc * risk_vix_roc +
            self.config.weight_vix_term * risk_vix_term +
            self.config.weight_credit * risk_credit
        )
        
        # Normalize to [0, 1]
        total_weight = (
            self.config.weight_vix_level +
            self.config.weight_vix_roc +
            self.config.weight_vix_term +
            self.config.weight_credit
        )
        leading_risk_score = leading_risk_score / total_weight
        
        return LeadingIndicatorState(
            timestamp=timestamp,
            vix_level=vix_close,
            vix_regime=vix_regime,
            vix_roc=vix_roc,
            vix_ewma=self._vix_ewma,
            vix_zscore=vix_zscore,
            vix_term_ratio=vix_term_ratio,
            term_inverted=term_inverted,
            credit_spread=credit_spread,
            credit_spread_zscore=credit_spread_zscore,
            leading_risk_score=leading_risk_score,
            risk_vix_level=risk_vix_level,
            risk_vix_roc=risk_vix_roc,
            risk_vix_term=risk_vix_term,
            risk_credit=risk_credit
        )
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        vix_col: str = 'VIX',
        vix3m_col: Optional[str] = None,
        hyg_col: Optional[str] = None,
        lqd_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process DataFrame with leading indicator data.
        
        Args:
            df: DataFrame indexed by date
            vix_col: Column name for VIX close
            vix3m_col: Column name for VIX3M close (optional)
            hyg_col: Column name for HYG close (optional)
            lqd_col: Column name for LQD close (optional)
            
        Returns:
            DataFrame with leading indicator signals
        """
        self.reset()
        results = []
        
        for idx, row in df.iterrows():
            timestamp = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
            
            vix = row.get(vix_col, np.nan)
            if pd.isna(vix):
                continue
            
            vix3m = row.get(vix3m_col) if vix3m_col else None
            hyg = row.get(hyg_col) if hyg_col else None
            lqd = row.get(lqd_col) if lqd_col else None
            
            if pd.isna(vix3m):
                vix3m = None
            if pd.isna(hyg):
                hyg = None
            if pd.isna(lqd):
                lqd = None
            
            state = self.update(timestamp, vix, vix3m, hyg, lqd)
            
            results.append({
                'timestamp': state.timestamp,
                'vix_level': state.vix_level,
                'vix_regime': state.vix_regime.value,
                'vix_roc': state.vix_roc,
                'vix_ewma': state.vix_ewma,
                'vix_zscore': state.vix_zscore,
                'vix_term_ratio': state.vix_term_ratio,
                'term_inverted': state.term_inverted,
                'credit_spread_zscore': state.credit_spread_zscore,
                'leading_risk_score': state.leading_risk_score,
                'risk_vix_level': state.risk_vix_level,
                'risk_vix_roc': state.risk_vix_roc,
                'risk_vix_term': state.risk_vix_term,
                'risk_credit': state.risk_credit
            })
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        return result_df


# =============================================================================
# INTEGRATED RISK SCORER (BOCPD + LEADING INDICATORS)
# =============================================================================

@dataclass
class CombinedRiskConfig:
    """Configuration for combining BOCPD and leading indicator risk."""
    
    # Weight for each risk source
    weight_bocpd: float = 0.4        # BOCPD-based risk (reactive but precise)
    weight_leading: float = 0.6      # Leading indicator risk (predictive)
    
    # Combination mode
    mode: str = "max"                # "max", "weighted", "product"
    
    # Fast override: if leading risk > this, immediately reduce
    leading_override_thr: float = 0.7
    override_position: float = 0.5
    
    # VIX emergency brake: if VIX > this, minimum position
    vix_emergency_level: float = 40.0
    emergency_position: float = 0.25


def combine_risk_scores(
    bocpd_risk: float,
    leading_risk: float,
    config: Optional[CombinedRiskConfig] = None
) -> Tuple[float, float]:
    """
    Combine BOCPD risk and leading indicator risk.
    
    Args:
        bocpd_risk: Risk score from BOCPD [0, 1]
        leading_risk: Risk score from leading indicators [0, 1]
        config: Combination configuration
        
    Returns:
        (combined_risk, position_adjustment)
        - combined_risk: Overall risk score [0, 1]
        - position_adjustment: Suggested position scalar [0, 1]
    """
    config = config or CombinedRiskConfig()
    
    if config.mode == "max":
        combined_risk = max(bocpd_risk, leading_risk)
    elif config.mode == "weighted":
        combined_risk = (
            config.weight_bocpd * bocpd_risk +
            config.weight_leading * leading_risk
        ) / (config.weight_bocpd + config.weight_leading)
    elif config.mode == "product":
        # P(either risk) = 1 - (1-p1)(1-p2)
        combined_risk = 1 - (1 - bocpd_risk) * (1 - leading_risk)
    else:
        combined_risk = max(bocpd_risk, leading_risk)
    
    # Check for fast override
    if leading_risk > config.leading_override_thr:
        position_adjustment = config.override_position
    else:
        # Smooth mapping from risk to position
        # risk=0 -> pos=1.0, risk=1 -> pos=0.25
        position_adjustment = 1.0 - 0.75 * combined_risk
    
    return combined_risk, position_adjustment


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LEADING INDICATORS MODULE - Test")
    print("=" * 70)
    
    try:
        import yfinance as yf
        
        # Download VIX and related data
        print("\nDownloading VIX data...")
        vix = yf.download("^VIX", start="2019-01-01", end="2024-12-31", progress=False)
        
        if vix.empty:
            print("Failed to download VIX data")
            exit(1)
        
        # Handle multi-index columns
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        print(f"VIX data: {len(vix)} bars")
        
        # Try to get VIX3M and credit data
        print("Downloading supplementary data...")
        vix3m = yf.download("^VIX3M", start="2019-01-01", end="2024-12-31", progress=False)
        hyg = yf.download("HYG", start="2019-01-01", end="2024-12-31", progress=False)
        lqd = yf.download("LQD", start="2019-01-01", end="2024-12-31", progress=False)
        
        # Combine into single DataFrame
        df = pd.DataFrame(index=vix.index)
        df['VIX'] = vix['Close']
        
        if not vix3m.empty:
            if isinstance(vix3m.columns, pd.MultiIndex):
                vix3m.columns = vix3m.columns.get_level_values(0)
            df['VIX3M'] = vix3m['Close']
        
        if not hyg.empty:
            if isinstance(hyg.columns, pd.MultiIndex):
                hyg.columns = hyg.columns.get_level_values(0)
            df['HYG'] = hyg['Close']
        
        if not lqd.empty:
            if isinstance(lqd.columns, pd.MultiIndex):
                lqd.columns = lqd.columns.get_level_values(0)
            df['LQD'] = lqd['Close']
        
        # Forward-fill any gaps
        df = df.ffill()
        
        print(f"\nCombined data: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Process with leading indicators
        engine = LeadingIndicatorEngine()
        
        results = engine.process_dataframe(
            df,
            vix_col='VIX',
            vix3m_col='VIX3M' if 'VIX3M' in df.columns else None,
            hyg_col='HYG' if 'HYG' in df.columns else None,
            lqd_col='LQD' if 'LQD' in df.columns else None
        )
        
        print(f"\nResults: {len(results)} rows")
        
        # Show COVID period
        covid_start = "2020-02-01"
        covid_end = "2020-04-30"
        covid_mask = (results.index >= covid_start) & (results.index <= covid_end)
        covid_results = results[covid_mask]
        
        print("\n" + "=" * 70)
        print("COVID CRASH PERIOD - Leading Indicator Signals")
        print("=" * 70)
        
        # Find key dates
        key_dates = ["2020-02-19", "2020-02-21", "2020-02-24", "2020-02-25", 
                     "2020-02-28", "2020-03-09", "2020-03-12", "2020-03-16"]
        
        print(f"\n{'Date':<12} {'VIX':>8} {'VIX_ROC':>10} {'Risk_ROC':>10} {'Lead_Risk':>10} {'Regime':<10}")
        print("-" * 70)
        
        for date in key_dates:
            if date in results.index.strftime('%Y-%m-%d').values:
                row = results.loc[date]
                print(f"{date:<12} {row['vix_level']:>8.1f} {row['vix_roc']:>+10.1%} "
                      f"{row['risk_vix_roc']:>10.2f} {row['leading_risk_score']:>10.2f} "
                      f"{row['vix_regime']:<10}")
        
        # Compare to SPY
        print("\n" + "=" * 70)
        print("TIMING COMPARISON vs SPY")
        print("=" * 70)
        
        spy = yf.download("SPY", start="2020-02-01", end="2020-04-30", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy['Return'] = spy['Close'].pct_change()
        
        # Align
        aligned = pd.DataFrame(index=spy.index)
        aligned['Return'] = spy['Return']
        aligned['VIX'] = results['vix_level']
        aligned['Lead_Risk'] = results['leading_risk_score']
        aligned['VIX_ROC'] = results['vix_roc']
        
        aligned = aligned.dropna()
        
        print(f"\n{'Date':<12} {'SPY_Ret':>10} {'VIX':>8} {'Lead_Risk':>10} {'Signal':<12}")
        print("-" * 60)
        
        for date_str in key_dates:
            date = pd.Timestamp(date_str)
            if date in aligned.index:
                row = aligned.loc[date]
                ret = row['Return']
                risk = row['Lead_Risk']
                signal = "REDUCE" if risk > 0.5 else "HOLD"
                if risk > 0.7:
                    signal = "HEAVY REDUCE"
                print(f"{date_str:<12} {ret:>+10.1%} {row['VIX']:>8.1f} {risk:>10.2f} {signal:<12}")
        
        print("\n" + "=" * 70)
        print("KEY INSIGHT: VIX ROC and leading risk score spike EARLY")
        print("By Feb 24, leading_risk > 0.5 (reduce position BEFORE worst days)")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
