"""
BOCPD Regime Detection package.

Adapted from regime_aware_portfolio_allocator for use as an agent tool.
Provides market regime classification (bull/bear/transition/crisis/consolidation)
using Bayesian Online Changepoint Detection with dynamic hazard rate.
"""

from models.bocpd_regime.detector import detect_current_regime
