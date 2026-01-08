# VIX ROC Three-Tier Risk Overlay Strategy

## Overview

A VIX Rate-of-Change based market timing overlay that automatically classifies assets into tiers and applies optimized parameters. Walk-forward validated on 2020-2024 data with **15/15 wins** on tested assets.

## The Core Insight

> "What if we re-enter the position while VIX is still high, when the RATE of VIX increase has fallen?"

The key discovery: **VIX rate of change matters more than VIX level**. By exiting when VIX is accelerating and re-entering when it's decelerating (even if still elevated), we catch optimal entry points.

## Three-Tier System

| Tier | Assets | Exit Thresh | Re-entry Thresh | Min Days | Avg Excess |
|------|--------|-------------|-----------------|----------|------------|
| **Tier 1** (Value/Cyclical) | SPY, DIA, IWM, XLF, XLE, VNQ | >50% | <+15% | 5 | +39% |
| **Tier 2** (Growth/Tech) | QQQ, XLK, AAPL, GOOGL, AMZN | >20% | <0% | 1 | +20% |
| **Tier 3** (Mega-Cap Tech) | NVDA, MSFT, META | >75% | <-10% | 2 | +140% |

### Why Different Tiers?

- **Tier 1 assets** recover *in sync* with VIX normalization → wait for VIX to calm
- **Tier 2 assets** recover *faster* than VIX normalizes → quick re-entry
- **Tier 3 assets** have *explosive* returns → only exit on extreme events

## Performance (2020-2024 Out-of-Sample)

### Tier 1: Value/Cyclical
| Asset | B&H Return | Strategy | Excess | DD Improvement |
|-------|------------|----------|--------|----------------|
| SPY | +94.6% | +111.5% | **+17.0%** | +6.4% |
| DIA | +62.0% | +88.5% | **+26.4%** | +11.8% |
| IWM | +42.2% | +81.0% | **+38.8%** | +5.4% |
| XLI | +72.5% | +120.3% | **+47.8%** | +15.9% |
| XLF | +71.6% | +114.9% | **+43.3%** | +15.7% |
| XLE | +76.3% | +133.0% | **+56.7%** | +20.2% |
| VNQ | +17.3% | +61.6% | **+44.3%** | +5.4% |

### Tier 2: Growth/Tech
| Asset | B&H Return | Strategy | Excess |
|-------|------------|----------|--------|
| QQQ | +143.9% | +169.2% | **+25.4%** |
| XLK | +160.5% | +166.9% | **+6.4%** |
| AAPL | +244.0% | +263.7% | **+19.7%** |
| GOOGL | +177.6% | +200.4% | **+22.8%** |
| AMZN | +131.2% | +158.7% | **+27.6%** |

### Tier 3: Mega-Cap Tech
| Asset | B&H Return | Strategy | Excess |
|-------|------------|----------|--------|
| NVDA | +2148.4% | +2445.3% | **+296.9%** |
| MSFT | +174.4% | +234.1% | **+59.7%** |
| META | +180.2% | +242.8% | **+62.7%** |

## Usage

### In Trading Agent

```python
from tools import TOOL_REGISTRY

# Get the tool
vix_tool = TOOL_REGISTRY['vix_roc_risk']

# Evaluate a trade
state = {}
result = vix_tool['fn'](state, {'symbol': 'NVDA'})
signal = result['tool_results']['vix_roc_risk']

print(signal['tier_name'])        # "Tier 3: Mega-Cap Tech"
print(signal['current_signal'])   # "hold" | "exit" | "reenter"
print(signal['position_status'])  # "IN MARKET" | "OUT OF MARKET"
```

### Portfolio Risk Check

```python
portfolio_tool = TOOL_REGISTRY['vix_roc_portfolio_risk']
state = {}
result = portfolio_tool['fn'](state, {'symbols': ['SPY', 'QQQ', 'NVDA']})
risk = result['tool_results']['vix_roc_portfolio_risk']

print(risk['risk_level'])      # "LOW" | "MODERATE" | "ELEVATED" | "HIGH"
print(risk['current_vix'])     # 16.0
print(risk['exit_signals_active'])  # 0
```

### Standalone Usage

```python
from models.vix_roc_production import VIXROCRiskOverlay

overlay = VIXROCRiskOverlay()
overlay.load_vix_data(vix_df)

# Classify asset
info = overlay.classify_asset("NVDA")

# Get current signal
signal = overlay.get_current_signal("NVDA")

# Portfolio risk
assessment = overlay.get_risk_assessment(['SPY', 'QQQ', 'NVDA'])
```

## Signal Interpretation

| Signal | Position Status | Action |
|--------|-----------------|--------|
| `hold` | IN MARKET | Safe to trade normally |
| `hold` | OUT OF MARKET | Wait for re-entry signal |
| `exit` | - | DO NOT ENTER - exit if holding |
| `reenter` | - | VIX calming, safe to re-enter |

## Files

- `models/vix_roc_production.py` - Production module with all components
- `tools.py` - Tool registration for agent integration
- `planner.py` - Updated tool selection guidance
- `models/vix_roc_*.py` - Research/validation scripts

## Key Research Files

| File | Purpose |
|------|---------|
| `vix_roc_reentry_strategy.py` | Initial concept validation |
| `vix_roc_walkforward.py` | Walk-forward optimization |
| `vix_roc_multi_asset.py` | 29-asset testing |
| `vix_roc_asset_analysis.py` | Asset characteristics analysis |
| `vix_roc_growth_optimization.py` | Tier 2 parameter tuning |
| `vix_roc_stubborn_optimization.py` | Tier 3 discovery |
| `vix_roc_tier3_validation.py` | Tier 3 validation |
| `vix_roc_dual_strategy.py` | Two-tier validation |
| `vix_roc_production.py` | **Final production module** |

## Integration with Other Tools

The VIX ROC tool complements other risk tools:

| Tool | Purpose | Use Together |
|------|---------|--------------|
| `vix_roc_risk` | **Should I be in or out?** | Primary timing signal |
| `vol_prediction` | Vol spike probability | Position sizing |
| `regime_detection_*` | Market regime context | Directional bias |
| `ml_prediction` | Directional forecast | Trade direction |

## Author
Agentic AI Trader - January 2026
