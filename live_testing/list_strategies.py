"""
Quick reference: All 17 trading strategies

Run this to see the current strategy configuration at a glance.
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import STRATEGIES

print("\n" + "="*70)
print("ALL 17 TRADING STRATEGIES")
print("="*70)

# Group by category
categories = {
    "Baseline": [k for k, v in STRATEGIES.items() if v.get("type") == "baseline"],
    "Regime Detection": [k for k, v in STRATEGIES.items() if v.get("type") == "regime"],
    "ML Consensus": [k for k, v in STRATEGIES.items() if v.get("type") == "ml"],
    "ML Individual Models": [k for k, v in STRATEGIES.items() if v.get("type") == "single_ml"],
    "Agent (RAG + ML + Regime)": [k for k, v in STRATEGIES.items() if v.get("type") == "agent"],
}

for category, strategy_names in categories.items():
    print(f"\n{category} ({len(strategy_names)}):")
    for name in sorted(strategy_names):
        config = STRATEGIES[name]
        desc = config.get("description", "")
        long_only = config.get("long_only", True)
        direction = "[Long-only]" if long_only else "[Long/Short]"
        
        # Special formatting for individual ML models
        if config.get("type") == "single_ml":
            model = config.get("model", "")
            print(f"  • {name:25} {direction:15} {model}")
        else:
            print(f"  • {name:25} {direction:15} {desc}")

print("\n" + "="*70)
print("TOTAL: 17 strategies")
print("="*70)
print("\nTo start paper trading with all strategies:")
print("  cd live_testing")
print("  python paper_trader.py")
print("\nTo view results:")
print("  python view_results.py")
print()
