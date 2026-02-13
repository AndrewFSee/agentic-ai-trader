"""
Verify Individual ML Model Strategies

Quick check to ensure the new strategies are properly configured.
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STRATEGIES

# Count strategies by type
strategy_types = {}
for name, config in STRATEGIES.items():
    stype = config.get("type", "unknown")
    if stype not in strategy_types:
        strategy_types[stype] = []
    strategy_types[stype].append(name)

print("=" * 60)
print("STRATEGY CONFIGURATION SUMMARY")
print("=" * 60)
print(f"\nTotal Strategies: {len(STRATEGIES)}")
print()

for stype, names in sorted(strategy_types.items()):
    print(f"{stype.upper()} ({len(names)} strategies):")
    for name in sorted(names):
        desc = STRATEGIES[name].get("description", "No description")
        print(f"  - {name}: {desc}")
    print()

# Specific check for individual ML models
print("=" * 60)
print("INDIVIDUAL ML MODEL STRATEGIES")
print("=" * 60)
individual_ml = [(name, config) for name, config in STRATEGIES.items() if config.get("type") == "single_ml"]
print(f"\nFound {len(individual_ml)} individual ML model strategies:\n")

for name, config in sorted(individual_ml):
    model = config.get("model", "Unknown")
    long_only = config.get("long_only", True)
    direction = "Long-only" if long_only else "Long/Short"
    print(f"  {name}")
    print(f"    Model: {model}")
    print(f"    Direction: {direction}")
    print(f"    Description: {config.get('description', 'N/A')}")
    print()

print("=" * 60)
print("✓ Configuration loaded successfully!")
print("=" * 60)
