#!/usr/bin/env python3
"""Debug probability relabeling logic."""

import numpy as np

# Simulate the scenario
raw_probs = np.array([3.46e-77, 2.59e-45, 1.00])
regime_mapping = {0: 0, 2: 1, 1: 2}  # old -> new

print("Raw probabilities:", raw_probs)
print("Regime mapping (old->new):", regime_mapping)
print()

# Method 1: Current implementation
relabeled_probs1 = np.zeros(3)
for old_label, new_label in regime_mapping.items():
    relabeled_probs1[new_label] = raw_probs[old_label]
    print(f"  old_label={old_label}, new_label={new_label}: raw_probs[{old_label}]={raw_probs[old_label]:.2e} -> relabeled_probs1[{new_label}]")

print()
print("Relabeled probabilities (method 1):", relabeled_probs1)
print("Argmax:", np.argmax(relabeled_probs1))
