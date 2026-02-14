"""
Utility functions for regime detection and labeling.

Adapted from regime_aware_portfolio_allocator.
"""

import numpy as np
import pandas as pd


def map_to_range(
    x: pd.Series,
    lo: float,
    hi: float,
    clip: bool = True,
) -> pd.Series:
    """Map values to [0, 1] range using linear scaling."""
    if hi == lo:
        return pd.Series(0.5, index=x.index)
    scaled = (x - lo) / (hi - lo)
    if clip:
        scaled = scaled.clip(0, 1)
    return scaled


def compute_erl_instability(
    erl: pd.Series,
    erl_floor: int = 5,
    erl_stable: int = 126,
) -> pd.Series:
    """Convert expected run length to instability score in [0, 1]."""
    erl_clipped = erl.clip(lower=erl_floor)
    instability = 1 - map_to_range(erl_clipped, erl_floor, erl_stable)
    instability.name = "erl_instability"
    return instability


def apply_run_length_filter(
    labels: pd.Series,
    min_duration: int,
) -> pd.Series:
    """Filter out short regime runs by replacing with neighbouring labels."""
    result = labels.copy()
    if len(result) == 0:
        return result

    runs = []
    current_label = result.iloc[0]
    run_start = 0

    for i, label in enumerate(result):
        if label != current_label:
            runs.append({
                "start": run_start,
                "end": i,
                "label": current_label,
                "length": i - run_start,
            })
            current_label = label
            run_start = i

    runs.append({
        "start": run_start,
        "end": len(result),
        "label": current_label,
        "length": len(result) - run_start,
    })

    for i, run in enumerate(runs):
        if run["length"] < min_duration:
            if i > 0:
                replacement = runs[i - 1]["label"]
            elif i < len(runs) - 1:
                replacement = runs[i + 1]["label"]
            else:
                continue
            result.iloc[run["start"]:run["end"]] = replacement

    return result
