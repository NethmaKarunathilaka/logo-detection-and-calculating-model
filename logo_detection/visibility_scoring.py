"""
Visibility Quality Score (VQS) Module

Combines all visibility factors into a single final score.
Factors: blur, partial visibility, size, rotation
"""

import numpy as np


def compute_visibility_score(
    blur_score: float,
    partial_score: float,
    size_score: float,
    rotation_score: float,
    weights: dict = None,
) -> float:
    """
    Visibility Quality Score (VQS) — weighted combination of all factors.

    Combines:
    - blur_score: How sharp the logo is (0=blurry, 1=sharp)
    - partial_score: How much logo is visible (0=occluded, 1=fully visible)
    - size_score: Logo prominence relative to frame (0=tiny, 1=prominent)
    - rotation_score: Logo orientation quality (0=heavily rotated, 1=upright)

    Default weights: Equal (0.25 each)
    Tune these in production based on your validation set performance.

    Args:
        blur_score: Blur factor (0.0-1.0)
        partial_score: Partial visibility factor (0.0-1.0)
        size_score: Size/prominence factor (0.0-1.0)
        rotation_score: Rotation/orientation factor (0.0-1.0)
        weights: Dict with keys ["blur", "partial", "size", "rotation"]
            Default: all weights = 0.25

    Returns:
        float: Final visibility score (0.0-1.0), rounded to 4 decimals
    """
    if weights is None:
        weights = {
            "blur": 0.25,
            "partial": 0.25,
            "size": 0.25,
            "rotation": 0.25,
        }

    # Validate weights sum to ~1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        # Normalize weights
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Weighted combination
    vqs = (
        weights["blur"] * blur_score
        + weights["partial"] * partial_score
        + weights["size"] * size_score
        + weights["rotation"] * rotation_score
    )

    return round(float(np.clip(vqs, 0.0, 1.0)), 4)


def get_default_weights() -> dict:
    """
    Returns default equal weights for all factors.
    Use this to verify or modify weights easily.

    Returns:
        dict: Weights dictionary
    """
    return {
        "blur": 0.25,
        "partial": 0.25,
        "size": 0.25,
        "rotation": 0.25,
    }
