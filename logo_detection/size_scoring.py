"""
Size Scoring Module

Computes prominence score based on logo size relative to frame.
Uses log-scale area ratio for realistic perception mapping.
"""

import numpy as np


def compute_size_score(bbox: list, frame_shape: tuple) -> float:
    """
    On-Screen Size Score — log-scale area ratio.
    
    - Logo area < 0.2% of frame -> score 0.0  (too small / distant)
    - Logo area > 3.0% of frame -> score 1.0  (large / prominent)
    - Linear interpolation in log-space for intermediate sizes
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        frame_shape: (height, width, channels) of the frame
    
    Returns:
        float: Size prominence score (0.0-1.0), rounded to 4 decimals
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate bbox area as percentage of frame
    bbox_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    frame_area = h * w
    
    if frame_area == 0:
        return 0.0
    
    ratio = bbox_area / frame_area
    
    # Define bounds for size scoring
    MIN_RATIO = 0.002  # 0.2% of frame
    MAX_RATIO = 0.03   # 3.0% of frame
    
    # Clamp to bounds
    if ratio <= MIN_RATIO:
        return 0.0
    if ratio >= MAX_RATIO:
        return 1.0
    
    # Log-scale interpolation between min and max
    score = (np.log(ratio) - np.log(MIN_RATIO)) / (np.log(MAX_RATIO) - np.log(MIN_RATIO))
    return round(float(np.clip(score, 0.0, 1.0)), 4)
