"""
Partial Visibility / Occlusion Scoring Module

Computes occlusion score based on YOLO detection confidence and bbox clipping.
High confidence + no boundary clipping = fully visible (score near 1.0)
Low confidence or clipped by frame = partially occluded (score near 0.0)
"""

import numpy as np


def compute_partial_score(bbox: list, frame_shape: tuple, confidence: float) -> float:
    """
    Partial Occlusion Score.
    
    Uses YOLO detection confidence + frame boundary clipping as proxy.
    - confidence near 1.0 = logo likely fully visible
    - confidence near 0.0 = logo likely partially hidden
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        frame_shape: (height, width, channels) of the frame
        confidence: YOLO detection confidence (0.0-1.0)
    
    Returns:
        float: Partial visibility score (0.0-1.0), rounded to 4 decimals
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Compute clipped bbox (intersection with frame bounds)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    
    # Calculate visible area ratio
    orig_area = max(1, (x2 - x1) * (y2 - y1))
    clip_area = max(0, (cx2 - cx1) * (cy2 - cy1))
    clip_ratio = clip_area / orig_area
    
    # Combine confidence with clipping ratio
    score = float(np.clip(confidence * clip_ratio, 0.0, 1.0))
    return round(score, 4)
