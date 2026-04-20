import cv2
import numpy as np


def compute_blur_score(crop: np.ndarray, threshold: float = 200.0) -> float:
    """
    Blur severity score from Variance of Laplacian.

    High variance => sharper region => score near 1.0
    Low variance  => blurrier region => score near 0.0
    """
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return round(float(np.clip(variance / threshold, 0.0, 1.0)), 4)
 

def compute_blur_scores_for_bboxes(
    image: np.ndarray,
    bboxes: list,
    threshold: float = 200.0,
) -> list:
    """
    Compute blur severity score for each detected bounding box.

    Returns a list of dict items:
    [{"logo_id": 1, "bbox": [x1, y1, x2, y2], "blur_score": 0.73}, ...]
    """
    if image is None:
        return []

    h, w = image.shape[:2]
    scores = []

    for idx, bbox in enumerate(bboxes, start=1):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Keep crop coordinates inside image bounds.
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        crop = image[y1:y2, x1:x2]
        blur_score = compute_blur_score(crop, threshold=threshold)

        scores.append({
            "logo_id": idx,
            "bbox": [x1, y1, x2, y2],
            "blur_score": blur_score,
        })

    return scores
