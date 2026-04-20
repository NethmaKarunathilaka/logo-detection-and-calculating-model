import cv2
import numpy as np
from typing import Optional


NEUTRAL_ROTATION_SCORE = 0.5


def _prepare_gray(crop: np.ndarray) -> np.ndarray:
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _find_best_contour(gray: np.ndarray, min_contour_area: float):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    merged = cv2.bitwise_or(mask, edges)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    ch, cw = gray.shape[:2]
    image_center = np.array([cw / 2.0, ch / 2.0], dtype=np.float32)
    best_contour = None
    best_quality = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        (cx, cy), (w, h), _ = cv2.minAreaRect(contour)
        if w <= 0 or h <= 0:
            continue

        rect_area = max(w * h, 1e-6)
        fill_ratio = float(np.clip(area / rect_area, 0.0, 1.0))

        center = np.array([cx, cy], dtype=np.float32)
        center_dist = float(np.linalg.norm(center - image_center))
        max_dist = float(np.linalg.norm(image_center)) + 1e-6
        center_score = 1.0 - float(np.clip(center_dist / max_dist, 0.0, 1.0))

        quality = 0.65 * fill_ratio + 0.35 * center_score
        if quality > best_quality:
            best_quality = quality
            best_contour = contour

    return best_contour


def _effective_angle_from_contour(contour) -> Optional[float]:
    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    if w <= 0 or h <= 0:
        return None

    if w < h:
        angle = angle + 90.0

    abs_angle = float(abs(np.clip(angle, -90.0, 90.0)))
    return min(abs_angle, abs(90.0 - abs_angle))


def compute_rotation_score(
    crop: np.ndarray,
    max_tolerable_angle: float = 45.0,
    min_contour_area: float = 100.0,
) -> float:
    """
    Compute a geometric distortion score for logo rotation.

    Score meaning:
    - 1.0 -> logo appears upright (near 0 degrees)
    - 0.0 -> logo appears heavily rotated (near or beyond max_tolerable_angle)

    Notes:
    - This estimates orientation from dominant contour geometry in the crop.
    - A neutral fallback score is used when orientation cannot be estimated reliably.
    """
    if crop is None or crop.size == 0:
        return NEUTRAL_ROTATION_SCORE

    if max_tolerable_angle <= 0:
        max_tolerable_angle = 45.0

    gray = _prepare_gray(crop)
    best_contour = _find_best_contour(gray, min_contour_area)
    if best_contour is None:
        return NEUTRAL_ROTATION_SCORE

    effective_angle = _effective_angle_from_contour(best_contour)
    if effective_angle is None:
        return NEUTRAL_ROTATION_SCORE

    effective_tolerance = float(np.clip(max_tolerable_angle, 1.0, 45.0))

    score = 1.0 - (min(effective_angle, effective_tolerance) / effective_tolerance)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def compute_rotation_scores_for_bboxes(
    image: np.ndarray,
    bboxes: list,
    max_tolerable_angle: float = 45.0,
    min_contour_area: float = 100.0,
) -> list:
    """
    Compute rotation score per detected logo bounding box.

    Returns a list of dict items:
    [{"logo_id": 1, "bbox": [x1, y1, x2, y2], "rotation_score": 0.91}, ...]
    """
    if image is None:
        return []

    h, w = image.shape[:2]
    scores = []

    for idx, bbox in enumerate(bboxes, start=1):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        crop = image[y1:y2, x1:x2]
        rotation_score = compute_rotation_score(
            crop,
            max_tolerable_angle=max_tolerable_angle,
            min_contour_area=min_contour_area,
        )

        scores.append(
            {
                "logo_id": idx,
                "bbox": [x1, y1, x2, y2],
                "rotation_score": rotation_score,
            }
        )

    return scores
