"""
Auto-labeling module using differential logic (current frame vs background)
Generates YOLO format annotations. ROI: rect (x1,y1,x2,y2) lub polygon (lista punktów).
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, List, Union
from django.conf import settings

logger = logging.getLogger(__name__)


def _roi_to_bbox(roi) -> Optional[Tuple[float, float, float, float]]:
    """Z roi dict zwraca (x1,y1,x2,y2) 0-1 lub None."""
    if roi is None:
        return None
    if isinstance(roi, dict):
        if roi.get('type') == 'rect':
            return tuple(roi['rect'])
        if roi.get('type') == 'polygon' and roi.get('points'):
            pts = roi['points']
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _apply_roi(image: np.ndarray, roi) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Przycina i maskuje obraz do ROI. roi = None | {'type':'rect','rect':(x1,y1,x2,y2)} | {'type':'polygon','points':[[x,y],...]}.
    Zwraca (obraz_do_analizy, x_off, y_off, w_full, h_full).
    """
    if image is None:
        return image, 0, 0, 0, 0
    h, w = image.shape[:2]
    if roi is None:
        return image, 0, 0, w, h
    bbox = _roi_to_bbox(roi)
    if bbox is None:
        return image, 0, 0, w, h
    x1, y1, x2, y2 = bbox
    x1_px = max(0, int(x1 * w))
    y1_px = max(0, int(y1 * h))
    x2_px = min(w, int(x2 * w))
    y2_px = min(h, int(y2 * h))
    if x2_px <= x1_px or y2_px <= y1_px:
        return image, 0, 0, w, h
    cropped = image[y1_px:y2_px, x1_px:x2_px].copy()
    if isinstance(roi, dict) and roi.get('type') == 'polygon' and roi.get('points'):
        pts = np.array(roi['points'], dtype=np.float32)
        pts[:, 0] = (pts[:, 0] - x1) / (x2 - x1) if x2 != x1 else 0
        pts[:, 1] = (pts[:, 1] - y1) / (y2 - y1) if y2 != y1 else 0
        h_c, w_c = cropped.shape[:2]
        pts[:, 0] *= w_c
        pts[:, 1] *= h_c
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    return cropped, x1_px, y1_px, w, h


def _crop_to_roi(image: np.ndarray, roi) -> Tuple[np.ndarray, int, int, int, int]:
    """Kompatybilność: roi jako tuple (x1,y1,x2,y2) lub dict. Zwraca (cropped, x_off, y_off, w_full, h_full)."""
    if roi is not None and isinstance(roi, (list, tuple)) and len(roi) == 4:
        roi = {'type': 'rect', 'rect': tuple(roi)}
    return _apply_roi(image, roi)


def detect_objects_differential(
    current_frame: np.ndarray,
    background: np.ndarray,
    hsv_min: Tuple[int, int, int],
    hsv_max: Tuple[int, int, int],
    min_area: int = 100,
    roi = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Detect objects using differential logic (everything that's not background).
    If roi (x1,y1,x2,y2 normalized 0-1) is given, only the region inside ROI is analyzed;
    returned boxes are in full-frame normalized coordinates.
    """
    if current_frame is None or background is None:
        return []

    frame_work, x_off, y_off, w_full, h_full = _apply_roi(current_frame, roi)
    bg_work, _, _, _, _ = _apply_roi(background, roi)
    if frame_work is None or frame_work.size == 0 or bg_work is None or bg_work.size == 0:
        return []

    h_roi, w_roi = frame_work.shape[:2]
    bbox = _roi_to_bbox(roi)

    current_hsv = cv2.cvtColor(frame_work, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(bg_work, cv2.COLOR_BGR2HSV)
    background_mask = cv2.inRange(background_hsv, np.array(hsv_min), np.array(hsv_max))
    diff = cv2.absdiff(current_hsv, background_hsv)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(thresh, cv2.bitwise_not(background_mask))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        x_center_roi = (x + bw / 2) / w_roi
        y_center_roi = (y + bh / 2) / h_roi
        width_norm = bw / w_full
        height_norm = bh / h_full
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x_center = x1 + (x2 - x1) * x_center_roi
            y_center = y1 + (y2 - y1) * y_center_roi
        else:
            x_center = x_center_roi
            y_center = y_center_roi
        boxes.append((x_center, y_center, width_norm, height_norm))
    return boxes


def save_yolo_annotation(
    boxes: List[Tuple[float, float, float, float]],
    annotation_path: str,
    class_id: int = 0
) -> bool:
    """
    Save bounding boxes in YOLO format
    
    Args:
        boxes: List of (x_center, y_center, width, height) normalized
        annotation_path: Path to save .txt file
        class_id: Class ID (0 for single class - cans)
    
    Returns:
        True if successful
    """
    try:
        annotation_file = Path(annotation_path)
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_file, 'w') as f:
            for box in boxes:
                x_center, y_center, width, height = box
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"Saved YOLO annotation: {annotation_path} with {len(boxes)} objects")
        return True
        
    except Exception as e:
        logger.error(f"Error saving YOLO annotation: {e}")
        return False


def auto_label_image(
    image_path: str,
    background_path: str,
    hsv_min: Tuple[int, int, int],
    hsv_max: Tuple[int, int, int],
    output_annotation_path: Optional[str] = None,
    min_area: int = 100
) -> Tuple[bool, int]:
    """
    Auto-label a single image
    
    Args:
        image_path: Path to image to label
        background_path: Path to background reference image
        hsv_min: Minimum HSV values
        hsv_max: Maximum HSV values
        output_annotation_path: Path to save annotation (if None, uses image_path with .txt extension)
        min_area: Minimum area for detection
    
    Returns:
        Tuple of (success, number_of_objects_detected)
    """
    try:
        # Load images
        current_frame = cv2.imread(image_path)
        background = cv2.imread(background_path)
        
        if current_frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return False, 0
        
        if background is None:
            logger.error(f"Failed to load background: {background_path}")
            return False, 0
        
        # Detect objects
        boxes = detect_objects_differential(current_frame, background, hsv_min, hsv_max, min_area)
        
        if not boxes:
            logger.info(f"No objects detected in {image_path}")
            # Still create empty annotation file
            if output_annotation_path is None:
                output_annotation_path = str(Path(image_path).with_suffix('.txt'))
            save_yolo_annotation([], output_annotation_path)
            return True, 0
        
        # Save annotation
        if output_annotation_path is None:
            output_annotation_path = str(Path(image_path).with_suffix('.txt'))
        
        success = save_yolo_annotation(boxes, output_annotation_path)
        return success, len(boxes)
        
    except Exception as e:
        logger.error(f"Error in auto-labeling: {e}")
        return False, 0
