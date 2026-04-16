"""
Background calibration module for calculating HSV parameters
"""
import cv2
import numpy as np
from pathlib import Path
import logging
import time
from typing import Tuple, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


def calculate_hsv_parameters(images: List[np.ndarray]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Calculate HSV min and max parameters from a list of background images

    Args:
        images: List of background images (BGR format)

    Returns:
        Tuple of (min_hsv, max_hsv) where each is (H, S, V)
    """
    if not images:
        raise ValueError("No images provided for calibration")

    all_hsv_values = []

    for img in images:
        if img is None:
            continue

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Flatten and collect all HSV values
        hsv_flat = hsv.reshape(-1, 3)
        all_hsv_values.append(hsv_flat)

    if not all_hsv_values:
        raise ValueError("No valid images for calibration")

    # Combine all HSV values
    all_hsv = np.vstack(all_hsv_values)

    # Calculate percentiles to get robust min/max (using 5th and 95th percentiles)
    min_hsv = np.percentile(all_hsv, 5, axis=0).astype(int)
    max_hsv = np.percentile(all_hsv, 95, axis=0).astype(int)

    # Ensure values are in valid ranges
    min_hsv[0] = max(0, min_hsv[0])  # H: 0-179
    min_hsv[1] = max(0, min_hsv[1])  # S: 0-255
    min_hsv[2] = max(0, min_hsv[2])  # V: 0-255

    max_hsv[0] = min(179, max_hsv[0])  # H: 0-179
    max_hsv[1] = min(255, max_hsv[1])  # S: 0-255
    max_hsv[2] = min(255, max_hsv[2])  # V: 0-255

    logger.info("Calculated HSV parameters - Min: %s, Max: %s", min_hsv.tolist(), max_hsv.tolist())
    return (tuple(int(x) for x in min_hsv), tuple(int(x) for x in max_hsv))


def save_background_images(images: List[np.ndarray], camera_id: int) -> List[str]:
    """
    Save background images to disk

    Args:
        images: List of background images
        camera_id: Camera ID for organizing files

    Returns:
        List of saved image paths
    """
    backgrounds_dir = Path(settings.BACKGROUNDS_ROOT) / f"camera_{camera_id}"
    backgrounds_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    timestamp = int(time.time())

    for i, img in enumerate(images):
        if img is None:
            continue

        filename = f"background_{timestamp}_{i:04d}.jpg"
        filepath = backgrounds_dir / filename

        cv2.imwrite(str(filepath), img)
        saved_paths.append(str(filepath))
        logger.info("Saved background image: %s", filepath)

    return saved_paths


def load_background_image(image_path: str) -> Optional[np.ndarray]:
    """Load background image from path."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to load image: %s", image_path)
        return img
    except Exception as e:
        logger.error("Error loading background image: %s", e)
        return None


def load_background_images_from_directory(dir_path: str) -> List[np.ndarray]:
    """
    Load all jpg/png images from a directory. Used for calibration from multiple background samples.
    """
    images = []
    path = Path(dir_path)
    if not path.is_dir():
        logger.warning("Background directory is not a directory: %s", dir_path)
        return images
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for f in path.glob(ext):
            try:
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
            except Exception as e:
                logger.debug("Skip %s: %s", f, e)
    logger.info("Loaded %d background images from %s", len(images), dir_path)
    return images


def get_one_background_for_detection(calibration) -> Optional[np.ndarray]:
    """
    Zwraca jeden obraz tła do użycia w pętli zbierania/detekcji.
    Jeśli ustawiono katalog – ładuje pierwszy obraz z katalogu; w przeciwnym razie pojedynczy plik.
    """
    if getattr(calibration, "background_images_directory", None) and calibration.background_images_directory.strip():
        images = load_background_images_from_directory(calibration.background_images_directory.strip())
        return images[0] if images else None
    if calibration.background_image_path:
        return load_background_image(calibration.background_image_path)
    return None
