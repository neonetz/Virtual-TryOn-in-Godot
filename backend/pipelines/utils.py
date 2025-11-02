"""
Utility Functions
NMS, timing, logging, visualization, IoU calculation
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1, box2: Dicts with keys {x, y, w, h}
        
    Returns:
        IoU value [0, 1]
    """
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def non_maximum_suppression(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection dicts with keys {bbox: {x, y, w, h}, score: float}
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by score (descending)
    sorted_dets = sorted(detections, key=lambda d: d['score'], reverse=True)
    
    keep = []
    while len(sorted_dets) > 0:
        # Take the detection with highest score
        current = sorted_dets.pop(0)
        keep.append(current)
        
        # Remove all detections with high IoU
        filtered = []
        for det in sorted_dets:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou <= iou_threshold:
                filtered.append(det)
        
        sorted_dets = filtered
    
    logger.debug(f"NMS: {len(detections)} → {len(keep)} detections")
    return keep


class Timer:
    """Simple timing context manager"""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        if self.verbose:
            logger.info(f"{self.name}: {self.elapsed_ms:.2f} ms")


def draw_detections(img: np.ndarray, detections: List[Dict], 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes and scores on image
    
    Args:
        img: Input image
        detections: List of detection dicts
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    vis_img = img.copy()
    
    for det in detections:
        bbox = det['bbox']
        score = det.get('score', 0.0)
        
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
        
        # Draw score label
        label = f"{score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for text
        cv2.rectangle(
            vis_img,
            (x, y - label_size[1] - baseline - 5),
            (x + label_size[0], y),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            vis_img,
            label,
            (x, y - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return vis_img


def resize_maintain_aspect(img: np.ndarray, max_width: int = 1280, 
                          max_height: int = 720) -> np.ndarray:
    """
    Resize image maintaining aspect ratio
    
    Args:
        img: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if w <= max_width and h <= max_height:
        return img
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_fps_counter():
    """Create FPS counter closure"""
    frame_times = []
    max_samples = 30
    
    def update() -> float:
        """Update and return current FPS"""
        current_time = time.time()
        frame_times.append(current_time)
        
        # Keep only recent samples
        if len(frame_times) > max_samples:
            frame_times.pop(0)
        
        if len(frame_times) < 2:
            return 0.0
        
        # Calculate FPS from time differences
        time_diff = frame_times[-1] - frame_times[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(frame_times) - 1) / time_diff
        return fps
    
    return update


def clamp_bbox(bbox: Dict, img_width: int, img_height: int) -> Dict:
    """
    Clamp bounding box to image boundaries
    
    Args:
        bbox: Dict with keys {x, y, w, h}
        img_width: Image width
        img_height: Image height
        
    Returns:
        Clamped bbox dict
    """
    x = max(0, min(bbox['x'], img_width - 1))
    y = max(0, min(bbox['y'], img_height - 1))
    w = max(1, min(bbox['w'], img_width - x))
    h = max(1, min(bbox['h'], img_height - y))
    
    return {'x': x, 'y': y, 'w': w, 'h': h}


def is_bbox_valid(bbox: Dict, min_size: int = 20) -> bool:
    """
    Check if bounding box is valid (not too small, positive dimensions)
    
    Args:
        bbox: Dict with keys {x, y, w, h}
        min_size: Minimum width/height
        
    Returns:
        True if valid
    """
    return bbox['w'] >= min_size and bbox['h'] >= min_size


def expand_bbox(bbox: Dict, expansion_factor: float = 1.2, 
               img_width: int = None, img_height: int = None) -> Dict:
    """
    Expand bounding box by a factor
    
    Args:
        bbox: Dict with keys {x, y, w, h}
        expansion_factor: Factor to expand (1.2 = 20% larger)
        img_width: Optional image width for clamping
        img_height: Optional image height for clamping
        
    Returns:
        Expanded bbox dict
    """
    cx = bbox['x'] + bbox['w'] / 2
    cy = bbox['y'] + bbox['h'] / 2
    
    new_w = bbox['w'] * expansion_factor
    new_h = bbox['h'] * expansion_factor
    
    new_x = int(cx - new_w / 2)
    new_y = int(cy - new_h / 2)
    
    expanded = {
        'x': new_x,
        'y': new_y,
        'w': int(new_w),
        'h': int(new_h)
    }
    
    if img_width is not None and img_height is not None:
        expanded = clamp_bbox(expanded, img_width, img_height)
    
    return expanded


def draw_fps(img: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30),
            color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw FPS counter on image"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    return img


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


if __name__ == "__main__":
    # Test utilities
    setup_logging("DEBUG")
    
    # Test IoU
    box1 = {'x': 10, 'y': 10, 'w': 50, 'h': 50}
    box2 = {'x': 30, 'y': 30, 'w': 50, 'h': 50}
    box3 = {'x': 100, 'y': 100, 'w': 50, 'h': 50}
    
    print(f"IoU(box1, box2): {calculate_iou(box1, box2):.3f}")
    print(f"IoU(box1, box3): {calculate_iou(box1, box3):.3f}")
    
    # Test NMS
    detections = [
        {'bbox': box1, 'score': 0.9},
        {'bbox': box2, 'score': 0.8},
        {'bbox': box3, 'score': 0.95}
    ]
    
    filtered = non_maximum_suppression(detections, iou_threshold=0.3)
    print(f"\nNMS: {len(detections)} → {len(filtered)} detections")
    
    # Test Timer
    with Timer("Sleep test"):
        time.sleep(0.1)
    
    # Test FPS counter
    fps_counter = create_fps_counter()
    for _ in range(10):
        time.sleep(0.033)  # ~30 FPS
        fps = fps_counter()
    print(f"\nFPS: {fps:.1f}")
