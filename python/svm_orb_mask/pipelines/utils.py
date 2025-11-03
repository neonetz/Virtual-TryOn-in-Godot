"""
Utility Functions Module
Provides common utilities for I/O, timing, NMS, and visualization.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        
    def get_elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds."""
        return self.elapsed * 1000


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Dictionary with configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config.json
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] Config saved to: {config_path}")


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: (x, y, w, h) first box
        box2: (x, y, w, h) second box
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Compute intersection area
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # Compute IoU
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def non_max_suppression(boxes: List[Tuple[int, int, int, int]], 
                        scores: List[float],
                        iou_threshold: float = 0.3) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: List of bounding boxes (x, y, w, h)
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for suppression (default: 0.3)
        
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Pick the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        
        # Filter out boxes with high IoU
        ious = [compute_iou(current_box, boxes[idx]) for idx in remaining_indices]
        mask = np.array(ious) < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return keep_indices


def draw_detection_box(image: np.ndarray, 
                       box: Tuple[int, int, int, int],
                       label: str = "Face",
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        box: Bounding box (x, y, w, h)
        label: Label text
        color: Box color in BGR
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    x, y, w, h = box
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y, label_size[1] + 10)
    
    cv2.rectangle(image, 
                 (x, label_y - label_size[1] - 10),
                 (x + label_size[0], label_y),
                 color, -1)
    
    # Draw label text
    cv2.putText(image, label, 
               (x, label_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (0, 0, 0), 1)
    
    return image


def visualize_keypoints(image: np.ndarray, 
                        keypoints: List[cv2.KeyPoint],
                        title: str = "ORB Keypoints") -> np.ndarray:
    """
    Visualize ORB keypoints on image.
    
    Args:
        image: Input image
        keypoints: List of cv2.KeyPoint objects
        title: Window title
        
    Returns:
        Image with drawn keypoints
    """
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, 
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Add text with keypoint count
    text = f"Keypoints: {len(keypoints)}"
    cv2.putText(img_with_kp, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_with_kp


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_cascade_path(cascade_name: str = "haarcascade_frontalface_default.xml") -> str:
    """
    Get path to OpenCV Haar Cascade XML file.
    First checks local assets/cascades/ folder, then falls back to OpenCV data.
    
    Args:
        cascade_name: Name of cascade file
        
    Returns:
        Path to cascade XML file
    """
    # Check local assets first
    local_path = Path(__file__).parent.parent / "assets" / "cascades" / cascade_name
    if local_path.exists():
        return str(local_path)
    
    # Fallback to OpenCV data
    opencv_data_path = Path(cv2.data.haarcascades)
    cascade_path = opencv_data_path / cascade_name
    
    if not cascade_path.exists():
        raise FileNotFoundError(
            f"Cascade file not found: {cascade_name}\n"
            f"Please place it in: {local_path}"
        )
    
    return str(cascade_path)


class FPSCounter:
    """
    FPS counter for real-time video processing.
    """
    
    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: Smoothing factor for exponential moving average
        """
        self.smoothing = smoothing
        self.fps = 0.0
        self.last_time = time.time()
        
    def update(self) -> float:
        """
        Update FPS counter and return current FPS.
        
        Returns:
            Current FPS value
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        
        self.last_time = current_time
        return self.fps
    
    def get_fps(self) -> float:
        """Returns current FPS value."""
        return self.fps


def resize_with_aspect_ratio(image: np.ndarray, 
                             target_size: Tuple[int, int],
                             pad: bool = False) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: (width, height) target size
        pad: If True, pad to exact target size; otherwise just scale
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Compute scale factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if not pad:
        return resized
    
    # Pad to exact target size
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
    
    # Center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded
