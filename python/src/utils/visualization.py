"""
Visualization Utilities
=======================
Draw bounding boxes, labels, and confidence scores on images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_detections(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw bounding boxes with optional scores and labels on image.
    
    Args:
        image: Input image (BGR)
        boxes: List of bounding boxes (x, y, w, h)
        scores: Optional confidence scores
        labels: Optional text labels
        color: Box color (BGR)
        thickness: Line thickness
        font_scale: Font size scale
        
    Returns:
        Image with drawn detections
    """
    output = image.copy()
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
        
        # Prepare label text
        text_parts = []
        if labels is not None and i < len(labels):
            text_parts.append(labels[i])
        if scores is not None and i < len(scores):
            text_parts.append(f"{scores[i]:.2f}")
        
        if text_parts:
            label_text = " ".join(text_parts)
            
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                1
            )
            
            # Draw label background
            cv2.rectangle(
                output,
                (x, y - text_h - baseline - 5),
                (x + text_w, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label_text,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return output


def draw_face_detections(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float]
) -> np.ndarray:
    """
    Draw face detections with green boxes.
    
    Args:
        image: Input image
        boxes: Face bounding boxes
        scores: Detection scores
        
    Returns:
        Image with drawn face boxes
    """
    return draw_detections(
        image,
        boxes,
        scores=scores,
        labels=['Face'] * len(boxes),
        color=(0, 255, 0),
        thickness=3
    )


def create_grid_visualization(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of images to display
        titles: Optional titles for each image
        grid_size: Grid dimensions (rows, cols). Auto-computed if None.
        
    Returns:
        Grid image
    """
    if len(images) == 0:
        raise ValueError("No images provided")
    
    # Determine grid size
    if grid_size is None:
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Create grid
    grid = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        if row >= rows:
            break
        
        # Resize image to fit cell
        img_resized = cv2.resize(img, (max_w, max_h))
        
        # Add title if provided
        if titles is not None and idx < len(titles):
            cv2.putText(
                img_resized,
                titles[idx],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
        
        # Place in grid
        y_start = row * max_h
        x_start = col * max_w
        grid[y_start:y_start+max_h, x_start:x_start+max_w] = img_resized
    
    return grid


def add_fps_overlay(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Add FPS counter overlay to image.
    
    Args:
        image: Input image
        fps: Frames per second
        position: Text position (x, y)
        color: Text color (BGR)
        
    Returns:
        Image with FPS overlay
    """
    output = image.copy()
    text = f"FPS: {fps:.1f}"
    
    cv2.putText(
        output,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )
    
    return output
