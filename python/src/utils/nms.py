"""
Non-Maximum Suppression (NMS)
==============================
Removes redundant overlapping bounding boxes based on IoU threshold.
Keeps only the highest-confidence detection in each cluster.
"""

import numpy as np
from typing import List, Tuple


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box (x, y, w, h)
        box2: Second box (x, y, w, h)
        
    Returns:
        IoU value in [0, 1]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Compute intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def non_max_suppression(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Algorithm:
    1. Sort boxes by scores (descending)
    2. Iterate through sorted boxes:
       a. Keep current box
       b. Remove all boxes with IoU > threshold with current box
    3. Return kept boxes and their scores
    
    Args:
        boxes: List of bounding boxes (x, y, w, h)
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression (default 0.3)
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores)
    """
    if len(boxes) == 0:
        return [], []
    
    # Convert to numpy arrays for easier manipulation
    boxes_array = np.array(boxes)
    scores_array = np.array(scores)
    
    # Step 1: Sort by scores (descending)
    sorted_indices = np.argsort(scores_array)[::-1]
    
    keep = []
    
    # Step 2: Iterate and suppress
    while len(sorted_indices) > 0:
        # Keep the highest scoring box
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        # Remove current box from list
        sorted_indices = sorted_indices[1:]
        
        if len(sorted_indices) == 0:
            break
        
        # Compute IoU of current box with remaining boxes
        current_box = boxes[current_idx]
        
        # Filter out boxes with high IoU
        remaining_indices = []
        for idx in sorted_indices:
            iou = compute_iou(current_box, boxes[idx])
            if iou < iou_threshold:
                remaining_indices.append(idx)
        
        sorted_indices = np.array(remaining_indices)
    
    # Extract kept boxes and scores
    filtered_boxes = [boxes[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    
    return filtered_boxes, filtered_scores


def nms_with_indices(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3
) -> List[int]:
    """
    Apply NMS and return indices of kept boxes.
    
    Args:
        boxes: List of bounding boxes
        scores: Confidence scores
        iou_threshold: IoU threshold
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    scores_array = np.array(scores)
    sorted_indices = np.argsort(scores_array)[::-1]
    
    keep = []
    
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        sorted_indices = sorted_indices[1:]
        
        if len(sorted_indices) == 0:
            break
        
        current_box = boxes[current_idx]
        
        remaining_indices = []
        for idx in sorted_indices:
            iou = compute_iou(current_box, boxes[idx])
            if iou < iou_threshold:
                remaining_indices.append(idx)
        
        sorted_indices = np.array(remaining_indices)
    
    return keep
