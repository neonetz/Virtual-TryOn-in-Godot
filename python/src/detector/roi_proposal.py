"""
ROI Proposal Generator
======================
Generates candidate face regions using multi-scale sliding window.
Pure classical CV approach - NO pre-trained models.

Methods:
- Multi-scale sliding window (primary): Exhaustive search at multiple scales (square windows for faces)
- Edge-based filtering: Filter out low-texture regions
- Image pyramid for scale invariance
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class ROIProposal:
    """
    Generates Region of Interest proposals for face detection.
    Uses pure sliding window approach with square windows - NO pre-trained models.
    
    High Cohesion: Focuses solely on candidate region generation.
    Low Coupling: Independent from feature extraction and classification.
    """
    
    def __init__(
        self,
        window_sizes: List[int] = None,
        step_size: int = 24,
        min_size: int = 40,
        max_size: int = 300,
        use_edge_detection: bool = True
    ):
        """
        Initialize ROI proposal generator with sliding window for face detection.
        
        Args:
            window_sizes: List of square window sizes for multi-scale search.
                         If None, uses [48, 64, 96, 128, 160, 192, 224, 256]
            step_size: Stride between windows (smaller = more proposals)
            min_size: Minimum ROI size (both width and height)
            max_size: Maximum ROI size (both width and height)
            use_edge_detection: Use edge density for filtering proposals
        """
        if window_sizes is None:
            # Default multi-scale square windows for faces
            self.window_sizes = [
                48,   # Very small face (far away)
                64,   # Small face
                96,   # Medium-small face
                128,  # Normal face
                160,  # Medium-large face
                192,  # Large face
                224,  # Very large face
                256   # Extra large face (very close)
            ]
        else:
            self.window_sizes = window_sizes
        
        self.step_size = step_size
        self.min_size = min_size
        self.max_size = max_size
        self.use_edge_detection = use_edge_detection
        
        print(f"ROI Proposal initialized for FACE detection:")
        print(f"  Square window sizes: {self.window_sizes}")
        print(f"  Step size: {self.step_size}")
        print(f"  Edge filtering: {self.use_edge_detection}")
    
    def propose(
        self, 
        image: np.ndarray,
        return_scores: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate face ROI proposals using multi-scale sliding window.
        
        Process:
        1. Convert to grayscale
        2. Calculate edge density map (optional filtering)
        3. Generate proposals at multiple scales (square windows)
        4. Filter by edge density and size
        
        Args:
            image: Input image (BGR or grayscale)
            return_scores: If True, return edge density scores
            
        Returns:
            List of bounding boxes as (x, y, w, h) tuples (square boxes)
        """
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Edge detection for filtering (optional)
        edge_map = None
        if self.use_edge_detection:
            edge_map = cv2.Canny(gray, 50, 150)
        
        # Step 3: Generate proposals at multiple scales
        h, w = gray.shape[:2]
        proposals = []
        
        for win_size in self.window_sizes:
            # Skip if window is larger than image
            if win_size > w or win_size > h:
                continue
            
            # Sliding window at this scale (SQUARE windows for faces)
            for y in range(0, h - win_size + 1, self.step_size):
                for x in range(0, w - win_size + 1, self.step_size):
                    # Check size constraints
                    if (win_size < self.min_size or win_size > self.max_size):
                        continue
                    
                    # Filter by edge density if enabled
                    if self.use_edge_detection and edge_map is not None:
                        roi_edges = edge_map[y:y+win_size, x:x+win_size]
                        edge_density = np.sum(roi_edges > 0) / (win_size * win_size)
                        
                        # Skip regions with very low edge density (likely background)
                        if edge_density < 0.05:  # Less than 5% edges
                            continue
                    
                    # Add square proposal (w = h for faces)
                    proposals.append((x, y, win_size, win_size))
        
        return proposals
    
    def propose_with_fallback(
        self, 
        image: np.ndarray,
        use_sliding_window: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate proposals with optional sliding window fallback.
        
        Args:
            image: Input image
            use_sliding_window: Use sliding window if Haar fails
            
        Returns:
            List of ROI bounding boxes
        """
        rois = self.propose(image)
        
        # Fallback to sliding window if no detections
        if len(rois) == 0 and use_sliding_window:
            rois = self._sliding_window(image)
        
        return rois
    
    def _sliding_window(
        self, 
        image: np.ndarray,
        window_size: Tuple[int, int] = (128, 128),
        step_size: int = 64
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fallback sliding window proposal generator.
        
        Args:
            image: Input image
            window_size: Window dimensions
            step_size: Stride between windows
            
        Returns:
            List of ROI bounding boxes
        """
        h, w = image.shape[:2]
        win_w, win_h = window_size
        rois = []
        
        for y in range(0, h - win_h, step_size):
            for x in range(0, w - win_w, step_size):
                rois.append((x, y, win_w, win_h))
        
        return rois
    
    def extract_roi(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract ROI from image given bounding box.
        
        Args:
            image: Source image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Cropped ROI image
        """
        x, y, w, h = bbox
        
        # Clamp to image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        return image[y:y+h, x:x+w].copy()
    
    def visualize_proposals(
        self, 
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize ROI proposals on image.
        
        Args:
            image: Input image
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        rois = self.propose(image)
        output = image.copy()
        
        for x, y, w, h in rois:
            cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
            # Add label
            cv2.putText(
                output, 
                f"ROI", 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return output


# Add Optional import at top
from typing import Optional
