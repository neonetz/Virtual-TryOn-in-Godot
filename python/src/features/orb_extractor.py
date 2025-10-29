"""
ORB Feature Extractor
=====================
Extracts ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors from image ROIs.
Provides rotation invariance and computational efficiency for real-time performance.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ORBExtractor:
    """
    Handles ORB feature extraction from image regions.
    
    High Cohesion: Focuses solely on ORB descriptor extraction.
    Low Coupling: Independent of classification or encoding logic.
    """
    
    def __init__(
        self, 
        n_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        wta_k: int = 2,
        patch_size: int = 31,
        fast_threshold: int = 20
    ):
        """
        Initialize ORB detector with configurable parameters.
        
        Args:
            n_features: Maximum number of features to retain per image
            scale_factor: Pyramid decimation ratio for scale invariance
            n_levels: Number of pyramid levels
            edge_threshold: Size of border where features are not detected
            first_level: Level of pyramid to start at
            wta_k: Number of points for descriptor computation (2, 3, or 4)
            patch_size: Size of patch used for descriptor
            fast_threshold: FAST detector threshold
        """
        self.n_features = n_features
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            patchSize=patch_size,
            fastThreshold=fast_threshold
        )
        
    def extract(
        self, 
        image: np.ndarray, 
        roi_size: Tuple[int, int] = (128, 128)
    ) -> Optional[np.ndarray]:
        """
        Extract ORB descriptors from an image ROI.
        
        Pipeline:
        1. Convert to grayscale if needed
        2. Resize to fixed dimensions for stability
        3. Detect ORB keypoints
        4. Compute 256-bit binary descriptors
        
        Args:
            image: Input image (color or grayscale)
            roi_size: Target size (width, height) for normalization
            
        Returns:
            descriptors: Array of shape (n_keypoints, 32) where each descriptor
                        is 32 bytes (256 bits), or None if no keypoints found
        """
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Resize to fixed dimensions for consistent feature scale
        if gray.shape[:2] != roi_size[::-1]:  # OpenCV uses (height, width)
            gray = cv2.resize(gray, roi_size, interpolation=cv2.INTER_AREA)
        
        # Step 3 & 4: Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Return None if no features detected (e.g., blank regions)
        if descriptors is None or len(descriptors) == 0:
            return None
            
        return descriptors
    
    def extract_batch(
        self, 
        images: List[np.ndarray], 
        roi_size: Tuple[int, int] = (128, 128)
    ) -> List[Optional[np.ndarray]]:
        """
        Extract ORB descriptors from multiple images.
        
        Args:
            images: List of input images
            roi_size: Target size for normalization
            
        Returns:
            List of descriptor arrays (or None for failed extractions)
        """
        return [self.extract(img, roi_size) for img in images]
    
    def visualize_keypoints(
        self, 
        image: np.ndarray, 
        roi_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Visualize detected ORB keypoints on the image.
        
        Args:
            image: Input image
            roi_size: Target size for normalization
            
        Returns:
            Image with keypoints drawn as colored circles
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if gray.shape[:2] != roi_size[::-1]:
            gray = cv2.resize(gray, roi_size, interpolation=cv2.INTER_AREA)
        
        keypoints = self.orb.detect(gray, None)
        
        # Draw keypoints with rich information
        output = cv2.drawKeypoints(
            gray, 
            keypoints, 
            None, 
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return output
