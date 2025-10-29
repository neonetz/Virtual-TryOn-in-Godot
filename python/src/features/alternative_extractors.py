"""
Alternative Feature Extractors
===============================
Provides BRISK and AKAZE feature extractors as alternatives to ORB.
Compatible with the same BoVW pipeline for easy benchmarking.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class BRISKExtractor:
    """
    BRISK (Binary Robust Invariant Scalable Keypoints) feature extractor.
    
    Advantages over ORB:
    - More keypoints at different scales
    - Better rotation invariance
    - Good for texture-rich images
    
    High Cohesion: Focuses solely on BRISK descriptor extraction.
    Low Coupling: Drop-in replacement for ORBExtractor.
    """
    
    def __init__(
        self,
        threshold: int = 30,
        octaves: int = 3,
        pattern_scale: float = 1.0
    ):
        """
        Initialize BRISK detector.
        
        Args:
            threshold: AGAST detection threshold
            octaves: Number of octaves for scale space
            pattern_scale: Pattern scaling factor
        """
        self.brisk = cv2.BRISK_create(
            thresh=threshold,
            octaves=octaves,
            patternScale=pattern_scale
        )
        
    def extract(
        self,
        image: np.ndarray,
        roi_size: Tuple[int, int] = (128, 128)
    ) -> Optional[np.ndarray]:
        """
        Extract BRISK descriptors from an image ROI.
        
        Args:
            image: Input image (color or grayscale)
            roi_size: Target size (width, height) for normalization
            
        Returns:
            descriptors: Array of shape (n_keypoints, 64) or None
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to fixed dimensions
        if gray.shape[:2] != roi_size[::-1]:
            gray = cv2.resize(gray, roi_size, interpolation=cv2.INTER_AREA)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.brisk.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None
            
        return descriptors
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        roi_size: Tuple[int, int] = (128, 128)
    ) -> List[Optional[np.ndarray]]:
        """Extract BRISK descriptors from multiple images."""
        return [self.extract(img, roi_size) for img in images]


class AKAZEExtractor:
    """
    AKAZE (Accelerated-KAZE) feature extractor.
    
    Advantages over ORB:
    - Nonlinear scale space (better edge preservation)
    - More distinctive features
    - Better for textured surfaces
    
    High Cohesion: Focuses solely on AKAZE descriptor extraction.
    Low Coupling: Drop-in replacement for ORBExtractor.
    """
    
    def __init__(
        self,
        descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size: int = 0,
        descriptor_channels: int = 3,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4
    ):
        """
        Initialize AKAZE detector.
        
        Args:
            descriptor_type: Type of descriptor
            descriptor_size: Size of the descriptor (0 = full)
            descriptor_channels: Number of channels
            threshold: Detector response threshold
            n_octaves: Maximum octave evolution
            n_octave_layers: Sublayers per octave
        """
        self.akaze = cv2.AKAZE_create(
            descriptor_type=descriptor_type,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers
        )
        
    def extract(
        self,
        image: np.ndarray,
        roi_size: Tuple[int, int] = (128, 128)
    ) -> Optional[np.ndarray]:
        """
        Extract AKAZE descriptors from an image ROI.
        
        Args:
            image: Input image (color or grayscale)
            roi_size: Target size (width, height) for normalization
            
        Returns:
            descriptors: Array of shape (n_keypoints, 61) or None
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to fixed dimensions
        if gray.shape[:2] != roi_size[::-1]:
            gray = cv2.resize(gray, roi_size, interpolation=cv2.INTER_AREA)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None
            
        return descriptors
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        roi_size: Tuple[int, int] = (128, 128)
    ) -> List[Optional[np.ndarray]]:
        """Extract AKAZE descriptors from multiple images."""
        return [self.extract(img, roi_size) for img in images]


class SIFTExtractor:
    """
    SIFT (Scale-Invariant Feature Transform) feature extractor.
    
    Note: SIFT is patented and may require licensing for commercial use.
    
    Advantages:
    - Most distinctive features
    - Best scale/rotation invariance
    - Gold standard for feature matching
    
    High Cohesion: Focuses solely on SIFT descriptor extraction.
    Low Coupling: Drop-in replacement for ORBExtractor.
    """
    
    def __init__(
        self,
        n_features: int = 500,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6
    ):
        """
        Initialize SIFT detector.
        
        Args:
            n_features: Maximum number of features
            n_octave_layers: Layers per octave
            contrast_threshold: Contrast threshold for filtering
            edge_threshold: Edge threshold for filtering
            sigma: Gaussian sigma
        """
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        
    def extract(
        self,
        image: np.ndarray,
        roi_size: Tuple[int, int] = (128, 128)
    ) -> Optional[np.ndarray]:
        """
        Extract SIFT descriptors from an image ROI.
        
        Args:
            image: Input image (color or grayscale)
            roi_size: Target size (width, height) for normalization
            
        Returns:
            descriptors: Array of shape (n_keypoints, 128) or None
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to fixed dimensions
        if gray.shape[:2] != roi_size[::-1]:
            gray = cv2.resize(gray, roi_size, interpolation=cv2.INTER_AREA)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None
            
        # Convert to uint8 for consistency with binary descriptors
        # Note: SIFT produces float descriptors, we keep them as float
        return descriptors
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        roi_size: Tuple[int, int] = (128, 128)
    ) -> List[Optional[np.ndarray]]:
        """Extract SIFT descriptors from multiple images."""
        return [self.extract(img, roi_size) for img in images]
