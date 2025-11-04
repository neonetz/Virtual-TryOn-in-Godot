"""
ORB Feature Extraction and Bag of Visual Words (BoVW) Encoding
Handles keypoint detection, descriptor computation, and codebook-based representation
"""

import cv2
import numpy as np
import joblib
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ORBFeatureExtractor:
    """Extract ORB features from image regions"""
    
    def __init__(self, max_keypoints: int = 500, patch_size: int = 31, 
                 scale_factor: float = 1.2, n_levels: int = 8):
        """
        Initialize ORB detector
        
        Args:
            max_keypoints: Maximum number of keypoints to detect
            patch_size: Size of the patch used by the oriented BRIEF descriptor
            scale_factor: Pyramid decimation ratio for multi-scale detection
            n_levels: Number of pyramid levels
        """
        self.max_keypoints = max_keypoints
        self.orb = cv2.ORB_create(
            nfeatures=max_keypoints,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=patch_size,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=patch_size,
            fastThreshold=20
        )
        logger.info(f"ORB detector initialized: max_kp={max_keypoints}, levels={n_levels}")
    
    def extract_descriptors(self, roi: np.ndarray, 
                          target_size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
        """
        Extract ORB descriptors from an ROI
        
        Args:
            roi: Input image region (BGR or grayscale)
            target_size: Resize ROI to this size for consistency
            
        Returns:
            ORB descriptors array (N × 32) or None if no keypoints found
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize to target size for descriptor stability
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Apply histogram equalization for better feature detection
        gray = cv2.equalizeHist(gray)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            logger.debug(f"No ORB keypoints detected in ROI")
            return None
        
        logger.debug(f"Extracted {len(descriptors)} ORB descriptors")
        return descriptors
    
    def visualize_keypoints(self, roi: np.ndarray) -> np.ndarray:
        """Draw ORB keypoints on image for debugging"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        gray = cv2.resize(gray, (128, 128))
        gray = cv2.equalizeHist(gray)
        
        keypoints = self.orb.detect(gray, None)
        vis_img = cv2.drawKeypoints(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
            keypoints, 
            None, 
            color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return vis_img


class BoVWEncoder:
    """
    Bag of Visual Words encoder using k-means codebook
    Converts variable-length ORB descriptors to fixed-length histogram
    """
    
    def __init__(self, codebook_path: str):
        """
        Load pre-trained k-means codebook
        
        Args:
            codebook_path: Path to bovw_encoder.pkl (k-means model)
        """
        self.kmeans = joblib.load(codebook_path)
        self.k = self.kmeans.n_clusters
        logger.info(f"BoVW encoder loaded: k={self.k} visual words")
    
    def encode(self, descriptors: Optional[np.ndarray], 
               normalize: str = 'l2') -> np.ndarray:
        """
        Encode ORB descriptors to BoVW histogram
        
        Args:
            descriptors: ORB descriptors array (N × 32) or None
            normalize: Normalization method ('l1', 'l2', or None)
            
        Returns:
            Histogram of visual words (k-dimensional vector)
        """
        # Handle case of zero keypoints
        if descriptors is None or len(descriptors) == 0:
            logger.debug("Zero descriptors, returning uniform histogram")
            histogram = np.ones(self.k, dtype=np.float32) / self.k
            return histogram
        
        # Predict visual word for each descriptor
        visual_words = self.kmeans.predict(descriptors)
        
        # Build histogram
        histogram = np.bincount(visual_words, minlength=self.k).astype(np.float32)
        
        # Normalize
        if normalize == 'l1':
            norm = np.sum(histogram)
            if norm > 0:
                histogram = histogram / norm
        elif normalize == 'l2':
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram = histogram / norm
        
        return histogram
    
    def encode_batch(self, descriptors_list: List[Optional[np.ndarray]], 
                    normalize: str = 'l2') -> np.ndarray:
        """
        Encode multiple ROIs to BoVW histograms
        
        Args:
            descriptors_list: List of descriptor arrays
            normalize: Normalization method
            
        Returns:
            Array of histograms (n_samples × k)
        """
        histograms = []
        for desc in descriptors_list:
            hist = self.encode(desc, normalize=normalize)
            histograms.append(hist)
        return np.array(histograms)


class FeaturePipeline:
    """Complete feature extraction pipeline: ROI → ORB → BoVW → Scaled features"""
    
    def __init__(self, bovw_encoder_path: str, scaler_path: str,
                 max_keypoints: int = 500, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize feature pipeline
        
        Args:
            bovw_encoder_path: Path to bovw_encoder.pkl
            scaler_path: Path to scaler.pkl
            max_keypoints: Max ORB keypoints per ROI
            target_size: ROI resize dimensions
        """
        self.orb_extractor = ORBFeatureExtractor(max_keypoints=max_keypoints)
        self.bovw_encoder = BoVWEncoder(bovw_encoder_path)
        self.scaler = joblib.load(scaler_path)
        self.target_size = target_size
        
        logger.info("Feature pipeline initialized")
        logger.info(f"  ORB: max_kp={max_keypoints}")
        logger.info(f"  BoVW: k={self.bovw_encoder.k}")
        logger.info(f"  Scaler: {type(self.scaler).__name__}")
    
    def extract_features(self, roi: np.ndarray) -> np.ndarray:
        """
        Extract and encode features from a single ROI
        
        Args:
            roi: Image region (BGR or grayscale)
            
        Returns:
            Scaled BoVW feature vector (k-dimensional)
        """
        # Extract ORB descriptors
        descriptors = self.orb_extractor.extract_descriptors(roi, self.target_size)
        
        # Encode to BoVW histogram
        histogram = self.bovw_encoder.encode(descriptors, normalize='l2')
        
        # Scale features
        features = self.scaler.transform([histogram])[0]
        
        return features
    
    def extract_features_batch(self, rois: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple ROIs
        
        Args:
            rois: List of image regions
            
        Returns:
            Array of scaled feature vectors (n_samples × k)
        """
        # Extract all descriptors
        descriptors_list = []
        for roi in rois:
            desc = self.orb_extractor.extract_descriptors(roi, self.target_size)
            descriptors_list.append(desc)
        
        # Encode to histograms
        histograms = self.bovw_encoder.encode_batch(descriptors_list, normalize='l2')
        
        # Scale features
        features = self.scaler.transform(histograms)
        
        return features


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) < 4:
        print("Usage: python features.py <test_img.jpg> <bovw_encoder.pkl> <scaler.pkl>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    bovw_path = sys.argv[2]
    scaler_path = sys.argv[3]
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image {img_path}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = FeaturePipeline(bovw_path, scaler_path)
    
    # Extract features
    features = pipeline.extract_features(img)
    
    print(f"\n✓ Features extracted: shape={features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
