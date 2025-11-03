"""
Feature Extraction Module
Handles ORB keypoint detection, descriptor extraction, and Bag of Visual Words encoding.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class ORBFeatureExtractor:
    """
    ORB-based feature extractor with Bag of Visual Words encoding.
    """
    
    def __init__(self, 
                 n_keypoints: int = 500,
                 roi_size: Tuple[int, int] = (128, 128),
                 codebook_size: int = 256):
        """
        Initialize ORB feature extractor.
        
        Args:
            n_keypoints: Maximum number of keypoints to detect per ROI
            roi_size: Standard ROI size (width, height) for preprocessing
            codebook_size: Number of visual words in BoVW codebook (k)
        """
        self.n_keypoints = n_keypoints
        self.roi_size = roi_size
        self.codebook_size = codebook_size
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=n_keypoints)
        
        # Codebook (k-means model) - will be loaded from file
        self.codebook = None
        
        # Feature scaler - will be loaded from file
        self.scaler = None
        
        print(f"[ORB] Initialized with {n_keypoints} keypoints, "
              f"ROI size {roi_size}, codebook size {codebook_size}")
    
    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI: convert to grayscale and resize to standard size.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Preprocessed grayscale ROI
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Resize to standard size
        roi_resized = cv2.resize(roi_gray, self.roi_size, 
                                interpolation=cv2.INTER_AREA)
        
        return roi_resized
    
    def extract_orb_descriptors(self, roi: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract ORB keypoints and descriptors from ROI.
        
        Args:
            roi: Input ROI (will be preprocessed)
            
        Returns:
            Tuple of (keypoints, descriptors)
            descriptors is None if no keypoints detected
        """
        # Preprocess ROI
        roi_processed = self.preprocess_roi(roi)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(roi_processed, None)
        
        return keypoints, descriptors
    
    def encode_bovw(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        """
        Encode ORB descriptors as Bag of Visual Words histogram.
        
        Args:
            descriptors: ORB descriptors (N x 32 array)
            
        Returns:
            Normalized BoVW histogram (codebook_size dimensional vector)
        """
        if self.codebook is None:
            raise ValueError("Codebook not loaded! Please load codebook.pkl first.")
        
        # Handle case with no descriptors
        if descriptors is None or len(descriptors) == 0:
            # Return uniform histogram as fallback
            uniform_hist = np.ones(self.codebook_size, dtype=np.float32)
            uniform_hist /= self.codebook_size
            return uniform_hist
        
        # Predict cluster assignments for each descriptor
        labels = self.codebook.predict(descriptors)
        
        # Build histogram
        histogram = np.zeros(self.codebook_size, dtype=np.float32)
        for label in labels:
            histogram[label] += 1
        
        # L1 normalization
        if histogram.sum() > 0:
            histogram /= histogram.sum()
        
        return histogram
    
    def extract_bovw_feature(self, roi: np.ndarray) -> np.ndarray:
        """
        Extract BoVW feature vector from ROI.
        Complete pipeline: ROI → ORB descriptors → BoVW encoding.
        
        Args:
            roi: Input ROI image
            
        Returns:
            BoVW feature vector (normalized histogram)
        """
        # Extract ORB descriptors
        keypoints, descriptors = self.extract_orb_descriptors(roi)
        
        # Encode as BoVW
        bovw_vector = self.encode_bovw(descriptors)
        
        return bovw_vector
    
    def extract_bovw_batch(self, rois: List[np.ndarray]) -> np.ndarray:
        """
        Extract BoVW features from a batch of ROIs.
        
        Args:
            rois: List of ROI images
            
        Returns:
            Array of BoVW feature vectors (n_rois x codebook_size)
        """
        features = []
        
        for roi in rois:
            bovw = self.extract_bovw_feature(roi)
            features.append(bovw)
        
        return np.array(features)
    
    def load_codebook(self, codebook_path: str):
        """
        Load pre-trained k-means codebook from file.
        
        Args:
            codebook_path: Path to codebook.pkl
        """
        codebook_file = Path(codebook_path)
        if not codebook_file.exists():
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        
        self.codebook = joblib.load(codebook_path)
        print(f"[ORB] Loaded codebook from: {codebook_path}")
        print(f"[ORB] Codebook size: {self.codebook.n_clusters}")
    
    def load_scaler(self, scaler_path: str):
        """
        Load pre-fitted feature scaler from file.
        
        Args:
            scaler_path: Path to scaler.pkl
        """
        scaler_file = Path(scaler_path)
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"[ORB] Loaded scaler from: {scaler_path}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using pre-fitted scaler.
        
        Args:
            features: BoVW feature vectors
            
        Returns:
            Normalized features
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded! Please load scaler.pkl first.")
        
        return self.scaler.transform(features)


class ROIProposer:
    """
    ROI (Region of Interest) proposer using Haar Cascade for face detection.
    """
    
    def __init__(self, 
                 cascade_path: Optional[str] = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize Haar Cascade face detector.
        
        Args:
            cascade_path: Path to haarcascade XML file (None = use OpenCV default)
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size (width, height)
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load Haar Cascade
        if cascade_path is None:
            # Use OpenCV's default frontal face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from: {cascade_path}")
        
        print(f"[ROI Proposer] Loaded cascade from: {cascade_path}")
    
    def propose_rois(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Propose face ROIs using Haar Cascade.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # Convert to list of tuples
        rois = [tuple(box) for box in faces]
        
        return rois


class EyeDetector:
    """
    Eye detector for mask rotation alignment (optional enhancement).
    """
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize eye detector.
        
        Args:
            cascade_path: Path to eye cascade XML file
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise ValueError(f"Failed to load eye cascade from: {cascade_path}")
        
        print(f"[Eye Detector] Loaded cascade from: {cascade_path}")
    
    def detect_eyes(self, 
                    face_roi: np.ndarray,
                    face_box: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Detect eyes in face ROI and compute rotation angle.
        
        Args:
            face_roi: Face region image
            face_box: Face bounding box (x, y, w, h) in original image
            
        Returns:
            Rotation angle in degrees (None if eyes not detected)
        """
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Detect eyes
        eyes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) < 2:
            return None
        
        # Get two largest eyes (by area)
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        eye1 = eyes[0]
        eye2 = eyes[1]
        
        # Compute eye centers
        eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
        eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)
        
        # Compute angle between eyes
        dx = eye2_center[0] - eye1_center[0]
        dy = eye2_center[1] - eye1_center[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
