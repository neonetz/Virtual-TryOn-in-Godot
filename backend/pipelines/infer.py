"""
Inference Pipeline
Haar Cascade proposal → ORB+BoVW feature extraction → SVM classification → NMS → Mask overlay
"""

import cv2
import numpy as np
import joblib
from typing import List, Dict, Optional, Tuple
import logging

from .features import FeaturePipeline
from .overlay import MaskOverlay
from .utils import non_maximum_suppression, clamp_bbox, is_bbox_valid, Timer

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Classical face detector using Haar Cascade + ORB + BoVW + SVM
    """
    
    def __init__(self, 
                 svm_model_path: str,
                 bovw_encoder_path: str,
                 scaler_path: str,
                 cascade_face_path: str,
                 max_keypoints: int = 500,
                 target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize face detector
        
        Args:
            svm_model_path: Path to svm_model.pkl
            bovw_encoder_path: Path to bovw_encoder.pkl
            scaler_path: Path to scaler.pkl
            cascade_face_path: Path to haarcascade_frontalface_default.xml
            max_keypoints: Max ORB keypoints per ROI
            target_size: ROI resize dimensions
        """
        # Load SVM classifier
        self.svm = joblib.load(svm_model_path)
        logger.info(f"✓ SVM model loaded: {svm_model_path}")
        
        # Load feature pipeline (ORB + BoVW + Scaler)
        self.feature_pipeline = FeaturePipeline(
            bovw_encoder_path,
            scaler_path,
            max_keypoints=max_keypoints,
            target_size=target_size
        )
        
        # Load Haar Cascade for face proposals
        self.face_cascade = cv2.CascadeClassifier(cascade_face_path)
        if self.face_cascade.empty():
            raise ValueError(f"Cannot load face cascade: {cascade_face_path}")
        logger.info(f"✓ Haar cascade loaded: {cascade_face_path}")
        
        # Detection parameters
        self.score_threshold = 0.0  # SVM decision threshold
        self.nms_threshold = 0.3
        self.min_face_size = 40
        
        logger.info("✓ Face detector initialized")
    
    def propose_rois(self, img: np.ndarray, 
                    scale_factor: float = 1.1,
                    min_neighbors: int = 5) -> List[Dict]:
        """
        Generate face ROI proposals using Haar Cascade
        
        Args:
            img: Input image (BGR)
            scale_factor: Haar cascade scale factor
            min_neighbors: Haar cascade min neighbors
            
        Returns:
            List of bbox dicts {x, y, w, h}
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Detect faces with Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of dicts
        proposals = []
        for (x, y, w, h) in faces:
            bbox = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            
            # Clamp to image boundaries
            bbox = clamp_bbox(bbox, img.shape[1], img.shape[0])
            
            # Validate
            if is_bbox_valid(bbox, min_size=self.min_face_size):
                proposals.append(bbox)
        
        logger.debug(f"Haar cascade proposed {len(proposals)} ROIs")
        return proposals
    
    def classify_rois(self, img: np.ndarray, rois: List[Dict]) -> List[Dict]:
        """
        Classify ROIs using ORB + BoVW + SVM
        
        Args:
            img: Input image (BGR)
            rois: List of bbox dicts
            
        Returns:
            List of detections {bbox, score} for faces (SVM positive)
        """
        if len(rois) == 0:
            return []
        
        detections = []
        
        for bbox in rois:
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Extract ROI
            roi_img = img[y:y+h, x:x+w]
            
            # Extract features
            try:
                features = self.feature_pipeline.extract_features(roi_img)
            except Exception as e:
                logger.debug(f"Feature extraction failed for ROI: {e}")
                continue
            
            # Classify with SVM
            prediction = self.svm.predict([features])[0]
            score = self.svm.decision_function([features])[0]
            
            # Keep only positive predictions (face)
            if prediction == 1 and score >= self.score_threshold:
                detections.append({
                    'bbox': bbox,
                    'score': float(score)
                })
        
        logger.debug(f"SVM classified: {len(detections)}/{len(rois)} as faces")
        return detections
    
    def detect(self, img: np.ndarray, 
              apply_nms: bool = True,
              scale_factor: float = 1.1,
              min_neighbors: int = 5) -> List[Dict]:
        """
        Full detection pipeline: Propose → Classify → NMS
        
        Args:
            img: Input image (BGR)
            apply_nms: Apply Non-Maximum Suppression
            scale_factor: Haar cascade scale factor
            min_neighbors: Haar cascade min neighbors
            
        Returns:
            List of final detections {bbox, score}
        """
        # Step 1: Generate ROI proposals with Haar Cascade
        with Timer("ROI proposal", verbose=False):
            proposals = self.propose_rois(img, scale_factor, min_neighbors)
        
        if len(proposals) == 0:
            logger.debug("No ROI proposals")
            return []
        
        # Step 2: Classify proposals with ORB+BoVW+SVM
        with Timer("SVM classification", verbose=False):
            detections = self.classify_rois(img, proposals)
        
        if len(detections) == 0:
            logger.debug("No faces detected by SVM")
            return []
        
        # Step 3: Apply NMS
        if apply_nms:
            with Timer("NMS", verbose=False):
                detections = non_maximum_suppression(detections, self.nms_threshold)
        
        logger.info(f"Detected {len(detections)} face(s)")
        return detections


class FaceDetectionSystem:
    """
    Complete system: Detection + Mask Overlay
    """
    
    def __init__(self,
                 svm_model_path: str,
                 bovw_encoder_path: str,
                 scaler_path: str,
                 cascade_face_path: str,
                 mask_path: str,
                 cascade_eye_path: Optional[str] = None,
                 max_keypoints: int = 500):
        """
        Initialize complete system
        
        Args:
            svm_model_path: Path to svm_model.pkl
            bovw_encoder_path: Path to bovw_encoder.pkl
            scaler_path: Path to scaler.pkl
            cascade_face_path: Path to haarcascade_frontalface_default.xml
            mask_path: Path to mask.png (RGBA)
            cascade_eye_path: Optional path to haarcascade_eye.xml
            max_keypoints: Max ORB keypoints per ROI
        """
        # Initialize detector
        self.detector = FaceDetector(
            svm_model_path,
            bovw_encoder_path,
            scaler_path,
            cascade_face_path,
            max_keypoints=max_keypoints
        )
        
        # Initialize mask overlay
        self.mask_overlay = MaskOverlay(mask_path, cascade_eye_path)
        
        logger.info("✓ Face detection system ready")
    
    def process_image(self, img: np.ndarray,
                     enable_rotation: bool = True,
                     alpha_multiplier: float = 0.95) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process image: detect faces and apply masks
        
        Args:
            img: Input image (BGR)
            enable_rotation: Enable mask rotation based on eyes
            alpha_multiplier: Mask opacity
            
        Returns:
            (result_image, detections)
        """
        # Detect faces
        detections = self.detector.detect(img)
        
        if len(detections) == 0:
            return img, []
        
        # Apply masks
        result = img.copy()
        for det in detections:
            result = self.mask_overlay.apply_mask(
                result,
                det['bbox'],
                alpha_multiplier=alpha_multiplier,
                enable_rotation=enable_rotation
            )
        
        return result, detections
    
    def process_video_frame(self, frame: np.ndarray,
                           enable_rotation: bool = True,
                           alpha_multiplier: float = 0.95,
                           draw_boxes: bool = False) -> np.ndarray:
        """
        Process video frame (optimized for real-time)
        
        Args:
            frame: Input frame (BGR)
            enable_rotation: Enable mask rotation
            alpha_multiplier: Mask opacity
            draw_boxes: Draw bounding boxes
            
        Returns:
            Processed frame
        """
        result, detections = self.process_image(frame, enable_rotation, alpha_multiplier)
        
        # Optionally draw boxes
        if draw_boxes and len(detections) > 0:
            for det in detections:
                bbox = det['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                score = det['score']
                
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    result,
                    f"{score:.2f}",
                    (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        
        return result


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 7:
        print("Usage: python infer.py <image.jpg> <svm_model.pkl> <bovw_encoder.pkl> "
              "<scaler.pkl> <haarcascade_face.xml> <mask.png>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    svm_path = sys.argv[2]
    bovw_path = sys.argv[3]
    scaler_path = sys.argv[4]
    cascade_path = sys.argv[5]
    mask_path = sys.argv[6]
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image {img_path}")
        sys.exit(1)
    
    print(f"Image size: {img.shape[1]}×{img.shape[0]}")
    
    # Initialize system
    print("\nInitializing system...")
    system = FaceDetectionSystem(
        svm_path,
        bovw_path,
        scaler_path,
        cascade_path,
        mask_path
    )
    
    # Process image
    print("\nProcessing image...")
    with Timer("Total processing"):
        result, detections = system.process_image(img)
    
    print(f"\n✓ Detected {len(detections)} face(s)")
    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        print(f"  Face {i}: bbox=({bbox['x']}, {bbox['y']}, {bbox['w']}, {bbox['h']}), "
              f"score={det['score']:.3f}")
    
    # Save result
    output_path = "output_with_mask.jpg"
    cv2.imwrite(output_path, result)
    print(f"\n✓ Result saved: {output_path}")
