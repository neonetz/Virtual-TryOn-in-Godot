"""
Inference Pipeline Module
Complete face detection and mask overlay pipeline using SVM+ORB.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Optional

from pipelines.features import ORBFeatureExtractor, ROIProposer, EyeDetector
from pipelines.mask_overlay import MaskOverlay
from pipelines.utils import non_max_suppression, draw_detection_box


class FaceDetectorSVM:
    """
    SVM-based face detector using ORB features with BoVW encoding.
    """
    
    def __init__(self, 
                 model_path: str,
                 feature_extractor: ORBFeatureExtractor,
                 confidence_threshold: float = 0.5):
        """
        Initialize SVM face detector.
        
        Args:
            model_path: Path to trained SVM model (svm_model.pkl)
            feature_extractor: ORB feature extractor instance
            confidence_threshold: Minimum confidence score for positive detection
        """
        self.feature_extractor = feature_extractor
        self.confidence_threshold = confidence_threshold
        
        # Load SVM model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"SVM model not found: {model_path}")
        
        self.svm_model = joblib.load(model_path)
        print(f"[SVM Detector] Loaded model from: {model_path}")
        print(f"[SVM Detector] Confidence threshold: {confidence_threshold}")
    
    def predict_roi(self, roi: np.ndarray) -> Tuple[int, float]:
        """
        Predict if ROI contains a face.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 = face, 0 = non-face
            confidence: Confidence score [0, 1]
        """
        # Extract BoVW feature
        bovw_feature = self.feature_extractor.extract_bovw_feature(roi)
        
        # Normalize feature
        bovw_normalized = self.feature_extractor.normalize_features(
            bovw_feature.reshape(1, -1)
        )
        
        # SVM prediction
        prediction = self.svm_model.predict(bovw_normalized)[0]
        
        # Get decision function score (distance to hyperplane)
        decision_score = self.svm_model.decision_function(bovw_normalized)[0]
        
        # Convert to confidence [0, 1] using sigmoid
        confidence = 1 / (1 + np.exp(-decision_score))
        
        return int(prediction), float(confidence)
    
    def predict_batch(self, rois: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Predict multiple ROIs in batch.
        
        Args:
            rois: List of ROI images
            
        Returns:
            List of (prediction, confidence) tuples
        """
        if not rois:
            return []
        
        # Extract BoVW features
        bovw_features = self.feature_extractor.extract_bovw_batch(rois)
        
        # Normalize features
        bovw_normalized = self.feature_extractor.normalize_features(bovw_features)
        
        # SVM predictions
        predictions = self.svm_model.predict(bovw_normalized)
        decision_scores = self.svm_model.decision_function(bovw_normalized)
        
        # Convert to confidences
        confidences = 1 / (1 + np.exp(-decision_scores))
        
        # Combine results
        results = list(zip(predictions.astype(int), confidences))
        
        return results


class FaceMaskPipeline:
    """
    Complete pipeline: Face detection → SVM classification → Mask overlay.
    """
    
    def __init__(self,
                 model_dir: str,
                 mask_path: str,
                 cascade_path: Optional[str] = None,
                 eye_cascade_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.3,
                 use_rotation: bool = True):
        """
        Initialize complete face-mask pipeline.
        
        Args:
            model_dir: Directory containing trained models (svm_model.pkl, codebook.pkl, scaler.pkl)
            mask_path: Path to mask PNG file
            cascade_path: Path to Haar Cascade XML (None = use default)
            eye_cascade_path: Path to eye cascade XML (None = use default)
            confidence_threshold: Minimum SVM confidence for face detection
            nms_threshold: IoU threshold for Non-Maximum Suppression
            use_rotation: Enable eye-based rotation alignment
        """
        model_dir = Path(model_dir)
        
        # Initialize ROI proposer (Haar Cascade)
        self.roi_proposer = ROIProposer(cascade_path=cascade_path)
        
        # Initialize feature extractor
        self.feature_extractor = ORBFeatureExtractor()
        
        # Load codebook and scaler
        codebook_path = model_dir / "codebook.pkl"
        scaler_path = model_dir / "scaler.pkl"
        
        self.feature_extractor.load_codebook(str(codebook_path))
        self.feature_extractor.load_scaler(str(scaler_path))
        
        # Initialize SVM detector
        svm_path = model_dir / "svm_model.pkl"
        self.svm_detector = FaceDetectorSVM(
            str(svm_path),
            self.feature_extractor,
            confidence_threshold
        )
        
        # Initialize mask overlay
        self.mask_overlay = MaskOverlay(mask_path, use_rotation=use_rotation)
        
        # Initialize eye detector (optional for rotation)
        self.eye_detector = None
        if use_rotation:
            try:
                self.eye_detector = EyeDetector(cascade_path=eye_cascade_path)
            except Exception as e:
                print(f"[Pipeline] Warning: Eye detector failed to load: {e}")
                print("[Pipeline] Rotation alignment disabled")
        
        self.nms_threshold = nms_threshold
        
        print("[Pipeline] Initialization complete")
        print(f"[Pipeline] NMS threshold: {nms_threshold}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in image using two-stage pipeline:
        1. Haar Cascade ROI proposal
        2. SVM classification with BoVW features
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of (bbox, confidence) tuples
            bbox: (x, y, w, h)
            confidence: SVM confidence score
        """
        # Stage 1: Propose ROIs using Haar Cascade
        candidate_rois = self.roi_proposer.propose_rois(image)
        
        if not candidate_rois:
            return []
        
        # Extract ROI patches
        roi_patches = []
        for (x, y, w, h) in candidate_rois:
            roi = image[y:y+h, x:x+w]
            roi_patches.append(roi)
        
        # Stage 2: SVM classification
        predictions = self.svm_detector.predict_batch(roi_patches)
        
        # Filter by confidence and positive prediction
        detections = []
        for (x, y, w, h), (pred, conf) in zip(candidate_rois, predictions):
            if pred == 1 and conf >= self.svm_detector.confidence_threshold:
                detections.append(((x, y, w, h), conf))
        
        # Apply Non-Maximum Suppression
        if len(detections) > 1:
            boxes = [det[0] for det in detections]
            scores = [det[1] for det in detections]
            
            nms_indices = non_max_suppression(boxes, scores, self.nms_threshold)
            detections = [detections[i] for i in nms_indices]
        
        return detections
    
    def detect_rotation_angles(self, 
                              image: np.ndarray,
                              face_boxes: List[Tuple[int, int, int, int]]) -> List[Optional[float]]:
        """
        Detect rotation angles for faces using eye detection.
        
        Args:
            image: Input image (BGR)
            face_boxes: List of face bounding boxes
            
        Returns:
            List of rotation angles (None if eyes not detected)
        """
        if self.eye_detector is None:
            return [None] * len(face_boxes)
        
        angles = []
        for (x, y, w, h) in face_boxes:
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Detect eyes and compute angle
            angle = self.eye_detector.detect_eyes(face_roi, (x, y, w, h))
            angles.append(angle)
        
        return angles
    
    def process_image(self, 
                     image: np.ndarray,
                     draw_boxes: bool = True,
                     mask_scale: float = 1.2,
                     mask_offset_y: float = 0.3) -> Tuple[np.ndarray, List[dict]]:
        """
        Complete processing: detect faces and apply masks.
        
        Args:
            image: Input image (BGR)
            draw_boxes: Draw bounding boxes on output
            mask_scale: Mask scale relative to face width
            mask_offset_y: Vertical offset for mask placement
            
        Returns:
            Tuple of (processed_image, detections_info)
            detections_info: List of detection metadata dicts
        """
        # Detect faces
        detections = self.detect_faces(image)
        
        if not detections:
            return image.copy(), []
        
        # Extract face boxes and confidences
        face_boxes = [det[0] for det in detections]
        confidences = [det[1] for det in detections]
        
        # Detect rotation angles (if enabled)
        rotation_angles = self.detect_rotation_angles(image, face_boxes)
        
        # Apply masks
        result = self.mask_overlay.apply_masks_batch(
            image,
            face_boxes,
            rotation_angles,
            scale_factor=mask_scale,
            offset_y=mask_offset_y
        )
        
        # Draw bounding boxes (optional)
        if draw_boxes:
            for (x, y, w, h), conf in detections:
                result = draw_detection_box(
                    result,
                    (x, y, x+w, y+h),
                    f"Face: {conf:.2f}",
                    color=(0, 255, 0)
                )
        
        # Build detection info
        detections_info = []
        for i, ((x, y, w, h), conf) in enumerate(detections):
            info = {
                'bbox': (x, y, w, h),
                'confidence': conf,
                'rotation_angle': rotation_angles[i]
            }
            detections_info.append(info)
        
        return result, detections_info
    
    def process_video(self,
                     video_path: str,
                     output_path: Optional[str] = None,
                     display: bool = True,
                     draw_boxes: bool = True,
                     mask_scale: float = 1.2,
                     mask_offset_y: float = 0.3) -> bool:
        """
        Process video file: detect faces and apply masks frame by frame.
        
        Args:
            video_path: Input video file path
            output_path: Output video path (None = no save)
            display: Display real-time processing window
            draw_boxes: Draw bounding boxes on frames
            mask_scale: Mask scale relative to face width
            mask_offset_y: Vertical offset for mask placement
            
        Returns:
            True if successful, False otherwise
        """
        from pipelines.utils import FPSCounter
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Pipeline] Error: Cannot open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[Pipeline] Processing video: {video_path}")
        print(f"[Pipeline] Resolution: {width}x{height}, FPS: {fps}")
        
        # Setup video writer (if output specified)
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # FPS counter
        fps_counter = FPSCounter()
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_image(
                    frame,
                    draw_boxes=draw_boxes,
                    mask_scale=mask_scale,
                    mask_offset_y=mask_offset_y
                )
                
                # Update FPS
                fps_counter.update()
                
                # Draw FPS
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps_counter.get_fps():.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                # Write to output
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if display:
                    cv2.imshow("Face Mask - SVM+ORB", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Pipeline] User interrupted")
                        break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[Pipeline] Processed {frame_count} frames, "
                          f"FPS: {fps_counter.get_fps():.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"[Pipeline] Video processing complete: {frame_count} frames")
        if output_path:
            print(f"[Pipeline] Output saved to: {output_path}")
        
        return True
    
    def process_webcam(self,
                      camera_id: int = 0,
                      draw_boxes: bool = True,
                      mask_scale: float = 1.2,
                      mask_offset_y: float = 0.3):
        """
        Process webcam stream in real-time.
        
        Args:
            camera_id: Webcam device ID
            draw_boxes: Draw bounding boxes on frames
            mask_scale: Mask scale relative to face width
            mask_offset_y: Vertical offset for mask placement
        """
        from pipelines.utils import FPSCounter
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[Pipeline] Error: Cannot open webcam {camera_id}")
            return
        
        print(f"[Pipeline] Webcam started (ID: {camera_id})")
        print("[Pipeline] Press 'q' to quit, 's' to save screenshot")
        
        fps_counter = FPSCounter()
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_image(
                    frame,
                    draw_boxes=draw_boxes,
                    mask_scale=mask_scale,
                    mask_offset_y=mask_offset_y
                )
                
                # Update FPS
                fps_counter.update()
                
                # Draw FPS and detection count
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps_counter.get_fps():.1f} | Faces: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                # Display
                cv2.imshow("Face Mask - SVM+ORB (Webcam)", processed_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Pipeline] Quit signal received")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{screenshot_count:04d}.png"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"[Pipeline] Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print("[Pipeline] Webcam stopped")
