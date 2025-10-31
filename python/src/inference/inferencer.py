"""
Inference Pipeline
==================
Performs face detection and mask overlay on images and video streams.

Pipeline:
1. Load trained models (codebook, SVM, scaler)
2. For each frame:
   a. Propose ROIs 
   b. Extract ORB descriptors from ROIs
   c. Encode to BoVW vectors
   d. Classify with SVM
   e. Apply NMS to remove duplicates
   f. Overlay mask on accepted faces
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import os

from ..features import ORBExtractor, BoVWEncoder
from ..detector import ROIProposal, SVMClassifier
from ..overlay import MaskOverlay
from ..utils import non_max_suppression, draw_face_detections, add_fps_overlay


class Inferencer:
    """
    Performs inference on images and video streams.
    
    High Cohesion: Handles complete inference workflow.
    Low Coupling: Coordinates trained components without tight dependencies.
    """
    
    def __init__(
        self,
        models_dir: str = 'models',
        mask_path: Optional[str] = None,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.0
    ):
        """
        Initialize inferencer with trained models.
        
        Args:
            models_dir: Directory containing trained models
            mask_path: Path to mask PNG (optional, can be set later)
            iou_threshold: NMS IoU threshold
            score_threshold: Minimum confidence score
        """
        self.models_dir = models_dir
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
        # Load models
        self._load_models()
        
        # Initialize mask overlay if path provided
        self.mask_overlay = None
        if mask_path is not None:
            self.set_mask(mask_path)
        
        print("Inferencer initialized")
    
    def _load_models(self) -> None:
        """Load all trained models."""
        print("\nLoading trained models...")
        
        # Load ORB extractor (OPTIMIZED: 300 features instead of 500)
        self.orb_extractor = ORBExtractor(n_features=300)
        
        # Load BoVW codebook
        codebook_path = os.path.join(self.models_dir, 'codebook.pkl')
        if not os.path.exists(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        self.bovw_encoder = BoVWEncoder.load(codebook_path)
        
        # Load SVM classifier
        svm_path = os.path.join(self.models_dir, 'svm.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if not os.path.exists(svm_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("SVM model or scaler not found")
        self.svm_classifier = SVMClassifier.load(svm_path, scaler_path)
        
        # Initialize ROI proposal (OPTIMIZED: 5 scales instead of 8)
        self.roi_proposal = ROIProposal(window_sizes=[64, 80, 96, 128, 160])
        
        print("✓ Models loaded successfully")
    
    def set_mask(self, mask_path: str) -> None:
        """
        Set mask for overlay.
        
        Args:
            mask_path: Path to mask PNG with alpha
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        self.mask_overlay = MaskOverlay(mask_path)
        print(f"Mask loaded: {mask_path}")
    
    def detect_faces(
        self,
        image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect faces in an image.
        
        Pipeline:
        1. Propose ROIs with sliding window
        2. Extract ORB features from each ROI
        3. Encode to BoVW vectors
        4. Classify with SVM
        5. Filter by score threshold
        6. Apply NMS
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (face_boxes, confidence_scores)
        """
        # Step 1: Propose ROIs
        roi_boxes = self.roi_proposal.propose(image)
        
        if len(roi_boxes) == 0:
            return [], []
        
        # Step 2 & 3: Extract and encode features
        descriptors_list = []
        for bbox in roi_boxes:
            roi = self.roi_proposal.extract_roi(image, bbox)
            descriptors = self.orb_extractor.extract(roi)
            descriptors_list.append(descriptors)
        
        features = self.bovw_encoder.encode_batch(descriptors_list)
        
        # Step 4: Classify
        scores = self.svm_classifier.predict_proba(features)
        predictions = (scores > 0).astype(int)  # Threshold at 0 for decision function
        
        # Step 5: Filter by threshold and positive predictions
        filtered_boxes = []
        filtered_scores = []
        
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == 1 and score >= self.score_threshold:
                filtered_boxes.append(roi_boxes[i])
                filtered_scores.append(float(score))
        
        # Step 6: Apply NMS
        if len(filtered_boxes) > 0:
            final_boxes, final_scores = non_max_suppression(
                filtered_boxes,
                filtered_scores,
                self.iou_threshold
            )
        else:
            final_boxes, final_scores = [], []
        
        return final_boxes, final_scores
    
    def infer_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        overlay_mask: bool = True,
        draw_boxes: bool = True
    ) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            overlay_mask: Whether to overlay mask
            draw_boxes: Whether to draw bounding boxes
            
        Returns:
            Processed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Detect faces
        boxes, scores = self.detect_faces(image)
        
        print(f"Detected {len(boxes)} faces in {image_path}")
        
        # Process output
        output = image.copy()
        
        if overlay_mask and self.mask_overlay is not None and len(boxes) > 0:
            output = self.mask_overlay.overlay_on_faces(output, boxes)
        
        if draw_boxes and len(boxes) > 0:
            output = draw_face_detections(output, boxes, scores)
        
        # Save if output path provided
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output)
            print(f"Output saved to {output_path}")
        
        return output
    
    def infer_webcam(
        self,
        camera_id: int = 0,
        show_fps: bool = True,
        overlay_mask: bool = True,
        draw_boxes: bool = False
    ) -> None:
        """
        Run inference on webcam stream with real-time mask overlay.
        
        Controls:
        - 'm': Toggle mask overlay
        - 'b': Toggle bounding boxes
        - 'q': Quit
        
        Args:
            camera_id: Camera device ID
            show_fps: Display FPS counter
            overlay_mask: Initial mask overlay state
            draw_boxes: Initial box drawing state
        """
        print("\nStarting webcam inference...")
        print("Controls:")
        print("  'h' - Toggle mask overlay (hide/show)")
        print("  'r' - Toggle eye-based rotation")
        print("  'b' - Toggle bounding boxes")
        print("  'q' - Quit")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set resolution (720p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS tracking
        fps_list = []
        
        # State
        show_mask = overlay_mask and self.mask_overlay is not None
        show_boxes = draw_boxes
        show_rotation = False  # Eye-based rotation (disabled by default for performance)
        
        # Frame skip for performance (process every 2 frames)
        frame_count = 0
        cached_boxes = []
        cached_scores = []
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # OPTIMIZATION: Process every 2 frames only
            if frame_count % 2 == 0:
                # Downscale for processing (50% size)
                small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                
                # Detect faces on small frame
                small_boxes, scores = self.detect_faces(small_frame)
                
                # Scale boxes back to original size
                cached_boxes = [(int(x*2), int(y*2), int(w*2), int(h*2)) 
                               for (x, y, w, h) in small_boxes]
                cached_scores = scores
                
                # OPTIMIZATION: Keep only the best detection (highest score)
                if len(cached_boxes) > 0:
                    best_idx = np.argmax(cached_scores)
                    cached_boxes = [cached_boxes[best_idx]]
                    cached_scores = [cached_scores[best_idx]]
            
            # Use cached detections
            boxes = cached_boxes
            scores = cached_scores
            
            # Process output
            output = frame.copy()
            
            if show_mask and self.mask_overlay is not None and len(boxes) > 0:
                # Eye-based rotation if enabled
                rotation_angles = None
                if show_rotation:
                    from ..detector.eye_detector import EyeDetector
                    eye_detector = EyeDetector()
                    rotation_angles = []
                    
                    for bbox in boxes:
                        x, y, w, h = bbox
                        face_roi = frame[y:y+h, x:x+w]
                        eye_result = eye_detector.detect_eyes(face_roi, bbox)
                        
                        if eye_result:
                            _, _, angle = eye_result
                            rotation_angles.append(angle)
                        else:
                            rotation_angles.append(None)
                
                output = self.mask_overlay.overlay_on_faces(output, boxes, rotation_angles)
            
            if show_boxes and len(boxes) > 0:
                output = draw_face_detections(output, boxes, scores)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(current_fps)
            
            # Keep last 30 frames for average
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            avg_fps = np.mean(fps_list)
            
            # Add FPS overlay
            if show_fps:
                output = add_fps_overlay(output, avg_fps)
            
            # Add status text
            status_y = 70
            if show_mask:
                cv2.putText(output, "Mask: ON", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(output, "Mask: OFF", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Rotation status
            if show_rotation:
                cv2.putText(output, "Rotation: ON", (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(output, "Rotation: OFF", (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Display
            cv2.imshow('Face Detection + Mask Overlay', output)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):  # 'h' untuk hide/show mask
                show_mask = not show_mask
                print(f"Mask overlay: {'ON' if show_mask else 'OFF'}")
            elif key == ord('r'):  # 'r' untuk rotation
                show_rotation = not show_rotation
                print(f"Eye-based rotation: {'ON' if show_rotation else 'OFF'}")
            elif key == ord('b'):
                show_boxes = not show_boxes
                print(f"Bounding boxes: {'ON' if show_boxes else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nAverage FPS: {np.mean(fps_list):.2f}")
        print("Webcam inference stopped")
    
    def infer_video(
        self,
        video_path: str,
        output_path: str,
        overlay_mask: bool = True,
        draw_boxes: bool = False
    ) -> None:
        """
        Run inference on video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            overlay_mask: Whether to overlay mask
            draw_boxes: Whether to draw bounding boxes
        """
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"  Processing frame {frame_count}/{total_frames}")
            
            # Detect faces
            boxes, scores = self.detect_faces(frame)
            
            # Process output
            output = frame.copy()
            
            if overlay_mask and self.mask_overlay is not None and len(boxes) > 0:
                output = self.mask_overlay.overlay_on_faces(output, boxes)
            
            if draw_boxes and len(boxes) > 0:
                output = draw_face_detections(output, boxes, scores)
            
            # Write frame
            out.write(output)
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✓ Video processing complete: {output_path}")
        print(f"  Processed {frame_count} frames")
