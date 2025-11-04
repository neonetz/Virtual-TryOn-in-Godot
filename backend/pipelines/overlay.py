"""
MASK Overlay Module
Alpha-blending face mask on detected faces with optional rotation based on eye detection
Supports both Haar Cascade and Custom Eye Detector
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MaskOverlay:
    """
    Handle face MASK overlay with alpha blending and optional rotation
    Mask is positioned on lower part of face (nose-mouth-chin area)
    Supports multiple eye detection methods
    """
    
    def __init__(self, mask_path: str, 
                 cascade_eye_path: Optional[str] = None,
                 smoothing_factor: float = 0.0,  # Smooting
                 use_custom_eye_detector: bool = False,
                 custom_eye_method: str = "orb",
                 tracker_type: str = "none"):
        """
        Initialize mask overlay
        
        Args:
            mask_path: Path to mask PNG with alpha channel (RGBA)
            cascade_eye_path: Optional path to haarcascade_eye.xml for rotation
            smoothing_factor: Temporal smoothing (0=instant, 1=full smooth). Default 0.3 (balanced)
            use_custom_eye_detector: If True, use custom eye detector instead of Haar
            custom_eye_method: Method for custom detector ("orb", "hough", "contour", "template")
            tracker_type: Tracking type - "exponential", "kalman", "velocity", or "none"
        """
        # Store mask path for dynamic reloading
        self.mask_path = mask_path
        
        # Load mask with alpha channel
        self.mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.mask_rgba is None:
            raise ValueError(f"Cannot load mask from {mask_path}")
        
        if self.mask_rgba.shape[2] != 4:
            raise ValueError(f"Mask must have alpha channel (RGBA), got {self.mask_rgba.shape[2]} channels")
        
        self.mask_h, self.mask_w = self.mask_rgba.shape[:2]
        logger.info(f"✓ Mask loaded: {mask_path} ({self.mask_w}×{self.mask_h})")
        
        # Mask directory for dynamic switching
        self.mask_dir = None
        import os
        if os.path.exists(mask_path):
            self.mask_dir = os.path.dirname(mask_path)
            logger.info(f"✓ Mask directory: {self.mask_dir}")
        
        # Temporal smoothing/tracking setup
        self.tracker_type = tracker_type
        self.smoothing_factor = smoothing_factor
        self.prev_bbox = None
        self.kalman_tracker = None
        self.velocity_predictor = None
        
        # Initialize tracker based on type
        if tracker_type == "kalman":
            try:
                from .kalman_tracker import KalmanBBoxTracker
                self.kalman_tracker = KalmanBBoxTracker(
                    process_noise=1e-3,
                    measurement_noise=1e-1
                )
                logger.info(f"✓ Kalman bbox tracker initialized (zero-lag predictive tracking)")
            except Exception as e:
                logger.warning(f"Cannot load Kalman tracker: {e}, falling back to exponential")
                self.tracker_type = "exponential"
        elif tracker_type == "velocity":
            try:
                from .kalman_tracker import SimpleVelocityPredictor
                self.velocity_predictor = SimpleVelocityPredictor(momentum=0.7)
                logger.info(f"✓ Velocity predictor initialized (momentum-based tracking)")
            except Exception as e:
                logger.warning(f"Cannot load velocity predictor: {e}, falling back to exponential")
                self.tracker_type = "exponential"
        elif tracker_type == "none":
            logger.info(f"✓ Tracking disabled (instant bbox, no smoothing)")
        else:  # exponential
            logger.info(f"✓ Exponential smoothing initialized (factor={smoothing_factor})")
        
        # Eye detection setup
        self.use_custom_eye_detector = use_custom_eye_detector
        self.eye_cascade = None
        self.custom_eye_detector = None
        
        if use_custom_eye_detector:
            # Use custom eye detector
            try:
                from .eye_detector import CustomEyeDetector
                self.custom_eye_detector = CustomEyeDetector(method=custom_eye_method)
                logger.info(f"✓ Custom eye detector loaded (method: {custom_eye_method})")
            except Exception as e:
                logger.warning(f"Cannot load custom eye detector: {e}")
                self.custom_eye_detector = None
        else:
            # Use Haar cascade (traditional)
            if cascade_eye_path is not None:
                try:
                    self.eye_cascade = cv2.CascadeClassifier(cascade_eye_path)
                    if self.eye_cascade.empty():
                        logger.warning(f"Eye cascade is empty: {cascade_eye_path}")
                        self.eye_cascade = None
                    else:
                        logger.info(f"✓ Haar eye cascade loaded")
                except Exception as e:
                    logger.warning(f"Cannot load eye cascade: {e}")
                    self.eye_cascade = None
        
        # Load mask configurations
        self.mask_configs = self._load_mask_configs()
        self.current_mask_config = self._get_mask_config(os.path.basename(mask_path))
        
        logger.info(f"✓ Mask config loaded: {self.current_mask_config}")
    
    def detect_eyes(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect two eyes in face ROI using configured method with geometry validation
        
        Args:
            face_roi: Face region image (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        # Use custom detector if enabled
        if self.use_custom_eye_detector and self.custom_eye_detector is not None:
            eyes = self.custom_eye_detector.detect(face_roi)
            if eyes is not None and self._validate_eye_geometry(eyes, face_roi.shape):
                return eyes
            return None
        
        # Fallback to Haar cascade
        if self.eye_cascade is None:
            return None
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Focus on upper half of face (eye region)
        h, w = gray.shape[:2]
        eye_region = gray[:int(h * 0.7), :]  # Top 70% of face
        
        # Detect eyes with relaxed parameters
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.05,  # Smaller steps for better detection
            minNeighbors=3,    # Relaxed for better detection
            minSize=(int(w * 0.1), int(h * 0.08)),  # Relative to face size
            maxSize=(int(w * 0.3), int(h * 0.25))
        )
        
        if len(eyes) < 2:
            logger.debug(f"Eye detection: found {len(eyes)} eyes (need 2)")
            return None
        
        # Sort by x coordinate to get left and right eyes
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Try to find best pair of eyes
        best_pair = None
        best_score = 0
        
        for i in range(len(eyes) - 1):
            for j in range(i + 1, len(eyes)):
                left_eye = eyes[i]
                right_eye = eyes[j]
                
                # Calculate centers
                left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
                right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
                
                # Validate geometry
                eyes_pair = (left_center, right_center)
                if self._validate_eye_geometry(eyes_pair, (h, w, 3)):
                    # Score based on horizontal alignment and spacing
                    y_diff = abs(left_center[1] - right_center[1])
                    x_diff = right_center[0] - left_center[0]
                    
                    # Good eyes should be: horizontally aligned, reasonable spacing
                    score = 1.0 / (1.0 + y_diff) * x_diff
                    
                    if score > best_score:
                        best_score = score
                        best_pair = eyes_pair
        
        if best_pair is not None:
            logger.debug(f"Eyes detected (Haar): left={best_pair[0]}, right={best_pair[1]}")
            return best_pair
        
        logger.debug("Eye detection: no valid eye pair found")
        return None
    
    def _validate_eye_geometry(self, eyes: Tuple[Tuple[int, int], Tuple[int, int]], 
                              face_shape: Tuple) -> bool:
        """
        Validate if detected eyes have reasonable geometry
        
        Args:
            eyes: ((left_x, left_y), (right_x, right_y))
            face_shape: Shape of face ROI (h, w, c)
            
        Returns:
            True if geometry is reasonable
        """
        left_eye, right_eye = eyes
        h, w = face_shape[:2]
        
        # Calculate distances
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        # Eyes should be horizontally separated
        if dx <= 0:
            logger.debug("Eye validation: right eye not to the right of left eye")
            return False
        
        # Eyes should be roughly on same horizontal line
        # Allow up to 20% vertical difference relative to horizontal distance
        if abs(dy) > dx * 0.2:
            logger.debug(f"Eye validation: too much vertical misalignment (dy={dy}, dx={dx})")
            return False
        
        # Inter-eye distance should be reasonable (20% to 60% of face width)
        eye_distance = np.sqrt(dx**2 + dy**2)
        if eye_distance < w * 0.2 or eye_distance > w * 0.6:
            logger.debug(f"Eye validation: unreasonable distance ({eye_distance:.1f} px, face_w={w})")
            return False
        
        # Eyes should be in upper half of face
        avg_y = (left_eye[1] + right_eye[1]) / 2
        if avg_y > h * 0.6:
            logger.debug(f"Eye validation: eyes too low (y={avg_y:.1f}, face_h={h})")
            return False
        
        return True
    
    def calculate_rotation_angle(self, left_eye: Tuple[int, int], 
                                right_eye: Tuple[int, int]) -> float:
        """
        Calculate rotation angle from two eye positions
        
        Args:
            left_eye: (x, y) of left eye center
            right_eye: (x, y) of right eye center
            
        Returns:
            Rotation angle in degrees
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def _load_mask_configs(self) -> Dict:
        """Load mask configuration from JSON file"""
        config_path = os.path.join("assets", "masks", "config.json")
        config_path = os.path.normpath(config_path)
        
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
            logger.info(f"✓ Loaded mask configs from: {config_path}")
            return configs
        except Exception as e:
            logger.warning(f"Could not load mask config: {e}")
            return {}

    def _get_mask_config(self, mask_filename: str) -> Dict:
        """Get configuration for specific mask"""
        default_config = {
            "scale_w": 1,
            "scale_h": 1,
            "offset_x": 1,
            "offset_y": 1,
            "description": "Default mask"
        }
        
        return self.mask_configs.get(mask_filename, default_config)

    def rotate_mask(self, mask_rgba: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate mask by angle around center
        
        Args:
            mask_rgba: RGBA mask image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated RGBA mask
        """
        h, w = mask_rgba.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with border value = transparent
        rotated = cv2.warpAffine(
            mask_rgba,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return rotated
    
    def apply_mask(self, img: np.ndarray, face_bbox: Dict, 
                  alpha_multiplier: float = 0.95,
                  enable_rotation: bool = True) -> np.ndarray:
        """
        Apply mask overlay on detected face
        
        Mask positioning (LARGER MASK for full face coverage):
        - mask_w = 0.95 × face_w  (was 0.9)
        - mask_h = 0.75 × face_h  (was 0.55)
        - mask_x = face_x + 0.025 × face_w
        - mask_y = face_y + 0.20 × face_h
        (covers from forehead to chin)
        
        Args:
            img: Input image (BGR)
            face_bbox: Face bounding box dict {x, y, w, h}
            alpha_multiplier: Opacity multiplier [0, 1]
            enable_rotation: If True and eye cascade loaded, rotate mask
            
        Returns:
            Image with mask overlay
        """
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']
        
        # Apply tracking/smoothing based on tracker type
        if self.tracker_type == "kalman" and self.kalman_tracker is not None:
            # Kalman filter: predictive tracking (zero lag)
            bbox_array = np.array([x, y, w, h], dtype=np.float32)
            smoothed = self.kalman_tracker.update(bbox_array)
            x, y, w, h = map(int, smoothed)
            logger.debug(f"Kalman tracking: ({x}, {y}, {w}, {h})")
            
        elif self.tracker_type == "velocity" and self.velocity_predictor is not None:
            # Velocity predictor: momentum-based tracking
            bbox_array = np.array([x, y, w, h], dtype=np.float32)
            smoothed = self.velocity_predictor.update(bbox_array)
            x, y, w, h = map(int, smoothed)
            logger.debug(f"Velocity tracking: ({x}, {y}, {w}, {h})")
            
        elif self.tracker_type == "exponential":
            # Exponential smoothing: reactive (has lag)
            if self.prev_bbox is not None and self.smoothing_factor > 0:
                alpha = self.smoothing_factor
                x = int(alpha * self.prev_bbox['x'] + (1 - alpha) * x)
                y = int(alpha * self.prev_bbox['y'] + (1 - alpha) * y)
                w = int(alpha * self.prev_bbox['w'] + (1 - alpha) * w)
                h = int(alpha * self.prev_bbox['h'] + (1 - alpha) * h)
            
            # Store current bbox for next frame
            self.prev_bbox = {'x': x, 'y': y, 'w': w, 'h': h}
        
        # else: tracker_type == "none" → use raw bbox (instant, no smoothing)

        # Get current mask configuration
        config = self.current_mask_config
        
        # Calculate mask position and size (BIGGER MASK)
        mask_w = int(config['scale_w'] * w)
        mask_h = int(config['scale_h'] * h)
        mask_x = int(x + config['offset_x'] * w)
        mask_y = int(y + config['offset_y'] * h)
        
        # Check if mask is within image bounds
        img_h, img_w = img.shape[:2]
        if mask_x < 0 or mask_y < 0 or mask_x + mask_w > img_w or mask_y + mask_h > img_h:
            logger.debug("Mask partially out of bounds, clamping")
            # Clamp to image boundaries
            mask_x = max(0, mask_x)
            mask_y = max(0, mask_y)
            mask_w = min(mask_w, img_w - mask_x)
            mask_h = min(mask_h, img_h - mask_y)
            
            if mask_w <= 0 or mask_h <= 0:
                logger.debug("Mask fully out of bounds, skipping")
                return img
        
        # Get rotation angle if enabled
        rotation_angle = 0.0
        if enable_rotation and (self.eye_cascade is not None or self.custom_eye_detector is not None):
            # Extract face ROI for eye detection
            face_roi = img[y:y+h, x:x+w]
            eyes = self.detect_eyes(face_roi)
            
            if eyes is not None:
                left_eye, right_eye = eyes
                rotation_angle = self.calculate_rotation_angle(left_eye, right_eye)
                logger.debug(f"Face rotation: {rotation_angle:.1f}°")
        
        # Prepare mask
        mask_to_use = self.mask_rgba.copy()
        
        # Rotate if needed (LOWER THRESHOLD - rotate even for small angles)
        if abs(rotation_angle) > 1.0:  # Changed from 2.0 to 1.0 - more sensitive
            mask_to_use = self.rotate_mask(mask_to_use, rotation_angle)
            logger.debug(f"Rotating mask by {rotation_angle:.1f}°")
        
        # Resize mask to target size
        mask_resized = cv2.resize(mask_to_use, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
        
        # Split into BGR and alpha channels
        mask_bgr = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3:4].astype(np.float32) / 255.0
        
        # Apply alpha multiplier
        mask_alpha = mask_alpha * alpha_multiplier
        
        # Extract ROI from original image
        roi = img[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
        
        # Handle potential size mismatch
        h_min = min(roi.shape[0], mask_bgr.shape[0])
        w_min = min(roi.shape[1], mask_bgr.shape[1])
        
        roi = roi[:h_min, :w_min]
        mask_bgr = mask_bgr[:h_min, :w_min]
        mask_alpha = mask_alpha[:h_min, :w_min]
        
        # Alpha blending: result = mask * alpha + roi * (1 - alpha)
        blended = (mask_bgr * mask_alpha + roi * (1 - mask_alpha)).astype(np.uint8)
        
        # Place blended region back into image
        result = img.copy()
        result[mask_y:mask_y+h_min, mask_x:mask_x+w_min] = blended

        logger.debug(f"Applied {os.path.basename(self.mask_path)} at ({mask_x}, {mask_y}) size ({mask_w}×{mask_h})")
        
        return result
    
    def change_mask(self, mask_filename: str) -> bool:
        """
        Change current mask to a different one
        
        Args:
            mask_filename: Name of mask file (e.g., "mask-2.png")
            
        Returns:
            True if successful, False otherwise
        """
        import os
        
        # Try mask in masks directory (backend/assets/masks/)
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, mask_filename)
        else:
            # Fallback to assets/masks folder
            mask_path = os.path.join("assets", mask_filename)
        
        if not os.path.exists(mask_path):
            logger.error(f"Mask not found: {mask_path}")
            return False
        
        # Load new mask
        new_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if new_mask is None:
            logger.error(f"Cannot load mask: {mask_path}")
            return False
        
        if new_mask.shape[2] != 4:
            logger.error(f"Mask must have alpha channel (RGBA)")
            return False
        
        # Update mask and configuration
        self.mask_rgba = new_mask
        self.mask_h, self.mask_w = new_mask.shape[:2]
        self.mask_path = mask_path
        self.current_mask_config = self._get_mask_config(mask_filename)
        
        logger.info(f"✓ Mask changed to: {mask_filename} ({self.mask_w}×{self.mask_h})")
        logger.info(f"✓ Config: scale=({self.current_mask_config['scale_w']:.2f}, {self.current_mask_config['scale_h']:.2f}), "
                   f"offset=({self.current_mask_config['offset_x']:.3f}, {self.current_mask_config['offset_y']:.3f})")
        return True
    
    def apply_mask_batch(self, img: np.ndarray, face_bboxes: list,
                        alpha_multiplier: float = 0.95,
                        enable_rotation: bool = True) -> np.ndarray:
        """
        Apply mask to multiple detected faces
        
        Args:
            img: Input image
            face_bboxes: List of face bbox dicts
            alpha_multiplier: Opacity multiplier
            enable_rotation: Enable rotation based on eyes
            
        Returns:
            Image with all masks applied
        """
        result = img.copy()
        
        for bbox in face_bboxes:
            result = self.apply_mask(result, bbox, alpha_multiplier, enable_rotation)
        
        return result


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) < 3:
        print("Usage: python overlay.py <image.jpg> <mask.png> [haarcascade_eye.xml]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    mask_path = sys.argv[2]
    eye_cascade_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image {img_path}")
        sys.exit(1)
    
    # Create dummy face detection
    h, w = img.shape[:2]
    dummy_face = {
        'x': w // 4,
        'y': h // 4,
        'w': w // 2,
        'h': h // 2
    }
    
    # Initialize overlay
    overlay = MaskOverlay(mask_path, eye_cascade_path)
    
    # Apply mask without rotation
    print("\n1. Applying mask without rotation...")
    result_no_rot = overlay.apply_mask(img, dummy_face, enable_rotation=False)
    cv2.imwrite("mask_no_rotation.jpg", result_no_rot)
    print("✓ Saved: mask_no_rotation.jpg")
    
    # Apply mask with rotation
    if eye_cascade_path:
        print("\n2. Applying mask with rotation...")
        result_rot = overlay.apply_mask(img, dummy_face, enable_rotation=True)
        cv2.imwrite("mask_with_rotation.jpg", result_rot)
        print("✓ Saved: mask_with_rotation.jpg")
    
    print("\n✓ Done!")
