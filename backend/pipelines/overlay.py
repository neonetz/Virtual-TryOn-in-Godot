"""
MASK Overlay Module
Alpha-blending face mask on detected faces with optional rotation based on eye detection
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MaskOverlay:
    """
    Handle face MASK overlay with alpha blending and optional rotation
    Mask is positioned on lower part of face (nose-mouth-chin area)
    """
    
    def __init__(self, mask_path: str, cascade_eye_path: Optional[str] = None):
        """
        Initialize mask overlay
        
        Args:
            mask_path: Path to mask PNG with alpha channel (RGBA)
            cascade_eye_path: Optional path to haarcascade_eye.xml for rotation
        """
        # Load mask with alpha channel
        self.mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.mask_rgba is None:
            raise ValueError(f"Cannot load mask from {mask_path}")
        
        if self.mask_rgba.shape[2] != 4:
            raise ValueError(f"Mask must have alpha channel (RGBA), got {self.mask_rgba.shape[2]} channels")
        
        self.mask_h, self.mask_w = self.mask_rgba.shape[:2]
        logger.info(f"✓ Mask loaded: {mask_path} ({self.mask_w}×{self.mask_h})")
        
        # Load eye cascade for rotation (optional)
        self.eye_cascade = None
        if cascade_eye_path is not None:
            try:
                self.eye_cascade = cv2.CascadeClassifier(cascade_eye_path)
                if self.eye_cascade.empty():
                    logger.warning(f"Eye cascade is empty: {cascade_eye_path}")
                    self.eye_cascade = None
                else:
                    logger.info(f"✓ Eye cascade loaded for rotation")
            except Exception as e:
                logger.warning(f"Cannot load eye cascade: {e}")
                self.eye_cascade = None
    
    def detect_eyes(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect two eyes in face ROI
        
        Args:
            face_roi: Face region image (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        if self.eye_cascade is None:
            return None
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        if len(eyes) < 2:
            return None
        
        # Sort by x coordinate to get left and right eyes
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Take first two (leftmost = left eye, next = right eye)
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        # Calculate eye centers
        left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
        
        logger.debug(f"Eyes detected: left={left_center}, right={right_center}")
        return (left_center, right_center)
    
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
        
        Mask positioning:
        - mask_w = 0.9 × face_w
        - mask_h = 0.55 × face_h
        - mask_x = face_x + 0.05 × face_w
        - mask_y = face_y + 0.40 × face_h
        (covers nose-mouth-chin area)
        
        Args:
            img: Input image (BGR)
            face_bbox: Face bounding box dict {x, y, w, h}
            alpha_multiplier: Opacity multiplier [0, 1]
            enable_rotation: If True and eye cascade loaded, rotate mask
            
        Returns:
            Image with mask overlay
        """
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']
        
        # Calculate mask position and size
        mask_w = int(0.9 * w)
        mask_h = int(0.55 * h)
        mask_x = int(x + 0.05 * w)
        mask_y = int(y + 0.40 * h)
        
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
        if enable_rotation and self.eye_cascade is not None:
            # Extract face ROI for eye detection
            face_roi = img[y:y+h, x:x+w]
            eyes = self.detect_eyes(face_roi)
            
            if eyes is not None:
                left_eye, right_eye = eyes
                rotation_angle = self.calculate_rotation_angle(left_eye, right_eye)
                logger.debug(f"Face rotation: {rotation_angle:.1f}°")
        
        # Prepare mask
        mask_to_use = self.mask_rgba.copy()
        
        # Rotate if needed
        if abs(rotation_angle) > 2.0:  # Only rotate if angle is significant
            mask_to_use = self.rotate_mask(mask_to_use, rotation_angle)
        
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
        
        return result
    
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
