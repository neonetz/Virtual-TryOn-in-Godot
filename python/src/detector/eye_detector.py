"""
Eye Detector
=============
Simple eye detection using template matching and symmetry analysis.
NO pre-trained models, uses classical CV techniques only.

Methods:
- Template matching with multiple eye templates
- Symmetry verification (eyes should be horizontally aligned)
- Distance constraints (inter-eye distance reasonable)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class EyeDetector:
    """
    Classical eye detector without pre-trained models.
    Uses template matching and geometric constraints.
    
    High Cohesion: Focuses on eye detection only.
    Low Coupling: Independent from face detection.
    """
    
    def __init__(
        self,
        min_eye_size: int = 20,
        max_eye_size: int = 60,
        symmetry_tolerance: float = 0.1
    ):
        """
        Initialize eye detector.
        
        Args:
            min_eye_size: Minimum eye region size
            max_eye_size: Maximum eye region size
            symmetry_tolerance: Tolerance for horizontal alignment (0-1)
        """
        self.min_eye_size = min_eye_size
        self.max_eye_size = max_eye_size
        self.symmetry_tolerance = symmetry_tolerance
        
    def detect_eyes(
        self,
        face_roi: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        Detect two eyes in face ROI using classical CV techniques.
        
        Strategy:
        1. Search in upper half of face (eyes are typically in top 60%)
        2. Use edge detection to find eye-like regions
        3. Verify symmetry (two eyes should be horizontally aligned)
        4. Compute angle between eyes
        
        Args:
            face_roi: Cropped face image
            face_bbox: Original face bounding box (x, y, w, h)
            
        Returns:
            (left_eye_center, right_eye_center, angle_degrees) or None
            Coordinates are in original image space
        """
        if face_roi is None or face_roi.size == 0:
            return None
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        h, w = gray.shape[:2]
        
        # Step 1: Focus on upper portion (eyes are in top 60% of face)
        upper_h = int(h * 0.6)
        upper_roi = gray[0:upper_h, :]
        
        # Step 2: Edge detection to find eye candidates
        edges = cv2.Canny(upper_roi, 50, 150)
        
        # Find contours (potential eye regions)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        eye_candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Size constraints
            if cw < self.min_eye_size or cw > self.max_eye_size:
                continue
            if ch < self.min_eye_size or ch > self.max_eye_size:
                continue
            
            # Aspect ratio (eyes are roughly circular/elliptical)
            aspect = cw / ch if ch > 0 else 0
            if aspect < 0.6 or aspect > 1.7:
                continue
            
            # Eye center in face ROI coordinates
            cx = x + cw // 2
            cy = y + ch // 2
            
            # Darkness check (eyes are typically darker)
            eye_region = upper_roi[y:y+ch, x:x+cw]
            darkness = np.mean(eye_region)
            
            eye_candidates.append({
                'center': (cx, cy),
                'bbox': (x, y, cw, ch),
                'darkness': darkness,
                'size': cw * ch
            })
        
        # Step 3: Find two eyes with symmetry
        if len(eye_candidates) < 2:
            return None
        
        # Sort by darkness (eyes are dark)
        eye_candidates.sort(key=lambda e: e['darkness'])
        
        # Try to find symmetric pair
        best_pair = None
        best_score = float('inf')
        
        for i in range(len(eye_candidates)):
            for j in range(i + 1, len(eye_candidates)):
                e1 = eye_candidates[i]
                e2 = eye_candidates[j]
                
                # Horizontal alignment check
                y_diff = abs(e1['center'][1] - e2['center'][1])
                avg_y = (e1['center'][1] + e2['center'][1]) / 2
                y_symmetry = y_diff / avg_y if avg_y > 0 else 1.0
                
                if y_symmetry > self.symmetry_tolerance:
                    continue
                
                # Distance check (inter-eye distance should be reasonable)
                x_dist = abs(e1['center'][0] - e2['center'][0])
                min_dist = w * 0.2  # At least 20% of face width
                max_dist = w * 0.7  # At most 70% of face width
                
                if x_dist < min_dist or x_dist > max_dist:
                    continue
                
                # Score based on symmetry and darkness
                score = y_symmetry + (e1['darkness'] + e2['darkness']) / 255.0
                
                if score < best_score:
                    best_score = score
                    best_pair = (e1, e2)
        
        if best_pair is None:
            return None
        
        # Step 4: Compute angle and return in original image coordinates
        e1, e2 = best_pair
        
        # Sort by x-coordinate (left eye first)
        if e1['center'][0] > e2['center'][0]:
            e1, e2 = e2, e1
        
        # Convert to original image coordinates
        fx, fy, fw, fh = face_bbox
        left_eye = (fx + e1['center'][0], fy + e1['center'][1])
        right_eye = (fx + e2['center'][0], fy + e2['center'][1])
        
        # Compute angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return (left_eye, right_eye, angle)
    
    def visualize_eyes(
        self,
        image: np.ndarray,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        angle: float
    ) -> np.ndarray:
        """
        Draw detected eyes and angle on image.
        
        Args:
            image: Input image
            left_eye: Left eye center (x, y)
            right_eye: Right eye center (x, y)
            angle: Rotation angle in degrees
            
        Returns:
            Image with eyes marked
        """
        output = image.copy()
        
        # Draw eye centers
        cv2.circle(output, left_eye, 5, (0, 255, 0), -1)
        cv2.circle(output, right_eye, 5, (0, 255, 0), -1)
        
        # Draw line between eyes
        cv2.line(output, left_eye, right_eye, (0, 255, 0), 2)
        
        # Draw angle text
        mid_x = (left_eye[0] + right_eye[0]) // 2
        mid_y = (left_eye[1] + right_eye[1]) // 2
        cv2.putText(
            output,
            f"Angle: {angle:.1f}°",
            (mid_x - 50, mid_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        return output
