"""
Custom Eye Detector using Classical Computer Vision
No deep learning - using ORB features, template matching, or Hough circles
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CustomEyeDetector:
    """
    Custom eye detector using classical CV methods
    Supports multiple detection strategies without deep learning
    """
    
    def __init__(self, method: str = "orb"):
        """
        Initialize custom eye detector
        
        Args:
            method: Detection method ("orb", "template", "hough", "contour")
        """
        self.method = method.lower()
        
        if self.method not in ["orb", "template", "hough", "contour"]:
            raise ValueError(f"Unknown method: {method}. Use 'orb', 'template', 'hough', or 'contour'")
        
        # ORB detector for feature-based detection
        if self.method == "orb":
            self.orb = cv2.ORB_create(nfeatures=100)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.eye_template = None  # Will be set if needed
        
        logger.info(f"✓ Custom eye detector initialized (method: {self.method})")
    
    def detect_eyes_orb(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes using ORB feature matching with eye region templates
        
        Strategy:
        1. Split face ROI into left/right halves
        2. Find regions with high ORB keypoint density in upper half
        3. These are likely eyes (high texture contrast)
        
        Args:
            face_roi: Face region (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        h, w = face_roi.shape[:2]
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Focus on upper 60% of face (eye region)
        eye_region = gray[:int(h * 0.6), :]
        
        # Enhance contrast for better keypoint detection
        eye_region = cv2.equalizeHist(eye_region)
        
        # Detect ORB keypoints
        keypoints = self.orb.detect(eye_region, None)
        
        if len(keypoints) < 10:
            logger.debug("Not enough ORB keypoints for eye detection")
            return None
        
        # Split into left and right halves
        left_kps = [kp for kp in keypoints if kp.pt[0] < w / 2]
        right_kps = [kp for kp in keypoints if kp.pt[0] >= w / 2]
        
        if len(left_kps) < 3 or len(right_kps) < 3:
            logger.debug("Not enough keypoints in left/right eye regions")
            return None
        
        # Find eye centers as centroid of strongest keypoints
        # Eyes have high response due to iris/pupil contrast
        left_kps_sorted = sorted(left_kps, key=lambda kp: kp.response, reverse=True)[:5]
        right_kps_sorted = sorted(right_kps, key=lambda kp: kp.response, reverse=True)[:5]
        
        left_eye_x = int(np.mean([kp.pt[0] for kp in left_kps_sorted]))
        left_eye_y = int(np.mean([kp.pt[1] for kp in left_kps_sorted]))
        
        right_eye_x = int(np.mean([kp.pt[0] for kp in right_kps_sorted]))
        right_eye_y = int(np.mean([kp.pt[1] for kp in right_kps_sorted]))
        
        # Sanity check: eyes should be roughly on same horizontal line
        if abs(left_eye_y - right_eye_y) > h * 0.15:
            logger.debug(f"Eyes not aligned (y_diff={abs(left_eye_y - right_eye_y)})")
            return None
        
        logger.debug(f"Eyes detected via ORB: left=({left_eye_x},{left_eye_y}), right=({right_eye_x},{right_eye_y})")
        return ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y))
    
    def detect_eyes_hough(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes using Hough Circle Transform (finds iris/pupil)
        
        Args:
            face_roi: Face region (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        h, w = face_roi.shape[:2]
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Focus on upper 60% (eye region)
        eye_region = gray[:int(h * 0.6), :]
        
        # Enhance contrast
        eye_region = cv2.equalizeHist(eye_region)
        
        # Blur to reduce noise
        eye_region = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        # Detect circles (iris/pupil)
        circles = cv2.HoughCircles(
            eye_region,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(w * 0.2),  # Minimum distance between eyes
            param1=50,
            param2=30,
            minRadius=int(w * 0.03),  # Iris size
            maxRadius=int(w * 0.08)
        )
        
        if circles is None or len(circles[0]) < 2:
            logger.debug("Could not detect 2 circles (irises)")
            return None
        
        # Sort circles by x coordinate
        circles = sorted(circles[0], key=lambda c: c[0])
        
        # Take first two (left and right eyes)
        left_circle = circles[0]
        right_circle = circles[1]
        
        left_eye = (int(left_circle[0]), int(left_circle[1]))
        right_eye = (int(right_circle[0]), int(right_circle[1]))
        
        logger.debug(f"Eyes detected via Hough: left={left_eye}, right={right_eye}")
        return (left_eye, right_eye)
    
    def detect_eyes_contour(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes using edge detection and contour analysis
        
        Strategy:
        1. Canny edge detection
        2. Find contours
        3. Filter contours by shape (elliptical eyes)
        
        Args:
            face_roi: Face region (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        h, w = face_roi.shape[:2]
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Focus on upper 60% (eye region)
        eye_region = gray[:int(h * 0.6), :]
        
        # Enhance contrast
        eye_region = cv2.equalizeHist(eye_region)
        
        # Blur
        eye_region = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(eye_region, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio (eyes are elliptical)
        eye_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > w * h * 0.05:  # Filter by size
                continue
            
            # Fit ellipse if possible
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (ma, MA), angle = ellipse
                    
                    # Check aspect ratio (eyes are roughly circular/elliptical)
                    aspect_ratio = ma / MA if MA > 0 else 0
                    if 0.4 < aspect_ratio < 2.5:
                        eye_contours.append((x, y, area))
                except:
                    pass
        
        if len(eye_contours) < 2:
            logger.debug("Could not find 2 eye-like contours")
            return None
        
        # Sort by area (largest first), then by x (left to right)
        eye_contours = sorted(eye_contours, key=lambda c: c[2], reverse=True)[:4]
        eye_contours = sorted(eye_contours, key=lambda c: c[0])
        
        # Take first two
        left_eye = (int(eye_contours[0][0]), int(eye_contours[0][1]))
        right_eye = (int(eye_contours[1][0]), int(eye_contours[1][1]))
        
        logger.debug(f"Eyes detected via contour: left={left_eye}, right={right_eye}")
        return (left_eye, right_eye)
    
    def detect_eyes_template(self, face_roi: np.ndarray, 
                            template_path: Optional[str] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes using template matching
        
        Note: Requires eye template image. If not provided, generates simple template.
        
        Args:
            face_roi: Face region (BGR)
            template_path: Optional path to eye template image
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        h, w = face_roi.shape[:2]
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Focus on upper 60%
        eye_region = gray[:int(h * 0.6), :]
        
        # Load or create template
        if template_path is not None:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create simple eye template (dark center with light surround)
            template_size = int(w * 0.12)
            template = self._create_simple_eye_template(template_size)
        
        if template is None:
            logger.warning("Could not load/create eye template")
            return None
        
        # Template matching
        result = cv2.matchTemplate(eye_region, template, cv2.TM_CCOEFF_NORMED)
        
        # Find peaks (2 highest matches = 2 eyes)
        threshold = 0.4
        locations = np.where(result >= threshold)
        
        if len(locations[0]) < 2:
            logger.debug("Template matching: not enough matches")
            return None
        
        # Get match coordinates
        matches = list(zip(*locations[::-1]))  # (x, y) tuples
        
        # Sort by match value, take top 2
        match_values = [result[y, x] for x, y in matches]
        top_matches = sorted(zip(matches, match_values), key=lambda m: m[1], reverse=True)[:2]
        
        # Sort by x coordinate (left to right)
        top_matches = sorted(top_matches, key=lambda m: m[0][0])
        
        # Get centers (add template offset)
        th, tw = template.shape[:2]
        left_eye = (top_matches[0][0][0] + tw // 2, top_matches[0][0][1] + th // 2)
        right_eye = (top_matches[1][0][0] + tw // 2, top_matches[1][0][1] + th // 2)
        
        logger.debug(f"Eyes detected via template: left={left_eye}, right={right_eye}")
        return (left_eye, right_eye)
    
    def _create_simple_eye_template(self, size: int) -> np.ndarray:
        """
        Create simple synthetic eye template (dark center = pupil, light surround)
        
        Args:
            size: Template size (square)
            
        Returns:
            Grayscale template image
        """
        template = np.ones((size, size), dtype=np.uint8) * 200  # Light gray background
        
        # Draw dark circle for pupil/iris
        center = (size // 2, size // 2)
        radius = size // 3
        cv2.circle(template, center, radius, 50, -1)  # Dark gray
        
        # Add highlight
        highlight_pos = (center[0] - radius // 3, center[1] - radius // 3)
        cv2.circle(template, highlight_pos, radius // 4, 180, -1)
        
        return template
    
    def detect(self, face_roi: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes using configured method
        
        Args:
            face_roi: Face region (BGR)
            
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        if self.method == "orb":
            return self.detect_eyes_orb(face_roi)
        elif self.method == "hough":
            return self.detect_eyes_hough(face_roi)
        elif self.method == "contour":
            return self.detect_eyes_contour(face_roi)
        elif self.method == "template":
            return self.detect_eyes_template(face_roi)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return None


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) < 2:
        print("Usage: python eye_detector.py <image.jpg> [method]")
        print("Methods: orb, hough, contour, template")
        sys.exit(1)
    
    img_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "orb"
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load {img_path}")
        sys.exit(1)
    
    print(f"\n🔍 Testing eye detector with method: {method}")
    print("=" * 50)
    
    # Initialize detector
    detector = CustomEyeDetector(method=method)
    
    # Detect eyes
    eyes = detector.detect(img)
    
    if eyes is not None:
        left_eye, right_eye = eyes
        print(f"\n✓ Eyes detected!")
        print(f"  Left eye:  {left_eye}")
        print(f"  Right eye: {right_eye}")
        
        # Draw on image
        result = img.copy()
        cv2.circle(result, left_eye, 5, (0, 255, 0), -1)
        cv2.circle(result, right_eye, 5, (0, 255, 0), -1)
        cv2.line(result, left_eye, right_eye, (0, 255, 0), 2)
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        print(f"  Rotation angle: {angle:.2f}°")
        
        # Save result
        output_path = f"eye_detection_{method}.jpg"
        cv2.imwrite(output_path, result)
        print(f"\n✓ Result saved: {output_path}")
    else:
        print("\n✗ Could not detect eyes")
