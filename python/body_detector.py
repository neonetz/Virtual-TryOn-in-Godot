"""
Body Detector Module
Handles upper-body detection using classical computer vision methods.
No deep learning required - uses background subtraction, skin detection, and contour analysis.
"""

import cv2
import numpy as np


class BodyDetector:
    """
    Detects the upper-body region (torso and shoulders) using classical CV methods.
    Employs background subtraction, skin color segmentation, and morphological operations.
    """
    
    def __init__(self):
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Skin color detection parameters (YCrCb color space)
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Morphological kernel for noise reduction
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Minimum area threshold for valid body detection (pixels)
        self.min_body_area = 5000
        
        # Frame counter for background learning
        self.frame_count = 0
        self.learning_frames = 30  # Number of frames to learn background
        
    def detect_upper_body(self, frame):
        """
        Detects upper-body region in the given frame.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            tuple: (x, y, w, h) bounding box coordinates, or None if no body detected
        """
        if frame is None or frame.size == 0:
            return None
            
        # Step 1: Apply background subtraction to detect moving regions
        fg_mask = self.bg_subtractor.apply(frame)
        self.frame_count += 1
        
        # Wait for background model to stabilize
        if self.frame_count < self.learning_frames:
            return None
        
        # Step 2: Skin color detection in YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        
        # Step 3: Combine foreground and skin masks
        # This helps isolate the person from background
        combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
        
        # Step 4: Morphological operations to reduce noise and fill gaps
        # Opening: removes small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)
        # Closing: fills small holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        # Dilation: expands the detected region slightly
        combined_mask = cv2.dilate(combined_mask, self.morph_kernel, iterations=2)
        
        # Step 5: Find contours in the processed mask
        contours, _ = cv2.findContours(
            combined_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Step 6: Find the largest contour (assumed to be the body)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter out small detections (noise)
        if area < self.min_body_area:
            return None
        
        # Step 7: Get bounding box for the upper-body region
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Focus on upper portion (torso and shoulders, not legs)
        # Take only top 60% of detected region
        upper_height = int(h * 0.6)
        
        return (x, y, w, upper_height)
    
    def draw_detection(self, frame, bbox):
        """
        Draws bounding box and label on the frame.
        
        Args:
            frame: BGR image to draw on
            bbox: (x, y, w, h) bounding box coordinates
            
        Returns:
            frame with drawn annotations
        """
        if bbox is None:
            return frame
        
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = "Upper Body"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame, 
            (x, y - label_size[1] - 10), 
            (x + label_size[0], y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            frame, 
            label, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0), 
            2
        )
        
        return frame
    
    def reset(self):
        """Resets the background model and frame counter."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        self.frame_count = 0
