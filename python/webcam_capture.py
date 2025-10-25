"""
Webcam Capture Module
Handles webcam initialization, frame capture, and resource management.
Provides clean interface for accessing camera frames.
"""

import cv2
import threading


class WebcamCapture:
    """
    Manages webcam access and frame capture in a thread-safe manner.
    Ensures smooth video streaming with proper resource cleanup.
    """
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        """
        Initializes webcam capture.
        
        Args:
            camera_index: Camera device index (0 for default camera)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        
    def start(self):
        """
        Opens the webcam and starts capturing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_running = True
        print(f"[INFO] Webcam started: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def read_frame(self):
        """
        Reads the latest frame from the webcam.
        
        Returns:
            numpy.ndarray: BGR frame, or None if capture failed
        """
        if not self.is_running or self.cap is None:
            return None
        
        with self.lock:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[WARNING] Failed to read frame from webcam")
                return None
            
            return frame
    
    def get_frame_size(self):
        """
        Returns the current frame dimensions.
        
        Returns:
            tuple: (width, height)
        """
        return (self.width, self.height)
    
    def stop(self):
        """
        Releases the webcam and cleans up resources.
        """
        self.is_running = False
        
        if self.cap is not None:
            with self.lock:
                self.cap.release()
                self.cap = None
        
        print("[INFO] Webcam stopped")
    
    def __del__(self):
        """Ensures proper cleanup when object is destroyed."""
        self.stop()
