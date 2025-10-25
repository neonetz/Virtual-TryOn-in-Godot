"""
Frame Encoder Module
Handles frame encoding to JPEG format for efficient network transmission.
Optimizes compression quality vs. size trade-off.
"""

import cv2
import struct


class FrameEncoder:
    """
    Encodes video frames to JPEG format with configurable quality.
    Prepares frames for UDP transmission with size headers.
    """
    
    def __init__(self, jpeg_quality=85):
        """
        Initializes the frame encoder.
        
        Args:
            jpeg_quality: JPEG compression quality (0-100, higher = better quality)
        """
        self.jpeg_quality = jpeg_quality
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        
    def encode_frame(self, frame):
        """
        Encodes a frame to JPEG format.
        
        Args:
            frame: BGR numpy array
            
        Returns:
            bytes: Encoded JPEG data, or None if encoding failed
        """
        if frame is None or frame.size == 0:
            return None
        
        # Encode frame to JPEG format
        ret, jpeg_data = cv2.imencode('.jpg', frame, self.encode_params)
        
        if not ret:
            print("[WARNING] Failed to encode frame to JPEG")
            return None
        
        # Convert to bytes
        return jpeg_data.tobytes()
    
    def encode_bbox(self, bbox):
        """
        Encodes bounding box coordinates to binary format.
        
        Args:
            bbox: tuple (x, y, w, h) or None
            
        Returns:
            bytes: Packed binary data (4 integers)
        """
        if bbox is None:
            # Send zeros if no detection
            return struct.pack('!IIII', 0, 0, 0, 0)
        
        x, y, w, h = bbox
        return struct.pack('!IIII', x, y, w, h)
    
    def create_packet(self, frame, bbox):
        """
        Creates a complete data packet with frame and bounding box.
        Packet structure:
        - 4 bytes: frame size
        - N bytes: JPEG frame data
        - 16 bytes: bounding box (x, y, w, h)
        
        Args:
            frame: BGR numpy array
            bbox: tuple (x, y, w, h) or None
            
        Returns:
            bytes: Complete packet ready for UDP transmission, or None if failed
        """
        # Encode frame to JPEG
        jpeg_data = self.encode_frame(frame)
        if jpeg_data is None:
            return None
        
        # Encode bounding box
        bbox_data = self.encode_bbox(bbox)
        
        # Create packet with size header
        frame_size = len(jpeg_data)
        size_header = struct.pack('!I', frame_size)
        
        # Combine all parts: size + frame + bbox
        packet = size_header + jpeg_data + bbox_data
        
        return packet
    
    def set_quality(self, quality):
        """
        Updates the JPEG compression quality.
        
        Args:
            quality: New quality value (0-100)
        """
        if 0 <= quality <= 100:
            self.jpeg_quality = quality
            self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
