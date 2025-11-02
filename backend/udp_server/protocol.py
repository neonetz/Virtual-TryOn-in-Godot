"""
UDP Protocol Definitions
JSON message schemas for communication between Godot and Python
"""

import json
import base64
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BoundingBox:
    """Face bounding box"""

    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Point2D:
    """2D point (for eye positions, center, etc.)"""

    x: float
    y: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EyeData:
    """Eye detection data"""

    left: Optional[Point2D]
    right: Optional[Point2D]
    detected: bool

    def to_dict(self) -> Dict:
        return {
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "detected": self.detected,
        }


@dataclass
class FaceDetection:
    """Single face detection result"""

    id: int
    bbox: BoundingBox
    center: Point2D
    rotation_deg: float
    confidence: float
    eyes: EyeData

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "center": self.center.to_dict(),
            "rotation_deg": self.rotation_deg,
            "confidence": self.confidence,
            "eyes": self.eyes.to_dict(),
        }


class MessageProtocol:
    """Protocol for encoding/decoding UDP messages"""

    @staticmethod
    def decode_frame_message(data: bytes) -> Optional[Dict]:
        """
        Decode incoming frame message from Godot

        Expected format:
        {
            "type": "frame",
            "timestamp": 1730563892.123,
            "frame": "base64_jpeg_string",
            "width": 640,
            "height": 480,
            "quality": 70
        }

        Returns:
            dict with 'image' as numpy array and metadata, or None if invalid
        """
        try:
            # Parse JSON
            msg = json.loads(data.decode("utf-8"))

            # Validate message type
            if msg.get("type") != "frame":
                return None

            # Decode base64 JPEG
            frame_b64 = msg.get("frame")
            if not frame_b64:
                return None

            # Decode base64 to bytes
            frame_bytes = base64.b64decode(frame_b64)

            # Decode JPEG to numpy array using OpenCV
            import cv2

            nparr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return None

            return {
                "image": image,
                "timestamp": msg.get("timestamp", time.time()),
                "width": msg.get("width"),
                "height": msg.get("height"),
                "quality": msg.get("quality", 70),
            }

        except Exception as e:
            print(f"Error decoding frame message: {e}")
            return None

    @staticmethod
    def encode_detection_message(
        faces: List[FaceDetection],
        processing_time_ms: float,
        frame_id: Optional[int] = None,
    ) -> bytes:
        """
        Encode detection results to send to Godot

        Args:
            faces: List of detected faces
            processing_time_ms: Time taken to process frame
            frame_id: Optional frame identifier

        Returns:
            JSON bytes ready to send via UDP
        """
        msg = {
            "type": "detection",
            "timestamp": time.time(),
            "processing_time_ms": round(processing_time_ms, 2),
            "faces": [face.to_dict() for face in faces],
            "frame_id": frame_id,
        }

        return json.dumps(msg).encode("utf-8")

    @staticmethod
    def encode_error_message(message: str, code: str = "ERROR") -> bytes:
        """
        Encode error message to send to Godot

        Args:
            message: Error description
            code: Error code identifier

        Returns:
            JSON bytes ready to send via UDP
        """
        msg = {
            "type": "error",
            "timestamp": time.time(),
            "message": message,
            "code": code,
        }

        return json.dumps(msg).encode("utf-8")

    @staticmethod
    def encode_status_message(status: str, details: Optional[Dict] = None) -> bytes:
        """
        Encode status message to send to Godot

        Args:
            status: Status string (e.g., "ready", "processing", "error")
            details: Optional additional details

        Returns:
            JSON bytes ready to send via UDP
        """
        msg = {
            "type": "status",
            "timestamp": time.time(),
            "status": status,
            "details": details or {},
        }

        return json.dumps(msg).encode("utf-8")


def create_face_detection_from_system_output(
    detection_dict: Dict, face_id: int
) -> FaceDetection:
    """
    Convert FaceDetectionSystem output to FaceDetection dataclass

    Args:
        detection_dict: Output from FaceDetectionSystem.process_image()
        face_id: Unique ID for this face

    Returns:
        FaceDetection object
    """
    bbox_data = detection_dict["bbox"]
    bbox = BoundingBox(
        x=bbox_data["x"], y=bbox_data["y"], w=bbox_data["w"], h=bbox_data["h"]
    )

    # Calculate center
    center = Point2D(
        x=bbox_data["x"] + bbox_data["w"] / 2, y=bbox_data["y"] + bbox_data["h"] / 2
    )

    # Extract eye data if available
    eye_data = detection_dict.get("eyes", {})
    left_eye = None
    right_eye = None
    eyes_detected = False

    if eye_data.get("left"):
        left_eye = Point2D(x=eye_data["left"]["x"], y=eye_data["left"]["y"])
        eyes_detected = True
    if eye_data.get("right"):
        right_eye = Point2D(x=eye_data["right"]["x"], y=eye_data["right"]["y"])
        eyes_detected = True

    eyes = EyeData(left=left_eye, right=right_eye, detected=eyes_detected)

    return FaceDetection(
        id=face_id,
        bbox=bbox,
        center=center,
        rotation_deg=detection_dict.get("rotation", 0.0),
        confidence=detection_dict.get("score", 0.0),
        eyes=eyes,
    )
