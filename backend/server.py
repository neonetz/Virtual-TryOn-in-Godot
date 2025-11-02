"""
UDP Server for Godot Integration
Receives images from Godot, processes with face detection, returns results
"""

import socket
import json
import base64
import cv2
import numpy as np
import argparse
import logging
from typing import Dict, Tuple, Optional

from pipelines.infer import FaceDetectionSystem
from pipelines.utils import setup_logging

logger = logging.getLogger(__name__)


class FaceDetectionUDPServer:
    """UDP server for face detection with mask overlay"""
    
    def __init__(self,
                 svm_model_path: str,
                 bovw_encoder_path: str,
                 scaler_path: str,
                 cascade_face_path: str,
                 mask_path: str,
                 cascade_eye_path: Optional[str] = None,
                 host: str = "0.0.0.0",
                 port: int = 5555,
                 buffer_size: int = 262144):
        """
        Initialize UDP server
        
        Args:
            svm_model_path: Path to svm_model.pkl
            bovw_encoder_path: Path to bovw_encoder.pkl
            scaler_path: Path to scaler.pkl
            cascade_face_path: Path to Haar cascade
            mask_path: Path to mask.png
            cascade_eye_path: Optional eye cascade
            host: Server host (0.0.0.0 = all interfaces)
            port: Server port
            buffer_size: UDP buffer size (large for images)
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        
        # Initialize detection system
        logger.info("Initializing face detection system...")
        self.detection_system = FaceDetectionSystem(
            svm_model_path,
            bovw_encoder_path,
            scaler_path,
            cascade_face_path,
            mask_path,
            cascade_eye_path
        )
        logger.info("✓ Detection system ready")
        
        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        logger.info(f"✓ UDP server listening on {self.host}:{self.port}")
        
        # Statistics
        self.total_requests = 0
        self.total_responses = 0
        self.total_errors = 0
    
    def decode_image(self, base64_str: str) -> Optional[np.ndarray]:
        """Decode base64 JPEG image to numpy array"""
        try:
            # Decode base64
            jpeg_bytes = base64.b64decode(base64_str)
            
            # Decode JPEG
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return img
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None
    
    def encode_image(self, img: np.ndarray, quality: int = 85) -> str:
        """Encode numpy array to base64 JPEG"""
        try:
            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, jpeg_buffer = cv2.imencode('.jpg', img, encode_param)
            
            # Encode to base64
            base64_str = base64.b64encode(jpeg_buffer).decode('utf-8')
            
            return base64_str
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return ""
    
    def process_request(self, request: Dict) -> Dict:
        """Process detection request"""
        frame_id = request.get("frame_id", -1)
        base64_image = request.get("image", "")
        mask_type = request.get("mask_type", "default")
        options = request.get("options", {})
        
        logger.debug(f"Processing frame {frame_id}")
        
        # Decode image
        img = self.decode_image(base64_image)
        
        if img is None:
            return {
                "frame_id": frame_id,
                "status": "error",
                "error": "Failed to decode image"
            }
        
        # Process image
        import time
        start_time = time.time()
        
        result_img, detections = self.detection_system.process_image(
            img,
            enable_rotation=True,
            alpha_multiplier=0.95
        )
        
        processing_ms = (time.time() - start_time) * 1000
        
        # Build response
        if len(detections) == 0:
            return {
                "frame_id": frame_id,
                "status": "no_face",
                "processing_time_ms": processing_ms
            }
        
        # Encode result image
        quality = options.get("quality", 85)
        overlay_base64 = self.encode_image(result_img, quality)
        
        # Format face data
        faces_data = []
        for det in detections:
            bbox = det['bbox']
            faces_data.append({
                "bbox": {
                    "x": bbox['x'],
                    "y": bbox['y'],
                    "w": bbox['w'],
                    "h": bbox['h']
                },
                "confidence": det['score']
            })
        
        return {
            "frame_id": frame_id,
            "status": "success",
            "faces": faces_data,
            "overlay_image": overlay_base64,
            "processing_time_ms": processing_ms
        }
    
    def run(self):
        """Run server main loop"""
        logger.info("="*60)
        logger.info("UDP SERVER READY")
        logger.info("="*60)
        logger.info(f"Listening on {self.host}:{self.port}")
        logger.info("Waiting for requests from Godot...")
        logger.info("="*60)
        
        try:
            while True:
                # Receive request
                data, client_addr = self.socket.recvfrom(self.buffer_size)
                self.total_requests += 1
                
                try:
                    # Parse JSON request
                    json_str = data.decode('utf-8')
                    request = json.loads(json_str)
                    
                    logger.info(f"Request from {client_addr}: frame_id={request.get('frame_id', '?')}")
                    
                    # Process request
                    response = self.process_request(request)
                    
                    # Send response
                    response_json = json.dumps(response)
                    response_bytes = response_json.encode('utf-8')
                    
                    self.socket.sendto(response_bytes, client_addr)
                    self.total_responses += 1
                    
                    logger.info(f"Response sent: status={response['status']}, "
                              f"time={response.get('processing_time_ms', 0):.1f}ms")
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    self.total_errors += 1
                    
                    error_response = {
                        "frame_id": -1,
                        "status": "error",
                        "error": "Invalid JSON"
                    }
                    self.socket.sendto(json.dumps(error_response).encode(), client_addr)
                
                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    self.total_errors += 1
                    
                    error_response = {
                        "frame_id": -1,
                        "status": "error",
                        "error": str(e)
                    }
                    self.socket.sendto(json.dumps(error_response).encode(), client_addr)
                
                # Log statistics every 10 requests
                if self.total_requests % 10 == 0:
                    logger.info(f"Stats: requests={self.total_requests}, "
                              f"responses={self.total_responses}, "
                              f"errors={self.total_errors}")
        
        except KeyboardInterrupt:
            logger.info("\n\nShutting down server...")
        
        finally:
            self.socket.close()
            logger.info("Server stopped")
            logger.info(f"Final stats: requests={self.total_requests}, "
                       f"responses={self.total_responses}, "
                       f"errors={self.total_errors}")


def get_opencv_cascade(name: str) -> str:
    """Get path to OpenCV cascade file"""
    import os
    cascade_path = os.path.join(cv2.data.haarcascades, name)
    
    if os.path.exists(cascade_path):
        return cascade_path
    
    # Try local path
    local_path = os.path.join("assets", "cascades", name)
    if os.path.exists(local_path):
        return local_path
    
    raise FileNotFoundError(f"Cascade not found: {name}")


def main():
    parser = argparse.ArgumentParser(description="UDP Server for Godot Face Detection")
    
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--svm-model", default="models/svm_model.pkl",
                       help="Path to SVM model")
    parser.add_argument("--bovw-encoder", default="models/bovw_encoder.pkl",
                       help="Path to BoVW encoder")
    parser.add_argument("--scaler", default="models/scaler.pkl",
                       help="Path to scaler")
    parser.add_argument("--mask", default="assets/mask.png",
                       help="Path to mask image")
    parser.add_argument("--enable-rotation", action="store_true",
                       help="Enable mask rotation (loads eye cascade)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get cascades
    cascade_face = get_opencv_cascade("haarcascade_frontalface_default.xml")
    cascade_eye = None
    if args.enable_rotation:
        try:
            cascade_eye = get_opencv_cascade("haarcascade_eye.xml")
        except FileNotFoundError:
            logger.warning("Eye cascade not found, rotation disabled")
    
    # Create and run server
    server = FaceDetectionUDPServer(
        svm_model_path=args.svm_model,
        bovw_encoder_path=args.bovw_encoder,
        scaler_path=args.scaler,
        cascade_face_path=cascade_face,
        mask_path=args.mask,
        cascade_eye_path=cascade_eye,
        host=args.host,
        port=args.port
    )
    
    server.run()


if __name__ == "__main__":
    main()
