"""
Webcam UDP Server for Godot Integration
Captures webcam, processes face detection with mask overlay, streams to Godot via UDP

Architecture:
- Backend (Python): Captures webcam → Face detection → Mask overlay → Send JPEG via UDP
- Godot (Client): Receives JPEG → Display in TextureRect

Protocol:
- Python sends: {"frame_id": int, "image": base64_jpeg, "fps": float, "faces": int}
- Godot receives and displays
"""

import cv2
import socket
import json
import base64
import time
import argparse
import logging
import numpy as np
from typing import Optional
import os

from pipelines.infer import FaceDetectionSystem
from pipelines.utils import setup_logging, create_fps_counter

logger = logging.getLogger(__name__)


class WebcamUDPServer:
    """
    Webcam capture + Face detection + UDP streaming server
    All processing done in Python, Godot just displays
    """
    
    def __init__(self,
                 detection_system: FaceDetectionSystem,
                 camera_id: int = 0,
                 host: str = "127.0.0.1",
                 port: int = 5555,
                 jpeg_quality: int = 85,
                 target_fps: int = 30):
        """
        Initialize webcam UDP server
        
        Args:
            detection_system: Face detection system
            camera_id: Webcam device ID
            host: Godot client host
            port: Godot client port
            jpeg_quality: JPEG compression quality (0-100)
            target_fps: Target frame rate
        """
        self.detection_system = detection_system
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"✓ UDP socket created (target: {host}:{port})")
        
        # Statistics
        self.frame_count = 0
        self.sent_count = 0
        self.error_count = 0
        self.fps_counter = create_fps_counter()
        
        # Camera
        self.cap = None
    
    def open_camera(self) -> bool:
        """Open webcam"""
        logger.info(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            return False
        
        # Get camera info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"✓ Camera opened: {width}×{height}")
        
        return True
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 JPEG with auto-resize for UDP compatibility"""
        try:
            # Resize if frame is too large (to keep UDP packet under 65KB)
            # 640x480 @ quality 60 = ~25-35KB (safe for UDP)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                new_w = 640
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized frame: {w}×{h} → {new_w}×{new_h}")
            
            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                logger.error("JPEG encoding failed")
                return ""
            
            # Encode to base64
            base64_str = base64.b64encode(jpeg_buffer).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return ""
    
    def send_frame(self, frame: np.ndarray, face_count: int, processing_time: float):
        """Send processed frame to Godot"""
        try:
            # Encode frame
            base64_image = self.encode_frame(frame)
            
            if not base64_image:
                self.error_count += 1
                return
            
            # Calculate FPS
            fps = self.fps_counter()  # Update FPS counter (no arguments!)
            
            # Build packet
            packet = {
                "frame_id": self.frame_count,
                "image": base64_image,
                "fps": round(fps, 1),
                "faces": face_count,
                "processing_ms": round(processing_time * 1000, 1),
                "timestamp": time.time()
            }
            
            # Convert to JSON
            json_str = json.dumps(packet)
            json_bytes = json_str.encode('utf-8')
            
            # Check packet size
            packet_size = len(json_bytes)
            if packet_size > 65000:  # UDP limit ~65KB
                logger.warning(f"Packet too large: {packet_size} bytes (reducing quality recommended)")
            
            # Send via UDP
            self.socket.sendto(json_bytes, (self.host, self.port))
            self.sent_count += 1
            
            logger.debug(f"Sent frame {self.frame_count}: {packet_size/1024:.1f} KB, "
                        f"{face_count} faces, {processing_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.error_count += 1
    
    def run(self, enable_rotation: bool = True, alpha: float = 0.95, show_preview: bool = False):
        """
        Run server main loop
        
        Args:
            enable_rotation: Enable mask rotation based on eyes
            alpha: Mask opacity (0-1)
            show_preview: Show local preview window
        """
        if not self.open_camera():
            return
        
        logger.info("="*60)
        logger.info("WEBCAM UDP SERVER RUNNING")
        logger.info("="*60)
        logger.info(f"Streaming to: {self.host}:{self.port}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"JPEG quality: {self.jpeg_quality}")
        logger.info(f"Preview: {'ON' if show_preview else 'OFF'}")
        logger.info("="*60)
        logger.info("Press 'q' to quit")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Process frame (face detection + mask overlay)
                process_start = time.time()
                result_frame, detections = self.detection_system.process_image(
                    frame,
                    enable_rotation=enable_rotation,
                    alpha_multiplier=alpha
                )
                process_time = time.time() - process_start
                
                face_count = len(detections)
                
                # Send to Godot
                self.send_frame(result_frame, face_count, process_time)
                
                # Show local preview if enabled
                if show_preview:
                    # Add info overlay
                    display_frame = result_frame.copy()
                    
                    # FPS counter - call function to get FPS
                    fps = self.fps_counter() if self.fps_counter else 0.0
                    cv2.putText(display_frame, f"FPS: {fps:.1f} (target: {self.target_fps})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Face count
                    cv2.putText(display_frame, f"Faces: {face_count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Processing time
                    cv2.putText(display_frame, f"Process: {process_time*1000:.1f}ms", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Sent count
                    cv2.putText(display_frame, f"Sent: {self.sent_count}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Webcam UDP Server (Preview)", display_frame)
                    
                    # Handle keyboard (BEFORE sleep calculation)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested")
                        break
                
                # Frame rate limiting - calculate AFTER all processing
                loop_time = time.time() - loop_start
                sleep_time = self.frame_time - loop_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Log if we're running slower than target
                    logger.debug(f"Running slow: {loop_time*1000:.1f}ms (target: {self.frame_time*1000:.1f}ms)")
                
                # Stats logging every 30 frames
                if self.frame_count % 30 == 0:
                    fps = self.fps_counter() if self.fps_counter else 0.0
                    logger.info(f"Stats: frames={self.frame_count}, sent={self.sent_count}, "
                              f"errors={self.error_count}, fps={fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info("\n\nShutting down...")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            self.socket.close()
            
            logger.info("="*60)
            logger.info("SERVER STOPPED")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Sent: {self.sent_count}")
            logger.info(f"Errors: {self.error_count}")
            logger.info("="*60)


def get_opencv_data_path(filename: str) -> str:
    """Get path to OpenCV Haar cascade XML file"""
    opencv_data = cv2.data.haarcascades
    cascade_path = os.path.join(opencv_data, filename)
    
    if os.path.exists(cascade_path):
        return cascade_path
    
    local_path = os.path.join("assets", "cascades", filename)
    if os.path.exists(local_path):
        return local_path
    
    raise FileNotFoundError(f"Cannot find cascade file: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Webcam UDP Server - Face Detection Streaming to Godot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream to local Godot (default - safe quality for UDP)
  python webcam_udp_server.py --camera 0 --host 127.0.0.1 --port 5555
  
  # With preview window
  python webcam_udp_server.py --camera 0 --preview
  
  # Higher quality (may exceed UDP limit if resolution is high)
  python webcam_udp_server.py --camera 0 --quality 80 --fps 24
        """
    )
    
    # Model files
    parser.add_argument("--svm-model", default="models/svm_model.pkl",
                       help="Path to SVM model")
    parser.add_argument("--bovw-encoder", default="models/bovw_encoder.pkl",
                       help="Path to BoVW encoder")
    parser.add_argument("--scaler", default="models/scaler.pkl",
                       help="Path to scaler")
    parser.add_argument("--mask", default="assets/mask.png",
                       help="Path to mask PNG")
    
    # Camera settings
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID")
    
    # Network settings
    parser.add_argument("--host", default="127.0.0.1",
                       help="Godot client host (127.0.0.1 for local)")
    parser.add_argument("--port", type=int, default=5555,
                       help="Godot client UDP port")
    
    # Stream settings
    parser.add_argument("--quality", type=int, default=60,
                       help="JPEG quality (0-100, default 60 for UDP compatibility)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS (default 30)")
    
    # Detection settings
    parser.add_argument("--alpha", type=float, default=0.95,
                       help="Mask opacity (0-1)")
    parser.add_argument("--enable-rotation", action="store_true",
                       help="Enable mask rotation based on eyes")
    parser.add_argument("--max-keypoints", type=int, default=500,
                       help="Max ORB keypoints")
    parser.add_argument("--smoothing", type=float, default=0.3,
                       help="Temporal smoothing factor (0=instant, 1=full smooth)")
    
    # Display
    parser.add_argument("--preview", action="store_true",
                       help="Show local preview window")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check model files
    if not os.path.exists(args.svm_model):
        print(f"✗ Error: SVM model not found: {args.svm_model}")
        return 1
    if not os.path.exists(args.bovw_encoder):
        print(f"✗ Error: BoVW encoder not found: {args.bovw_encoder}")
        return 1
    if not os.path.exists(args.scaler):
        print(f"✗ Error: Scaler not found: {args.scaler}")
        return 1
    if not os.path.exists(args.mask):
        print(f"✗ Error: Mask not found: {args.mask}")
        return 1
    
    # Get cascades
    cascade_face = get_opencv_data_path("haarcascade_frontalface_default.xml")
    cascade_eye = None  # Use custom eye detector
    
    logger.info("\n" + "="*60)
    logger.info("INITIALIZING WEBCAM UDP SERVER")
    logger.info("="*60)
    
    # Initialize detection system
    logger.info("Loading detection system...")
    logger.info(f"  SVM model: {args.svm_model}")
    logger.info(f"  BoVW encoder: {args.bovw_encoder}")
    logger.info(f"  Scaler: {args.scaler}")
    logger.info(f"  Mask: {args.mask}")
    logger.info(f"  Smoothing factor: {args.smoothing}")
    logger.info(f"  Eye detection: Custom ORB detector")
    
    try:
        detection_system = FaceDetectionSystem(
            svm_model_path=args.svm_model,
            bovw_encoder_path=args.bovw_encoder,
            scaler_path=args.scaler,
            cascade_face_path=cascade_face,
            mask_path=args.mask,
            cascade_eye_path=cascade_eye,
            max_keypoints=args.max_keypoints,
            use_custom_eye_detector=True,
            custom_eye_method="orb"
        )
    except Exception as e:
        logger.error(f"✗ Failed to initialize detection system: {e}")
        return 1
    
    logger.info("✓ Detection system ready")
    
    # Create and run server
    server = WebcamUDPServer(
        detection_system=detection_system,
        camera_id=args.camera,
        host=args.host,
        port=args.port,
        jpeg_quality=args.quality,
        target_fps=args.fps
    )
    
    server.run(
        enable_rotation=args.enable_rotation,
        alpha=args.alpha,
        show_preview=args.preview
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
