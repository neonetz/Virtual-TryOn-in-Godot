"""
UDP Server for Virtual Try-On
Handles communication with Godot frontend via JSON over UDP
"""

import socket
import json
import base64
import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.face_detector import FaceDetector
from inference.mask_overlay import MaskManager, LandmarkEstimator


class UDPServer:
    """UDP Server for real-time face detection and mask overlay"""

    def __init__(self, config: Dict):
        """
        Initialize UDP server

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 5001)
        self.buffer_size = config.get("buffer_size", 131072)

        # Initialize socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))

        # Threading
        self.running = False
        self.request_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.response_queue = queue.Queue(maxsize=5)

        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_stats_time = time.time()

        # ROI tracking for performance
        self.previous_detection = None
        self.frame_counter = 0
        self.full_detection_interval = 10  # Full detection every N frames

        # Initialize detector and mask manager
        self.detector = None
        self.mask_manager = None

        print(f"✓ UDP Server initialized")
        print(f"  Listening on {self.host}:{self.port}")
        print(f"  Buffer size: {self.buffer_size} bytes")

    def load_models(self, model_path: str, scaler_path: str, config_path: str):
        """Load face detection model"""
        print("\nLoading face detection model...")
        self.detector = FaceDetector(model_path, scaler_path, config_path)
        print("✓ Model loaded")

    def load_masks(self, mask_configs: Dict):
        """
        Load Halloween masks

        Args:
            mask_configs: Dictionary mapping mask_id to mask_path
                Example: {'zombie': 'assets/masks/zombie.png'}
        """
        print("\nLoading masks...")
        self.mask_manager = MaskManager()

        for mask_id, mask_path in mask_configs.items():
            self.mask_manager.load_mask(mask_id, mask_path)

        print(f"✓ Loaded {len(self.mask_manager.list_masks())} masks")

    def decode_image(self, base64_str: str) -> Optional[np.ndarray]:
        """Decode base64 image string to numpy array"""
        try:
            # Decode base64
            img_bytes = base64.b64decode(base64_str)

            # Convert to numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)

            # Decode image
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def encode_image(self, img: np.ndarray, quality: int = 85) -> str:
        """Encode numpy array to base64 JPEG string"""
        try:
            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode(".jpg", img, encode_params)

            # Convert to base64
            base64_str = base64.b64encode(buffer).decode("utf-8")

            return base64_str
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""

    def process_request(self, request: Dict) -> Dict:
        """
        Process a single request

        Args:
            request: Request dictionary with keys:
                - frame_id: int
                - image: base64 encoded string
                - mask_type: str
                - options: dict (optional)

        Returns:
            Response dictionary
        """
        start_time = time.time()

        frame_id = request.get("frame_id", 0)
        mask_type = request.get("mask_type", "zombie")
        options = request.get("options", {})

        # Decode image
        img = self.decode_image(request["image"])
        if img is None:
            return {
                "frame_id": frame_id,
                "status": "error",
                "error": "Failed to decode image",
                "faces": [],
                "overlay_image": "",
                "processing_time_ms": 0,
            }

        # Detect faces with ROI tracking
        self.frame_counter += 1
        use_roi_tracking = self.frame_counter % self.full_detection_interval != 0

        if use_roi_tracking and self.previous_detection is not None:
            detections = self.detector.detect_with_roi_tracking(
                img, self.previous_detection
            )
        else:
            detections = self.detector.detect(img, single_face=True)

        # Update previous detection
        if len(detections) > 0:
            self.previous_detection = detections[0]

        # Check if face found
        if len(detections) == 0:
            return {
                "frame_id": frame_id,
                "status": "no_face",
                "faces": [],
                "overlay_image": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # Add landmarks to detections
        for detection in detections:
            landmarks = LandmarkEstimator.estimate_5_points(detection["bbox"])
            detection["landmarks"] = landmarks

        # Apply mask
        result_img = self.mask_manager.apply_mask(
            img, detections[0], mask_type, use_perspective=True, alpha=0.95
        )

        if result_img is None:
            return {
                "frame_id": frame_id,
                "status": "error",
                "error": f"Mask {mask_type} not found",
                "faces": detections,
                "overlay_image": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # Encode result
        quality = options.get("quality", 85)
        overlay_image_b64 = self.encode_image(result_img, quality)

        processing_time = (time.time() - start_time) * 1000

        # Update stats
        self.frame_count += 1
        self.total_processing_time += processing_time

        # Prepare response
        response = {
            "frame_id": frame_id,
            "status": "success",
            "faces": detections,
            "overlay_image": overlay_image_b64,
            "processing_time_ms": processing_time,
        }

        return response

    def receive_thread(self):
        """Thread for receiving requests"""
        print("✓ Receive thread started")

        while self.running:
            try:
                # Receive data
                data, addr = self.socket.recvfrom(self.buffer_size)

                # Parse JSON
                request = json.loads(data.decode("utf-8"))

                # Add to queue (drop if full)
                try:
                    self.request_queue.put_nowait((request, addr))
                except queue.Full:
                    print("Warning: Request queue full, dropping frame")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving request: {e}")

    def process_thread(self):
        """Thread for processing requests"""
        print("✓ Process thread started")

        while self.running:
            try:
                # Get request from queue
                request, addr = self.request_queue.get(timeout=0.1)

                # Process request
                response = self.process_request(request)

                # Add to response queue
                try:
                    self.response_queue.put_nowait((response, addr))
                except queue.Full:
                    print("Warning: Response queue full, dropping frame")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing request: {e}")

    def send_thread(self):
        """Thread for sending responses"""
        print("✓ Send thread started")

        while self.running:
            try:
                # Get response from queue
                response, addr = self.response_queue.get(timeout=0.1)

                # Serialize response
                response_json = json.dumps(response)
                response_bytes = response_json.encode("utf-8")

                # Send response
                self.socket.sendto(response_bytes, addr)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error sending response: {e}")

    def stats_thread(self):
        """Thread for printing performance statistics"""
        while self.running:
            time.sleep(5)  # Print stats every 5 seconds

            current_time = time.time()
            elapsed = current_time - self.last_stats_time

            if self.frame_count > 0:
                fps = self.frame_count / elapsed
                avg_time = self.total_processing_time / self.frame_count

                print(f"\n--- Performance Stats ---")
                print(f"FPS: {fps:.1f}")
                print(f"Avg processing time: {avg_time:.1f}ms")
                print(f"Frames processed: {self.frame_count}")
                print(
                    f"Queue sizes - Request: {self.request_queue.qsize()}, Response: {self.response_queue.qsize()}"
                )

                # Reset counters
                self.frame_count = 0
                self.total_processing_time = 0
                self.last_stats_time = current_time

    def start(self):
        """Start the UDP server"""
        if self.detector is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if self.mask_manager is None:
            raise RuntimeError("Masks not loaded. Call load_masks() first.")

        print("\n" + "=" * 50)
        print("Starting UDP Server")
        print("=" * 50)

        self.running = True

        # Set socket timeout for graceful shutdown
        self.socket.settimeout(1.0)

        # Start threads
        threads = [
            threading.Thread(target=self.receive_thread, daemon=True),
            threading.Thread(target=self.process_thread, daemon=True),
            threading.Thread(target=self.send_thread, daemon=True),
            threading.Thread(target=self.stats_thread, daemon=True),
        ]

        for thread in threads:
            thread.start()

        print("\n✓ Server is running")
        print("Press Ctrl+C to stop\n")

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            self.stop()

    def stop(self):
        """Stop the UDP server"""
        self.running = False
        time.sleep(2)  # Wait for threads to finish
        self.socket.close()
        print("✓ Server stopped")


def main():
    """Main entry point for UDP server"""
    import argparse

    parser = argparse.ArgumentParser(description="Virtual Try-On UDP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument(
        "--model", required=True, help="Path to face detector model (.pkl)"
    )
    parser.add_argument("--scaler", required=True, help="Path to scaler (.pkl)")
    parser.add_argument("--config", required=True, help="Path to config (.json)")
    parser.add_argument("--masks", required=True, help="Path to masks directory")

    args = parser.parse_args()

    # Server configuration
    server_config = {"host": args.host, "port": args.port, "buffer_size": 131072}

    # Initialize server
    server = UDPServer(server_config)

    # Load models
    server.load_models(args.model, args.scaler, args.config)

    # Load masks
    import glob

    mask_files = glob.glob(os.path.join(args.masks, "*.png"))
    mask_configs = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}
    server.load_masks(mask_configs)

    # Start server
    server.start()


if __name__ == "__main__":
    main()
