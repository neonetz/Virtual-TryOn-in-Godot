"""
UDP Face Detection Server
Main server that receives frames from Godot and sends back detection results
"""

import socket
import sys
import os
import time
import logging
import signal
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from udp_server.config import ServerConfig, DevelopmentConfig, ProductionConfig
from udp_server.protocol import (
    MessageProtocol,
    create_face_detection_from_system_output,
)
from udp_server.frame_handler import FrameProcessor
from pipelines.infer import FaceDetectionSystem
from pipelines.utils import setup_logging

logger = logging.getLogger(__name__)


class FaceDetectionUDPServer:
    """
    UDP Server for real-time face detection
    Receives frames from Godot, processes them, sends back detection results
    """

    def __init__(self, config: ServerConfig):
        """Initialize server with configuration"""
        self.config = config
        self.running = False

        # Sockets
        self.recv_socket: Optional[socket.socket] = None
        self.send_socket: Optional[socket.socket] = None
        self.client_address: Optional[tuple] = None

        # Face detection system
        self.face_system: Optional[FaceDetectionSystem] = None

        # Frame processor
        self.frame_processor: Optional[FrameProcessor] = None

        # Statistics
        self.frames_received = 0
        self.messages_sent = 0
        self.start_time = None

        logger.info("Face Detection UDP Server initialized")

    def _init_detection_system(self):
        """Initialize face detection system"""
        logger.info("Initializing face detection system...")

        # Get cascade path
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from face_detector_cli import get_opencv_data_path

        cascade_face = get_opencv_data_path(self.config.CASCADE_FACE)

        try:
            # Import here to avoid circular dependency
            from pipelines.infer import FaceDetectionSystem

            self.face_system = FaceDetectionSystem(
                svm_model_path=self.config.SVM_MODEL_PATH,
                bovw_encoder_path=self.config.BOVW_ENCODER_PATH,
                scaler_path=self.config.SCALER_PATH,
                cascade_face_path=cascade_face,
                mask_path=self.config.MASK_PATH,  # Use mask path from config
                cascade_eye_path=self.config.CASCADE_EYE,
                max_keypoints=self.config.MAX_KEYPOINTS,
                use_custom_eye_detector=self.config.USE_CUSTOM_EYE_DETECTOR,
                custom_eye_method=self.config.CUSTOM_EYE_METHOD,
            )
            logger.info("✓ Face detection system ready (mask not used for UDP)")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to initialize face detection: {e}")
            logger.error(f"   Troubleshooting:")
            logger.error(f"   1. Make sure you're running from backend/ directory")
            logger.error(f"   2. Check mask file exists: {self.config.MASK_PATH}")
            logger.error(f"   3. Check models exist in models/ directory")
            import traceback

            traceback.print_exc()
            return False

    def _process_frame_wrapper(self, image):
        """
        Wrapper for face detection processing
        Returns detections list
        """
        try:
            # Process without mask overlay
            _, detections = self.face_system.process_image(
                image,
                enable_rotation=self.config.ENABLE_ROTATION,
                alpha_multiplier=0,  # No mask overlay
            )
            return detections
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def _setup_sockets(self):
        """Setup UDP sockets"""
        try:
            # Socket for receiving frames from Godot
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.recv_socket.bind((self.config.HOST, self.config.PORT_RECEIVE))
            # Set timeout for graceful shutdown
            self.recv_socket.settimeout(1.0)

            # Socket for sending results to Godot
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            logger.info(f"✓ Sockets created:")
            logger.info(
                f"  Receiving on: {self.config.HOST}:{self.config.PORT_RECEIVE}"
            )
            logger.info(f"  Sending to: <client>:{self.config.PORT_SEND}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup sockets: {e}")
            return False

    def start(self):
        """Start the UDP server"""
        logger.info("=" * 60)
        logger.info("STARTING FACE DETECTION UDP SERVER")
        logger.info("=" * 60)

        # Validate config
        errors = self.config.validate()
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        # Initialize detection system
        if not self._init_detection_system():
            return False

        # Setup sockets
        if not self._setup_sockets():
            return False

        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            process_func=self._process_frame_wrapper,
            max_queue_size=self.config.MAX_QUEUE_SIZE,
            worker_threads=self.config.WORKER_THREADS,
        )
        self.frame_processor.start()

        # Start main loop
        self.running = True
        self.start_time = time.time()

        logger.info("\n" + "=" * 60)
        logger.info("SERVER READY - Waiting for frames from Godot...")
        logger.info("=" * 60)
        logger.info(f"Listening on: {self.config.HOST}:{self.config.PORT_RECEIVE}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60 + "\n")

        self._main_loop()

        return True

    def _main_loop(self):
        """Main server loop"""
        last_result_check = time.time()
        last_stats_print = time.time()

        while self.running:
            try:
                # Receive frame from Godot
                data, address = self.recv_socket.recvfrom(65535)  # Max UDP packet size

                # Store client address for sending responses
                if self.client_address != address:
                    self.client_address = address
                    logger.info(f"New client connected: {address}")

                # Decode frame
                frame_data = MessageProtocol.decode_frame_message(data)

                if frame_data:
                    self.frames_received += 1

                    # Submit for processing
                    success = self.frame_processor.submit_frame(
                        frame_data["image"],
                        metadata={"timestamp": frame_data["timestamp"]},
                    )

                    if not success:
                        logger.warning("Failed to queue frame")
                else:
                    logger.warning("Invalid frame data received")

                # Check for processing results (non-blocking)
                current_time = time.time()
                if current_time - last_result_check > 0.01:  # Check every 10ms
                    result = self.frame_processor.get_latest_result()

                    if result and self.client_address:
                        self._send_detection_result(result)

                    last_result_check = current_time

                # Print stats periodically
                if current_time - last_stats_print > 10.0:  # Every 10 seconds
                    self._print_stats()
                    last_stats_print = current_time

            except socket.timeout:
                # Check for results even when no frames received
                result = self.frame_processor.get_latest_result()
                if result and self.client_address:
                    self._send_detection_result(result)
                continue

            except KeyboardInterrupt:
                logger.info("\nShutdown requested...")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

    def _send_detection_result(self, result: dict):
        """Send detection result to Godot"""
        try:
            # Convert detections to protocol format
            detections_raw = result["result"]
            faces = []

            for i, det in enumerate(detections_raw):
                face = create_face_detection_from_system_output(det, face_id=i)
                faces.append(face)

            # Encode message
            message = MessageProtocol.encode_detection_message(
                faces=faces, processing_time_ms=result["processing_time_ms"]
            )

            # Send to client
            self.send_socket.sendto(
                message, (self.client_address[0], self.config.PORT_SEND)
            )
            self.messages_sent += 1

            # Log detection details
            if faces:
                logger.debug(
                    f"Sent {len(faces)} face(s) to {self.client_address[0]}:{self.config.PORT_SEND}"
                )

        except Exception as e:
            logger.error(f"Error sending result: {e}")

    def _print_stats(self):
        """Print server statistics"""
        uptime = time.time() - self.start_time
        proc_stats = self.frame_processor.get_stats()

        logger.info("\n" + "=" * 60)
        logger.info("SERVER STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Uptime: {uptime:.1f}s")
        logger.info(f"Frames received: {self.frames_received}")
        logger.info(f"Messages sent: {self.messages_sent}")
        logger.info(f"Frames processed: {proc_stats['frames_processed']}")
        logger.info(f"Frames dropped: {proc_stats['frames_dropped']}")
        logger.info(
            f"Avg processing time: {proc_stats['avg_processing_time_ms']:.1f}ms"
        )
        logger.info(f"Queue size: {proc_stats['queue_size']}")

        if self.frames_received > 0:
            fps = self.frames_received / uptime
            logger.info(f"Receive rate: {fps:.1f} FPS")

        logger.info("=" * 60 + "\n")

    def stop(self):
        """Stop the server gracefully"""
        logger.info("Stopping server...")

        self.running = False

        # Stop frame processor
        if self.frame_processor:
            self.frame_processor.stop()

        # Close sockets
        if self.recv_socket:
            self.recv_socket.close()
        if self.send_socket:
            self.send_socket.close()

        # Final stats
        self._print_stats()

        logger.info("Server stopped")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Face Detection UDP Server for Godot Integration"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Server mode (dev=localhost, prod=LAN)",
    )
    parser.add_argument("--host", help="Override host address")
    parser.add_argument("--port-recv", type=int, help="Override receive port")
    parser.add_argument("--port-send", type=int, help="Override send port")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Select config
    if args.mode == "dev":
        config = DevelopmentConfig()
    else:
        config = ProductionConfig()

    # Override from args
    if args.host:
        config.HOST = args.host
    if args.port_recv:
        config.PORT_RECEIVE = args.port_recv
    if args.port_send:
        config.PORT_SEND = args.port_send

    config.LOG_LEVEL = args.log_level

    # Setup logging
    setup_logging(config.LOG_LEVEL)

    # Create and start server
    server = FaceDetectionUDPServer(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    try:
        server.start()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
