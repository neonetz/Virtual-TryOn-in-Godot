"""
Godot Integration Bridge - UDP Version
=======================================
High-performance UDP server for real-time face detection and mask overlay.
UDP is faster than HTTP for video streaming (no TCP overhead).

Usage:
    python godot_bridge_udp.py --port 8888 --models_dir models --mask assets/mask.png

Godot will send UDP packets with image data to port 8888.
Server responds with processed image data via UDP.

Protocol:
---------
CLIENT → SERVER (Godot → Python):
    [4 bytes: packet_id][4 bytes: data_length][N bytes: JPEG image data]

SERVER → CLIENT (Python → Godot):
    [4 bytes: packet_id][4 bytes: num_faces][4 bytes: data_length][face_data][JPEG image data]
    
Face data format (per face):
    [4 bytes: x][4 bytes: y][4 bytes: w][4 bytes: h][4 bytes: score (float)]
"""

import sys
import os
import socket
import struct
import threading
import time
import cv2
import numpy as np
import argparse
from queue import Queue, Empty

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import Inferencer


class UDPFaceDetectorServer:
    """
    UDP server for real-time face detection and mask overlay.
    Uses non-blocking UDP socket for high performance.
    """
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 8888,
        models_dir: str = 'models',
        mask_path: str = None,
        buffer_size: int = 65535
    ):
        """
        Initialize UDP server.
        
        Args:
            host: Host address to bind to
            port: UDP port number
            models_dir: Directory with trained models
            mask_path: Path to mask PNG
            buffer_size: UDP buffer size (max 65535 bytes)
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.running = False
        
        # Initialize inferencer
        print("\nInitializing face detector...")
        try:
            self.inferencer = Inferencer(
                models_dir=models_dir,
                mask_path=mask_path
            )
            print("✓ Face detector ready")
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            raise
        
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.1)  # Non-blocking with 100ms timeout
        
        # Statistics
        self.packets_received = 0
        self.packets_sent = 0
        self.errors = 0
        self.start_time = time.time()
        
        print(f"✓ UDP server bound to {self.host}:{self.port}")
    
    def decode_image(self, data: bytes) -> np.ndarray:
        """
        Decode JPEG image from bytes.
        
        Args:
            data: JPEG image bytes
            
        Returns:
            OpenCV image (BGR)
        """
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def encode_image(self, img: np.ndarray, quality: int = 85) -> bytes:
        """
        Encode OpenCV image to JPEG bytes.
        
        Args:
            img: OpenCV image (BGR)
            quality: JPEG quality (0-100)
            
        Returns:
            JPEG bytes
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        return buffer.tobytes()
    
    def process_request(self, data: bytes, addr: tuple) -> bytes:
        """
        Process detection request and create response.
        
        Args:
            data: Request data from client
            addr: Client address (host, port)
            
        Returns:
            Response bytes to send back
        """
        try:
            # Parse request: [packet_id][data_length][image_data]
            if len(data) < 8:
                raise ValueError("Invalid packet: too short")
            
            packet_id = struct.unpack('I', data[0:4])[0]
            data_length = struct.unpack('I', data[4:8])[0]
            image_data = data[8:8+data_length]
            
            # Decode image
            img = self.decode_image(image_data)
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Detect faces
            boxes, scores = self.inferencer.detect_faces(img)
            
            # Overlay mask if available
            output = img.copy()
            if self.inferencer.mask_overlay is not None and len(boxes) > 0:
                output = self.inferencer.mask_overlay.overlay_on_faces(output, boxes)
            
            # Encode result image
            result_image_data = self.encode_image(output, quality=85)
            
            # Build response: [packet_id][num_faces][data_length][face_data][image_data]
            num_faces = len(boxes)
            
            # Build face data: for each face: [x][y][w][h][score]
            face_data = b''
            for (x, y, w, h), score in zip(boxes, scores):
                face_data += struct.pack('iiiii', int(x), int(y), int(w), int(h), int(score * 1000))  # score as int (x1000)
            
            # Build full response
            response = struct.pack('III', packet_id, num_faces, len(result_image_data))
            response += face_data
            response += result_image_data
            
            return response
            
        except Exception as e:
            print(f"Error processing request: {e}")
            self.errors += 1
            # Return error response: packet_id=0, num_faces=0, data_length=0
            return struct.pack('III', 0, 0, 0)
    
    def start(self):
        """Start UDP server."""
        self.running = True
        print("\n" + "="*60)
        print("UDP FACE DETECTOR SERVER RUNNING")
        print("="*60)
        print(f"Listening on {self.host}:{self.port}")
        print("Waiting for packets from Godot...")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while self.running:
                try:
                    # Receive packet
                    data, addr = self.sock.recvfrom(self.buffer_size)
                    self.packets_received += 1
                    
                    # Process request
                    response = self.process_request(data, addr)
                    
                    # Send response back to client
                    self.sock.sendto(response, addr)
                    self.packets_sent += 1
                    
                    # Print statistics every 30 packets
                    if self.packets_received % 30 == 0:
                        self._print_stats()
                    
                except socket.timeout:
                    # Timeout is normal (non-blocking mode)
                    continue
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    self.errors += 1
                    
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop UDP server."""
        self.running = False
        self.sock.close()
        self._print_stats()
        print("\n✓ Server stopped")
    
    def _print_stats(self):
        """Print server statistics."""
        elapsed = time.time() - self.start_time
        fps = self.packets_received / elapsed if elapsed > 0 else 0
        print(f"[Stats] Packets: RX={self.packets_received} TX={self.packets_sent} "
              f"Errors={self.errors} FPS={fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Godot-Python CV Bridge (UDP)')
    parser.add_argument('--port', type=int, default=8888,
                       help='UDP port to listen on (default: 8888)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory with trained models')
    parser.add_argument('--mask', type=str, default=None,
                       help='Initial mask PNG path')
    parser.add_argument('--buffer_size', type=int, default=65535,
                       help='UDP buffer size (max 65535)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GODOT-PYTHON CV BRIDGE (UDP)")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Models: {args.models_dir}")
    if args.mask:
        print(f"Mask: {args.mask}")
    print(f"Buffer size: {args.buffer_size} bytes")
    print("="*60)
    
    # Create and start server
    server = UDPFaceDetectorServer(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        mask_path=args.mask,
        buffer_size=args.buffer_size
    )
    
    server.start()


if __name__ == '__main__':
    main()
