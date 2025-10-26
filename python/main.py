"""
Virtual Try-On Server Application
Main entry point for the Python webcam server.
Orchestrates webcam capture, body detection, encoding, and UDP streaming.
"""

import time
import signal
import sys
from webcam_capture import WebcamCapture
from body_detector import BodyDetector
from frame_encoder import FrameEncoder
from udp_server import UDPServer
from clothing_overlay import ClothingOverlay


class VirtualTryOnServer:
    """
    Main server application that coordinates all components.
    Captures webcam frames, detects upper-body, and streams to Godot clients.
    """
    
    def __init__(self):
        # Configuration
        self.target_fps = 30
        self.frame_delay = 1.0 / self.target_fps
        
        # Component initialization
        self.webcam = WebcamCapture(
            camera_index=0,
            width=640,
            height=480,
            fps=self.target_fps
        )
        self.detector = BodyDetector()
        self.clothing = ClothingOverlay(clothes_dir="clothes")
        self.encoder = FrameEncoder(jpeg_quality=85)
        self.server = UDPServer(host='127.0.0.1', port=9999)
        
        # Runtime state
        self.is_running = False
        
        # Statistics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def start(self):
        """
        Starts all components and begins the main processing loop.
        """
        print("=" * 60)
        print("Virtual Try-On Server - Python Side")
        print("=" * 60)
        
        # Start webcam
        if not self.webcam.start():
            print("[ERROR] Failed to start webcam")
            return False
        
        # Start UDP server
        if not self.server.start():
            print("[ERROR] Failed to start UDP server")
            self.webcam.stop()
            return False
        
        print("\n[INFO] Server is running...")
        print("[INFO] Waiting for Godot clients to connect...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        self.is_running = True
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start main processing loop
        self._main_loop()
        
        return True
    
    def _main_loop(self):
        """
        Main processing loop: capture → detect → overlay → encode → broadcast.
        Maintains target FPS and displays performance statistics.
        """
        while self.is_running:
            loop_start_time = time.time()
            
            # Step 1: Capture frame from webcam
            frame = self.webcam.read_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Step 2: Detect upper-body region
            bbox = self.detector.detect_upper_body(frame)
            
            # Step 3: Draw detection on frame for visualization
            display_frame = self.detector.draw_detection(frame.copy(), bbox)
            
            # Step 4: Overlay clothing on frame (NEW!)
            display_frame = self.clothing.overlay_clothing(display_frame, bbox)
            
            # Step 5: Encode frame to packet (bbox still sent for future use)
            packet = self.encoder.create_packet(display_frame, bbox)
            
            # Step 6: Broadcast packet to all connected clients
            if packet is not None:
                self.server.broadcast_packet(packet)
            
            # Update statistics
            self._update_statistics()
            
            # Maintain target FPS
            elapsed = time.time() - loop_start_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)
    
    def _update_statistics(self):
        """
        Updates and displays performance statistics.
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            
            # Display statistics
            client_count = self.server.get_client_count()
            print(f"\r[STATS] FPS: {self.current_fps:.1f} | Clients: {client_count} | Frames: {self.frame_count}", end="")
            
            # Reset counters
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _signal_handler(self, signum, frame):
        """
        Handles Ctrl+C for graceful shutdown.
        """
        print("\n\n[INFO] Shutdown signal received...")
        self.stop()
    
    def stop(self):
        """
        Stops all components and cleans up resources.
        """
        print("[INFO] Stopping server...")
        self.is_running = False
        
        self.webcam.stop()
        self.server.stop()
        
        print("[INFO] Server stopped successfully")
        print("=" * 60)
        sys.exit(0)


def main():
    """
    Entry point for the application.
    """
    server = VirtualTryOnServer()
    server.start()


if __name__ == "__main__":
    main()
