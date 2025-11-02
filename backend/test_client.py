"""
UDP Test Client
Test the UDP server without needing Godot
Simulates Godot's camera by sending test images or webcam frames
"""

import socket
import json
import base64
import time
import cv2
import argparse
import sys
import os
from pathlib import Path


class UDPTestClient:
    """Test client that simulates Godot"""

    def __init__(self, server_host="127.0.0.1", server_port=5555, receive_port=5556):
        self.server_host = server_host
        self.server_port = server_port
        self.receive_port = receive_port

        # Sockets
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_socket.bind(("0.0.0.0", receive_port))
        self.recv_socket.settimeout(0.1)  # Non-blocking

        print(f"✓ Test client initialized")
        print(f"  Sending to: {server_host}:{server_port}")
        print(f"  Receiving on: 0.0.0.0:{receive_port}")

    def encode_frame(self, image, width=640, height=480, quality=70):
        """Encode frame to JSON message"""
        # Resize image
        resized = cv2.resize(image, (width, height))

        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", resized, encode_param)

        # Base64 encode
        jpg_b64 = base64.b64encode(buffer).decode("utf-8")

        # Create message
        msg = {
            "type": "frame",
            "timestamp": time.time(),
            "frame": jpg_b64,
            "width": width,
            "height": height,
            "quality": quality,
        }

        return json.dumps(msg).encode("utf-8")

    def send_frame(self, image, width=640, height=480, quality=70):
        """Send frame to server"""
        data = self.encode_frame(image, width, height, quality)
        self.send_socket.sendto(data, (self.server_host, self.server_port))

    def receive_detection(self):
        """Receive detection result (non-blocking)"""
        try:
            data, addr = self.recv_socket.recvfrom(65535)
            msg = json.loads(data.decode("utf-8"))
            return msg
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving: {e}")
            return None

    def test_single_image(self, image_path):
        """Test with a single image"""
        print(f"\n{'=' * 60}")
        print("TEST MODE: Single Image")
        print(f"{'=' * 60}\n")

        # Load image
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Cannot load image: {image_path}")
            return

        print(f"✓ Loaded image: {img.shape[1]}x{img.shape[0]} px")
        print(f"\nSending frame to server...")

        # Send frame
        start_time = time.time()
        self.send_frame(img)

        print(f"✓ Frame sent, waiting for response...")

        # Wait for response (with timeout)
        timeout = 5.0
        elapsed = 0
        result = None

        while elapsed < timeout:
            result = self.receive_detection()
            if result:
                break
            time.sleep(0.01)
            elapsed = time.time() - start_time

        if result:
            round_trip_ms = (time.time() - start_time) * 1000
            print(f"✓ Response received in {round_trip_ms:.1f}ms")
            self._print_detection_result(result)
        else:
            print(f"✗ No response after {timeout}s")

    def test_webcam(self, camera_id=0, width=640, height=480, quality=70):
        """Test with webcam (simulates real-time Godot usage)"""
        print(f"\n{'=' * 60}")
        print("TEST MODE: Webcam (Real-time)")
        print(f"{'=' * 60}\n")

        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✓ Camera opened: {actual_w}x{actual_h}")
        print(f"\nStreaming frames to server...")
        print("Press 'q' to quit, 'd' to toggle detection display\n")

        # Stats
        frames_sent = 0
        detections_received = 0
        show_detections = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Cannot read frame")
                    break

                # Send frame
                self.send_frame(frame, width, height, quality)
                frames_sent += 1

                # Check for detection results
                result = self.receive_detection()

                # Display frame with detections
                display_frame = frame.copy()

                if result:
                    detections_received += 1

                    if show_detections and result.get("type") == "detection":
                        self._draw_detections(display_frame, result)

                    # Show stats on frame
                    proc_time = result.get("processing_time_ms", 0)
                    cv2.putText(
                        display_frame,
                        f"Proc: {proc_time:.1f}ms | Faces: {len(result.get('faces', []))}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Show stats
                cv2.putText(
                    display_frame,
                    f"Sent: {frames_sent} | Recv: {detections_received}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("UDP Test Client - Webcam", display_frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    show_detections = not show_detections
                    print(f"Detection display: {'ON' if show_detections else 'OFF'}")

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            print(f"\n{'=' * 60}")
            print("TEST STATISTICS")
            print(f"{'=' * 60}")
            print(f"Frames sent: {frames_sent}")
            print(f"Detections received: {detections_received}")
            if frames_sent > 0:
                print(f"Response rate: {detections_received / frames_sent * 100:.1f}%")
            print(f"{'=' * 60}\n")

    def _draw_detections(self, image, result):
        """Draw detection results on image"""
        faces = result.get("faces", [])

        for face in faces:
            bbox = face["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw confidence
            conf = face["confidence"]
            cv2.putText(
                image,
                f"{conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw eyes if detected
            eyes = face.get("eyes", {})
            if eyes.get("detected"):
                if eyes.get("left"):
                    lx, ly = int(eyes["left"]["x"]), int(eyes["left"]["y"])
                    cv2.circle(image, (lx, ly), 3, (255, 0, 0), -1)
                if eyes.get("right"):
                    rx, ry = int(eyes["right"]["x"]), int(eyes["right"]["y"])
                    cv2.circle(image, (rx, ry), 3, (255, 0, 0), -1)

            # Draw rotation indicator
            rotation = face.get("rotation_deg", 0)
            if abs(rotation) > 5:
                cv2.putText(
                    image,
                    f"Rot: {rotation:.1f}°",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

    def _print_detection_result(self, result):
        """Pretty print detection result"""
        print(f"\n{'=' * 60}")
        print("DETECTION RESULT")
        print(f"{'=' * 60}")

        msg_type = result.get("type")
        print(f"Type: {msg_type}")

        if msg_type == "detection":
            proc_time = result.get("processing_time_ms", 0)
            faces = result.get("faces", [])

            print(f"Processing time: {proc_time:.1f}ms")
            print(f"Faces detected: {len(faces)}")

            for i, face in enumerate(faces, 1):
                print(f"\n  Face {i}:")
                bbox = face["bbox"]
                print(f"    Position: ({bbox['x']}, {bbox['y']})")
                print(f"    Size: {bbox['w']}x{bbox['h']} px")
                print(f"    Confidence: {face['confidence']:.3f}")
                print(f"    Rotation: {face['rotation_deg']:.1f}°")

                eyes = face.get("eyes", {})
                if eyes.get("detected"):
                    print(f"    Eyes detected:")
                    if eyes.get("left"):
                        print(
                            f"      Left: ({eyes['left']['x']:.0f}, {eyes['left']['y']:.0f})"
                        )
                    if eyes.get("right"):
                        print(
                            f"      Right: ({eyes['right']['x']:.0f}, {eyes['right']['y']:.0f})"
                        )

        elif msg_type == "error":
            print(f"Error: {result.get('message')}")
            print(f"Code: {result.get('code')}")

        print(f"{'=' * 60}\n")

    def close(self):
        """Close sockets"""
        self.send_socket.close()
        self.recv_socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test UDP Face Detection Server (No Godot Required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with single image
  python test_client.py image --image test.jpg

  # Test with webcam (real-time)
  python test_client.py webcam --camera 0

  # Test with custom server address
  python test_client.py webcam --host 192.168.1.100 --camera 0
        """,
    )

    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server receive port")
    parser.add_argument(
        "--recv-port", type=int, default=5556, help="Client receive port"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Test mode")

    # Image mode
    img_parser = subparsers.add_parser("image", help="Test with single image")
    img_parser.add_argument("--image", required=True, help="Image path")

    # Webcam mode
    webcam_parser = subparsers.add_parser("webcam", help="Test with webcam")
    webcam_parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    webcam_parser.add_argument("--width", type=int, default=640, help="Frame width")
    webcam_parser.add_argument("--height", type=int, default=480, help="Frame height")
    webcam_parser.add_argument("--quality", type=int, default=70, help="JPEG quality")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return 1

    # Create client
    client = UDPTestClient(args.host, args.port, args.recv_port)

    try:
        if args.mode == "image":
            client.test_single_image(args.image)
        elif args.mode == "webcam":
            client.test_webcam(args.camera, args.width, args.height, args.quality)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
