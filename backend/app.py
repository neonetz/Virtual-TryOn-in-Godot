"""
Virtual Try-On Main Application
Command-line interface for face detection and mask overlay
"""

import cv2
import argparse
import sys
import os
import time

from inference.face_detector import FaceDetector
from inference.mask_overlay import MaskManager, LandmarkEstimator


def infer_command(args):
    """Run inference on a single image"""
    print("\n" + "=" * 50)
    print("INFERENCE MODE")
    print("=" * 50)

    # Load models
    print("\nLoading models...")
    detector = FaceDetector(args.model, args.scaler, args.config)

    # Load mask
    print(f"\nLoading mask: {args.mask}")
    mask_manager = MaskManager()
    mask_id = "custom_mask"
    mask_manager.load_mask(mask_id, args.mask)

    # Load input image
    print(f"\nLoading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return 1

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Detect faces
    print("\nDetecting faces...")
    start_time = time.time()
    detections = detector.detect(img, single_face=True)
    detection_time = (time.time() - start_time) * 1000

    print(f"✓ Detection completed in {detection_time:.1f}ms")
    print(f"✓ Found {len(detections)} face(s)")

    if len(detections) == 0:
        print("\nNo faces detected in image")
        return 1

    # Print detection details
    for i, det in enumerate(detections):
        print(f"\nFace {i + 1}:")
        print(f"  Position: ({det['bbox']['x']}, {det['bbox']['y']})")
        print(f"  Size: {det['bbox']['w']}x{det['bbox']['h']}")
        print(f"  Confidence: {det['confidence']:.3f}")

    # Add landmarks
    for detection in detections:
        landmarks = LandmarkEstimator.estimate_5_points(detection["bbox"])
        detection["landmarks"] = landmarks

    # Apply mask
    print("\nApplying mask overlay...")
    start_time = time.time()
    result = mask_manager.apply_mask(
        img, detections[0], mask_id, use_perspective=True, alpha=0.95
    )
    overlay_time = (time.time() - start_time) * 1000

    print(f"✓ Overlay completed in {overlay_time:.1f}ms")

    # Save result
    cv2.imwrite(args.out, result)
    print(f"\n✓ Result saved to: {args.out}")

    # Show if requested
    if args.show:
        print("\nDisplaying result (press any key to close)...")
        cv2.imshow("Virtual Try-On Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nTotal processing time: {:.1f}ms".format(detection_time + overlay_time))
    print("=" * 50)

    return 0


def webcam_command(args):
    """Run real-time webcam overlay"""
    print("\n" + "=" * 50)
    print("WEBCAM MODE")
    print("=" * 50)

    # Load models
    print("\nLoading models...")
    detector = FaceDetector(args.model, args.scaler, args.config)

    # Load mask
    print(f"\nLoading mask: {args.mask}")
    mask_manager = MaskManager()
    mask_id = "webcam_mask"
    mask_manager.load_mask(mask_id, args.mask)

    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return 1

    # Set resolution if specified
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened: {actual_width}x{actual_height}")

    # Performance tracking
    frame_count = 0
    total_time = 0
    previous_detection = None

    print("\n" + "=" * 50)
    print("Press 'q' to quit, 's' to save screenshot")
    print("=" * 50 + "\n")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            start_time = time.time()

            # Detect with ROI tracking
            if frame_count % 10 == 0 or previous_detection is None:
                # Full detection every 10 frames
                detections = detector.detect(frame, single_face=True)
            else:
                # ROI tracking
                detections = detector.detect_with_roi_tracking(
                    frame, previous_detection
                )

            if len(detections) > 0:
                previous_detection = detections[0]

                # Add landmarks
                landmarks = LandmarkEstimator.estimate_5_points(detections[0]["bbox"])
                detections[0]["landmarks"] = landmarks

                # Apply mask
                frame = mask_manager.apply_mask(
                    frame, detections[0], mask_id, use_perspective=True, alpha=0.95
                )
            else:
                previous_detection = None

            processing_time = (time.time() - start_time) * 1000
            frame_count += 1
            total_time += processing_time

            # Draw FPS
            fps = 1000 / processing_time if processing_time > 0 else 0
            avg_fps = frame_count / (total_time / 1000) if total_time > 0 else 0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} (avg: {avg_fps:.1f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Time: {processing_time:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if len(detections) == 0:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Show frame
            if args.show:
                cv2.imshow("Virtual Try-On - Webcam", frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n✓ Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        if frame_count > 0:
            print("\n" + "=" * 50)
            print("SESSION STATISTICS")
            print("=" * 50)
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {frame_count / (total_time / 1000):.1f}")
            print(f"Average processing time: {total_time / frame_count:.1f}ms")
            print("=" * 50)

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Virtual Try-On: Halloween Mask Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on single image
  python app.py infer --image input.jpg --out output.jpg --mask assets/zombie.png
  
  # Live webcam with overlay
  python app.py webcam --camera 0 --mask assets/zombie.png --show
  
  # Start UDP server for Godot
  python -m src.server.udp_server --model models/face_detector_svm.pkl \\
                                   --scaler models/scaler.pkl \\
                                   --config models/config.json \\
                                   --masks assets/masks/
        """,
    )

    # Common arguments
    parser.add_argument(
        "--model",
        default="models/face_detector_svm.pkl",
        help="Path to face detector model",
    )
    parser.add_argument(
        "--scaler", default="models/scaler.pkl", help="Path to feature scaler"
    )
    parser.add_argument(
        "--config", default="models/config.json", help="Path to configuration file"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on single image")
    infer_parser.add_argument("--image", required=True, help="Input image path")
    infer_parser.add_argument("--out", required=True, help="Output image path")
    infer_parser.add_argument(
        "--mask", required=True, help="Mask image path (PNG with alpha)"
    )
    infer_parser.add_argument("--show", action="store_true", help="Display result")

    # Webcam command
    webcam_parser = subparsers.add_parser("webcam", help="Run real-time webcam overlay")
    webcam_parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    webcam_parser.add_argument(
        "--mask", required=True, help="Mask image path (PNG with alpha)"
    )
    webcam_parser.add_argument(
        "--show", action="store_true", default=True, help="Display webcam feed"
    )
    webcam_parser.add_argument("--width", type=int, help="Camera width")
    webcam_parser.add_argument("--height", type=int, help="Camera height")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    if not os.path.exists(args.scaler):
        print(f"Error: Scaler file not found: {args.scaler}")
        return 1
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Run command
    if args.command == "infer":
        return infer_command(args)
    elif args.command == "webcam":
        return webcam_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
