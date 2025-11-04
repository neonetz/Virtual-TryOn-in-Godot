"""
SVM+ORB Face Detector with MASK Overlay
Classical CV pipeline: Haar proposal → ORB+BoVW → SVM → Mask overlay

CLI Commands:
  - infer: Process single image
  - webcam: Real-time webcam processing
"""

import cv2
import argparse
import sys
import os
import time
import logging

from pipelines.infer import FaceDetectionSystem
from pipelines.utils import setup_logging, create_fps_counter, draw_fps, resize_maintain_aspect


def get_opencv_data_path(filename: str) -> str:
    """Get path to OpenCV Haar cascade XML file"""
    # Try OpenCV data directory
    opencv_data = cv2.data.haarcascades
    cascade_path = os.path.join(opencv_data, filename)
    
    if os.path.exists(cascade_path):
        return cascade_path
    
    # Try local assets/cascades directory
    local_path = os.path.join("assets", "cascades", filename)
    if os.path.exists(local_path):
        return local_path
    
    raise FileNotFoundError(f"Cannot find cascade file: {filename}")


def infer_command(args):
    """Process single image"""
    print("\n" + "="*60)
    print("INFERENCE MODE: Single Image")
    print("="*60)
    
    # Check input image
    if not os.path.exists(args.image):
        print(f"✗ Error: Image not found: {args.image}")
        return 1
    
    # Load image
    print(f"\nLoading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"✗ Error: Cannot load image")
        return 1
    
    print(f"✓ Image loaded: {img.shape[1]}×{img.shape[0]} px")
    
    # Get cascade paths
    cascade_face = get_opencv_data_path("haarcascade_frontalface_default.xml")
    cascade_eye = None
    if args.enable_rotation:
        try:
            cascade_eye = get_opencv_data_path("haarcascade_eye.xml")
        except FileNotFoundError:
            print("⚠ Warning: Eye cascade not found, rotation disabled")
    
    # Initialize system
    print("\nInitializing detection system...")
    print(f"  SVM model: {args.svm_model}")
    print(f"  BoVW encoder: {args.bovw_encoder}")
    print(f"  Scaler: {args.scaler}")
    print(f"  Mask: {args.mask}")
    
    try:
        system = FaceDetectionSystem(
            svm_model_path=args.svm_model,
            bovw_encoder_path=args.bovw_encoder,
            scaler_path=args.scaler,
            cascade_face_path=cascade_face,
            mask_path=args.mask,
            cascade_eye_path=cascade_eye,
            max_keypoints=args.max_keypoints
        )
    except Exception as e:
        print(f"✗ Error initializing system: {e}")
        return 1
    
    print("✓ System initialized")
    
    # Process image
    print("\nProcessing...")
    start_time = time.time()
    
    result, detections = system.process_image(
        img,
        enable_rotation=args.enable_rotation,
        alpha_multiplier=args.alpha
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"✓ Processing completed in {elapsed_ms:.1f} ms")
    print(f"✓ Detected {len(detections)} face(s)")
    
    # Print detection details
    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        print(f"\n  Face {i}:")
        print(f"    Position: ({bbox['x']}, {bbox['y']})")
        print(f"    Size: {bbox['w']}×{bbox['h']} px")
        print(f"    Score: {det['score']:.3f}")
    
    # Save result
    cv2.imwrite(args.out, result)
    print(f"\n✓ Result saved: {args.out}")
    
    # Show result
    if args.show:
        print("\nDisplaying result (press any key to close)...")
        # Resize for display if too large
        display_img = resize_maintain_aspect(result, 1280, 720)
        cv2.imshow("Face Detection with Mask", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    return 0


def webcam_command(args):
    """Real-time webcam processing"""
    print("\n" + "="*60)
    print("WEBCAM MODE: Real-time Face Detection")
    print("="*60)
    
    # Get cascade paths
    cascade_face = get_opencv_data_path("haarcascade_frontalface_default.xml")
    
    # Don't use Haar eye cascade - use custom ORB detector instead
    cascade_eye = None
    
    # Initialize system
    print("\nInitializing detection system...")
    print(f"  SVM model: {args.svm_model}")
    print(f"  BoVW encoder: {args.bovw_encoder}")
    print(f"  Scaler: {args.scaler}")
    print(f"  Mask: {args.mask}")
    print(f"  Eye detection: Custom ORB detector (no Haar cascade)")
    
    try:
        system = FaceDetectionSystem(
            svm_model_path=args.svm_model,
            bovw_encoder_path=args.bovw_encoder,
            scaler_path=args.scaler,
            cascade_face_path=cascade_face,
            mask_path=args.mask,
            cascade_eye_path=cascade_eye,
            max_keypoints=args.max_keypoints,
            use_custom_eye_detector=True,  # Enable custom detector
            custom_eye_method="orb"  # Use ORB method
        )
    except Exception as e:
        print(f"✗ Error initializing system: {e}")
        return 1
    
    print("✓ System initialized")
    
    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"✗ Error: Cannot open camera {args.camera}")
        return 1
    
    # Set resolution if specified
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened: {actual_w}×{actual_h}")
    
    # FPS counter and limiter
    fps_counter = create_fps_counter()
    target_fps = args.fps
    frame_time = 1.0 / target_fps
    print(f"✓ Target FPS: {target_fps}")
    
    print("\n" + "="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'b' - Toggle bounding boxes")
    print("  'r' - Toggle rotation")
    print("  'd' - Toggle debug info (eye detection)")
    print("="*60 + "\n")
    
    # State
    show_boxes = args.show_boxes
    enable_rotation = args.enable_rotation
    frame_count = 0
    show_debug = False  # Toggle with 'd' key
    
    try:
        while True:
            loop_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Cannot read frame")
                break
            
            # Process frame
            start_time = time.time()
            
            result = system.process_video_frame(
                frame,
                enable_rotation=enable_rotation,
                alpha_multiplier=args.alpha,
                draw_boxes=show_boxes,
                draw_eye_debug=show_debug
            )
            
            processing_ms = (time.time() - start_time) * 1000
            
            # Update FPS
            fps = fps_counter()
            
            # Frame rate limiting (BEFORE display to ensure stable FPS)
            loop_time = time.time() - loop_start
            sleep_time = frame_time - loop_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Draw FPS and info
            result = draw_fps(result, fps, position=(10, 30))
            
            cv2.putText(
                result,
                f"Time: {processing_ms:.1f}ms (target: {target_fps} FPS)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Status indicators
            status_y = 90
            if show_boxes:
                cv2.putText(result, "[Boxes: ON]", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                status_y += 25
            
            if enable_rotation:
                cv2.putText(result, "[Rotation: ON]", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                status_y += 25
            
            if show_debug:
                cv2.putText(result, "[Debug: ON]", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Show frame
            if args.show:
                cv2.imshow("SVM+ORB Face Detector - Webcam", result)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, result)
                print(f"\n✓ Screenshot saved: {filename}")
            elif key == ord('b'):
                show_boxes = not show_boxes
                print(f"\n→ Bounding boxes: {'ON' if show_boxes else 'OFF'}")
            elif key == ord('r'):
                enable_rotation = not enable_rotation
                print(f"\n→ Rotation: {'ON' if enable_rotation else 'OFF'}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"\n→ Debug mode (eye detection): {'ON' if show_debug else 'OFF'}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        if frame_count > 0:
            print(f"Average FPS: {fps:.1f}")
        print("="*60)
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SVM+ORB Face Detector with MASK Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python face_detector_cli.py infer --image input.jpg --out output.jpg

  # Real-time webcam
  python face_detector_cli.py webcam --camera 0 --show

  # Webcam with custom mask
  python face_detector_cli.py webcam --camera 0 --mask assets/custom_mask.png
        """
    )
    
    # Common arguments
    parser.add_argument("--svm-model", default="models/svm_model.pkl",
                       help="Path to SVM model file")
    parser.add_argument("--bovw-encoder", default="models/bovw_encoder.pkl",
                       help="Path to BoVW encoder (k-means codebook)")
    parser.add_argument("--scaler", default="models/scaler.pkl",
                       help="Path to feature scaler")
    parser.add_argument("--mask", default="assets/masks/mask-1.png",
                       help="Path to mask PNG with alpha channel")
    parser.add_argument("--max-keypoints", type=int, default=500,
                       help="Maximum ORB keypoints per ROI")
    parser.add_argument("--alpha", type=float, default=0.95,
                       help="Mask opacity (0-1)")
    parser.add_argument("--enable-rotation", action="store_true",
                       help="Enable mask rotation based on eye detection")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Process single image")
    infer_parser.add_argument("--image", required=True, help="Input image path")
    infer_parser.add_argument("--out", required=True, help="Output image path")
    infer_parser.add_argument("--show", action="store_true", help="Display result")
    
    # Webcam command
    webcam_parser = subparsers.add_parser("webcam", help="Real-time webcam processing")
    webcam_parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    webcam_parser.add_argument("--show", action="store_true", default=True,
                              help="Display webcam feed")
    webcam_parser.add_argument("--show-boxes", action="store_true",
                              help="Show bounding boxes")
    webcam_parser.add_argument("--width", type=int, help="Camera width")
    webcam_parser.add_argument("--height", type=int, help="Camera height")
    webcam_parser.add_argument("--fps", type=int, default=30,
                              help="Target FPS (default 30)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check command
    if args.command is None:
        parser.print_help()
        return 1
    
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
    
    # Run command
    if args.command == "infer":
        return infer_command(args)
    elif args.command == "webcam":
        return webcam_command(args)
    else:
        print(f"✗ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
