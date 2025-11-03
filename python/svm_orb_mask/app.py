"""
SVM+ORB Face Detector with Mask Overlay - CLI Application
Complete inference pipeline for images, videos, and webcam.
"""

import argparse
import sys
from pathlib import Path

import cv2

from pipelines.infer import FaceMaskPipeline
from pipelines.utils import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SVM+ORB Face Detector with Mask Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process image
  python app.py --mode image --input test.jpg --output result.jpg
  
  # Process video
  python app.py --mode video --input video.mp4 --output output.mp4
  
  # Webcam (default camera)
  python app.py --mode webcam
  
  # Webcam with custom camera ID
  python app.py --mode webcam --camera-id 1
  
  # Custom model directory and mask
  python app.py --mode image --input test.jpg --models-dir ./my_models --mask ./my_mask.png
  
  # Adjust mask scale and position
  python app.py --mode webcam --mask-scale 1.5 --mask-offset 0.4
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['image', 'video', 'webcam'],
        help='Processing mode'
    )
    
    # Input/output paths
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (required for image/video modes)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional for video mode, ignored for webcam)'
    )
    
    # Model configuration
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models/)'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        default='assets/mask.png',
        help='Path to mask PNG file (default: assets/mask.png)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for face detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.3,
        help='NMS IoU threshold (default: 0.3)'
    )
    
    parser.add_argument(
        '--no-rotation',
        action='store_true',
        help='Disable eye-based rotation alignment'
    )
    
    # Mask parameters
    parser.add_argument(
        '--mask-scale',
        type=float,
        default=1.2,
        help='Mask scale factor relative to face width (default: 1.2)'
    )
    
    parser.add_argument(
        '--mask-offset',
        type=float,
        default=0.3,
        help='Mask vertical offset as fraction of face height (default: 0.3)'
    )
    
    # Display options
    parser.add_argument(
        '--no-boxes',
        action='store_true',
        help='Do not draw bounding boxes'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display video/webcam window (useful for headless processing)'
    )
    
    # Webcam options
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Webcam device ID (default: 0)'
    )
    
    # Cascade paths (optional overrides)
    parser.add_argument(
        '--face-cascade',
        type=str,
        help='Path to Haar Cascade face detector XML (default: OpenCV built-in)'
    )
    
    parser.add_argument(
        '--eye-cascade',
        type=str,
        help='Path to Haar Cascade eye detector XML (default: OpenCV built-in)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check input file requirement
    if args.mode in ['image', 'video'] and not args.input:
        print(f"Error: --input is required for mode '{args.mode}'")
        return False
    
    # Check input file exists
    if args.input and not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return False
    
    # Check models directory
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory not found: {args.models_dir}")
        print("Please provide trained models (svm_model.pkl, codebook.pkl, scaler.pkl)")
        return False
    
    required_models = ['svm_model.pkl', 'codebook.pkl', 'scaler.pkl']
    for model_file in required_models:
        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"Error: Required model file not found: {model_path}")
            return False
    
    # Check mask file
    if not Path(args.mask).exists():
        print(f"Error: Mask file not found: {args.mask}")
        return False
    
    return True


def process_image_mode(pipeline, args):
    """Process single image."""
    print(f"\n[APP] Processing image: {args.input}")
    
    # Read image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Failed to read image: {args.input}")
        return False
    
    # Process
    result, detections = pipeline.process_image(
        image,
        draw_boxes=not args.no_boxes,
        mask_scale=args.mask_scale,
        mask_offset_y=args.mask_offset
    )
    
    # Print detection results
    print(f"[APP] Detected {len(detections)} face(s)")
    for i, det in enumerate(detections):
        print(f"  Face {i+1}: bbox={det['bbox']}, "
              f"confidence={det['confidence']:.3f}, "
              f"rotation={det['rotation_angle']}")
    
    # Save output
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"[APP] Output saved to: {args.output}")
    else:
        # Display result
        cv2.imshow("Result", result)
        print("[APP] Press any key to close window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True


def process_video_mode(pipeline, args):
    """Process video file."""
    print(f"\n[APP] Processing video: {args.input}")
    
    success = pipeline.process_video(
        args.input,
        output_path=args.output,
        display=not args.no_display,
        draw_boxes=not args.no_boxes,
        mask_scale=args.mask_scale,
        mask_offset_y=args.mask_offset
    )
    
    return success


def process_webcam_mode(pipeline, args):
    """Process webcam stream."""
    print(f"\n[APP] Starting webcam mode (Camera ID: {args.camera_id})")
    
    pipeline.process_webcam(
        camera_id=args.camera_id,
        draw_boxes=not args.no_boxes,
        mask_scale=args.mask_scale,
        mask_offset_y=args.mask_offset
    )
    
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("SVM+ORB Face Detector with Mask Overlay")
    print("Classical Computer Vision - Inference Pipeline")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Load configuration (if exists)
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"[APP] Loaded configuration from: {args.config}")
    
    # Initialize pipeline
    try:
        print("\n[APP] Initializing pipeline...")
        pipeline = FaceMaskPipeline(
            model_dir=args.models_dir,
            mask_path=args.mask,
            cascade_path=args.face_cascade,
            eye_cascade_path=args.eye_cascade,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold,
            use_rotation=not args.no_rotation
        )
        print("[APP] Pipeline initialized successfully")
    except Exception as e:
        print(f"[APP] Error: Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process based on mode
    try:
        if args.mode == 'image':
            success = process_image_mode(pipeline, args)
        elif args.mode == 'video':
            success = process_video_mode(pipeline, args)
        elif args.mode == 'webcam':
            success = process_webcam_mode(pipeline, args)
        else:
            print(f"Error: Unknown mode: {args.mode}")
            success = False
        
        if success:
            print("\n[APP] Processing completed successfully")
        else:
            print("\n[APP] Processing failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[APP] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
