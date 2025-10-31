"""
ORB+SVM Face Detector with Mask Overlay
========================================
Classical Computer Vision pipeline for face detection and virtual try-on.

⚠️ TRAINING: Use Google Colab (see train_colab.ipynb)
   This script is for INFERENCE ONLY.

Usage:
    python app.py infer --image input.jpg --out output.jpg --mask assets/mask.png
    python app.py webcam --camera 0 --mask assets/mask.png --show
    python app.py video --input video.mp4 --output output.mp4

Prerequisites:
    - Trained models from Colab: codebook.pkl, svm.pkl, scaler.pkl
    - Place models in: models/ folder

Author: Computer Vision Expert
Date: 2025
"""

import argparse
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import Inferencer


def infer_command(args):
    """Run inference on image."""
    print("\n" + "="*60)
    print("INFERENCE ON IMAGE")
    print("="*60)
    
    # Check if models exist
    models_dir = args.models_dir
    if not os.path.exists(os.path.join(models_dir, 'svm.pkl')):
        print("❌ Error: No trained models found!")
        print("\n📋 Steps to get models:")
        print("  1. Open train_colab.ipynb in Google Colab")
        print("  2. Upload dataset and run training (2-5 minutes)")
        print("  3. Download: codebook.pkl, svm.pkl, scaler.pkl")
        print(f"  4. Place models in: {models_dir}/")
        sys.exit(1)
    
    # Check if mask provided
    if args.mask is None:
        print("Warning: No mask provided. Will only show detection boxes.")
        overlay_mask = False
    else:
        overlay_mask = True
    
    # Initialize inferencer
    inferencer = Inferencer(
        models_dir=args.models_dir,
        mask_path=args.mask if overlay_mask else None,
        iou_threshold=args.iou,
        score_threshold=args.threshold
    )
    
    # Run inference
    output = inferencer.infer_image(
        image_path=args.image,
        output_path=args.out,
        overlay_mask=overlay_mask,
        draw_boxes=True
    )
    
    print(f"\n✓ Inference complete!")
    if args.out:
        print(f"  Output saved: {args.out}")


def webcam_command(args):
    """Run webcam inference."""
    print("\n" + "="*60)
    print("WEBCAM INFERENCE")
    print("="*60)
    
    # Check if mask provided
    if args.mask is None:
        print("Warning: No mask provided. Running detection only.")
        mask_path = None
    else:
        mask_path = args.mask
    
    # Initialize inferencer
    inferencer = Inferencer(
        models_dir=args.models_dir,
        mask_path=mask_path,
        iou_threshold=args.iou,
        score_threshold=args.threshold
    )
    
    # Run webcam
    inferencer.infer_webcam(
        camera_id=args.camera,
        show_fps=True,
        overlay_mask=args.show,
        draw_boxes=False
    )


def video_command(args):
    """Run inference on video."""
    print("\n" + "="*60)
    print("VIDEO INFERENCE")
    print("="*60)
    
    # Check if mask provided
    if args.mask is None:
        print("Warning: No mask provided. Will only show detection boxes.")
        overlay_mask = False
    else:
        overlay_mask = True
    
    # Initialize inferencer
    inferencer = Inferencer(
        models_dir=args.models_dir,
        mask_path=args.mask if overlay_mask else None,
        iou_threshold=args.iou,
        score_threshold=args.threshold
    )
    
    # Run inference
    inferencer.infer_video(
        video_path=args.video,
        output_path=args.out,
        overlay_mask=overlay_mask,
        draw_boxes=True
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ORB+SVM Face Detector with Mask Overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters
  python app.py train --pos_dir data/face --neg_dir data/non_face

  # Train with custom parameters and grid search
  python app.py train --pos_dir data/face --neg_dir data/non_face \\
                       --k 256 --max_desc 200000 --svm linear --C 1.0 --grid_search

  # Evaluate trained model
  python app.py eval

  # Inference on image
  python app.py infer --image input.jpg --out output.jpg --mask assets/mask.png

  # Live webcam with mask overlay
  python app.py webcam --camera 0 --mask assets/mask.png --show

  # Process video
  python app.py video --video input.mp4 --out output.mp4 --mask assets/mask.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ORB+SVM classifier')
    train_parser.add_argument('--pos_dir', type=str, required=True,
                             help='Directory with positive (face) samples')
    train_parser.add_argument('--neg_dir', type=str, required=True,
                             help='Directory with negative (non-face) samples')
    train_parser.add_argument('--k', type=int, default=256,
                             help='Number of visual words (codebook size)')
    train_parser.add_argument('--max_desc', type=int, default=200000,
                             help='Max descriptors for k-means clustering')
    train_parser.add_argument('--n_features', type=int, default=500,
                             help='Max ORB features per image')
    train_parser.add_argument('--svm', type=str, default='linear',
                             choices=['linear', 'rbf'],
                             help='SVM kernel type')
    train_parser.add_argument('--C', type=float, default=1.0,
                             help='SVM regularization parameter')
    train_parser.add_argument('--grid_search', action='store_true',
                             help='Use GridSearchCV for hyperparameter tuning')
    train_parser.add_argument('--models_dir', type=str, default='models',
                             help='Directory to save trained models')
    train_parser.add_argument('--reports_dir', type=str, default='reports',
                             help='Directory to save reports')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--models_dir', type=str, default='models',
                            help='Directory with trained models')
    eval_parser.add_argument('--reports_dir', type=str, default='reports',
                            help='Directory with saved reports')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on image')
    infer_parser.add_argument('--image', type=str, required=True,
                             help='Input image path')
    infer_parser.add_argument('--out', type=str, required=True,
                             help='Output image path')
    infer_parser.add_argument('--mask', type=str, default=None,
                             help='Mask PNG path with alpha channel')
    infer_parser.add_argument('--models_dir', type=str, default='models',
                             help='Directory with trained models')
    infer_parser.add_argument('--iou', type=float, default=0.3,
                             help='NMS IoU threshold')
    infer_parser.add_argument('--threshold', type=float, default=0.8,
                             help='Confidence score threshold (default: 0.8 - high precision)')
    
    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Run webcam inference')
    webcam_parser.add_argument('--camera', type=int, default=0,
                              help='Camera device ID')
    webcam_parser.add_argument('--mask', type=str, default=None,
                              help='Mask PNG path with alpha channel')
    webcam_parser.add_argument('--show', action='store_true',
                              help='Show mask overlay by default')
    webcam_parser.add_argument('--models_dir', type=str, default='models',
                              help='Directory with trained models')
    webcam_parser.add_argument('--iou', type=float, default=0.3,
                              help='NMS IoU threshold')
    webcam_parser.add_argument('--threshold', type=float, default=0.8,
                              help='Confidence score threshold (default: 0.8 - high precision)')
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Run inference on video')
    video_parser.add_argument('--video', type=str, required=True,
                             help='Input video path')
    video_parser.add_argument('--out', type=str, required=True,
                             help='Output video path')
    video_parser.add_argument('--mask', type=str, default=None,
                             help='Mask PNG path with alpha channel')
    video_parser.add_argument('--models_dir', type=str, default='models',
                             help='Directory with trained models')
    video_parser.add_argument('--iou', type=float, default=0.3,
                             help='NMS IoU threshold')
    video_parser.add_argument('--threshold', type=float, default=0.5,
                             help='Confidence score threshold (default: 0.5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Set random seed
    np.random.seed(42)
    
    # Execute command
    if args.command == 'train':
        print("\n❌ Training is NOT available in local VS Code!")
        print("\n📋 To train the model:")
        print("  1. Open train_colab.ipynb in Google Colab (local or online)")
        print("  2. Upload your dataset")
        print("  3. Run training (2-5 minutes)")
        print("  4. Download models: codebook.pkl, svm.pkl, scaler.pkl")
        print("  5. Place them in models/ folder")
        print("\n✅ Then you can run inference here!")
        sys.exit(1)
    elif args.command == 'eval':
        print("\n❌ Evaluation is NOT available in local VS Code!")
        print("\n📋 Model metrics are shown during training in Colab.")
        print("  Check training output for accuracy, precision, recall, F1.")
        sys.exit(1)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'webcam':
        webcam_command(args)
    elif args.command == 'video':
        video_command(args)


if __name__ == '__main__':
    main()
