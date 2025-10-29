"""
ORB+SVM Face Detector with Mask Overlay
========================================
Classical Computer Vision pipeline for face detection and virtual try-on.

Usage:
    python app.py train --pos_dir data/face --neg_dir data/non_face
    python app.py eval
    python app.py infer --image input.jpg --out output.jpg --mask assets/mask.png
    python app.py webcam --camera 0 --mask assets/mask.png --show

Author: Computer Vision Expert
Date: 2025
"""

import argparse
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training import Trainer
from src.inference import Inferencer


def train_command(args):
    """Execute training pipeline."""
    print("\n" + "="*60)
    print("TRAINING ORB+SVM FACE DETECTOR")
    print("="*60)
    
    # Initialize trainer
    trainer = Trainer(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        random_state=args.seed
    )
    
    # Run full pipeline
    metrics = trainer.run_full_pipeline(
        k=args.k,
        max_descriptors=args.max_desc,
        n_features=args.n_features,
        kernel=args.svm,
        C=args.C,
        use_grid_search=args.grid_search
    )
    
    print("\n✓ Training complete!")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics['f1']:.4f}")
    print(f"  Test ROC AUC:  {metrics.get('roc_auc', 0):.4f}")


def eval_command(args):
    """Evaluate trained model."""
    print("\n" + "="*60)
    print("EVALUATING TRAINED MODEL")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists(os.path.join(args.models_dir, 'svm.pkl')):
        print("Error: No trained model found. Run 'train' first.")
        sys.exit(1)
    
    # Load trainer with existing models
    print("Loading trained models and test data...")
    print("Note: This command requires re-loading the dataset for evaluation.")
    print("For quick evaluation, metrics are already saved during training.")
    
    # Read saved metrics
    metrics_path = os.path.join(args.reports_dir, 'test_metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\nTest Set Metrics:")
        print("="*60)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print("="*60)
        
        print(f"\nReports saved in: {args.reports_dir}")
    else:
        print(f"Error: Metrics file not found at {metrics_path}")


def infer_command(args):
    """Run inference on image."""
    print("\n" + "="*60)
    print("INFERENCE ON IMAGE")
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
    infer_parser.add_argument('--threshold', type=float, default=0.0,
                             help='Confidence score threshold')
    
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
    webcam_parser.add_argument('--threshold', type=float, default=0.0,
                              help='Confidence score threshold')
    
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
    video_parser.add_argument('--threshold', type=float, default=0.0,
                             help='Confidence score threshold')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Set random seed
    np.random.seed(42)
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'webcam':
        webcam_command(args)
    elif args.command == 'video':
        video_command(args)


if __name__ == '__main__':
    main()
