"""
SVM+ORB Face Detector Pipeline Package
Classical computer vision approach for face detection and mask overlay.
"""

from .features import ORBFeatureExtractor, ROIProposer, EyeDetector
from .mask_overlay import MaskOverlay, MaskLibrary
from .infer import FaceDetectorSVM, FaceMaskPipeline
from .utils import (
    Timer,
    FPSCounter,
    load_config,
    save_config,
    compute_iou,
    non_max_suppression,
    draw_detection_box,
    visualize_keypoints,
    get_cascade_path,
    resize_with_aspect_ratio
)

__all__ = [
    # Feature extraction
    'ORBFeatureExtractor',
    'ROIProposer',
    'EyeDetector',
    
    # Mask overlay
    'MaskOverlay',
    'MaskLibrary',
    
    # Inference
    'FaceDetectorSVM',
    'FaceMaskPipeline',
    
    # Utilities
    'Timer',
    'FPSCounter',
    'load_config',
    'save_config',
    'compute_iou',
    'non_max_suppression',
    'draw_detection_box',
    'visualize_keypoints',
    'get_cascade_path',
    'resize_with_aspect_ratio'
]

__version__ = '1.0.0'
__author__ = 'Computer Vision Expert'
__description__ = 'SVM+ORB Face Detector with Mask Overlay'
