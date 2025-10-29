"""
Quick Test Script
=================
Verifies that all modules can be imported and basic functionality works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.features import ORBExtractor, BoVWEncoder
        print("✓ Features module")
    except Exception as e:
        print(f"✗ Features module: {e}")
        return False
    
    try:
        from src.detector import ROIProposal, SVMClassifier
        print("✓ Detector module")
    except Exception as e:
        print(f"✗ Detector module: {e}")
        return False
    
    try:
        from src.overlay import MaskOverlay
        print("✓ Overlay module")
    except Exception as e:
        print(f"✗ Overlay module: {e}")
        return False
    
    try:
        from src.training import Trainer
        print("✓ Training module")
    except Exception as e:
        print(f"✗ Training module: {e}")
        return False
    
    try:
        from src.inference import Inferencer
        print("✓ Inference module")
    except Exception as e:
        print(f"✗ Inference module: {e}")
        return False
    
    try:
        from src.utils import non_max_suppression, compute_metrics
        print("✓ Utils module")
    except Exception as e:
        print(f"✗ Utils module: {e}")
        return False
    
    return True


def test_orb_extractor():
    """Test ORB feature extraction."""
    print("\nTesting ORB extractor...")
    
    try:
        import cv2
        import numpy as np
        from src.features import ORBExtractor
        
        # Create dummy image
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Extract features
        extractor = ORBExtractor(n_features=100)
        descriptors = extractor.extract(img)
        
        if descriptors is not None:
            print(f"✓ Extracted {len(descriptors)} ORB descriptors")
            return True
        else:
            print("✗ No descriptors extracted")
            return False
            
    except Exception as e:
        print(f"✗ ORB extractor error: {e}")
        return False


def test_nms():
    """Test Non-Maximum Suppression."""
    print("\nTesting NMS...")
    
    try:
        from src.utils import non_max_suppression
        
        # Test boxes
        boxes = [
            (10, 10, 50, 50),
            (15, 15, 50, 50),  # Overlaps with first
            (100, 100, 50, 50)  # Separate
        ]
        scores = [0.9, 0.8, 0.95]
        
        filtered_boxes, filtered_scores = non_max_suppression(boxes, scores, 0.3)
        
        print(f"✓ NMS: {len(boxes)} boxes → {len(filtered_boxes)} boxes")
        return len(filtered_boxes) == 2  # Should keep first and third
        
    except Exception as e:
        print(f"✗ NMS error: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required = [
        'cv2',
        'numpy',
        'sklearn',
        'joblib',
        'matplotlib'
    ]
    
    missing = []
    
    for module in required:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_structure():
    """Check if directory structure exists."""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'src/features',
        'src/detector',
        'src/overlay',
        'src/training',
        'src/inference',
        'src/utils',
        'assets',
        'data/body',
        'data/non_body',
        'models',
        'reports'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("="*60)
    print("ORB+SVM Body Detector - System Check")
    print("="*60)
    
    results = []
    
    # Check dependencies first
    results.append(("Dependencies", check_dependencies()))
    
    # Check structure
    results.append(("Directory Structure", check_structure()))
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test functionality
    results.append(("ORB Extractor", test_orb_extractor()))
    results.append(("NMS", test_nms()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! System ready.")
        print("\nNext steps:")
        print("1. Add training images to data/body and data/non_body")
        print("2. Run: python app.py train --pos_dir data/body --neg_dir data/non_body")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
