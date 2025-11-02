"""
Installation Test Script
Verifies all dependencies and components are working correctly
"""

import sys
import os


def test_python_version():
    """Test Python version"""
    print("\n" + "=" * 50)
    print("Testing Python Version")
    print("=" * 50)

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK (>= 3.8)")
        return True
    else:
        print("✗ Python version too old. Need >= 3.8")
        return False


def test_dependencies():
    """Test all required dependencies"""
    print("\n" + "=" * 50)
    print("Testing Dependencies")
    print("=" * 50)

    deps = {
        "numpy": "numpy",
        "opencv": "cv2",
        "scikit-learn": "sklearn",
        "scikit-image": "skimage",
        "PIL": "PIL",
        "scipy": "scipy",
    }

    all_ok = True

    for name, module in deps.items():
        try:
            exec(f"import {module}")
            version = eval(f"{module}.__version__")
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"✗ {name:20s} NOT INSTALLED")
            all_ok = False
        except Exception as e:
            print(f"? {name:20s} Error: {e}")
            all_ok = False

    return all_ok


def test_opencv_camera():
    """Test OpenCV camera access"""
    print("\n" + "=" * 50)
    print("Testing Camera Access")
    print("=" * 50)

    try:
        import cv2

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ No camera found or camera in use")
            return False

        ret, frame = cap.read()
        cap.release()

        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Camera OK: {w}x{h}")
            return True
        else:
            print("✗ Failed to capture frame")
            return False

    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_hog_extraction():
    """Test HOG feature extraction"""
    print("\n" + "=" * 50)
    print("Testing HOG Feature Extraction")
    print("=" * 50)

    try:
        import numpy as np
        from skimage.feature import hog

        # Create dummy image
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        # Extract HOG features
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True,
        )

        print(f"✓ HOG extraction OK: {len(features)} features")
        return True

    except Exception as e:
        print(f"✗ HOG extraction failed: {e}")
        return False


def test_svm():
    """Test SVM training"""
    print("\n" + "=" * 50)
    print("Testing SVM Classifier")
    print("=" * 50)

    try:
        import numpy as np
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler

        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train
        svm = LinearSVC(max_iter=100, dual=False)
        svm.fit(X_scaled, y)

        # Predict
        pred = svm.predict(X_scaled[:5])

        print(f"✓ SVM training OK: {svm.n_iter_} iterations")
        return True

    except Exception as e:
        print(f"✗ SVM test failed: {e}")
        return False


def test_image_processing():
    """Test image processing operations"""
    print("\n" + "=" * 50)
    print("Testing Image Processing")
    print("=" * 50)

    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test resize
        resized = cv2.resize(img, (64, 64))

        # Test grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Test histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Test JPEG encoding
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Test PIL
        pil_img = Image.fromarray(img)

        print("✓ Image processing OK")
        print(f"  - Resize: {resized.shape}")
        print(f"  - Grayscale: {gray.shape}")
        print(f"  - JPEG encode: {len(buffer)} bytes")
        return True

    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False


def test_udp_socket():
    """Test UDP socket creation"""
    print("\n" + "=" * 50)
    print("Testing UDP Socket")
    print("=" * 50)

    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))  # Bind to any available port

        addr = sock.getsockname()
        sock.close()

        print(f"✓ UDP socket OK: bound to {addr[0]}:{addr[1]}")
        return True

    except Exception as e:
        print(f"✗ UDP socket failed: {e}")
        return False


def test_json_serialization():
    """Test JSON serialization"""
    print("\n" + "=" * 50)
    print("Testing JSON Serialization")
    print("=" * 50)

    try:
        import json
        import base64
        import numpy as np

        # Create test data
        data = {
            "frame_id": 123,
            "timestamp": 1234567890.123,
            "image": base64.b64encode(b"test_image_data").decode("utf-8"),
            "mask_type": "zombie",
            "options": {"quality": 85},
        }

        # Serialize
        json_str = json.dumps(data)

        # Deserialize
        parsed = json.loads(json_str)

        # Verify
        assert parsed["frame_id"] == 123
        assert parsed["mask_type"] == "zombie"

        print(f"✓ JSON serialization OK: {len(json_str)} bytes")
        return True

    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


def test_file_structure():
    """Test project file structure"""
    print("\n" + "=" * 50)
    print("Testing File Structure")
    print("=" * 50)

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "inference/face_detector.py",
        "inference/mask_overlay.py",
        "server/udp_server.py",
    ]

    all_ok = True

    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT FOUND")
            all_ok = False

    return all_ok


def test_model_files():
    """Test if model files exist (optional)"""
    print("\n" + "=" * 50)
    print("Testing Model Files (Optional)")
    print("=" * 50)

    model_files = [
        "models/face_detector_svm.pkl",
        "models/scaler.pkl",
        "models/config.json",
    ]

    found = 0

    for file in model_files:
        if os.path.exists(file):
            print(f"✓ {file}")
            found += 1
        else:
            print(f"○ {file} (not yet trained)")

    if found == 0:
        print("\nNote: No models found. Run training notebook first.")
        return None
    elif found < 3:
        print("\nWarning: Incomplete model files. Ensure all 3 files are present.")
        return False
    else:
        print("\n✓ All model files present")
        return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("   VIRTUAL TRY-ON - INSTALLATION TEST")
    print("=" * 60)

    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Camera Access", test_opencv_camera),
        ("HOG Extraction", test_hog_extraction),
        ("SVM Classifier", test_svm),
        ("Image Processing", test_image_processing),
        ("UDP Socket", test_udp_socket),
        ("JSON Serialization", test_json_serialization),
        ("File Structure", test_file_structure),
        ("Model Files", test_model_files),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("   TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "○ SKIP"
        print(f"{status:10s} {name}")

    print("\n" + "-" * 60)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print(f"Skipped: {skipped}/{len(tests)}")
    print("-" * 60)

    if failed == 0:
        print("\n✅ All critical tests passed!")
        print("You're ready to start training and using the system.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Check camera connection and permissions")
        print("  3. Verify file structure is complete")
        print("  4. See README.md for detailed setup instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
