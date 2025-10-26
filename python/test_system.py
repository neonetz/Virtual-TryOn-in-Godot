"""
Test Script for Virtual Try-On System
Tests each component individually and verifies integration
"""

import sys
import time


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*60)
    print("TEST 1: Checking Python Dependencies")
    print("="*60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    print("\n✓ All dependencies installed correctly")
    return True


def test_webcam():
    """Test webcam access"""
    print("\n" + "="*60)
    print("TEST 2: Testing Webcam Access")
    print("="*60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Cannot open webcam")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Cannot read from webcam")
            return False
        
        print(f"✓ Webcam accessible")
        print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
        return True
        
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
        return False


def test_components():
    """Test if all Python modules can be imported"""
    print("\n" + "="*60)
    print("TEST 3: Testing Python Modules")
    print("="*60)
    
    modules = [
        'webcam_capture',
        'body_detector',
        'frame_encoder',
        'udp_server',
        'clothing_overlay'
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}.py loaded successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
    
    print("\n✓ All modules loaded successfully")
    return True


def test_body_detection():
    """Test body detection with a sample frame"""
    print("\n" + "="*60)
    print("TEST 4: Testing Body Detection")
    print("="*60)
    
    try:
        import cv2
        from body_detector import BodyDetector
        
        detector = BodyDetector()
        print("✓ BodyDetector initialized")
        
        # Capture a test frame
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Cannot access webcam for detection test")
            return False
        
        print("  Capturing test frames (this takes ~2 seconds)...")
        
        # Let background model learn
        for i in range(35):
            ret, frame = cap.read()
            if ret:
                detector.detect_upper_body(frame)
        
        cap.release()
        
        print("✓ Body detection system operational")
        return True
        
    except Exception as e:
        print(f"✗ Body detection test failed: {e}")
        return False


def test_udp_server():
    """Test UDP server initialization"""
    print("\n" + "="*60)
    print("TEST 5: Testing UDP Server")
    print("="*60)
    
    try:
        from udp_server import UDPServer
        
        server = UDPServer(host='127.0.0.1', port=9999)
        print("✓ UDPServer created")
        
        if server.start():
            print("✓ UDP Server started on 127.0.0.1:9999")
            time.sleep(0.5)
            server.stop()
            print("✓ UDP Server stopped cleanly")
            return True
        else:
            print("✗ Failed to start UDP server")
            return False
            
    except Exception as e:
        print(f"✗ UDP server test failed: {e}")
        return False


def test_frame_encoding():
    """Test frame encoding"""
    print("\n" + "="*60)
    print("TEST 6: Testing Frame Encoding")
    print("="*60)
    
    try:
        import cv2
        import numpy as np
        from frame_encoder import FrameEncoder
        
        encoder = FrameEncoder(jpeg_quality=85)
        print("✓ FrameEncoder initialized")
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 300)
        
        # Encode frame
        jpeg_data = encoder.encode_frame(test_frame)
        if jpeg_data is None:
            print("✗ Failed to encode frame")
            return False
        
        print(f"✓ Frame encoded to JPEG ({len(jpeg_data)} bytes)")
        
        # Create packet
        packet = encoder.create_packet(test_frame, test_bbox)
        if packet is None:
            print("✗ Failed to create packet")
            return False
        
        print(f"✓ Packet created ({len(packet)} bytes)")
        return True
        
    except Exception as e:
        print(f"✗ Frame encoding test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("VIRTUAL TRY-ON SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Dependencies", test_imports),
        ("Webcam", test_webcam),
        ("Modules", test_components),
        ("Body Detection", test_body_detection),
        ("UDP Server", test_udp_server),
        ("Frame Encoding", test_frame_encoding),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print("\n" + "-"*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - System ready to run!")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Launch Godot and start the Try-On scene")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please fix errors before running")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
