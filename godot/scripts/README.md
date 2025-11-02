# 🎮 Godot Integration for SVM+ORB Face Detector

GDScript files for integrating the face detection backend with Godot 4.

## 📁 Files

- `FaceDetectorClient.gd` - UDP client untuk komunikasi dengan Python backend
- `WebcamFaceDetector.gd` - Controller untuk webcam capture dan display
- `face_detector_demo.tscn` - Demo scene (example)

## 🚀 Quick Start

### 1. Start Python Backend Server

First, start the detection backend (you'll need to create a UDP server):

```bash
cd backend
python server.py  # You need to create this
```

### 2. Use in Godot

Add `FaceDetectorClient.gd` to your scene:

```gdscript
# In your main scene
extends Node

var face_detector: FaceDetectorClient

func _ready():
    face_detector = FaceDetectorClient.new()
    add_child(face_detector)
    
    # Connect to server
    face_detector.connect_to_server("127.0.0.1", 5555)
    
    # Process image
    var img = Image.load_from_file("test.jpg")
    face_detector.detect_faces(img)
```

## 📝 Usage Examples

See individual `.gd` files for detailed API documentation.

## ⚠️ Note

This is a basic integration. For production use with Godot, you may want to:

1. Create a dedicated UDP server in Python (see backend README)
2. Optimize image encoding/decoding
3. Add error handling
4. Implement frame buffering for smoother video
