# 🎭 Virtual Try-On: Halloween Mask Overlay

Real-time Halloween mask overlay system using classical Machine Learning (HOG + SVM) for face detection. Designed for integration with Godot v4 frontend via UDP.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Godot Integration Guide](#godot-integration-guide)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Classical ML Face Detection**: HOG + Linear SVM (no deep learning)
- **Real-time Processing**: 20-30 FPS on standard hardware
- **Perspective-Correct Overlay**: Masks follow face orientation
- **UDP Server**: Fast communication with Godot frontend
- **ROI Tracking**: Optimized performance for video streams
- **Multi-mask Support**: Easy switching between different Halloween masks
- **CLI Interface**: Test with single images or live webcam

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GODOT FRONTEND (v4)                   │
│                     (UDP Client)                        │
└──────────────────────┬──────────────────────────────────┘
                       │ JSON over UDP
                       │ (Base64 images)
┌──────────────────────▼──────────────────────────────────┐
│                   PYTHON BACKEND                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ UDP Server (port 5001)                          │   │
│  │  - Receive: frame + mask_type                   │   │
│  │  - Send: overlay result + face coordinates      │   │
│  └────────────┬────────────────────────────────────┘   │
│               │                                         │
│  ┌────────────▼────────────────────────────────────┐   │
│  │ Face Detector (HOG + SVM)                       │   │
│  │  - Multi-scale sliding window                   │   │
│  │  - Non-maximum suppression                      │   │
│  │  - ROI tracking for performance                 │   │
│  └────────────┬────────────────────────────────────┘   │
│               │                                         │
│  ┌────────────▼────────────────────────────────────┐   │
│  │ Landmark Estimator                              │   │
│  │  - 5-point geometric estimation                 │   │
│  │  - Eyes, nose, mouth corners                    │   │
│  └────────────┬────────────────────────────────────┘   │
│               │                                         │
│  ┌────────────▼────────────────────────────────────┐   │
│  │ Mask Overlay                                    │   │
│  │  - Perspective transformation                   │   │
│  │  - Alpha blending                               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Requirements

### System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Webcam (for real-time testing)

### Python Dependencies

See `requirements.txt`:

```bash
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
Pillow>=9.0.0
tqdm>=4.62.0
```

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd virtual-tryon
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import cv2; import sklearn; import numpy; print('✓ All dependencies installed')"
```

---

## 🎓 Training the Model

### Option 1: Google Colab (Recommended)

1. **Upload Dataset to Google Drive**

   ```
   MyDrive/
   └── virtual_tryon/
       └── datasets/
           ├── faces/          # 2500+ face images
           └── non_faces/      # 2500+ non-face images
   ```

2. **Open Training Notebook**
   - Upload `01_train_face_detector.ipynb` to Google Colab
   - Or use: [Open in Colab](link-to-colab)

3. **Run All Cells**
   - The notebook will guide you through:
     - Mounting Google Drive
     - Loading dataset
     - Feature extraction (HOG)
     - Training SVM
     - Hard negative mining
     - Model evaluation
     - Saving models

4. **Download Trained Models**

   ```
   models/
   ├── face_detector_svm.pkl    # Trained SVM model
   ├── scaler.pkl               # Feature scaler
   └── config.json              # Configuration
   ```

### Option 2: Local Training

```bash
# Convert notebook to Python script
jupyter nbconvert --to python 01_train_face_detector.ipynb

# Run training
python 01_train_face_detector.py
```

### Expected Training Time

- Dataset preparation: 10-20 minutes
- Feature extraction: 15-30 minutes
- SVM training: 5-10 minutes
- Hard negative mining: 10-20 minutes per round
- **Total: 40-80 minutes** (on Google Colab with GPU)

### Expected Performance

- **Accuracy**: 85-92% on test set
- **Inference Speed**: 55-100ms per frame (without optimization)
- **With ROI Tracking**: 20-35ms per frame

---

## 💻 Usage

### 1. Command Line Interface (CLI)

#### Inference on Single Image

```bash
python app.py infer \
  --image input.jpg \
  --out output.jpg \
  --mask assets/masks/zombie.png \
  --show
```

**Arguments:**

- `--image`: Input image path
- `--out`: Output image path
- `--mask`: Halloween mask PNG (with alpha channel)
- `--show`: Display result window (optional)

#### Live Webcam

```bash
python app.py webcam \
  --camera 0 \
  --mask assets/masks/zombie.png \
  --show
```

**Arguments:**

- `--camera`: Camera device ID (default: 0)
- `--mask`: Halloween mask PNG
- `--show`: Display webcam feed (default: true)
- `--width`: Camera resolution width (optional)
- `--height`: Camera resolution height (optional)

**Controls:**

- Press `q` to quit
- Press `s` to save screenshot

### 2. UDP Server (for Godot)

```bash
python -m src.server.udp_server \
  --host 0.0.0.0 \
  --port 5001 \
  --model models/face_detector_svm.pkl \
  --scaler models/scaler.pkl \
  --config models/config.json \
  --masks assets/masks/
```

**Arguments:**

- `--host`: Server host (0.0.0.0 = all interfaces)
- `--port`: UDP port (default: 5001)
- `--model`: Path to trained SVM model
- `--scaler`: Path to feature scaler
- `--config`: Path to configuration
- `--masks`: Directory containing mask PNG files

**Server will automatically load all PNG files from masks directory.**

---

## 📡 API Documentation

### UDP Protocol

The server uses **JSON over UDP** for communication.

#### Request Format (Godot → Python)

```json
{
  "frame_id": 12345,
  "timestamp": 1698765432.123,
  "image": "base64_encoded_jpeg_string",
  "mask_type": "zombie",
  "options": {
    "resolution": [640, 480],
    "quality": 85
  }
}
```

**Fields:**

- `frame_id` (int): Unique frame identifier for synchronization
- `timestamp` (float, optional): Unix timestamp
- `image` (string): Base64-encoded JPEG image
- `mask_type` (string): Mask ID (filename without extension)
- `options` (object, optional):
  - `resolution` (array): Image dimensions [width, height]
  - `quality` (int): JPEG quality for response (0-100)

#### Response Format (Python → Godot)

```json
{
  "frame_id": 12345,
  "status": "success",
  "faces": [
    {
      "bbox": {
        "x": 120,
        "y": 80,
        "w": 150,
        "h": 150
      },
      "confidence": 0.95,
      "landmarks": {
        "left_eye": [140, 100],
        "right_eye": [220, 100],
        "nose": [180, 130],
        "left_mouth": [160, 160],
        "right_mouth": [200, 160]
      }
    }
  ],
  "overlay_image": "base64_encoded_jpeg_result",
  "processing_time_ms": 45.2
}
```

**Fields:**

- `frame_id` (int): Same as request for synchronization
- `status` (string): `"success"`, `"no_face"`, or `"error"`
- `faces` (array): List of detected faces
  - `bbox` (object): Bounding box `{x, y, w, h}`
  - `confidence` (float): Detection confidence (0-1)
  - `landmarks` (object): Facial keypoints
- `overlay_image` (string): Base64-encoded result with mask applied
- `processing_time_ms` (float): Server processing time

#### Status Codes

| Status    | Description                          |
| --------- | ------------------------------------ |
| `success` | Face detected, mask applied          |
| `no_face` | No face detected in frame            |
| `error`   | Processing error (see `error` field) |

---

## 🎮 Godot Integration Guide

### GDScript UDP Client Example

```gdscript
# UDPClient.gd
extends Node

var udp := PacketPeerUDP.new()
var server_host := "127.0.0.1"  # Change for production
var server_port := 5001
var frame_id := 0

func _ready():
    # Connect to server
    udp.set_dest_address(server_host, server_port)
    print("✓ Connected to Python server: %s:%d" % [server_host, server_port])

func send_frame(image: Image, mask_type: String) -> int:
    # Convert image to JPEG bytes
    var buffer := image.save_jpg_to_buffer(85)
    var base64_image := Marshalls.raw_to_base64(buffer)

    # Create request
    var request := {
        "frame_id": frame_id,
        "timestamp": Time.get_unix_time_from_system(),
        "image": base64_image,
        "mask_type": mask_type,
        "options": {
            "quality": 85
        }
    }

    # Serialize to JSON
    var json := JSON.stringify(request)

    # Send via UDP
    udp.put_packet(json.to_utf8_buffer())

    frame_id += 1
    return frame_id - 1

func receive_response() -> Dictionary:
    if udp.get_available_packet_count() > 0:
        var packet := udp.get_packet()
        var json_str := packet.get_string_from_utf8()

        # Parse JSON
        var json := JSON.new()
        var error := json.parse(json_str)

        if error == OK:
            var response: Dictionary = json.data
            return response

    return {}

func process_response(response: Dictionary) -> Image:
    if response.get("status") == "success":
        # Decode base64 image
        var base64_image: String = response.get("overlay_image", "")
        var buffer := Marshalls.base64_to_raw(base64_image)

        # Create image from JPEG buffer
        var image := Image.new()
        image.load_jpg_from_buffer(buffer)

        return image

    return null
```

### Webcam Capture Example

```gdscript
# WebcamOverlay.gd
extends Control

@onready var camera_feed := $CameraFeed  # TextureRect
@onready var result_display := $ResultDisplay  # TextureRect
@onready var udp_client := $UDPClient  # UDPClient node

var camera: CameraFeed
var pending_frames := {}
var current_mask := "zombie"

func _ready():
    # Get default camera
    var cameras := CameraServer.get_feed_count()
    if cameras > 0:
        camera = CameraServer.get_feed(0)
        camera.feed_is_active = true

func _process(delta):
    if camera and camera.feed_is_active:
        # Get camera image
        var image := camera.get_frame()

        if image:
            # Display original
            camera_feed.texture = ImageTexture.create_from_image(image)

            # Send to server (every other frame for performance)
            if Engine.get_frames_drawn() % 2 == 0:
                var sent_frame_id := udp_client.send_frame(image, current_mask)
                pending_frames[sent_frame_id] = Time.get_ticks_msec()

            # Check for response
            var response := udp_client.receive_response()
            if not response.is_empty():
                process_server_response(response)

func process_server_response(response: Dictionary):
    var frame_id := response.get("frame_id", -1)

    # Calculate latency
    if pending_frames.has(frame_id):
        var latency := Time.get_ticks_msec() - pending_frames[frame_id]
        print("Latency: %d ms" % latency)
        pending_frames.erase(frame_id)

    # Display result
    if response.get("status") == "success":
        var result_image := udp_client.process_response(response)
        if result_image:
            result_display.texture = ImageTexture.create_from_image(result_image)
    else:
        print("No face detected")

func change_mask(mask_name: String):
    current_mask = mask_name
    print("Switched to mask: %s" % mask_name)
```

### Error Handling

```gdscript
func send_frame_with_timeout(image: Image, mask_type: String, timeout_ms := 200):
    var frame_id := udp_client.send_frame(image, mask_type)
    var start_time := Time.get_ticks_msec()

    # Wait for response with timeout
    while Time.get_ticks_msec() - start_time < timeout_ms:
        var response := udp_client.receive_response()
        if not response.is_empty() and response.get("frame_id") == frame_id:
            return response
        await get_tree().create_timer(0.01).timeout

    # Timeout - display original frame
    print("Warning: Server timeout for frame %d" % frame_id)
    return {"status": "timeout"}
```

### Performance Optimization Tips

1. **Frame Skipping**: Process every 2nd or 3rd frame

   ```gdscript
   if Engine.get_frames_drawn() % 2 == 0:
       udp_client.send_frame(image, mask_type)
   ```

2. **Resolution Reduction**: Send 640x480 instead of 1080p

   ```gdscript
   image.resize(640, 480, Image.INTERPOLATE_BILINEAR)
   ```

3. **JPEG Quality**: Use 75-85 instead of 100

   ```gdscript
   var buffer := image.save_jpg_to_buffer(80)
   ```

4. **Async Processing**: Don't wait for each frame

   ```gdscript
   # Send frame and continue
   udp_client.send_frame(image, mask_type)

   # Check for any available response
   var response := udp_client.receive_response()
   ```

---

## ⚡ Performance Tuning

### Server-Side Optimizations

#### 1. Adjust Detection Parameters

Edit `src/inference/face_detector.py`:

```python
# Faster but less accurate
self.window_stride = 24  # Default: 16
self.scale_factor = 1.5  # Default: 1.2
self.min_scale = 0.8     # Default: 0.5

# More accurate but slower
self.window_stride = 8
self.scale_factor = 1.1
self.min_scale = 0.3
```

#### 2. ROI Tracking Frequency

Edit `src/server/udp_server.py`:

```python
# Full detection every N frames (higher = faster)
self.full_detection_interval = 15  # Default: 10
```

#### 3. Threading Configuration

```python
# Increase queue sizes for high-throughput
self.request_queue = queue.Queue(maxsize=10)  # Default: 5
```

### Client-Side Optimizations (Godot)

1. **Reduce Resolution**: 640x480 is ideal for real-time
2. **Frame Skip**: Process every 2-3 frames
3. **JPEG Quality**: 75-85 for balance
4. **Batch Requests**: Don't send if previous frame still pending

### Expected Performance

| Configuration              | FPS   | Latency | Quality    |
| -------------------------- | ----- | ------- | ---------- |
| **High Quality**           | 15-20 | 50-70ms | Excellent  |
| **Balanced** (Recommended) | 25-30 | 30-50ms | Good       |
| **High Performance**       | 35-45 | 20-30ms | Acceptable |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No faces detected"

**Possible causes:**

- Poor lighting
- Face too small in frame
- Extreme angle (>45°)
- Partial face visibility

**Solutions:**

- Improve lighting conditions
- Move closer to camera
- Face camera more directly
- Retrain model with more diverse angles

#### 2. Slow detection (< 15 FPS)

**Solutions:**

- Reduce image resolution
- Increase `window_stride`
- Enable ROI tracking
- Use frame skipping
- Check CPU usage

#### 3. UDP packets lost

**Solutions:**

- Increase buffer size: `buffer_size: 262144`
- Check network connection
- Reduce image size/quality
- Add frame dropping logic

#### 4. Mask alignment incorrect

**Solutions:**

- Check landmark estimation
- Adjust mask anchor points in `mask_overlay.py`
- Use simpler overlay mode: `use_perspective=False`

#### 5. Model accuracy low (< 80%)

**Solutions:**

- Add more training data
- Perform more hard negative mining rounds
- Adjust SVM `C` parameter
- Check data quality

### Debug Mode

Enable verbose logging:

```bash
# Add to server startup
python -m src.server.udp_server --verbose
```

Or in code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📁 Project Structure

```
virtual-tryon/
├── assets/
│   └── masks/
│       ├── zombie.png
│       ├── vampire.png
│       └── skeleton.png
├── datasets/
│   ├── faces/
│   └── non_faces/
├── models/
│   ├── face_detector_svm.pkl
│   ├── scaler.pkl
│   └── config.json
├── notebooks/
│   └── 01_train_face_detector.ipynb
├── src/
│   ├── inference/
│   │   ├── face_detector.py
│   │   └── mask_overlay.py
│   └── server/
│       └── udp_server.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- Dataset: [Your dataset source]
- Inspiration: Classical computer vision techniques
- Built for Godot v4 integration

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Email: [your-email]
- Discord: [your-discord]

---

**Happy Halloween! 🎃**
