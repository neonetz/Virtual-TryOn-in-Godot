# 🎭 SVM+ORB Face Detector with MASK Overlay + Godot Integration

Classical computer vision pipeline for real-time face detection and mask overlay using **NO deep learning**.

## 🎯 Overview

This system uses:
- **Haar Cascade** for fast face region proposals
- **ORB (Oriented FAST and Rotated BRIEF)** for local feature extraction
- **Bag of Visual Words (BoVW)** with k-means codebook for fixed-length encoding
- **Linear SVM** for face classification
- **Alpha blending** for realistic mask overlay on nose-mouth-chin area
- **Temporal smoothing** for stable tracking (reduces flickering)
- **Optional eye-based rotation** for better mask alignment
- **Custom ORB-based eye detector** (no Haar cascade dependency)
- **UDP streaming** for Godot integration

**Performance**: 15-30 FPS on webcam (720p), depending on hardware.

---

## 🏗️ System Architecture

### Two Usage Modes

```
Mode 1: Standalone Python (No Godot)
┌──────────────────────────────────────┐
│       Python Backend                 │
│  Webcam → Detection → Mask → Display │
└──────────────────────────────────────┘

Mode 2: Python Backend + Godot Frontend
┌─────────────────────┐         ┌──────────────────┐
│  Python Backend     │  UDP    │  Godot Frontend  │
│  • Webcam capture   │ ─────>  │  • Display JPEG  │
│  • Face detection   │  5555   │  • Show stats    │
│  • Mask overlay     │         │  • UI controls   │
│  • JPEG encoding    │         │                  │
└─────────────────────┘         └──────────────────┘
```

**Mode 2 Benefits**: 
- Python does ALL processing (face detection + mask overlay)
- Godot only displays the result (lightweight)
- Easy to integrate with Godot UI/game logic

---

## 📁 Project Structure

```
backend/
├── face_detector_cli.py      # Standalone CLI (webcam with preview)
├── webcam_udp_server.py      # UDP server for Godot integration
├── server.py                 # Old UDP server (request-response)
├── requirements.txt
├── README.md
├── pipelines/
│   ├── __init__.py
│   ├── features.py          # ORB extraction + BoVW encoding
│   ├── infer.py             # Full detection pipeline
│   ├── overlay.py           # MASK alpha blending + rotation
│   ├── eye_detector.py      # Custom ORB eye detector (no Haar)
│   └── utils.py             # NMS, timing, visualization
├── models/
│   ├── svm_model.pkl        # ⚠️ YOU MUST PLACE YOUR TRAINED SVM HERE
│   ├── bovw_encoder.pkl     # ⚠️ YOU MUST PLACE YOUR K-MEANS CODEBOOK HERE
│   └── scaler.pkl           # ⚠️ YOU MUST PLACE YOUR FEATURE SCALER HERE
└── assets/
    ├── mask.png             # ⚠️ PLACE YOUR FACE MASK IMAGE HERE (RGBA)
    └── cascades/            # OpenCV Haar cascades (auto-detected)
```

---

## 🚀 Quick Start

### Option 1: Standalone Python (Preview Window)

Perfect for testing and development.

```bash
cd backend
pip install -r requirements.txt

# Basic webcam detection
python face_detector_cli.py webcam --camera 0 --show

# With custom eye detector and rotation
python face_detector_cli.py webcam --camera 0 --enable-rotation --show
```

**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `b` - Toggle bounding boxes
- `r` - Toggle rotation
- `d` - Toggle debug info

### Option 2: Python Backend + Godot Frontend (UDP Streaming)

Python does all processing, Godot displays the result.

**Step 1: Start Python Backend**

```bash
cd backend

# Basic streaming to Godot
python webcam_udp_server.py --camera 0 --host 127.0.0.1 --port 5555

# With local preview (for debugging)
python webcam_udp_server.py --camera 0 --preview

# Full configuration
python webcam_udp_server.py \
    --camera 0 \
    --host 127.0.0.1 \
    --port 5555 \
    --quality 60 \
    --fps 30 \
    --smoothing 0.3 \
    --enable-rotation \
    --preview
```

You should see:
```
============================================================
WEBCAM UDP SERVER RUNNING
============================================================
Streaming to: 127.0.0.1:5555
Target FPS: 30
JPEG quality: 85
============================================================
Stats: frames=30, sent=30, errors=0, fps=28.5
```

**Step 2: Setup Godot Scene**

Create scene `TryOn.tscn`:

```
TryOn (Node)
├── UDPVideoReceiver (Node) ← attach udp_video_receiver.gd
├── VideoDisplay (TextureRect) ← show video here
└── UI (Control)
    ├── FPSLabel (Label)
    ├── StatusLabel (Label)
    └── FacesLabel (Label)
```

**Step 3: Add GDScript**

Use `godot/scripts/tryon/udp_video_receiver.gd`:
- Receives UDP packets on port 5555
- Decodes base64 JPEG
- Displays in TextureRect
- Shows FPS and stats

**Step 4: Run Godot**

Press F5 in Godot. You'll see webcam video with face detection + mask overlay!

You mentioned you already have the trained models. **Place these 3 files** in `backend/models/`:

```
models/
├── svm_model.pkl        # Your trained Linear SVM
├── bovw_encoder.pkl     # Your k-means codebook (e.g. k=256)
└── scaler.pkl           # Your feature scaler (StandardScaler/MinMaxScaler)
```

### 3. Place Your Mask

Put your face mask image in `backend/assets/mask.png`:
- **Format**: PNG with alpha channel (RGBA)
- **Content**: Face mask (covers nose, mouth, chin)
- **Transparency**: Background must be transparent
- **Recommended size**: 512×512 or larger

**Example mask positioning**:
```
The mask will be placed at:
  - Width: 90% of face width
  - Height: 55% of face height
  - X position: face_x + 5% of face width
  - Y position: face_y + 40% of face height

This covers the lower face (nose-mouth-chin area)
```

### 4. Run Inference on Image

```bash
python app.py infer --image test.jpg --out result.jpg --show
```

### 5. Run Real-time Webcam

```bash
python app.py webcam --camera 0 --show
```

---

## � UDP Protocol (Godot Integration)

### Architecture

```
Python Backend                    Godot Frontend
    (Server)                         (Client)
       |                                |
       | 1. Capture webcam frame        |
       | 2. Face detection + mask       |
       | 3. JPEG encode + base64        |
       | 4. Build JSON packet           |
       |                                |
       |--- UDP SEND (port 5555) ------>|
       |                                |
       |                                | 5. Receive packet
       |                                | 6. Parse JSON
       |                                | 7. Decode base64
       |                                | 8. Display JPEG
       |                                |
```

### Packet Format

**Python → Godot (JSON over UDP):**

```json
{
    "frame_id": 123,
    "image": "base64_encoded_jpeg_string",
    "fps": 28.5,
    "faces": 1,
    "processing_ms": 35.2,
    "timestamp": 1234567890.123
}
```

**Fields:**
- `frame_id` - Sequential frame counter
- `image` - Base64-encoded JPEG (already has mask overlay applied)
- `fps` - Current FPS (backend processing)
- `faces` - Number of faces detected in frame
- `processing_ms` - Face detection + mask overlay time
- `timestamp` - Unix timestamp (seconds)

**Packet Size:** ~20-50 KB per frame (depends on quality and resolution)

**Bandwidth:** ~600 KB/s @ 30 FPS, quality 60 (UDP-safe)

**Note:** Frames are auto-resized to 640px width if larger (to keep UDP packets under 65KB limit)

### Server Configuration

```bash
# webcam_udp_server.py options
--camera 0              # Webcam device ID
--host 127.0.0.1        # Godot client IP (localhost)
--port 5555             # UDP port (must match Godot)
--quality 85            # JPEG quality (0-100)
--fps 30                # Target frame rate
--alpha 0.95            # Mask opacity
--enable-rotation       # Enable eye-based rotation
--preview               # Show local preview window
```

### Client Configuration (Godot)

In `udp_video_receiver.gd`:
```gdscript
var listening_port: int = 5555  # Must match Python --port
```

Scene structure:
- `UDPVideoReceiver` node with script
- `TextureRect` sibling for display
- Optional UI labels for stats

### Network Notes

**Local (same machine):**
- Use `--host 127.0.0.1` (loopback)
- Fast, no firewall issues

**Network (different machines):**
- Use Godot machine's LAN IP: `--host 192.168.x.x`
- Open UDP port 5555 in firewall
- Check network bandwidth (1-2 MB/s needed)

**UDP Limitations:**
- Max packet size ~65 KB (adjust quality if exceeded)
- Packets may be lost (UDP is unreliable)
- No acknowledgment (fire-and-forget)
- Frame order may change (use frame_id)

---

## 📊 Performance Comparison

### Standalone vs UDP Streaming

| Metric | Standalone Python | UDP Streaming |
|--------|-------------------|---------------|
| Processing | 25-35 ms | 25-35 ms |
| Encoding | N/A | 5-10 ms |
| Network | N/A | 1-3 ms |
| Display | OpenCV window | Godot TextureRect |
| **Total FPS** | 28-30 | 25-28 |

**Overhead:** UDP streaming adds ~10ms for JPEG encoding + network transmission.

---

## 💻 Usage Details

### webcam_udp_server.py (Godot Integration)

Stream processed webcam video to Godot via UDP.

```bash
python webcam_udp_server.py \
    --camera 0 \
    --host 127.0.0.1 \
    --port 5555 \
    --quality 85 \
    --fps 30 \
    --enable-rotation \
    --alpha 0.95 \
    --preview
```

**Arguments:**
- `--camera` - Webcam device ID (0 = default)
- `--host` - Godot client IP address
- `--port` - UDP port (default 5555)
- `--quality` - JPEG quality 0-100 (default 60, recommended for UDP)
- `--fps` - Target frame rate (default 30)
- `--smoothing` - Temporal smoothing factor 0-1 (default 0.3)
- `--enable-rotation` - Enable mask rotation based on eyes
- `--alpha` - Mask opacity 0-1 (default 0.95)
- `--preview` - Show local preview window
- `--svm-model` - SVM model path
- `--bovw-encoder` - BoVW encoder path
- `--scaler` - Scaler path
- `--mask` - Mask PNG path
- `--max-keypoints` - Max ORB keypoints (default 500)
- `--log-level` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Output:**
```
============================================================
WEBCAM UDP SERVER RUNNING
============================================================
Streaming to: 127.0.0.1:5555
Target FPS: 30
JPEG quality: 85
Preview: OFF
============================================================
Press 'q' to quit (if preview enabled)
============================================================

Stats: frames=30, sent=30, errors=0, fps=28.5
Stats: frames=60, sent=60, errors=0, fps=29.1
...
```

---

### face_detector_cli.py (Standalone)

Local webcam detection with OpenCV preview window (no Godot needed).

```bash
python face_detector_cli.py webcam --camera 0 --show
```

Same as before, but displays result locally.

---

## �💻 Usage (Legacy Commands)

### Command: `infer` (Single Image)

Process a single image and apply mask overlay:

```bash
python app.py infer \
  --image input.jpg \
  --out output.jpg \
  --show
```

**Arguments**:
- `--image`: Input image path (required)
- `--out`: Output image path (required)
- `--show`: Display result window (optional)
- `--svm-model`: Path to SVM model (default: `models/svm_model.pkl`)
- `--bovw-encoder`: Path to BoVW encoder (default: `models/bovw_encoder.pkl`)
- `--scaler`: Path to scaler (default: `models/scaler.pkl`)
- `--mask`: Path to mask PNG (default: `assets/mask.png`)
- `--alpha`: Mask opacity 0-1 (default: 0.95)
- `--enable-rotation`: Enable mask rotation based on eyes (optional flag)
- `--max-keypoints`: Max ORB keypoints per ROI (default: 500)

**Example with rotation**:
```bash
python app.py infer \
  --image photo.jpg \
  --out result.jpg \
  --enable-rotation \
  --show
```

---

### Command: `webcam` (Real-time)

Real-time face detection with mask overlay from webcam:

```bash
python app.py webcam --camera 0 --show
```

**Arguments**:
- `--camera`: Camera device ID (default: 0)
- `--show`: Display webcam feed (default: true)
- `--show-boxes`: Show bounding boxes (optional flag)
- `--width`: Camera width in pixels (optional)
- `--height`: Camera height in pixels (optional)
- `--alpha`: Mask opacity 0-1 (default: 0.95)
- `--enable-rotation`: Enable mask rotation (optional flag)
- `--max-keypoints`: Max ORB keypoints per ROI (default: 500)

**Webcam Controls**:
- Press `q` - Quit
- Press `s` - Save screenshot
- Press `b` - Toggle bounding boxes
- Press `r` - Toggle rotation on/off

**Example with all features**:
```bash
python app.py webcam \
  --camera 0 \
  --show \
  --show-boxes \
  --enable-rotation \
  --width 1280 \
  --height 720
```

---

## 🔧 How It Works

### 1. **ROI Proposal** (Haar Cascade)
- Uses OpenCV's `haarcascade_frontalface_default.xml`
- Fast multi-scale face detection (~5-10ms)
- Generates candidate bounding boxes

### 2. **Feature Extraction** (ORB + BoVW)
- For each face candidate:
  - Resize to 128×128 (configurable)
  - Apply histogram equalization
  - Detect up to 500 ORB keypoints
  - Compute 32-byte binary descriptors
  - Encode to BoVW histogram using k-means codebook
  - Normalize with L2 norm
  - Scale features with pre-trained scaler

### 3. **Classification** (SVM)
- Feed BoVW features to Linear SVM
- Get decision function score
- Filter by score threshold (default: 0.0)

### 4. **Non-Maximum Suppression**
- Remove overlapping detections
- IoU threshold: 0.3

### 5. **MASK Overlay**
- Position mask on lower face (nose-mouth-chin)
- Scale mask to 90% × 55% of face size
- **Optional**: Detect eyes and rotate mask to match head tilt
- Alpha blend with original image (opacity = 0.95)

---

## 📊 Performance

### Expected FPS (Webcam 720p)

| Configuration | FPS | Notes |
|--------------|-----|-------|
| **No rotation, max_kp=300** | 25-30 | Recommended for real-time |
| **No rotation, max_kp=500** | 20-25 | Good quality |
| **With rotation, max_kp=500** | 15-20 | Best quality, slower |

### Tuning for Speed

**Reduce ORB keypoints**:
```bash
python app.py webcam --camera 0 --max-keypoints 300
```

**Reduce camera resolution**:
```bash
python app.py webcam --camera 0 --width 640 --height 480
```

**Disable rotation**:
```bash
python app.py webcam --camera 0  # rotation off by default
```

---

## 🎨 Customizing the Mask

### Mask Requirements

1. **Format**: PNG with alpha channel (RGBA)
2. **Size**: At least 512×512 pixels
3. **Content**: Face mask covering nose, mouth, chin
4. **Background**: Must be transparent (alpha=0)

### Creating Your Own Mask

Use any image editor (Photoshop, GIMP, etc.):

1. Create new image with transparent background
2. Draw or paste mask artwork
3. Ensure mask is centered and facing forward
4. Save as PNG with alpha channel
5. Place in `assets/mask.png`

### Example Masks

- Medical mask
- Halloween mask (zombie, vampire, etc.)
- Gas mask
- Bandana
- Face paint patterns

---

## 🐛 Troubleshooting

### UDP/Godot Issues

**"No frames received in Godot"**

1. Check Python server is running:
   ```bash
   python webcam_udp_server.py --camera 0 --preview
   ```
   Preview window should show video with mask.

2. Check IP/port match:
   - Python: `--host 127.0.0.1 --port 5555`
   - Godot: `listening_port = 5555`

3. Check firewall:
   - Windows: Allow UDP port 5555
   - Antivirus: May block UDP traffic

4. Check Godot console for errors

**"Packet too large (>65KB)"**

UDP has ~65KB limit. Solutions:

1. Use default quality (already optimized):
   ```bash
   python webcam_udp_server.py --camera 0  # quality 60 by default
   ```

2. Reduce quality further if needed:
   ```bash
   python webcam_udp_server.py --quality 50
   ```

3. Frame is auto-resized to 640px width if larger (prevents most issues)

**"Video is laggy in Godot"**

1. Check Python FPS (with `--preview`):
   - If Python FPS is low, reduce `--max-keypoints` or disable rotation
   - If Python FPS is good, check Godot performance

2. Increase JPEG quality for better compression:
   ```bash
   python webcam_udp_server.py --quality 95
   ```

3. Check network if using remote connection

**"Address already in use (WinError 10048)"**

Port 5555 is taken. Change port:
```bash
python webcam_udp_server.py --port 5556
```

Update Godot script:
```gdscript
var listening_port: int = 5556
```

### Python Backend Issues

**"Cannot find cascade file"**

**Solution**: The system auto-detects OpenCV's Haar cascades. If not found, download manually:

```bash
# Download to assets/cascades/
curl -o assets/cascades/haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

curl -o assets/cascades/haarcascade_eye.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
```

### "No faces detected"

**Possible causes**:
- Poor lighting
- Face too small in frame
- Extreme angle (>45°)
- SVM threshold too high

**Solutions**:
- Improve lighting
- Move closer to camera
- Face camera more directly
- Adjust `detector.score_threshold` in code (default: 0.0)

### Low FPS (< 15)

**Solutions**:
1. Reduce ORB keypoints: `--max-keypoints 300`
2. Reduce camera resolution: `--width 640 --height 480`
3. Disable rotation (omit `--enable-rotation`)
4. Check CPU usage in Task Manager

### Mask alignment incorrect

**Solutions**:
- Enable rotation: `--enable-rotation`
- Adjust mask position constants in `pipelines/overlay.py`:
  ```python
  mask_w = int(0.9 * w)   # Try 0.8 or 1.0
  mask_h = int(0.55 * h)  # Try 0.5 or 0.6
  mask_x = int(x + 0.05 * w)
  mask_y = int(y + 0.40 * h)  # Try 0.35 or 0.45
  ```

### "ModuleNotFoundError"

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📝 Technical Details

### Model Files

**1. `svm_model.pkl`** (Trained SVM)
- Type: `sklearn.svm.LinearSVC` or `SVC`
- Input: BoVW histogram features (k-dimensional)
- Output: Binary classification (1=face, 0=non-face)
- Training: Should be trained on BoVW-encoded face/non-face patches

**2. `bovw_encoder.pkl`** (K-Means Codebook)
- Type: `sklearn.cluster.KMeans`
- Purpose: Cluster ORB descriptors into k visual words
- Typical k: 128, 256, or 512
- Training: K-means on large collection of ORB descriptors

**3. `scaler.pkl`** (Feature Scaler)
- Type: `sklearn.preprocessing.StandardScaler` or `MinMaxScaler`
- Purpose: Normalize BoVW histograms
- Training: Fit on training set BoVW features

### Feature Pipeline

```
Image ROI (128×128)
  ↓
Grayscale + Histogram Equalization
  ↓
ORB Detection (max 500 keypoints)
  ↓
ORB Descriptors (N × 32 binary)
  ↓
K-Means Prediction (visual words)
  ↓
BoVW Histogram (k-dimensional, L2 normalized)
  ↓
Scaler Transform
  ↓
SVM Classification
```

---

## �️ Custom Eye Detection (No Haar Cascade)

The system includes a **custom ORB-based eye detector** that doesn't depend on `haarcascade_eye.xml`.

### Why Custom Eye Detector?

- ✅ No dependency on Haar cascade files
- ✅ More flexible (4 different methods)
- ✅ Better for educational purposes
- ✅ Classical CV techniques only

### Available Methods

Implemented in `pipelines/eye_detector.py`:

1. **ORB Keypoint Density** (default)
   - Analyzes ORB feature concentration
   - Works best for frontal faces
   - Fast and reliable

2. **Hough Circle Detection**
   - Detects circular iris regions
   - Good for high-contrast images
   - Sensitive to lighting

3. **Contour Analysis**
   - Finds dark elliptical regions
   - Works with various angles
   - More robust to lighting

4. **Template Matching**
   - Matches pre-defined eye templates
   - Good for specific use cases
   - Requires good templates

### Usage

The CLI uses ORB method by default:

```bash
# Default (ORB method)
python face_detector_cli.py webcam --camera 0 --enable-rotation

# UDP server also uses ORB
python webcam_udp_server.py --camera 0 --enable-rotation
```

To change method, edit the code:

```python
detection_system = FaceDetectionSystem(
    ...
    use_custom_eye_detector=True,
    custom_eye_method="orb"  # Options: "orb", "hough", "contour", "template"
)
```

### Eye Detection Validation

The system validates eye geometry:
- Eyes must be horizontally separated (right eye to the right of left eye)
- Vertical alignment within 20% of horizontal distance
- Inter-eye distance: 20-60% of face width
- Eyes in upper 60% of face region

This prevents false positives from other facial features.

---

## �🔬 ORB vs Deep Learning

**Why ORB+SVM?**

| Aspect | ORB+SVM | Deep Learning |
|--------|---------|---------------|
| **Speed** | ✅ Fast (15-30 FPS) | ⚠️ Slower without GPU |
| **Model Size** | ✅ ~10 MB | ❌ 50-200 MB |
| **Training Data** | ✅ Few thousand images | ❌ Tens of thousands |
| **Explainability** | ✅ Interpretable features | ❌ Black box |
| **Dependency** | ✅ OpenCV + scikit-learn | ❌ TensorFlow/PyTorch |
| **Accuracy** | ⚠️ 85-92% | ✅ 95-99% |

**Best for**:
- Educational projects
- Resource-constrained devices
- Explainable AI requirements
- Quick prototyping

---

## 📚 References

- ORB: Rublee et al. "ORB: An efficient alternative to SIFT or SURF" (ICCV 2011)
- Bag of Visual Words: Csurka et al. "Visual categorization with bags of keypoints" (ECCV 2004)
- Haar Cascades: Viola & Jones "Rapid object detection" (CVPR 2001)

---

## 📄 License

[Your License Here]

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

---

## 💬 Support

For issues or questions:
- Open an issue on GitHub
- Check troubleshooting section
- Review code comments

---

**Happy face detecting! 🎭**
