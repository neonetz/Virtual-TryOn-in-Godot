# 🎭 SVM+ORB Face Detector with MASK Overlay

Classical computer vision pipeline for real-time face detection and mask overlay using **NO deep learning**.

## 🎯 Overview

This system uses:
- **Haar Cascade** for fast face region proposals
- **ORB (Oriented FAST and Rotated BRIEF)** for local feature extraction
- **Bag of Visual Words (BoVW)** with k-means codebook for fixed-length encoding
- **Linear SVM** for face classification
- **Alpha blending** for realistic mask overlay on nose-mouth-chin area
- **Optional eye-based rotation** for better mask alignment

**Performance**: 15-30 FPS on webcam (720p), depending on hardware.

---

## 📁 Project Structure

```
backend/
├── app.py                    # CLI entry point (infer, webcam)
├── requirements.txt
├── README.md
├── pipelines/
│   ├── __init__.py
│   ├── features.py          # ORB extraction + BoVW encoding
│   ├── infer.py             # Full detection pipeline
│   ├── overlay.py           # MASK alpha blending + rotation
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

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Place Your Models

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

## 💻 Usage

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

### "Cannot find cascade file"

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

## 🔬 ORB vs Deep Learning

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
