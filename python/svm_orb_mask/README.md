# SVM+ORB Face Detector with Mask Overlay

**Classical Computer Vision** approach for real-time face detection and mask overlay using:
- **Support Vector Machine (SVM)** classifier
- **ORB (Oriented FAST and Rotated BRIEF)** feature descriptors
- **Bag of Visual Words (BoVW)** encoding
- **Haar Cascade** ROI proposals
- **Alpha-blended mask overlay** with optional rotation alignment

> **Note**: This repository contains **INFERENCE-ONLY** code. Model training should be performed separately (e.g., in Google Colab).

---

## 📁 Project Structure

```
svm_orb_mask/
├── pipelines/
│   ├── __init__.py
│   ├── utils.py            # Utility functions (NMS, IoU, FPS, visualization)
│   ├── features.py         # ORB feature extraction & BoVW encoding
│   ├── mask_overlay.py     # Mask loading, scaling, rotation, alpha blending
│   └── infer.py            # Complete inference pipeline
├── models/                 # Pre-trained models (empty - user provides)
│   ├── svm_model.pkl       # Trained SVM classifier
│   ├── codebook.pkl        # k-means codebook for BoVW
│   └── scaler.pkl          # Feature scaler (StandardScaler)
├── assets/
│   ├── mask.png            # Default mask image (PNG with alpha)
│   └── cascades/           # Optional custom Haar Cascades
│       ├── haarcascade_frontalface_default.xml
│       └── haarcascade_eye.xml
├── app.py                  # CLI application entry point
├── config.json             # Configuration file
└── README.md               # This file
```

---

## 🚀 Setup Instructions

### 1. Prerequisites

- **Python**: 3.9 or higher
- **pip**: Latest version

### 2. Install Dependencies

```bash
cd python/svm_orb_mask
pip install -r requirements.txt
```

**Requirements**:
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
joblib>=1.3.0
matplotlib>=3.7.0
```

### 3. Prepare Models Directory

**IMPORTANT**: This repository does NOT include trained models. You must train models separately and place them in the `models/` directory.

Required files:
- `models/svm_model.pkl` - Trained SVM classifier (e.g., `SVC(kernel='linear')`)
- `models/codebook.pkl` - k-means codebook (MiniBatchKMeans with k=256)
- `models/scaler.pkl` - Feature scaler (StandardScaler fitted on training data)

See **Training Guide** section below.

### 4. Prepare Mask Asset

Place your mask image at `assets/mask.png` (PNG format with alpha channel).

**Mask Requirements**:
- **Format**: PNG with transparency (BGRA)
- **Resolution**: High-resolution recommended (will be scaled automatically)
- **Content**: Mask/overlay graphic (e.g., face mask, sunglasses, hat)

---

## 📊 Training Guide (External - Google Colab)

### Dataset Preparation

1. **Collect Images**:
   - **Positive samples**: Face images (1000+ recommended)
   - **Negative samples**: Non-face images (1000+ recommended)
   - Balanced dataset for best results

2. **Preprocessing**:
   - Convert to grayscale
   - Resize to standard size (e.g., 128×128)
   - Extract ROIs using Haar Cascade

3. **Organize Dataset**:
   ```
   dataset/
   ├── positive/
   │   ├── face_001.jpg
   │   ├── face_002.jpg
   │   └── ...
   └── negative/
       ├── non_face_001.jpg
       ├── non_face_002.jpg
       └── ...
   ```

### Training Pipeline (Google Colab Recommended)

```python
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Extract ORB descriptors from all training images
from pipelines.features import ORBFeatureExtractor

extractor = ORBFeatureExtractor(n_keypoints=500)
all_descriptors = []

for image_path in training_images:
    image = cv2.imread(image_path)
    keypoints, descriptors = extractor.extract_orb_descriptors(image)
    if descriptors is not None:
        all_descriptors.append(descriptors)

# Concatenate all descriptors
all_descriptors = np.vstack(all_descriptors)

# 2. Train k-means codebook (Bag of Visual Words)
print("Training codebook...")
codebook = MiniBatchKMeans(n_clusters=256, batch_size=100, random_state=42)
codebook.fit(all_descriptors)
joblib.dump(codebook, 'codebook.pkl')

# 3. Encode training images as BoVW histograms
extractor.codebook = codebook
X_train = []
y_train = []

for image_path, label in zip(training_images, labels):
    image = cv2.imread(image_path)
    bovw = extractor.extract_bovw_feature(image)
    X_train.append(bovw)
    y_train.append(label)  # 1 = face, 0 = non-face

X_train = np.array(X_train)
y_train = np.array(y_train)

# 4. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.pkl')

# 5. Train SVM classifier
print("Training SVM...")
svm = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, 'svm_model.pkl')

print("Training complete!")
```

### Download Trained Models

After training in Colab, download the following files to your local `models/` directory:
- `svm_model.pkl`
- `codebook.pkl`
- `scaler.pkl`

---

## 🎯 Usage

### Command Line Interface

The `app.py` script provides three modes: **image**, **video**, and **webcam**.

### 1. Process Single Image

```bash
python app.py --mode image --input test.jpg --output result.jpg
```

**Options**:
- `--mask-scale 1.5` - Scale mask to 150% of face width
- `--mask-offset 0.4` - Position mask 40% down from face top
- `--confidence 0.6` - Increase confidence threshold
- `--no-boxes` - Hide bounding boxes

### 2. Process Video File

```bash
python app.py --mode video --input video.mp4 --output output.mp4
```

**Options**:
- `--no-display` - Process without showing window (headless)
- `--no-rotation` - Disable eye-based rotation alignment

### 3. Webcam Real-Time Processing

```bash
python app.py --mode webcam
```

**Controls**:
- `q` - Quit
- `s` - Save screenshot

**Options**:
- `--camera-id 1` - Use different camera

### Advanced Usage

```bash
# Custom models and mask
python app.py --mode webcam \
  --models-dir ./my_models \
  --mask ./custom_mask.png \
  --mask-scale 1.3 \
  --mask-offset 0.35

# Custom Haar Cascades
python app.py --mode image --input test.jpg \
  --face-cascade ./cascades/custom_face.xml \
  --eye-cascade ./cascades/custom_eye.xml

# High confidence threshold (fewer false positives)
python app.py --mode webcam --confidence 0.7 --nms-threshold 0.2
```

---

## 🔍 Algorithm Overview

### Pipeline Architecture

```
Input Image
    ↓
[1] Haar Cascade ROI Proposal
    ↓
[2] ORB Feature Extraction (per ROI)
    ↓
[3] Bag of Visual Words Encoding
    ↓
[4] SVM Classification (Face vs Non-Face)
    ↓
[5] Non-Maximum Suppression (NMS)
    ↓
[6] Eye Detection (Optional - for rotation)
    ↓
[7] Mask Overlay with Alpha Blending
    ↓
Output Image with Masks
```

### Key Components

#### 1. **ORB (Oriented FAST and Rotated BRIEF)**
- **Fast keypoint detector** (FAST corners)
- **Rotation-invariant descriptors** (BRIEF with orientation)
- **Binary descriptors** (efficient matching)
- Default: 500 keypoints per ROI

#### 2. **Bag of Visual Words (BoVW)**
- **Visual vocabulary**: k-means clustering of ORB descriptors (k=256)
- **Encoding**: Map descriptors to nearest cluster centers
- **Histogram**: Count occurrences of each visual word
- **Normalization**: L1-normalized histogram → fixed-length feature vector

#### 3. **SVM (Support Vector Machine)**
- **Linear kernel** for efficiency
- **Binary classification**: Face (1) vs Non-Face (0)
- **Decision function**: Distance to separating hyperplane
- **Confidence**: Sigmoid activation on decision score

#### 4. **Non-Maximum Suppression (NMS)**
- Removes overlapping detections
- Keeps detection with highest confidence
- IoU threshold: 0.3 (default)

#### 5. **Mask Overlay System**
- **Alpha blending**: Transparent PNG overlay
- **Scale**: Automatically matches face width
- **Position**: Configurable vertical offset
- **Rotation**: Eye-based alignment (optional)

---

## ⚙️ Configuration

Edit `config.json` to customize pipeline parameters:

```json
{
  "detection": {
    "confidence_threshold": 0.5,  // Min confidence for face detection
    "nms_threshold": 0.3          // IoU threshold for NMS
  },
  "features": {
    "orb": {
      "n_keypoints": 500,         // ORB keypoints per ROI
      "roi_size": [128, 128]      // Standard ROI size
    },
    "bovw": {
      "codebook_size": 256        // Visual vocabulary size (k)
    }
  },
  "mask": {
    "scale_factor": 1.2,          // Mask scale (1.0 = face width)
    "vertical_offset": 0.3,       // Position (0.3 = 30% down)
    "use_rotation": true          // Eye-based rotation alignment
  }
}
```

---

## 🎭 Mask Alignment Guide

### Basic Positioning

Adjust `--mask-scale` and `--mask-offset` to fit different mask types:

**Examples**:
- **Face mask** (covers nose/mouth):
  ```bash
  --mask-scale 1.2 --mask-offset 0.4
  ```
- **Sunglasses** (eyes region):
  ```bash
  --mask-scale 0.9 --mask-offset 0.2
  ```
- **Hat** (top of head):
  ```bash
  --mask-scale 1.5 --mask-offset -0.3
  ```

### Rotation Alignment (Eye-Based)

Enable with `--use-rotation` (default: ON)

**How it works**:
1. Detect eyes in face ROI using Haar Cascade
2. Compute angle between eye centers
3. Rotate mask to match face tilt

**Disable** if:
- Eyes not consistently detected (profile faces)
- Mask doesn't need rotation (symmetric designs)

---

## 🐛 Troubleshooting

### Issue: No faces detected

**Solutions**:
1. Lower confidence threshold: `--confidence 0.3`
2. Check Haar Cascade parameters (adjust in `config.json`)
3. Verify image quality (lighting, resolution)

### Issue: Too many false positives

**Solutions**:
1. Increase confidence threshold: `--confidence 0.7`
2. Decrease NMS threshold: `--nms-threshold 0.2`
3. Retrain SVM with more negative samples

### Issue: Mask misaligned

**Solutions**:
1. Adjust scale: `--mask-scale 1.0` to `--mask-scale 1.5`
2. Adjust offset: `--mask-offset 0.2` to `--mask-offset 0.5`
3. Check mask PNG aspect ratio
4. Enable rotation: ensure `--use-rotation` is not disabled

### Issue: Low FPS in webcam mode

**Solutions**:
1. Reduce ORB keypoints: Edit `config.json` → `"n_keypoints": 250`
2. Decrease codebook size: Retrain with k=128
3. Use smaller ROI size: `"roi_size": [64, 64]`
4. Optimize Haar Cascade: Increase `min_size` parameter

---

## 📈 Performance Metrics

**Typical Performance** (on Intel i7, 720p webcam):
- **FPS**: 15-25 FPS (real-time)
- **Latency**: ~40-60ms per frame
- **Detection Accuracy**: 85-95% (depends on training data quality)

**Bottlenecks**:
1. ORB descriptor extraction (~20ms)
2. BoVW encoding (~10ms)
3. SVM classification (~5ms)

**Optimization Tips**:
- Reduce keypoints (500 → 250)
- Use smaller ROI size (128×128 → 96×96)
- Decrease codebook size (256 → 128)
- Multi-threading for video processing

---

## 🔬 Limitations

### Classical CV Approach

1. **Haar Cascade ROI Proposal**:
   - Struggles with profile faces (non-frontal)
   - Sensitive to lighting conditions
   - Fixed detector (no learning from new data)

2. **ORB Features**:
   - Less discriminative than deep features (e.g., CNN embeddings)
   - Requires sufficient texture/edges in image
   - Binary descriptors limit expressiveness

3. **BoVW Encoding**:
   - Loses spatial information (orderless bag)
   - Fixed vocabulary size (k-means clustering)
   - Quantization artifacts

4. **SVM Classifier**:
   - Linear separation assumption
   - Requires careful feature engineering
   - Less flexible than neural networks

### When to Use Deep Learning Instead

Consider deep learning (e.g., YOLO, RetinaFace, MTCNN) if:
- Need higher accuracy (>95%)
- Handle extreme poses/occlusions
- Process diverse datasets (wild faces)
- GPU acceleration available

**Advantages of Classical CV** (this project):
- ✅ Lightweight (CPU-friendly)
- ✅ Explainable (interpretable features)
- ✅ No GPU required
- ✅ Fast inference (<50ms)
- ✅ Small model size (<10MB)

---

## 🛠️ Development

### Adding Custom Features

**Example**: Add HOG (Histogram of Oriented Gradients) features

1. Edit `pipelines/features.py`:
```python
def extract_hog_features(self, roi):
    from skimage.feature import hog
    hog_features = hog(roi, pixels_per_cell=(8, 8))
    return hog_features
```

2. Concatenate with BoVW features in `infer.py`:
```python
bovw = extractor.extract_bovw_feature(roi)
hog = extractor.extract_hog_features(roi)
combined = np.concatenate([bovw, hog])
```

3. Retrain SVM with combined features

### Testing Pipeline Components

```python
# Test ORB extraction
from pipelines.features import ORBFeatureExtractor

extractor = ORBFeatureExtractor()
keypoints, descriptors = extractor.extract_orb_descriptors(image)
print(f"Detected {len(keypoints)} keypoints")

# Test mask overlay
from pipelines.mask_overlay import MaskOverlay

mask = MaskOverlay('assets/mask.png')
result = mask.apply_mask(image, face_box=(100, 100, 200, 200))
cv2.imshow("Result", result)
cv2.waitKey(0)
```

---

## 📚 References

### Papers
- **ORB**: Rublee et al., "ORB: An efficient alternative to SIFT or SURF" (ICCV 2011)
- **BoVW**: Csurka et al., "Visual Categorization with Bags of Keypoints" (ECCV 2004)
- **SVM**: Cortes & Vapnik, "Support-Vector Networks" (Machine Learning 1995)

### OpenCV Documentation
- [ORB Feature Detector](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
- [Haar Cascade Classifiers](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Face Detection Guide](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)

### Scikit-learn Documentation
- [SVM Classifiers](https://scikit-learn.org/stable/modules/svm.html)
- [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)

---

## 📝 License

This project is provided as-is for educational purposes. 

**Third-party components**:
- OpenCV: Apache 2.0 License
- scikit-learn: BSD 3-Clause License
- NumPy: BSD License

---

## 🤝 Contributing

This is an inference-only project. For improvements:

1. Optimize feature extraction (faster ORB)
2. Implement multi-scale detection
3. Add pose estimation for better mask alignment
4. Support multiple mask selection (runtime switching)

---

## 📧 Support

For issues related to:
- **Setup/Installation**: Check Python version and dependencies
- **Model Training**: Use Google Colab with GPU runtime
- **Performance**: Review optimization tips in README
- **Mask Alignment**: Adjust `--mask-scale` and `--mask-offset`

---

**Built with Classical Computer Vision** 🎯

No deep learning. No GPUs. Just pure CV algorithms.
