# 🚀 Quick Start Guide

Get up and running with Virtual Try-On in 15 minutes!

---

## ✅ Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Webcam connected
- [ ] 2GB free disk space
- [ ] Internet connection (for Colab training)
- [ ] Google Drive account (for dataset storage)

---

## 📝 Step-by-Step Setup

### Step 1: Install Dependencies (2 minutes)

```bash
# Clone repository
git clone <repository-url>
cd virtual-tryon

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset (5 minutes)

**Option A: Use your existing dataset**

Organize your dataset:

```
datasets/
├── faces/          # Your 2500+ face images
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── non_faces/      # Your non-face images
    ├── wall001.jpg
    ├── lamp001.jpg
    └── ...
```

**Option B: Download sample dataset** (if you don't have one)

```bash
# Download from provided link or use your own
# Ensure at least 2000+ images in each category
```

### Step 3: Upload to Google Drive (3 minutes)

1. Create folder structure in Google Drive:

   ```
   MyDrive/
   └── virtual_tryon/
       └── datasets/
           ├── faces/
           └── non_faces/
   ```

2. Upload your images (this may take a while depending on size)

### Step 4: Train Model on Google Colab (40-80 minutes)

1. **Open Colab**
   - Go to [Google Colab](https://colab.research.google.com)
   - Upload `01_train_face_detector.ipynb`

2. **Configure Runtime**
   - Runtime → Change runtime type
   - Select: GPU (T4 or better)

3. **Run Training**
   - Update paths in Cell 1:
     ```python
     DATASET_BASE = '/content/drive/MyDrive/virtual_tryon/datasets'
     ```
   - Run all cells (Runtime → Run all)
   - ☕ Take a break! Training takes 40-80 minutes

4. **Download Models**
   After training completes, download these files:
   - `face_detector_svm.pkl`
   - `scaler.pkl`
   - `config.json`

### Step 5: Setup Project Structure (1 minute)

Place downloaded models in your project:

```
virtual-tryon/
├── models/
│   ├── face_detector_svm.pkl  ← Place here
│   ├── scaler.pkl              ← Place here
│   └── config.json             ← Place here
└── assets/
    └── masks/
        └── zombie.png          ← Your mask PNG files
```

### Step 6: Prepare Mask Images (2 minutes)

Create or download Halloween masks:

- Must be PNG with alpha channel (transparency)
- Recommended size: 512x512 or larger
- Face should be centered

Place in `assets/masks/`:

```
assets/masks/
├── zombie.png
├── vampire.png
├── skeleton.png
└── witch.png
```

### Step 7: Test CLI Interface (2 minutes)

**Test on single image:**

```bash
# Download or use your own test image
python app.py infer \
  --image test_face.jpg \
  --out result.jpg \
  --mask assets/masks/zombie.png \
  --show
```

**Test webcam:**

```bash
python app.py webcam \
  --camera 0 \
  --mask assets/masks/zombie.png \
  --show
```

Press `q` to quit, `s` to save screenshot.

---

## 🎮 For Godot Developers

### Step 8: Start UDP Server (1 minute)

```bash
python -m src.server.udp_server \
  --host 0.0.0.0 \
  --port 5001 \
  --model models/face_detector_svm.pkl \
  --scaler models/scaler.pkl \
  --config models/config.json \
  --masks assets/masks/
```

Server should print:

```
✓ Face detector loaded
✓ Loaded 4 masks
✓ UDP Server initialized
✓ Server is running
```

### Step 9: Test with Godot

1. Open provided Godot project or create new one
2. Add `UDPClient.gd` script (from GODOT_INTEGRATION.md)
3. Create simple test scene with TextureRect
4. Run and verify connection

**Minimum test code:**

```gdscript
extends Node

var udp := PacketPeerUDP.new()

func _ready():
    udp.set_dest_address("127.0.0.1", 5001)
    print("Connected to server")
```

---

## 🎯 Verification Checklist

After setup, verify everything works:

- [ ] **Model trained**: Accuracy > 80% on test set
- [ ] **CLI works**: Can detect face in test image
- [ ] **Webcam works**: Real-time overlay visible
- [ ] **UDP server starts**: No errors on startup
- [ ] **Godot connects**: Can send/receive UDP packets
- [ ] **Mask overlay**: Mask appears on face correctly
- [ ] **Performance**: 20+ FPS with mask overlay

---

## 🔧 Common First-Time Issues

### Issue 1: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**

```bash
pip install opencv-python
```

### Issue 2: "No cameras found!"

**Solution:**

- Check webcam is connected
- On Linux, check permissions: `sudo chmod 666 /dev/video0`
- Test webcam: `ffplay /dev/video0` (Linux) or use Camera app (Windows)

### Issue 3: "Could not load model"

**Solution:**

- Verify file paths are correct
- Check models exist in `models/` directory
- Ensure you downloaded all 3 files from Colab

### Issue 4: "Connection refused" (Godot)

**Solution:**

- Ensure Python server is running
- Check firewall settings
- Verify correct IP address (127.0.0.1 for local)
- Check port 5001 is not in use: `netstat -an | grep 5001`

### Issue 5: "Accuracy too low" (< 70%)

**Solution:**

- Check dataset quality (faces should be visible)
- Increase training data
- Run more hard negative mining rounds
- Adjust SVM C parameter in notebook

---

## 📊 Expected Results

After setup, you should see:

**Training:**

- Validation accuracy: 85-92%
- Test accuracy: 83-90%
- Precision: 88-94%
- Recall: 82-90%

**Performance:**

- Detection time: 55-100ms per frame
- With ROI tracking: 20-35ms per frame
- Webcam FPS: 20-30 (without optimization), 30-45 (with optimization)
- UDP latency: 5-15ms (local), 20-50ms (network)

**Visual Quality:**

- Mask aligns with face orientation
- Smooth transitions between frames
- Minimal jitter or jumping
- Natural blending with skin tones

---

## 🎓 Next Steps

Once basic setup works:

1. **Optimize Performance**
   - Read GODOT_INTEGRATION.md → Performance section
   - Implement ROI tracking
   - Add frame skipping

2. **Add More Masks**
   - Create or download additional masks
   - Test switching between masks
   - Implement UI for mask selection

3. **Enhance Features**
   - Add screenshot capability
   - Record video with mask overlay
   - Multiple face support
   - Real-time mask editing

4. **Deploy**
   - Move server to separate machine
   - Configure production settings
   - Add error recovery
   - Implement logging

---

## 📚 Additional Resources

- **Main README**: Comprehensive documentation
- **GODOT_INTEGRATION.md**: Complete Godot guide with code
- **Training Notebook**: Step-by-step ML training
- **API Documentation**: Request/response formats

---

## 🆘 Getting Help

If you're stuck:

1. Check [Troubleshooting](#-common-first-time-issues) above
2. Review main README.md
3. Check server logs for errors
4. Open issue on GitHub with:
   - Error messages
   - System info (OS, Python version)
   - Steps to reproduce

---

## ⏱️ Time Estimates

| Task                 | Time            | Can Skip?                 |
| -------------------- | --------------- | ------------------------- |
| Install dependencies | 2 min           | No                        |
| Prepare dataset      | 5 min           | No                        |
| Upload to Drive      | 3 min           | No                        |
| Train model (Colab)  | 40-80 min       | No                        |
| Setup project        | 1 min           | No                        |
| Prepare masks        | 2 min           | No                        |
| Test CLI             | 2 min           | Yes (if only using Godot) |
| Start UDP server     | 1 min           | No (for Godot)            |
| Test Godot           | 2 min           | No (for Godot)            |
| **Total**            | **~60-100 min** |                           |

**Active time**: ~15 minutes
**Waiting time**: ~45-85 minutes (training)

---

## ✨ Quick Commands Reference

```bash
# Training (Google Colab)
# Just run cells in notebook!

# CLI Testing
python app.py infer --image test.jpg --out result.jpg --mask assets/masks/zombie.png --show
python app.py webcam --camera 0 --mask assets/masks/zombie.png

# Start Server (for Godot)
python -m src.server.udp_server \
  --model models/face_detector_svm.pkl \
  --scaler models/scaler.pkl \
  --config models/config.json \
  --masks assets/masks/

# Production Server
python -m src.server.udp_server \
  --host 0.0.0.0 \
  --port 5001 \
  --model models/face_detector_svm.pkl \
  --scaler models/scaler.pkl \
  --config models/config.json \
  --masks assets/masks/
```

---

**Congratulations! 🎉 You're now ready to build your Virtual Try-On application!**
