# Godot Integration Scripts

Scripts GDScript untuk integrasi dengan Python backend face detection system.

---

## 📁 Files

### face_detector_client.gd
HTTP client untuk komunikasi dengan Flask server Python.

**Fungsi:**
- `detect_faces(image_base64: String)` - Kirim gambar ke server, terima hasil deteksi
- Async HTTP request dengan callback
- Base64 encoding untuk transfer gambar

**Usage:**
```gdscript
var detector = FaceDetectorClient.new()
add_child(detector)
detector.detection_complete.connect(_on_detection_complete)
detector.detect_faces(image_base64)
```

### virtual_tryon_demo.gd
Demo scene dengan webcam integration.

**Fungsi:**
- Capture frame dari CameraServer
- Process setiap 10 frames (performance optimization)
- Display hasil dengan mask overlay
- Real-time FPS monitoring

**Usage:**
Attach ke scene node dengan Camera2D child.

---

## 🚀 Setup

### 1. Start Python Server

```bash
cd d:\PCDVirtualTryOn\Virtual-TryOn-in-Godot\python
python godot_bridge.py
```

Server berjalan di `http://localhost:5000`

### 2. Godot Scene Setup

1. Buat scene baru
2. Add `Camera2D` node
3. Add script `virtual_tryon_demo.gd`
4. Run scene

### 3. API Endpoint

**POST** `http://localhost:5000/detect`

**Request:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANS..."
}
```

**Response:**
```json
{
  "faces": [
    [x, y, width, height, confidence],
    [x, y, width, height, confidence]
  ],
  "result_image": "iVBORw0KGgoAAAANS..."
}
```

---

## 🔧 Configuration

### face_detector_client.gd

```gdscript
# Change server URL if needed
var api_url = "http://localhost:5000/detect"  # Default

# Timeout settings
http_request.timeout = 5.0  # seconds
```

### virtual_tryon_demo.gd

```gdscript
# Processing interval
var frame_count = 0
var process_every = 10  # Process every 10 frames

# Camera settings
var camera_index = 0  # Change if multiple cameras
```

---

## 📊 Performance

- **Latency**: ~50-100ms per request (local server)
- **FPS**: 18-25 FPS (with processing every 10 frames)
- **Network**: Uses localhost (no internet required)

---

## 🐛 Troubleshooting

### Error: "Connection refused"

**Solusi:**
1. Pastikan Flask server berjalan: `python godot_bridge.py`
2. Check port 5000 tidak dipakai aplikasi lain
3. Verify firewall settings

### Error: "Timeout"

**Solusi:**
1. Increase timeout: `http_request.timeout = 10.0`
2. Check server performance (CPU usage)
3. Reduce image resolution

### No Detection Results

**Solusi:**
1. Verify models trained: `python app.py train ...`
2. Check image format (harus RGB)
3. Lower confidence threshold di Python side

---

## 📚 Integration Guide

### Complete Workflow

1. **Python Side:**
   ```bash
   # Train model
   python app.py train --pos_dir data/face --neg_dir data/non_face
   
   # Start server
   python godot_bridge.py
   ```

2. **Godot Side:**
   ```gdscript
   # In your scene
   var detector = preload("res://godot/scripts/face_detector_client.gd").new()
   add_child(detector)
   detector.detection_complete.connect(_on_detection_complete)
   
   # Capture and send
   var img = get_viewport().get_texture().get_image()
   var base64 = Marshalls.raw_to_base64(img.save_png_to_buffer())
   detector.detect_faces(base64)
   
   # Handle result
   func _on_detection_complete(faces, result_image):
       print("Detected ", faces.size(), " faces")
   ```

---

## ✅ Requirements

- Godot Engine 4.x
- Python 3.10+ dengan Flask server running
- Trained models di `python/models/`

---

Selamat mengintegrasikan! 🎮
