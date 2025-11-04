# 🎮 Integrasi UDP Virtual Try-On ke Godot

Panduan lengkap untuk mengintegrasikan sistem face detection UDP dengan Godot Engine untuk Virtual Try-On real-time.

---

## 📋 Daftar Isi

1. [Overview](#overview)
2. [Arsitektur Sistem](#arsitektur-sistem)
3. [Persiapan](#persiapan)
4. [Setup Python Server](#setup-python-server)
5. [Setup Godot Client](#setup-godot-client)
6. [Protocol UDP](#protocol-udp)
7. [Implementasi Detail](#implementasi-detail)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

Sistem ini memisahkan **processing** dan **rendering**:

- **Python Backend**: Face detection, mask overlay processing (heavy ML work)
- **Godot Frontend**: Display hasil processing saja (lightweight rendering)
- **UDP Protocol**: Streaming video frames dan hasil detection

### Keuntungan Arsitektur Ini:

✅ **Performa**: ML processing di Python (optimized dengan NumPy/OpenCV)  
✅ **Fleksibel**: Godot hanya perlu display, tidak perlu ML  
✅ **Scalable**: Bisa deploy Python server terpisah  
✅ **Real-time**: UDP protocol untuk latency rendah  

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                        GODOT CLIENT                         │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Webcam     │ ──────> │ UDP Sender   │ ───┐            │
│  │   Capture    │         │ (Port 5555)  │    │            │
│  └──────────────┘         └──────────────┘    │            │
│                                                 │            │
│  ┌──────────────┐         ┌──────────────┐    │            │
│  │   Display    │ <────── │ UDP Receiver │    │            │
│  │   Processed  │         │ (Port 5556)  │ <──┼──┐         │
│  └──────────────┘         └──────────────┘    │  │         │
└────────────────────────────────────────────────┼──┼─────────┘
                                                  │  │
                                        UDP       │  │ UDP
                                        Frames    │  │ Results
                                                  │  │
┌────────────────────────────────────────────────┼──┼─────────┐
│                       PYTHON SERVER             │  │         │
│  ┌──────────────┐         ┌──────────────┐    │  │         │
│  │ UDP Receiver │ ──────> │ Face Detect  │ ───┘  │         │
│  │ (Port 5555)  │         │ + Haar + SVM │       │         │
│  └──────────────┘         └──────────────┘       │         │
│                                  │                │         │
│                                  ↓                │         │
│                           ┌──────────────┐       │         │
│                           │ Mask Overlay │       │         │
│                           │   + Resize   │       │         │
│                           └──────────────┘       │         │
│                                  │                │         │
│  ┌──────────────┐         ┌──────────────┐      │         │
│  │ UDP Sender   │ <────── │ JPEG Encode  │ ─────┘         │
│  │ (Port 5556)  │         │  + Base64    │                │
│  └──────────────┘         └──────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow:

1. **Godot** capture webcam → JPEG → Base64 → UDP (5555) → **Python**
2. **Python** process (detect + overlay) → JPEG → Base64 → UDP (5556) → **Godot**
3. **Godot** decode → display
4. Loop @ 30 FPS

---

## 🛠️ Persiapan

### Requirements:

**Python Side:**
```bash
- Python 3.8+
- OpenCV (cv2)
- NumPy
- scikit-learn
- Trained models (SVM, BoVW, Scaler)
```

**Godot Side:**
```bash
- Godot 4.x
- Webcam plugin (recommended: godot-camera-capture)
- Mask asset (PNG with alpha channel)
```

### File Structure:

```
Virtual-TryOn-in-Godot/
├── backend/
│   ├── webcam_udp_server.py          # Main server
│   ├── pipelines/
│   │   ├── infer.py                   # Face detection
│   │   └── overlay.py                 # Mask overlay
│   ├── models/
│   │   ├── svm_model.pkl
│   │   ├── bovw_encoder.pkl
│   │   └── scaler.pkl
│   └── assets/
│       └── mask.png
└── godot/
    ├── scripts/tryon/
    │   └── udp_video_receiver.gd      # UDP client
    └── scenes/tryon/
        └── TryOn.tscn                 # Main scene
```

---

## 🐍 Setup Python Server

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare Models

Pastikan file model ada di `backend/models/`:
```
models/
├── svm_model.pkl        # Trained SVM classifier
├── bovw_encoder.pkl     # K-means codebook (e.g. k=256)
└── scaler.pkl           # Feature scaler
```

### 3. Prepare Mask Asset

Place mask PNG di `backend/assets/mask.png`:
- Format: PNG dengan alpha channel (RGBA)
- Size: 512×512 atau 1024×1024 px
- Background: Transparent
- Content: Mask untuk area hidung-mulut-dagu

### 4. Run Server

```bash
# Basic (localhost)
python webcam_udp_server.py --camera 0 --preview

# Custom settings
python webcam_udp_server.py \
  --camera 0 \
  --host 127.0.0.1 \
  --port 5555 \
  --quality 60 \
  --fps 30 \
  --preview
```

### Command Line Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera device ID |
| `--host` | 127.0.0.1 | Godot client host |
| `--port` | 5555 | UDP port untuk kirim ke Godot |
| `--quality` | 60 | JPEG quality (0-100) |
| `--fps` | 30 | Target FPS |
| `--alpha` | 0.95 | Mask opacity (0-1) |
| `--enable-rotation` | False | Enable mask rotation |
| `--preview` | False | Show preview window |
| `--log-level` | INFO | Logging level |

### Server Output:

```
============================================================
WEBCAM UDP SERVER RUNNING
============================================================
Streaming to: 127.0.0.1:5555
Target FPS: 30
JPEG quality: 60
Preview: ON
============================================================
Press 'q' to quit
============================================================

✓ Camera opened: 640×480
📤 Sent frame 1: 25.3 KB, 1 faces, 28.5ms
📤 Sent frame 2: 24.8 KB, 1 faces, 26.2ms
...
```

---

## 🎮 Setup Godot Client

### 1. Create Scene Structure

Buat scene dengan hierarchy berikut:

```
TryOn (Control)
├── VideoDisplay (TextureRect)
│   └── Properties:
│       - Expand Mode: Ignore Size
│       - Stretch Mode: Keep Aspect Centered
│       - Anchor: Full Rect
│
├── UDPReceiver (Node)
│   └── Script: udp_video_receiver.gd
│
└── UI (CanvasLayer)
    ├── FPSLabel (Label)
    │   └── Text: "FPS: 0"
    ├── FaceCountLabel (Label)
    │   └── Text: "Faces: 0"
    └── StatusLabel (Label)
        └── Text: "Connecting..."
```

### 2. Setup UDP Receiver Script

Create `udp_video_receiver.gd`:

```gdscript
extends Node

# ============================================================================
# CONFIGURATION
# ============================================================================

## UDP port untuk menerima video dari Python
@export var udp_port: int = 5555

## Target TextureRect untuk display video
@export var video_display: TextureRect

## Labels untuk stats (optional)
@export var fps_label: Label
@export var face_count_label: Label
@export var status_label: Label

# ============================================================================
# NETWORKING
# ============================================================================

var socket: PacketPeerUDP
var is_connected: bool = false

# ============================================================================
# STATISTICS
# ============================================================================

var frames_received: int = 0
var last_fps: float = 0.0
var last_face_count: int = 0

# ============================================================================
# LIFECYCLE
# ============================================================================

func _ready():
	_setup_udp()

func _process(_delta: float):
	_receive_packets()

# ============================================================================
# UDP SETUP
# ============================================================================

func _setup_udp():
	print("🔌 Setting up UDP receiver on port %d..." % udp_port)
	
	socket = PacketPeerUDP.new()
	var result = socket.bind(udp_port)
	
	if result == OK:
		is_connected = true
		print("✅ UDP receiver ready")
		_update_status("✅ Connected - Waiting for frames...")
	else:
		print("❌ Failed to bind UDP port %d" % udp_port)
		_update_status("❌ Failed to bind port %d" % udp_port)

# ============================================================================
# PACKET RECEIVING
# ============================================================================

func _receive_packets():
	if not is_connected:
		return
	
	while socket.get_available_packet_count() > 0:
		var packet = socket.get_packet()
		_process_packet(packet)

func _process_packet(packet: PackedByteArray):
	# Parse JSON
	var json_string = packet.get_string_from_utf8()
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	
	if parse_result != OK:
		print("❌ JSON parse error: %s" % json.get_error_message())
		return
	
	var data = json.data
	
	# Validate data
	if not _validate_packet(data):
		return
	
	# Decode and display frame
	_display_frame(data)
	
	# Update stats
	_update_stats(data)

func _validate_packet(data: Dictionary) -> bool:
	if not data.has("image"):
		print("⚠️ Packet missing 'image' field")
		return false
	
	return true

# ============================================================================
# FRAME DISPLAY
# ============================================================================

func _display_frame(data: Dictionary):
	# Decode base64
	var base64_image = data.image
	var jpeg_buffer = Marshalls.base64_to_raw(base64_image)
	
	# Create image from JPEG
	var image = Image.new()
	var error = image.load_jpg_from_buffer(jpeg_buffer)
	
	if error != OK:
		print("❌ Failed to decode JPEG: error %d" % error)
		return
	
	# Create texture and display
	var texture = ImageTexture.create_from_image(image)
	
	if video_display:
		video_display.texture = texture
	
	frames_received += 1

# ============================================================================
# STATISTICS
# ============================================================================

func _update_stats(data: Dictionary):
	# Extract data
	if data.has("fps"):
		last_fps = data.fps
	
	if data.has("faces"):
		last_face_count = data.faces
	
	# Update UI labels
	if fps_label:
		fps_label.text = "FPS: %.1f" % last_fps
	
	if face_count_label:
		face_count_label.text = "Faces: %d" % last_face_count
	
	# Update status
	if frames_received % 30 == 0:
		_update_status("📺 Receiving (Frame: %d)" % frames_received)

func _update_status(message: String):
	if status_label:
		status_label.text = message

# ============================================================================
# CLEANUP
# ============================================================================

func _exit_tree():
	if socket:
		socket.close()
	print("🔌 UDP socket closed")
```

### 3. Attach Script to Scene

1. Select `UDPReceiver` node
2. Attach `udp_video_receiver.gd`
3. Configure exports:
   - `Udp Port`: `5555`
   - `Video Display`: Drag `VideoDisplay` TextureRect node
   - `Fps Label`: Drag `FPSLabel` node (optional)
   - `Face Count Label`: Drag `FaceCountLabel` node (optional)
   - `Status Label`: Drag `StatusLabel` node (optional)

### 4. Configure Video Display

Select `VideoDisplay` (TextureRect):
- Layout → Anchors Preset: **Full Rect**
- Layout → Expand: **ON**
- Texture → Expand Mode: **Ignore Size**
- Texture → Stretch Mode: **Keep Aspect Centered**
- Modulate → Alpha: `1.0`

### 5. Style UI Labels (Optional)

Example styling for labels:

```gdscript
# FPSLabel
Theme Overrides → Font Size: 24
Theme Overrides → Font Color: #00FF00 (green)
Theme Overrides → Shadow Size: 2
Position: (10, 10)

# FaceCountLabel
Theme Overrides → Font Size: 20
Theme Overrides → Font Color: #FFFF00 (yellow)
Position: (10, 40)

# StatusLabel
Theme Overrides → Font Size: 18
Theme Overrides → Font Color: #FFFFFF (white)
Anchor Preset: Bottom Left
Position: (10, -40)
```

---

## 📡 Protocol UDP

### Packet Format: Python → Godot

```json
{
  "frame_id": 123,
  "image": "/9j/4AAQSkZJRgABAQAA...",
  "fps": 30.5,
  "faces": 1,
  "processing_ms": 28.3,
  "timestamp": 1730563892.145
}
```

### Field Descriptions:

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Sequential frame number |
| `image` | string | Base64-encoded JPEG image |
| `fps` | float | Current FPS di server |
| `faces` | int | Jumlah wajah terdeteksi |
| `processing_ms` | float | Waktu processing (ms) |
| `timestamp` | float | Unix timestamp |

### Packet Size:

- Resolution 640×480, Quality 60: **~25-35 KB**
- Resolution 320×240, Quality 50: **~10-15 KB**
- UDP limit: **~65 KB** per packet

**⚠️ Important:** Jika packet > 65KB, akan error. Gunakan quality 60 atau lebih rendah.

---

## 🔧 Implementasi Detail

### Python Side: Encoding & Sending

```python
def encode_frame(self, frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG"""
    # Auto-resize jika terlalu besar
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        new_w = 640
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    # Encode JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
    success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_param)
    
    # Base64 encode
    base64_str = base64.b64encode(jpeg_buffer).decode('utf-8')
    return base64_str

def send_frame(self, frame: np.ndarray, face_count: int, processing_time: float):
    """Send frame via UDP"""
    base64_image = self.encode_frame(frame)
    
    packet = {
        "frame_id": self.frame_count,
        "image": base64_image,
        "fps": round(self.fps_counter(), 1),
        "faces": face_count,
        "processing_ms": round(processing_time * 1000, 1),
        "timestamp": time.time()
    }
    
    json_bytes = json.dumps(packet).encode('utf-8')
    self.socket.sendto(json_bytes, (self.host, self.port))
```

### Godot Side: Receiving & Decoding

```gdscript
func _process_packet(packet: PackedByteArray):
    # Parse JSON
    var json_string = packet.get_string_from_utf8()
    var json = JSON.new()
    var parse_result = json.parse(json_string)
    
    if parse_result != OK:
        return
    
    var data = json.data
    
    # Decode base64
    var jpeg_buffer = Marshalls.base64_to_raw(data.image)
    
    # Load JPEG
    var image = Image.new()
    var error = image.load_jpg_from_buffer(jpeg_buffer)
    
    if error != OK:
        return
    
    # Display
    var texture = ImageTexture.create_from_image(image)
    video_display.texture = texture
```

---

## 🧪 Testing

### Test 1: Python Server Standalone

```bash
# Test dengan preview window
python webcam_udp_server.py --camera 0 --preview

# Expected output:
✓ Camera opened: 640×480
📤 Sent frame 1: 25.3 KB, 1 faces, 28.5ms
📤 Sent frame 2: 24.8 KB, 1 faces, 26.2ms
```

**✅ Success criteria:**
- Camera opens
- Preview window shows video with mask
- "Sent" count increases
- No errors

### Test 2: Godot Client Standalone

1. Run Python server first
2. Play Godot scene
3. Check Output console

**✅ Success criteria:**
```
🔌 Setting up UDP receiver on port 5555...
✅ UDP receiver ready
📺 Receiving (Frame: 30)
📺 Receiving (Frame: 60)
```

### Test 3: End-to-End

1. Run Python server
2. Run Godot client
3. Observe:
   - Godot displays processed video
   - FPS shows ~30
   - Face count updates
   - No lag/stuttering

### Test 4: Network Test

```bash
# Check UDP port
netstat -an | findstr 5555

# Expected:
UDP    0.0.0.0:5555           *:*
```

---

## 🐛 Troubleshooting

### Problem: Tidak Ada Video di Godot

**Symptoms:**
- Godot scene hitam
- Status label shows "Waiting for frames"
- No errors

**Solutions:**

1. **Check Python server running:**
   ```bash
   # Should show "Sent frame X" messages
   python webcam_udp_server.py --camera 0 --preview
   ```

2. **Check UDP port:**
   ```bash
   netstat -an | findstr 5555
   ```

3. **Check firewall:**
   - Windows: Allow Python & Godot through firewall
   - Add exception untuk port 5555

4. **Check host/port match:**
   - Python: `--host 127.0.0.1 --port 5555`
   - Godot: `udp_port = 5555`

### Problem: Video Lag/Stuttering

**Symptoms:**
- Video tersendat
- Low FPS (< 15)
- Processing time > 80ms

**Solutions:**

1. **Reduce resolution:**
   ```python
   # In webcam_udp_server.py
   # Frame auto-resizes to 640px, try 480px:
   if w > 480:
       scale = 480.0 / w
   ```

2. **Lower quality:**
   ```bash
   python webcam_udp_server.py --quality 50
   ```

3. **Reduce FPS:**
   ```bash
   python webcam_udp_server.py --fps 24
   ```

4. **Check CPU usage:**
   - Face detection takes ~25-35ms per frame
   - If > 50ms, CPU might be overloaded

### Problem: Packet Size Error

**Symptoms:**
```
[WinError 10040] A message sent on a datagram socket was larger 
than the internal message buffer
```

**Solutions:**

1. **Lower quality:**
   ```bash
   python webcam_udp_server.py --quality 40
   ```

2. **Reduce resolution:**
   - Frames auto-resize to 640px
   - Modify code untuk 480px or 320px

3. **Check packet size in logs:**
   ```
   📤 Sent frame 1: 25.3 KB  ← Should be < 60 KB
   ```

### Problem: Wrong Colors

**Symptoms:**
- Video tampak weird colors
- Blue faces, red tint, etc.

**Solutions:**

- OpenCV uses BGR, Godot uses RGB
- Should be handled automatically
- If not, add conversion in Python:
  ```python
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  ```

### Problem: Connection Timeout

**Symptoms:**
- Godot status: "Connection lost"
- No frames after few seconds

**Solutions:**

1. **Check Python server still running**
2. **Check network:**
   ```bash
   ping 127.0.0.1
   ```
3. **Restart both Python & Godot**

---

## ⚡ Performance Optimization

### 1. Adjust Resolution

**Trade-off:** Quality vs Speed

```bash
# High quality (slower)
python webcam_udp_server.py --camera 0  # 640×480

# Medium quality (balanced)
# Modify code: resize to 480×360

# Low quality (faster)
# Modify code: resize to 320×240
```

### 2. Adjust JPEG Quality

```bash
# High quality (larger packets)
python webcam_udp_server.py --quality 80

# Medium quality (recommended)
python webcam_udp_server.py --quality 60

# Low quality (smaller packets)
python webcam_udp_server.py --quality 40
```

### 3. FPS Control

```bash
# Target 30 FPS (smooth)
python webcam_udp_server.py --fps 30

# Target 24 FPS (cinematic)
python webcam_udp_server.py --fps 24

# Target 15 FPS (slower devices)
python webcam_udp_server.py --fps 15
```

### 4. Skip Detection Frames

Modify Python code untuk skip detection setiap N frames:

```python
def run(self):
    skip_counter = 0
    
    while True:
        ret, frame = self.cap.read()
        
        skip_counter += 1
        
        if skip_counter >= 2:  # Process every 2nd frame
            result_frame, detections = self.detection_system.process_image(frame)
            skip_counter = 0
        else:
            result_frame = frame  # Use raw frame
            detections = []
        
        self.send_frame(result_frame, len(detections), 0)
```

### 5. Godot-Side Optimization

```gdscript
# In udp_video_receiver.gd

# Skip frame updates if FPS too low
var frame_skip = 0

func _display_frame(data: Dictionary):
    frame_skip += 1
    
    if frame_skip >= 2:  # Display every 2nd frame
        # ... decode and display
        frame_skip = 0
```

### Performance Targets:

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| FPS | 30 | 20-30 | < 20 |
| Processing Time | < 35ms | 35-50ms | > 50ms |
| Packet Size | < 30 KB | 30-50 KB | > 50 KB |
| Latency | < 100ms | 100-200ms | > 200ms |

---

## 📊 Monitoring & Debug

### Enable Debug Logs

**Python:**
```bash
python webcam_udp_server.py --log-level DEBUG
```

**Godot:**
```gdscript
# Add in _process_packet()
print("Received frame %d: %d KB" % [data.frame_id, packet.size() / 1024.0])
```

### Monitor Statistics

**Python side:**
```
Stats: frames=300, sent=300, errors=0, fps=30.2
```

**Godot side:**
```gdscript
func _process(_delta):
    if frames_received % 30 == 0:
        print("Received %d frames, FPS: %.1f" % [frames_received, last_fps])
```

### Visualize Packet Flow

Add packet visualization:

```gdscript
extends Control

@export var packet_history_label: Label
var packet_times: Array = []

func _process_packet(packet: PackedByteArray):
    packet_times.append(Time.get_ticks_msec())
    
    # Keep only last 30 packets
    if packet_times.size() > 30:
        packet_times.pop_front()
    
    # Calculate receive rate
    if packet_times.size() >= 2:
        var time_diff = packet_times[-1] - packet_times[0]
        var rate = packet_times.size() / (time_diff / 1000.0)
        packet_history_label.text = "Receive rate: %.1f packets/sec" % rate
```

---

## 🎯 Best Practices

### 1. Error Handling

**Python:**
```python
try:
    self.send_frame(result_frame, face_count, process_time)
except Exception as e:
    logger.error(f"Send error: {e}")
    self.error_count += 1
```

**Godot:**
```gdscript
func _process_packet(packet: PackedByteArray):
    var json_string = packet.get_string_from_utf8()
    var json = JSON.new()
    var parse_result = json.parse(json_string)
    
    if parse_result != OK:
        push_error("JSON parse failed: %s" % json.get_error_message())
        return
```

### 2. Graceful Degradation

```python
# If camera fails, continue running
if not ret:
    logger.warning("Frame read failed, using last frame")
    frame = self.last_frame
```

### 3. Resource Cleanup

**Python:**
```python
def __del__(self):
    if self.cap:
        self.cap.release()
    self.socket.close()
```

**Godot:**
```gdscript
func _exit_tree():
    if socket:
        socket.close()
```

### 4. Configuration Management

Create config file:

**config.json:**
```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 5555,
    "fps": 30,
    "quality": 60
  },
  "detection": {
    "min_face_size": 40,
    "enable_rotation": true,
    "alpha": 0.95
  }
}
```

---

## 🚀 Deployment

### Development (Localhost)

```bash
# Python
python webcam_udp_server.py --host 127.0.0.1 --port 5555

# Godot
udp_port = 5555
```

### Production (LAN)

**Server machine:**
```bash
# Find IP address
ipconfig  # Windows
ifconfig  # Linux/Mac

# Example: 192.168.1.100
python webcam_udp_server.py --host 0.0.0.0 --port 5555
```

**Client machine (Godot):**
```gdscript
# Modify udp_video_receiver.gd
var server_host = "192.168.1.100"  # Server's IP
var udp_port = 5555
```

### Firewall Configuration

**Windows (PowerShell as Admin):**
```powershell
# Allow Python
New-NetFirewallRule -DisplayName "Python UDP" -Direction Inbound -Protocol UDP -LocalPort 5555 -Action Allow

# Allow Godot
New-NetFirewallRule -DisplayName "Godot UDP" -Direction Inbound -Protocol UDP -LocalPort 5555 -Action Allow
```

---

## ✅ Checklist

### Pre-Launch:

- [ ] Python dependencies installed
- [ ] Models trained and placed in `models/`
- [ ] Mask asset created and placed in `assets/`
- [ ] Godot scene structure correct
- [ ] UDP ports configured (5555)
- [ ] Firewall rules added (if needed)

### Testing:

- [ ] Python server starts without errors
- [ ] Camera opens successfully
- [ ] Preview window shows video with mask
- [ ] Godot receives frames
- [ ] Video displays correctly
- [ ] FPS is acceptable (> 20)
- [ ] Face count updates
- [ ] No packet size errors
- [ ] Network latency acceptable

### Optimization:

- [ ] Resolution optimized (640×480 or lower)
- [ ] Quality optimized (60 or lower)
- [ ] FPS stable (30 target)
- [ ] Processing time < 50ms
- [ ] Packet size < 50 KB

---

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Godot UDP Networking](https://docs.godotengine.org/en/stable/classes/class_packetpeerudp.html)
- [JSON in Godot](https://docs.godotengine.org/en/stable/classes/class_json.html)
- [Base64 Encoding](https://docs.godotengine.org/en/stable/classes/class_marshalls.html)

---

## 💬 Support

**Issues?**
1. Check troubleshooting section
2. Enable debug logs
3. Monitor packet flow
4. Check network connectivity

**Need help?**
- Check server logs: `--log-level DEBUG`
- Check Godot output console
- Test with preview window: `--preview`
- Verify UDP ports: `netstat -an | findstr 5555`

---

**Happy coding! 🎮✨**
