# Quick Start Guide - UDP Face Detection Server

**Get the face detection server running in 5 minutes**

---

## Prerequisites

```bash
# Check Python version (need 3.8+)
python --version

# Check required packages
python -c "import cv2, numpy, sklearn; print('✓ All packages installed')"
```

If packages missing:

```bash
pip install opencv-python numpy scikit-learn
```

---

## Step 1: Setup Project Structure

```bash
# Create UDP server directory
mkdir -p udp_server
touch udp_server/__init__.py

# Verify models exist
ls models/
# Should show: svm_model.pkl  bovw_encoder.pkl  scaler.pkl
```

Copy these files into your project:

- `config.py` → `udp_server/config.py`
- `protocol.py` → `udp_server/protocol.py`
- `frame_handler.py` → `udp_server/frame_handler.py`
- `server.py` → `udp_server/server.py`
- `test_client.py` → `test_client.py` (root directory)

---

## Step 2: Start the Server

```bash
# Development mode (localhost only)
python -m udp_server.server --mode dev
```

**Expected output:**

```
============================================================
STARTING FACE DETECTION UDP SERVER
============================================================
Initializing face detection system...
✓ Face detection system ready
✓ Sockets created:
  Receiving on: 127.0.0.1:5555
  Sending to: <client>:5556

============================================================
SERVER READY - Waiting for frames from Godot...
============================================================
```

**Server is now running!** ✅

---

## Step 3: Test the Server

### Test with an image

```bash
# In a NEW terminal (keep server running)
python test_client.py image --image path/to/your/test.jpg
```

**Expected:**

```
✓ Loaded image: 1920x1080 px
Sending frame to server...
✓ Response received in 87.3ms

Faces detected: 1
  Face 1:
    Position: (120, 80)
    Size: 180x220 px
    Confidence: 0.943
```

### Test with webcam

```bash
python test_client.py webcam --camera 0
```

- Window will show your webcam
- Green boxes around detected faces
- Press `q` to quit

**If you see detections, everything works!** ✅

---

## Step 4: Connect Godot (For Frontend Team)

**In your Godot project:**

1. **Setup UDP sockets:**

```gdscript
var send_socket = PacketPeerUDP.new()
var recv_socket = PacketPeerUDP.new()

func _ready():
    send_socket.set_dest_address("127.0.0.1", 5555)
    recv_socket.bind(5556)
```

2. **Send camera frame:**

```gdscript
func send_frame(image: Image):
    image.resize(640, 480)
    var jpeg = image.save_jpg_to_buffer(0.7)
    var b64 = Marshalls.raw_to_base64(jpeg)

    var msg = {
        "type": "frame",
        "timestamp": Time.get_unix_time_from_system(),
        "frame": b64,
        "width": 640,
        "height": 480,
        "quality": 70
    }

    send_socket.put_packet(JSON.stringify(msg).to_utf8_buffer())
```

3. **Receive detections:**

```gdscript
func _process(delta):
    if recv_socket.get_available_packet_count() > 0:
        var packet = recv_socket.get_packet()
        var json = JSON.new()
        json.parse(packet.get_string_from_utf8())
        var data = json.data

        for face in data.faces:
            # Position mask at face.center.x, face.center.y
            # Rotate by face.rotation_deg
            # Scale to face.bbox.w, face.bbox.h
```

**See `GODOT_INTEGRATION.md` for complete examples!**

---

## Common Issues

### "Address already in use"

```bash
# Kill process on port 5555
# Linux/Mac:
lsof -ti:5555 | xargs kill -9
# Windows:
netstat -ano | findstr :5555
taskkill /PID <PID> /F
```

### "Model files not found"

```bash
# Check models directory
ls models/
# Must have: svm_model.pkl  bovw_encoder.pkl  scaler.pkl
```

### Test client shows "No response"

```bash
# Check if server is running
# Look for "SERVER READY" message
# Check firewall (allow ports 5555, 5556)
```

### Low FPS

```bash
# Start server with debug logging
python -m udp_server.server --mode dev --log-level DEBUG

# Check "Avg processing time" in server logs
# Should be < 50ms
# If higher: reduce resolution or increase threads
```

---

## Next Steps

1. ✅ Server running and tested
2. 📖 Read `GODOT_INTEGRATION.md` for Godot implementation
3. 🎭 Add your mask images to Godot project
4. 🚀 Start development!

---

## Production Deployment

**For live demo (separate machines):**

```bash
# On server machine:
python -m udp_server.server --mode prod --host 0.0.0.0

# Note the server's IP address (e.g., 192.168.1.100)
```

**In Godot:**

```gdscript
const SERVER_HOST = "192.168.1.100"  # Server's IP
```

**Test from client machine:**

```bash
python test_client.py webcam --host 192.168.1.100 --camera 0
```

---

## Help

**Test not working?**

1. Check server is running (see "SERVER READY")
2. Run test client: `python test_client.py image --image test.jpg`
3. Check firewall settings
4. Check server logs with `--log-level DEBUG`

**Need more details?**

- Server setup: `README_UDP_SERVER.md`
- Godot integration: `GODOT_INTEGRATION.md`
- Protocol details: `udp_server/protocol.py`

---

**You're ready to go!** 🎉
