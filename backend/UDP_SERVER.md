# Face Detection UDP Server for Godot Integration

Complete UDP-based face detection system for Virtual Try-On applications using Godot v4.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Testing](#testing)
- [For Frontend Team](#for-frontend-team)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This UDP server enables real-time face detection for Godot-based Virtual Try-On applications. The system uses classical computer vision techniques:

- **Haar Cascade** for face region proposals
- **ORB features** for robust feature extraction
- **Bag of Visual Words** encoding
- **SVM classifier** for face verification
- **Eye detection** for mask rotation

### Key Features

- ✅ Real-time face detection (15-30 FPS)
- ✅ Multi-face support
- ✅ Rotation detection based on eye positions
- ✅ Low latency (<100ms)
- ✅ UDP-based for minimal overhead
- ✅ Works on localhost or LAN

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│       GODOT (Frontend)          │
│                                 │
│  1. Capture camera frame        │
│  2. Resize & compress to JPEG   │
│  3. Encode to base64            │
│  4. Send via UDP (port 5555)    │
│                                 │
└────────────┬────────────────────┘
             │
             │ UDP (JSON)
             ▼
┌─────────────────────────────────┐
│      PYTHON (ML Backend)        │
│                                 │
│  1. Receive & decode frame      │
│  2. Run face detection:         │
│     - Haar proposals            │
│     - ORB feature extraction    │
│     - BoVW encoding             │
│     - SVM classification        │
│     - Eye detection (rotation)  │
│  3. Send results via UDP (5556) │
│                                 │
└────────────┬────────────────────┘
             │
             │ UDP (JSON)
             ▼
┌─────────────────────────────────┐
│       GODOT (Frontend)          │
│                                 │
│  1. Receive detection data      │
│  2. Parse face positions        │
│  3. Render masks at positions   │
│  4. Apply rotation & scaling    │
│                                 │
└─────────────────────────────────┘
```

### Data Flow

**Godot → Python (Frame Data):**

- JPEG-compressed camera frame (base64 encoded)
- ~20-50 KB per frame at 640x480, quality 70

**Python → Godot (Detection Results):**

- Face bounding boxes, rotation angles, confidence scores
- ~500 bytes per response (JSON metadata only)

---

## 📦 Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Required packages (already in your project)
pip install opencv-python numpy scikit-learn
```

### Project Structure

```
your-project/
├── face_detector_cli.py          # Existing CLI
├── pipelines/
│   ├── infer.py                  # Face detection system
│   └── utils.py
├── models/                        # Pre-trained models
│   ├── svm_model.pkl
│   ├── bovw_encoder.pkl
│   └── scaler.pkl
├── udp_server/                    # NEW - UDP Server
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── protocol.py               # Message protocol
│   ├── frame_handler.py          # Async processing
│   └── server.py                 # Main server
└── test_client.py                # NEW - Testing tool
```

### Setup

1. **Create UDP server directory:**

```bash
mkdir -p udp_server
touch udp_server/__init__.py
```

2. **Copy the provided files:**
   - `config.py` → `udp_server/config.py`
   - `protocol.py` → `udp_server/protocol.py`
   - `frame_handler.py` → `udp_server/frame_handler.py`
   - `server.py` → `udp_server/server.py`
   - `test_client.py` → `test_client.py` (project root)

3. **Verify models exist:**

```bash
ls models/
# Should show: svm_model.pkl  bovw_encoder.pkl  scaler.pkl
```

---

## 🚀 Quick Start

### Start the Server

**Development Mode (localhost only):**

```bash
python -m udp_server.server --mode dev
```

**Production Mode (LAN accessible):**

```bash
python -m udp_server.server --mode prod
```

**With custom settings:**

```bash
python -m udp_server.server \
  --mode prod \
  --host 0.0.0.0 \
  --port-recv 5555 \
  --port-send 5556 \
  --log-level INFO
```

### Expected Output

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
Listening on: 127.0.0.1:5555
Press Ctrl+C to stop
============================================================
```

---

## ⚙️ Configuration

### Configuration File: `udp_server/config.py`

Key settings you can modify:

```python
class ServerConfig:
    # Network
    HOST = "0.0.0.0"          # Listen on all interfaces
    PORT_RECEIVE = 5555       # Receive frames from Godot
    PORT_SEND = 5556          # Send results to Godot

    # Performance
    MAX_QUEUE_SIZE = 2        # Max frames in queue (drop old)
    WORKER_THREADS = 2        # Processing threads

    # Detection
    MAX_KEYPOINTS = 500       # ORB keypoints per face
    ENABLE_ROTATION = True    # Enable rotation detection

    # Logging
    LOG_LEVEL = "INFO"        # DEBUG, INFO, WARNING, ERROR
```

### Environment Variables (Override)

```bash
# Network
export UDP_HOST="192.168.1.100"
export UDP_PORT_RECEIVE=5555
export UDP_PORT_SEND=5556

# Models
export SVM_MODEL_PATH="models/svm_model.pkl"
export BOVW_ENCODER_PATH="models/bovw_encoder.pkl"
export SCALER_PATH="models/scaler.pkl"

# Logging
export LOG_LEVEL="DEBUG"
```

### Presets

**Development:**

- Localhost only (`127.0.0.1`)
- Debug logging
- 2 worker threads

**Production:**

- LAN accessible (`0.0.0.0`)
- Info logging
- 4 worker threads

---

## 🧪 Testing

### Test Without Godot

Use the test client to verify server functionality:

#### 1. Test with Single Image

```bash
python test_client.py image --image path/to/test.jpg
```

**Expected Output:**

```
============================================================
TEST MODE: Single Image
============================================================

✓ Loaded image: 1920x1080 px

Sending frame to server...
✓ Frame sent, waiting for response...
✓ Response received in 87.3ms

============================================================
DETECTION RESULT
============================================================
Type: detection
Processing time: 45.2ms
Faces detected: 1

  Face 1:
    Position: (120, 80)
    Size: 180x220 px
    Confidence: 0.943
    Rotation: -12.5°
    Eyes detected:
      Left: (170, 150)
      Right: (250, 145)
============================================================
```

#### 2. Test with Webcam (Real-time)

```bash
python test_client.py webcam --camera 0
```

**Controls:**

- `q` - Quit
- `d` - Toggle detection display

**Expected Behavior:**

- Window showing webcam feed
- Green bounding boxes around detected faces
- Blue circles at eye positions
- Real-time stats: processing time, faces detected

#### 3. Test with Custom Resolution

```bash
python test_client.py webcam \
  --camera 0 \
  --width 640 \
  --height 480 \
  --quality 70
```

#### 4. Test Remote Server

```bash
# Test server on another machine
python test_client.py webcam \
  --host 192.168.1.100 \
  --camera 0
```

### Verify Server is Running

```bash
# Check if ports are open
netstat -an | grep 5555
netstat -an | grep 5556

# Or on Linux/Mac
lsof -i :5555
lsof -i :5556
```

### Performance Testing

The test client shows real-time statistics:

- **Frames sent**: Total frames sent to server
- **Detections received**: Total responses received
- **Response rate**: % of frames that got responses
- **Processing time**: ML processing time per frame

Target metrics:

- ✅ Processing time: <50ms
- ✅ Response rate: >90%
- ✅ End-to-end latency: <100ms

---

## 👥 For Frontend Team

See dedicated documentation: **[GODOT_INTEGRATION.md](GODOT_INTEGRATION.md)**

Quick links:

- [JSON Protocol Specification](GODOT_INTEGRATION.md#json-protocol)
- [Connection Guide](GODOT_INTEGRATION.md#connecting-to-server)
- [Mask Rendering Examples](GODOT_INTEGRATION.md#rendering-masks)
- [Error Handling](GODOT_INTEGRATION.md#error-handling)

---

## 🐛 Troubleshooting

### Server won't start

**Error: "Address already in use"**

```bash
# Kill process using the port
# Linux/Mac:
lsof -ti:5555 | xargs kill -9

# Windows:
netstat -ano | findstr :5555
taskkill /PID <PID> /F
```

**Error: "Model files not found"**

```bash
# Verify models exist
ls models/
# Should show: svm_model.pkl  bovw_encoder.pkl  scaler.pkl

# Check paths in config.py
python -c "from udp_server.config import ServerConfig; print(ServerConfig.validate())"
```

### No detections received

**Check server logs:**

```bash
# Start with debug logging
python -m udp_server.server --mode dev --log-level DEBUG
```

**Verify network connectivity:**

```bash
# Test UDP connectivity
nc -u 127.0.0.1 5555
# Or use the test client
python test_client.py image --image test.jpg
```

**Check firewall:**

```bash
# Linux: Allow UDP ports
sudo ufw allow 5555/udp
sudo ufw allow 5556/udp

# Windows: Add firewall rules for Python
```

### Low FPS / High latency

**Reduce frame size:**

- Use 640x480 instead of 1920x1080
- Lower JPEG quality (50-70)

**Increase worker threads:**

```python
# In config.py
WORKER_THREADS = 4  # Default is 2
```

**Reduce keypoints:**

```python
# In config.py
MAX_KEYPOINTS = 300  # Default is 500
```

**Check system resources:**

```bash
# Monitor CPU/RAM
htop  # Linux
top   # Mac
taskmgr  # Windows
```

### Frames being dropped

**Check server statistics:**

```
# Server prints stats every 10 seconds:
============================================================
SERVER STATISTICS
============================================================
Frames received: 450
Messages sent: 448
Frames processed: 448
Frames dropped: 12  ← Check this
Avg processing time: 45.2ms
Queue size: 1
============================================================
```

**Solution: Increase queue size**

```python
# In config.py
MAX_QUEUE_SIZE = 5  # Default is 2 (increase if dropping)
```

### Server crashes

**Check logs:**

```bash
# Run with full error output
python -m udp_server.server --mode dev --log-level DEBUG 2>&1 | tee server.log
```

**Common causes:**

- Out of memory (reduce resolution/quality)
- Model corruption (re-download models)
- OpenCV version mismatch

---

## 📊 Server Statistics

The server logs statistics every 10 seconds:

```
============================================================
SERVER STATISTICS
============================================================
Uptime: 125.3s
Frames received: 3750           ← Frames received from Godot
Messages sent: 3742             ← Responses sent to Godot
Frames processed: 3742          ← Faces detected
Frames dropped: 8               ← Frames skipped (queue full)
Avg processing time: 42.3ms     ← ML processing time
Queue size: 1                   ← Current queue depth
Receive rate: 29.9 FPS          ← Incoming frame rate
============================================================
```

### Interpretation

**Healthy System:**

- Frames dropped: <5%
- Avg processing time: <50ms
- Receive rate: 15-30 FPS

**System Under Load:**

- Frames dropped: >10% → Reduce resolution or increase threads
- Avg processing time: >80ms → Reduce keypoints or resolution
- Queue size: Always at max → Processing too slow

---

## 🔧 Advanced Configuration

### Custom Model Paths

```python
# In config.py or via environment variables
SVM_MODEL_PATH = "custom/path/to/svm.pkl"
BOVW_ENCODER_PATH = "custom/path/to/bovw.pkl"
SCALER_PATH = "custom/path/to/scaler.pkl"
```

### Network Tuning

```python
# config.py
class ServerConfig:
    # Increase for high-latency networks
    PROCESS_TIMEOUT_SEC = 2.0  # Default: 1.0

    # Buffer size (bytes)
    UDP_BUFFER_SIZE = 65535  # Max UDP packet size
```

### Logging to File

```python
# config.py
LOG_FILE = "server.log"  # Enable file logging
```

Or command line:

```bash
python -m udp_server.server --mode prod 2>&1 | tee server.log
```

---

## 📝 API Reference

See **[GODOT_INTEGRATION.md](GODOT_INTEGRATION.md)** for complete API documentation.

---

## 🆘 Support

**Common Issues:**

1. Server won't start → Check ports and model files
2. No detections → Verify network connectivity
3. Low FPS → Reduce resolution/quality
4. High latency → Increase worker threads

**Getting Help:**

- Check server logs with `--log-level DEBUG`
- Use test client to isolate issues
- Verify network with `netstat`/`lsof`

---

## 📄 License

Same as the main project.
