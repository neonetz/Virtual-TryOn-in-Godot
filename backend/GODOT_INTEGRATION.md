# 🎮 Godot Integration Guide for Virtual Try-On

Complete guide for frontend developers to integrate the Python backend with Godot v4.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the API](#understanding-the-api)
3. [Implementation Steps](#implementation-steps)
4. [Complete Code Examples](#complete-code-examples)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)
7. [Testing Guide](#testing-guide)

---

## 🚀 Quick Start

### Prerequisites

- Godot 4.x installed
- Python backend running (see main README)
- Webcam connected

### 5-Minute Setup

1. **Start Python Server**

   ```bash
   python -m src.server.udp_server \
     --model models/face_detector_svm.pkl \
     --scaler models/scaler.pkl \
     --config models/config.json \
     --masks assets/masks/
   ```

2. **Create Godot Project**
   - New project in Godot
   - Add the scripts below
   - Run and see mask overlay!

---

## 🔌 Understanding the API

### Communication Flow

```
Godot                                Python Server
  |                                       |
  |  1. Capture webcam frame             |
  |------------------------------------>  |
  |  2. Send JSON request (UDP)          |
  |     {frame_id, image, mask_type}     |
  |                                       |
  |                          3. Detect face
  |                          4. Apply mask
  |                          5. Create response
  |                                       |
  |  <------------------------------------|
  |  6. Receive JSON response (UDP)      |
  |     {faces, overlay_image, status}   |
  |                                       |
  |  7. Display result                   |
```

### Request Structure

```json
{
  "frame_id": 12345,
  "timestamp": 1698765432.123,
  "image": "base64_jpeg_string",
  "mask_type": "zombie",
  "options": {
    "quality": 85
  }
}
```

**Important Fields:**

- `frame_id`: For matching responses to requests
- `image`: Your webcam frame as base64-encoded JPEG
- `mask_type`: Mask filename without .png (e.g., "zombie" for zombie.png)

### Response Structure

```json
{
  "frame_id": 12345,
  "status": "success",
  "faces": [
    {
      "bbox": { "x": 120, "y": 80, "w": 150, "h": 150 },
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
  "overlay_image": "base64_jpeg_result",
  "processing_time_ms": 45.2
}
```

**Status Values:**

- `"success"`: Face detected, mask applied
- `"no_face"`: No face found
- `"error"`: Something went wrong

---

## 🛠️ Implementation Steps

### Step 1: Create UDP Client Node

**File: `UDPClient.gd`**

```gdscript
extends Node
class_name UDPClient

# Server configuration
@export var server_host := "127.0.0.1"  # Change to server IP for production
@export var server_port := 5001

# UDP socket
var udp := PacketPeerUDP.new()

# Frame tracking
var frame_id := 0
var pending_requests := {}  # {frame_id: send_timestamp}

signal response_received(response: Dictionary)
signal connection_error(error: String)

func _ready():
    # Set destination
    var error := udp.set_dest_address(server_host, server_port)

    if error != OK:
        push_error("Failed to set UDP destination: %s" % error)
        connection_error.emit("Failed to connect to server")
        return

    print("✓ UDP Client connected to %s:%d" % [server_host, server_port])

func send_frame(image: Image, mask_type: String, quality: int = 85) -> int:
    """
    Send a frame to the server for processing

    Args:
        image: The captured image
        mask_type: Name of mask to apply (without .png)
        quality: JPEG quality (0-100)

    Returns:
        frame_id of sent frame
    """

    # Convert image to JPEG
    var buffer := image.save_jpg_to_buffer(quality)
    if buffer.size() == 0:
        push_error("Failed to encode image as JPEG")
        return -1

    # Encode to base64
    var base64_image := Marshalls.raw_to_base64(buffer)

    # Create request
    var request := {
        "frame_id": frame_id,
        "timestamp": Time.get_unix_time_from_system(),
        "image": base64_image,
        "mask_type": mask_type,
        "options": {
            "quality": quality
        }
    }

    # Serialize to JSON
    var json_string := JSON.stringify(request)
    var packet := json_string.to_utf8_buffer()

    # Send via UDP
    var error := udp.put_packet(packet)

    if error != OK:
        push_error("Failed to send packet: %s" % error)
        return -1

    # Track pending request
    pending_requests[frame_id] = Time.get_ticks_msec()

    # Increment frame ID
    frame_id += 1

    return frame_id - 1

func poll() -> void:
    """
    Check for incoming responses
    Call this every frame in _process()
    """

    while udp.get_available_packet_count() > 0:
        var packet := udp.get_packet()

        if packet.size() == 0:
            continue

        # Decode JSON
        var json_string := packet.get_string_from_utf8()
        var json := JSON.new()
        var error := json.parse(json_string)

        if error != OK:
            push_error("Failed to parse JSON response: %s" % json.get_error_message())
            continue

        var response: Dictionary = json.data

        # Calculate latency
        var received_frame_id := response.get("frame_id", -1)
        if pending_requests.has(received_frame_id):
            var latency := Time.get_ticks_msec() - pending_requests[received_frame_id]
            response["latency_ms"] = latency
            pending_requests.erase(received_frame_id)

        # Emit signal
        response_received.emit(response)

func decode_response_image(response: Dictionary) -> Image:
    """
    Decode the base64 overlay image from response

    Returns:
        Decoded Image or null if failed
    """

    if response.get("status") != "success":
        return null

    var base64_image: String = response.get("overlay_image", "")
    if base64_image.is_empty():
        return null

    # Decode base64
    var buffer := Marshalls.base64_to_raw(base64_image)

    # Load JPEG
    var image := Image.new()
    var error := image.load_jpg_from_buffer(buffer)

    if error != OK:
        push_error("Failed to decode response image")
        return null

    return image

func get_latency_ms(frame_id: int) -> int:
    """Get latency for a specific frame"""
    if pending_requests.has(frame_id):
        return Time.get_ticks_msec() - pending_requests[frame_id]
    return -1

func clear_old_requests(max_age_ms: int = 1000) -> void:
    """Clear requests older than max_age_ms"""
    var current_time := Time.get_ticks_msec()
    var to_remove := []

    for req_frame_id in pending_requests:
        if current_time - pending_requests[req_frame_id] > max_age_ms:
            to_remove.append(req_frame_id)

    for req_frame_id in to_remove:
        pending_requests.erase(req_frame_id)
        push_warning("Request %d timed out" % req_frame_id)
```

---

### Step 2: Create Main Scene

**File: `VirtualTryOn.gd`**

```gdscript
extends Control

# Node references
@onready var camera_feed: TextureRect = $UI/CameraFeed
@onready var result_display: TextureRect = $UI/ResultDisplay
@onready var fps_label: Label = $UI/FPSLabel
@onready var status_label: Label = $UI/StatusLabel
@onready var mask_selector: OptionButton = $UI/MaskSelector
@onready var udp_client: UDPClient = $UDPClient

# Camera
var camera: CameraFeed
var camera_index := 0

# Settings
@export var target_fps := 30
@export var process_every_n_frames := 2  # Process every Nth frame for performance
@export var camera_resolution := Vector2i(640, 480)
@export var jpeg_quality := 85

# State
var current_mask := "zombie"
var frame_counter := 0
var last_result_image: Image = null
var processing_enabled := true

# Performance tracking
var frame_times := []
var max_frame_times := 60

func _ready():
    # Initialize camera
    setup_camera()

    # Setup mask selector
    setup_mask_selector()

    # Connect UDP signals
    udp_client.response_received.connect(_on_response_received)
    udp_client.connection_error.connect(_on_connection_error)

    # Start processing
    set_process(true)

    print("✓ Virtual Try-On ready")

func setup_camera() -> void:
    """Initialize camera feed"""
    var camera_count := CameraServer.get_feed_count()

    if camera_count == 0:
        push_error("No cameras found!")
        status_label.text = "ERROR: No camera"
        return

    print("Found %d camera(s)" % camera_count)

    # Get first camera
    camera = CameraServer.get_feed(camera_index)

    if camera:
        camera.feed_is_active = true
        print("✓ Camera activated: %s" % camera.get_name())
        status_label.text = "Camera: %s" % camera.get_name()
    else:
        push_error("Failed to activate camera")
        status_label.text = "ERROR: Camera activation failed"

func setup_mask_selector() -> void:
    """Setup mask selection dropdown"""
    mask_selector.clear()

    # Add available masks (must match server-side mask names)
    var masks := ["zombie", "vampire", "skeleton", "witch", "ghost"]

    for mask in masks:
        mask_selector.add_item(mask.capitalize())

    mask_selector.select(0)
    mask_selector.item_selected.connect(_on_mask_selected)

func _process(delta: float) -> void:
    # Track performance
    var frame_start := Time.get_ticks_usec()

    # Poll UDP client for responses
    udp_client.poll()

    # Process camera frame
    if camera and camera.feed_is_active and processing_enabled:
        process_camera_frame()

    # Update FPS display
    update_fps_display(delta)

    # Track frame time
    var frame_time := (Time.get_ticks_usec() - frame_start) / 1000.0
    frame_times.append(frame_time)
    if frame_times.size() > max_frame_times:
        frame_times.pop_front()

func process_camera_frame() -> void:
    """Capture and process camera frame"""
    frame_counter += 1

    # Get camera image
    var image := camera.get_frame()

    if not image:
        return

    # Display original (always)
    camera_feed.texture = ImageTexture.create_from_image(image)

    # Process every Nth frame for performance
    if frame_counter % process_every_n_frames != 0:
        return

    # Resize if needed
    if image.get_size() != camera_resolution:
        image.resize(camera_resolution.x, camera_resolution.y, Image.INTERPOLATE_BILINEAR)

    # Send to server
    var sent_frame_id := udp_client.send_frame(image, current_mask, jpeg_quality)

    if sent_frame_id < 0:
        status_label.text = "ERROR: Failed to send frame"

func _on_response_received(response: Dictionary) -> void:
    """Handle response from server"""
    var status: String = response.get("status", "unknown")
    var frame_id: int = response.get("frame_id", -1)
    var processing_time: float = response.get("processing_time_ms", 0.0)
    var latency: float = response.get("latency_ms", 0.0)

    match status:
        "success":
            # Decode and display result
            var result_image := udp_client.decode_response_image(response)

            if result_image:
                result_display.texture = ImageTexture.create_from_image(result_image)
                last_result_image = result_image

                # Update status
                var total_time := processing_time + latency
                status_label.text = "Processing: %.1fms | Latency: %.1fms | Total: %.1fms" % [
                    processing_time, latency, total_time
                ]
            else:
                status_label.text = "ERROR: Failed to decode image"

        "no_face":
            status_label.text = "No face detected"
            # Optionally display original frame
            if camera and camera.feed_is_active:
                var image := camera.get_frame()
                if image:
                    result_display.texture = ImageTexture.create_from_image(image)

        "error":
            var error_msg: String = response.get("error", "Unknown error")
            status_label.text = "ERROR: %s" % error_msg
            push_error("Server error: %s" % error_msg)

func _on_connection_error(error: String) -> void:
    """Handle connection errors"""
    status_label.text = "CONNECTION ERROR: %s" % error
    push_error("UDP connection error: %s" % error)

func _on_mask_selected(index: int) -> void:
    """Handle mask selection change"""
    var mask_name := mask_selector.get_item_text(index).to_lower()
    current_mask = mask_name
    print("Switched to mask: %s" % mask_name)
    status_label.text = "Mask: %s" % mask_name.capitalize()

func update_fps_display(delta: float) -> void:
    """Update FPS label"""
    var fps := Engine.get_frames_per_second()

    var avg_frame_time := 0.0
    if frame_times.size() > 0:
        for t in frame_times:
            avg_frame_time += t
        avg_frame_time /= frame_times.size()

    fps_label.text = "FPS: %d | Frame: %.1fms" % [fps, avg_frame_time]

func _on_save_button_pressed() -> void:
    """Save current result as image"""
    if not last_result_image:
        push_warning("No result to save")
        return

    var filename := "screenshot_%d.png" % Time.get_unix_time_from_system()
    var error := last_result_image.save_png("user://%s" % filename)

    if error == OK:
        print("✓ Saved: %s" % filename)
        status_label.text = "Saved: %s" % filename
    else:
        push_error("Failed to save screenshot")

func _on_toggle_processing_pressed() -> void:
    """Toggle processing on/off"""
    processing_enabled = !processing_enabled
    status_label.text = "Processing: %s" % ("ON" if processing_enabled else "OFF")
```

---

### Step 3: Create Scene Tree

```
VirtualTryOn (Control)
├── UDPClient (Node) - Attach UDPClient.gd
├── UI (Control)
│   ├── CameraFeed (TextureRect) - Shows original camera
│   ├── ResultDisplay (TextureRect) - Shows result with mask
│   ├── FPSLabel (Label)
│   ├── StatusLabel (Label)
│   ├── MaskSelector (OptionButton)
│   ├── SaveButton (Button)
│   └── ToggleButton (Button)
```

---

## ⚡ Performance Optimization

### 1. Frame Skipping

```gdscript
# Process every 2nd frame (doubles FPS)
if frame_counter % 2 == 0:
    udp_client.send_frame(image, current_mask)
```

### 2. Resolution Management

```gdscript
# Lower resolution = faster processing
var target_resolution := Vector2i(640, 480)  # Good balance
# var target_resolution := Vector2i(320, 240)  # Very fast
# var target_resolution := Vector2i(800, 600)  # Higher quality

image.resize(target_resolution.x, target_resolution.y, Image.INTERPOLATE_BILINEAR)
```

### 3. JPEG Quality Tuning

```gdscript
# Lower quality = smaller packets = faster transmission
var quality := 75   # Fast, good quality
# var quality := 85   # Balanced (recommended)
# var quality := 95   # High quality, slower
```

### 4. Async Processing

```gdscript
# Don't wait for previous response before sending next frame
func process_async():
    # Send frame immediately
    udp_client.send_frame(image, mask_type)

    # Display any available response
    # (may be from previous frame)
    udp_client.poll()
```

### 5. Request Throttling

```gdscript
# Limit pending requests
const MAX_PENDING := 3

func can_send_request() -> bool:
    return udp_client.pending_requests.size() < MAX_PENDING
```

---

## 🛡️ Error Handling

### Network Errors

```gdscript
func send_with_retry(image: Image, mask_type: String, max_retries := 3) -> bool:
    for attempt in range(max_retries):
        var frame_id := udp_client.send_frame(image, mask_type)

        if frame_id >= 0:
            return true

        print("Retry %d/%d" % [attempt + 1, max_retries])
        await get_tree().create_timer(0.1).timeout

    return false
```

### Timeout Handling

```gdscript
func wait_for_response(frame_id: int, timeout_ms := 200) -> Dictionary:
    var start_time := Time.get_ticks_msec()

    while Time.get_ticks_msec() - start_time < timeout_ms:
        udp_client.poll()

        # Check if response received (via signal)
        if last_response.get("frame_id") == frame_id:
            return last_response

        await get_tree().create_timer(0.01).timeout

    return {"status": "timeout"}
```

### Graceful Degradation

```gdscript
func display_frame_safe(image: Image) -> void:
    """Display frame with fallback"""
    # Try to get mask overlay
    var frame_id := udp_client.send_frame(image, current_mask)

    # Wait briefly for response
    await get_tree().create_timer(0.05).timeout
    udp_client.poll()

    # If no response, display original
    if not last_response or last_response.get("frame_id") != frame_id:
        result_display.texture = ImageTexture.create_from_image(image)
```

---

## 🧪 Testing Guide

### Unit Testing

```gdscript
# test_udp_client.gd
extends GutTest

var udp_client: UDPClient

func before_each():
    udp_client = UDPClient.new()
    add_child(udp_client)

func test_connection():
    assert_not_null(udp_client.udp)
    assert_eq(udp_client.server_port, 5001)

func test_encode_decode():
    var image := Image.create(100, 100, false, Image.FORMAT_RGB8)
    image.fill(Color.RED)

    var frame_id := udp_client.send_frame(image, "zombie")
    assert_ge(frame_id, 0)
```

### Integration Testing

1. **Start Python server**

   ```bash
   python -m src.server.udp_server --model ... --masks ...
   ```

2. **Run Godot scene**
   - Verify camera displays
   - Check UDP connection
   - Test mask switching
   - Monitor FPS

3. **Performance Testing**
   - Measure average FPS
   - Check latency (should be < 100ms)
   - Monitor packet loss

### Debug Tools

```gdscript
# Add to scene for debugging
func _ready():
    # Performance monitor
    Performance.add_custom_monitor("pending_requests",
        func(): return udp_client.pending_requests.size())

    Performance.add_custom_monitor("frame_latency",
        func(): return last_latency_ms)

    # Enable debug overlay
    $DebugOverlay.visible = true
```

---

## 📊 Performance Targets

| Metric          | Target  | Notes                                 |
| --------------- | ------- | ------------------------------------- |
| **FPS**         | 25-30   | Smooth real-time experience           |
| **Latency**     | < 50ms  | Server processing time                |
| **Total Time**  | < 100ms | End-to-end (send + process + receive) |
| **Packet Loss** | < 1%    | UDP is lossy, handle gracefully       |

---

## 🔧 Troubleshooting

### "Server not responding"

1. Check server is running: `netstat -an | grep 5001`
2. Verify IP address: Use `127.0.0.1` for local, actual IP for remote
3. Check firewall: Allow UDP port 5001

### "Poor performance"

1. Reduce resolution: 640x480 or lower
2. Increase frame skip: Process every 3rd frame
3. Lower JPEG quality: 70-80
4. Check CPU usage on server

### "Mask not aligned"

This is a server-side issue:

1. Check landmark estimation in Python
2. Verify mask anchor points
3. Try `use_perspective=False` for testing

---

## 🎯 Quick Reference

### Minimum Working Example

```gdscript
# Absolute minimum code to get started
extends Node

var udp := PacketPeerUDP.new()
var camera: CameraFeed

func _ready():
    udp.set_dest_address("127.0.0.1", 5001)
    camera = CameraServer.get_feed(0)
    camera.feed_is_active = true

func _process(delta):
    var image := camera.get_frame()
    if image and Engine.get_frames_drawn() % 3 == 0:
        var buffer := image.save_jpg_to_buffer(80)
        var b64 := Marshalls.raw_to_base64(buffer)
        var request := JSON.stringify({
            "frame_id": Engine.get_frames_drawn(),
            "image": b64,
            "mask_type": "zombie"
        })
        udp.put_packet(request.to_utf8_buffer())

    if udp.get_available_packet_count() > 0:
        var packet := udp.get_packet()
        var response := JSON.parse_string(packet.get_string_from_utf8())
        if response and response.status == "success":
            var result_buffer := Marshalls.base64_to_raw(response.overlay_image)
            var result_image := Image.new()
            result_image.load_jpg_from_buffer(result_buffer)
            # Display result_image
```

---

**Need help?** Check the main README or open an issue on GitHub!
