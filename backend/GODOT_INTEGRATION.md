# Godot Integration Guide - Virtual Try-On Face Detection

**For Frontend: Complete guide to integrating with the UDP face detection server**

---

## 📋 Quick Reference

| Item                    | Value                                   |
| ----------------------- | --------------------------------------- |
| **Server Receive Port** | `5555` (Godot sends frames here)        |
| **Server Send Port**    | `5556` (Godot receives detections here) |
| **Protocol**            | UDP (JSON messages)                     |
| **Message Format**      | JSON over UDP                           |
| **Localhost**           | `127.0.0.1` (development)               |
| **LAN**                 | Server's IP address (production)        |

---

## 🎯 Overview

### What You Need to Do

1. **Capture camera frames** in Godot
2. **Compress to JPEG** and encode to base64
3. **Send via UDP** to port `5555`
4. **Listen on UDP** port `5556` for detection results
5. **Parse JSON** to get face positions
6. **Render masks** at detected positions with rotation

### What the Python Server Does

- ✅ Receives your camera frames
- ✅ Detects faces using ML (Haar + ORB + SVM)
- ✅ Detects eyes for rotation
- ✅ Sends back positions, sizes, and rotation angles
- ❌ Does NOT render masks (you handle that)

---

## 📡 JSON Protocol Specification

### Message 1: Godot → Python (Send Frame)

**When to send:** Every camera frame (30 FPS recommended)

**Format:**

```json
{
  "type": "frame",
  "timestamp": 1730563892.123,
  "frame": "/9j/4AAQSkZJRgABAQAA...",
  "width": 640,
  "height": 480,
  "quality": 70
}
```

**Fields:**

| Field       | Type   | Description               | Required                  |
| ----------- | ------ | ------------------------- | ------------------------- |
| `type`      | string | Always `"frame"`          | ✅ Yes                    |
| `timestamp` | float  | Unix timestamp (seconds)  | ✅ Yes                    |
| `frame`     | string | Base64-encoded JPEG image | ✅ Yes                    |
| `width`     | int    | Frame width in pixels     | ✅ Yes                    |
| `height`    | int    | Frame height in pixels    | ✅ Yes                    |
| `quality`   | int    | JPEG quality (1-100)      | ⚠️ Optional (default: 70) |

**Recommended Settings:**

- **Resolution:** 640x480 (balance between quality and performance)
- **Quality:** 70 (good compression, acceptable quality)
- **Frame Rate:** 30 FPS max (send every frame)

---

### Message 2: Python → Godot (Detection Results)

**When received:** After each frame is processed (~30-50ms later)

**Format:**

```json
{
  "type": "detection",
  "timestamp": 1730563892.145,
  "processing_time_ms": 45.2,
  "faces": [
    {
      "id": 0,
      "bbox": {
        "x": 120,
        "y": 80,
        "w": 180,
        "h": 220
      },
      "center": {
        "x": 210,
        "y": 190
      },
      "rotation_deg": -12.5,
      "confidence": 0.94,
      "eyes": {
        "left": { "x": 170, "y": 150 },
        "right": { "x": 250, "y": 145 },
        "detected": true
      }
    }
  ],
  "frame_id": 12345
}
```

**Top-Level Fields:**

| Field                | Type     | Description                          |
| -------------------- | -------- | ------------------------------------ |
| `type`               | string   | Always `"detection"`                 |
| `timestamp`          | float    | When result was generated            |
| `processing_time_ms` | float    | ML processing time (info only)       |
| `faces`              | array    | Array of face objects (can be empty) |
| `frame_id`           | int/null | Optional frame identifier            |

**Face Object Fields:**

| Field           | Type        | Description                | How to Use             |
| --------------- | ----------- | -------------------------- | ---------------------- |
| `id`            | int         | Face ID (0, 1, 2...)       | Use as instance ID     |
| `bbox.x`        | int         | Top-left X coordinate      | Position mask here     |
| `bbox.y`        | int         | Top-left Y coordinate      | Position mask here     |
| `bbox.w`        | int         | Bounding box width         | Scale mask to this     |
| `bbox.h`        | int         | Bounding box height        | Scale mask to this     |
| `center.x`      | float       | Face center X              | Easier for centering   |
| `center.y`      | float       | Face center Y              | Easier for centering   |
| `rotation_deg`  | float       | Rotation angle (degrees)   | Rotate mask by this    |
| `confidence`    | float       | Detection confidence (0-1) | Filter low scores      |
| `eyes.left`     | object/null | Left eye position          | Debug/validation       |
| `eyes.right`    | object/null | Right eye position         | Debug/validation       |
| `eyes.detected` | bool        | Were eyes found?           | Check rotation quality |

**Coordinate System:**

- Origin: Top-left corner (0, 0)
- X-axis: Right is positive
- Y-axis: Down is positive
- Rotation: Clockwise is positive (same as most graphics APIs)

---

### Message 3: Python → Godot (Error)

**When received:** If processing fails

**Format:**

```json
{
  "type": "error",
  "timestamp": 1730563892.15,
  "message": "Processing failed: Invalid frame data",
  "code": "PROCESSING_ERROR"
}
```

**Error Codes:**

| Code               | Meaning                  | Action                                |
| ------------------ | ------------------------ | ------------------------------------- |
| `PROCESSING_ERROR` | ML pipeline failed       | Continue, skip frame                  |
| `INVALID_FRAME`    | Cannot decode frame      | Check encoding                        |
| `TIMEOUT`          | Processing took too long | Increase timeout or reduce resolution |

---

## 🔌 Connecting to Server

### Step 1: Setup UDP Sockets

**GDScript Example:**

```gdscript
extends Node

var send_socket: PacketPeerUDP
var recv_socket: PacketPeerUDP

const SERVER_HOST = "127.0.0.1"  # Change for LAN
const SERVER_PORT = 5555
const RECEIVE_PORT = 5556

func _ready():
    # Socket for sending frames
    send_socket = PacketPeerUDP.new()
    send_socket.set_dest_address(SERVER_HOST, SERVER_PORT)

    # Socket for receiving detections
    recv_socket = PacketPeerUDP.new()
    recv_socket.bind(RECEIVE_PORT)

    print("Connected to face detection server")
```

### Step 2: Capture & Send Frames

**GDScript Example:**

```gdscript
func _process(delta):
    # Get camera frame
    var camera_texture = $Camera.get_texture()
    var image = camera_texture.get_image()

    # Resize if needed (recommended: 640x480)
    if image.get_width() > 640:
        image.resize(640, 480)

    # Convert to JPEG
    var jpeg_buffer = image.save_jpg_to_buffer(0.7)  # 70% quality

    # Encode to base64
    var base64_frame = Marshalls.raw_to_base64(jpeg_buffer)

    # Create JSON message
    var message = {
        "type": "frame",
        "timestamp": Time.get_unix_time_from_system(),
        "frame": base64_frame,
        "width": image.get_width(),
        "height": image.get_height(),
        "quality": 70
    }

    # Send via UDP
    var json_string = JSON.stringify(message)
    send_socket.put_packet(json_string.to_utf8_buffer())
```

### Step 3: Receive Detection Results

**GDScript Example:**

```gdscript
func _process(delta):
    # ... send frame code ...

    # Check for detection results
    if recv_socket.get_available_packet_count() > 0:
        var packet = recv_socket.get_packet()
        var json_string = packet.get_string_from_utf8()
        var json = JSON.new()
        var parse_result = json.parse(json_string)

        if parse_result == OK:
            var data = json.data
            handle_detection_result(data)

func handle_detection_result(data: Dictionary):
    if data.type == "detection":
        var faces = data.faces
        print("Detected %d face(s)" % faces.size())

        # Update mask positions
        for face in faces:
            update_mask(face)

    elif data.type == "error":
        print("Error: ", data.message)
```

---

## 🎭 Rendering Masks

### Understanding the Data

When you receive a face object:

```json
{
  "id": 0,
  "bbox": { "x": 120, "y": 80, "w": 180, "h": 220 },
  "center": { "x": 210, "y": 190 },
  "rotation_deg": -12.5,
  "confidence": 0.94
}
```

**Visual representation:**

```
Camera Frame (640×480)
┌─────────────────────────────────┐
│                                 │
│    (120,80) ┌──────180px──────┐ │
│             │                  │ │
│             │  DETECTED FACE   │ │
│             │                  │ │
│             │   center(210,190)│ │ 220px
│             │        •         │ │
│             │                  │ │
│             │  rotation:-12.5° │ │
│             └──────────────────┘ │
│                                 │
└─────────────────────────────────┘
```

### Basic Mask Rendering (2D Sprite)

**GDScript Example:**

```gdscript
func update_mask(face_data: Dictionary):
    # Get or create mask sprite
    var mask_sprite = get_mask_sprite(face_data.id)

    # Position at center
    mask_sprite.position = Vector2(
        face_data.center.x,
        face_data.center.y
    )

    # Scale to face size
    var mask_base_size = Vector2(mask_sprite.texture.get_width(),
                                  mask_sprite.texture.get_height())
    var scale_x = face_data.bbox.w / mask_base_size.x
    var scale_y = face_data.bbox.h / mask_base_size.y
    mask_sprite.scale = Vector2(scale_x, scale_y)

    # Rotate
    mask_sprite.rotation_degrees = face_data.rotation_deg

    # Optional: Set opacity based on confidence
    mask_sprite.modulate.a = face_data.confidence

    # Show sprite
    mask_sprite.visible = true
```

### Advanced: 3D Mask with Depth

**GDScript Example:**

```gdscript
func update_mask_3d(face_data: Dictionary):
    var mask_3d = get_mask_3d_node(face_data.id)

    # Convert 2D screen position to 3D
    var camera = get_viewport().get_camera_3d()
    var depth = estimate_depth_from_bbox_size(face_data.bbox.w)
    var world_pos = camera.project_position(
        Vector2(face_data.center.x, face_data.center.y),
        depth
    )

    mask_3d.global_position = world_pos

    # Rotate around Z-axis
    mask_3d.rotation.z = deg_to_rad(face_data.rotation_deg)

    # Scale based on distance
    var scale_factor = depth / 2.0
    mask_3d.scale = Vector3.ONE * scale_factor

func estimate_depth_from_bbox_size(width: float) -> float:
    # Rough estimate: larger bbox = closer to camera
    var avg_face_width_cm = 15.0
    var focal_length_px = 500.0  # Calibrate for your camera
    return (avg_face_width_cm * focal_length_px) / width
```

### Multi-Face Support

**GDScript Example:**

```gdscript
var active_masks = {}  # Dictionary: face_id -> mask_node

func handle_detection_result(data: Dictionary):
    var current_face_ids = []

    # Update/create masks for detected faces
    for face in data.faces:
        var face_id = face.id
        current_face_ids.append(face_id)

        # Get or create mask
        if not active_masks.has(face_id):
            active_masks[face_id] = create_new_mask()

        # Update mask
        update_mask(face, active_masks[face_id])

    # Hide masks for faces no longer detected
    for face_id in active_masks.keys():
        if face_id not in current_face_ids:
            active_masks[face_id].visible = false

func create_new_mask() -> Sprite2D:
    var sprite = Sprite2D.new()
    sprite.texture = preload("res://masks/halloween_mask.png")
    sprite.centered = true
    add_child(sprite)
    return sprite
```

---

## 🎨 Mask Asset Guidelines

### File Format

- **Format:** PNG with alpha channel
- **Recommended size:** 512×512 px or 1024×1024 px
- **Color space:** RGBA

### Design Considerations

1. **Centered:** Design mask centered in image
2. **Face coverage:** Should cover entire face area
3. **Transparent background:** Use alpha channel
4. **Aspect ratio:** Design for typical face (~0.8 ratio, width:height)

### Example Mask Structure

```
mask.png (512×512)
┌─────────────────────┐
│ Transparent         │
│                     │
│   ┌───────────┐     │
│   │           │     │
│   │   MASK    │     │ ← Face area
│   │  DESIGN   │     │
│   │           │     │
│   └───────────┘     │
│                     │
│ Transparent         │
└─────────────────────┘
```

### Testing Your Mask

1. Load mask in Godot scene
2. Test at different scales
3. Test rotation (-30° to +30°)
4. Test with multiple instances (multi-face)

---

## ⚡ Performance Optimization

### 1. Frame Transmission

**Recommended settings:**

```gdscript
const FRAME_WIDTH = 640
const FRAME_HEIGHT = 480
const JPEG_QUALITY = 0.7  # 70%
const TARGET_FPS = 30
```

**Don't send every frame if struggling:**

```gdscript
var frame_skip = 0

func _process(delta):
    frame_skip += 1
    if frame_skip >= 2:  # Send every 2nd frame (15 FPS)
        send_frame()
        frame_skip = 0
```

### 2. Adaptive Quality

**Lower quality when processing is slow:**

```gdscript
var current_quality = 0.7
var last_processing_time = 0.0

func handle_detection_result(data: Dictionary):
    last_processing_time = data.processing_time_ms

    # Adaptive quality
    if last_processing_time > 80:
        current_quality = max(0.5, current_quality - 0.05)
    elif last_processing_time < 40:
        current_quality = min(0.8, current_quality + 0.05)
```

### 3. Smooth Mask Movement

**Use interpolation to avoid jittery masks:**

```gdscript
var mask_target_pos = Vector2.ZERO
var mask_target_rot = 0.0

func _process(delta):
    # Smooth position
    mask_sprite.position = mask_sprite.position.lerp(
        mask_target_pos,
        10.0 * delta  # Smoothing factor
    )

    # Smooth rotation
    mask_sprite.rotation_degrees = lerp_angle(
        mask_sprite.rotation_degrees,
        mask_target_rot,
        10.0 * delta
    )

func update_mask_targets(face_data: Dictionary):
    mask_target_pos = Vector2(face_data.center.x, face_data.center.y)
    mask_target_rot = face_data.rotation_deg
```

---

## 🔍 Debugging & Testing

### Enable Debug Visualization

**Show bounding boxes and detection info:**

```gdscript
var debug_mode = true

func handle_detection_result(data: Dictionary):
    if debug_mode:
        # Clear previous debug
        queue_redraw()

        # Draw bounding boxes
        for face in data.faces:
            draw_rect(
                Rect2(face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h),
                Color.GREEN,
                false,
                2.0
            )

            # Draw center
            draw_circle(
                Vector2(face.center.x, face.center.y),
                5,
                Color.RED
            )

            # Draw eyes
            if face.eyes.detected:
                if face.eyes.left:
                    draw_circle(
                        Vector2(face.eyes.left.x, face.eyes.left.y),
                        3,
                        Color.BLUE
                    )
                if face.eyes.right:
                    draw_circle(
                        Vector2(face.eyes.right.x, face.eyes.right.y),
                        3,
                        Color.BLUE
                    )
```

### Log Detection Stats

```gdscript
func handle_detection_result(data: Dictionary):
    print("=== Detection Result ===")
    print("Processing time: %.1f ms" % data.processing_time_ms)
    print("Faces detected: %d" % data.faces.size())

    for i in range(data.faces.size()):
        var face = data.faces[i]
        print("  Face %d:" % i)
        print("    Position: (%d, %d)" % [face.bbox.x, face.bbox.y])
        print("    Size: %dx%d" % [face.bbox.w, face.bbox.h])
        print("    Confidence: %.2f" % face.confidence)
        print("    Rotation: %.1f°" % face.rotation_deg)
```

### Network Diagnostics

```gdscript
var frames_sent = 0
var detections_received = 0
var last_detection_time = 0.0

func _process(delta):
    # Send frame
    send_frame()
    frames_sent += 1

    # Check timeout
    if Time.get_ticks_msec() - last_detection_time > 1000:
        print("WARNING: No detection for 1 second!")
        print("  Frames sent: %d" % frames_sent)
        print("  Detections received: %d" % detections_received)

func handle_detection_result(data: Dictionary):
    detections_received += 1
    last_detection_time = Time.get_ticks_msec()

    # Calculate response rate
    var response_rate = float(detections_received) / float(frames_sent)
    if response_rate < 0.8:
        print("WARNING: Low response rate: %.1f%%" % (response_rate * 100))
```

---

## ❌ Error Handling

### Connection Loss

```gdscript
var connection_timeout = 2.0  # seconds
var last_response_time = 0.0

func _process(delta):
    var current_time = Time.get_unix_time_from_system()

    # Check timeout
    if current_time - last_response_time > connection_timeout:
        handle_connection_lost()

func handle_connection_lost():
    print("Connection lost to face detection server")

    # Hide all masks
    for mask in active_masks.values():
        mask.visible = false

    # Show reconnecting UI
    $UI/ReconnectingLabel.visible = true

    # Try to reconnect
    attempt_reconnect()
```

### Invalid Data

```gdscript
func handle_detection_result(data: Dictionary):
    # Validate data structure
    if not data.has("type"):
        print("ERROR: Invalid message format")
        return

    if data.type == "detection":
        if not data.has("faces"):
            print("ERROR: Missing faces array")
            return

        # Validate each face
        for face in data.faces:
            if not validate_face_data(face):
                continue  # Skip invalid face
            update_mask(face)

func validate_face_data(face: Dictionary) -> bool:
    # Check required fields
    if not (face.has("bbox") and face.has("center") and
            face.has("rotation_deg") and face.has("confidence")):
        return false

    # Validate confidence
    if face.confidence < 0.5:  # Filter low confidence
        return false

    # Validate bbox
    var bbox = face.bbox
    if bbox.w <= 0 or bbox.h <= 0:
        return false

    return true
```

### Graceful Degradation

```gdscript
func handle_detection_result(data: Dictionary):
    if data.type == "error":
        print("Server error: ", data.message)

        # Don't crash - just continue with last known positions
        # Masks will stay at their current positions
        return

    # Handle detection normally
    # ...
```

---

## 🎯 Best Practices

### 1. Always validate received data

```gdscript
# ✅ Good
if data.has("faces") and data.faces.size() > 0:
    for face in data.faces:
        if validate_face_data(face):
            update_mask(face)

# ❌ Bad
for face in data.faces:  # Crashes if faces doesn't exist
    update_mask(face)
```

### 2. Handle missing faces gracefully

```gdscript
# ✅ Good
func handle_detection_result(data: Dictionary):
    if data.faces.size() == 0:
        # Hide all masks with fade-out
        fade_out_all_masks()
    else:
        update_masks(data.faces)

# ❌ Bad
# Leaving masks visible when no faces detected
```

### 3. Use confidence thresholds

```gdscript
# ✅ Good
const MIN_CONFIDENCE = 0.7

func validate_face_data(face: Dictionary) -> bool:
    return face.confidence >= MIN_CONFIDENCE

# ❌ Bad
# Showing masks for all detections, including false positives
```

### 4. Smooth transitions

```gdscript
# ✅ Good
mask_sprite.position = mask_sprite.position.lerp(target_pos, 0.3)

# ❌ Bad
mask_sprite.position = target_pos  # Jittery movement
```

### 5. Handle multi-face correctly

```gdscript
# ✅ Good
# Track face IDs, reuse mask instances

# ❌ Bad
# Creating new mask for every detection
```

---

## 🔧 Troubleshooting

### No detections received

**Check network:**

```gdscript
func test_connection():
    print("Testing connection to %s:%d" % [SERVER_HOST, SERVER_PORT])
    # Send test frame
    send_frame()
    # Wait 2 seconds
    await get_tree().create_timer(2.0).timeout
    if detections_received == 0:
        print("ERROR: No response from server")
        print("  Check if server is running")
        print("  Check firewall settings")
```

### Masks appear in wrong position

**Verify coordinate system:**

- Server uses: Top-left origin, X-right, Y-down
- Check if your camera texture needs flipping

### Rotation is inverted

**Fix rotation direction:**

```gdscript
# If rotation appears inverted:
mask_sprite.rotation_degrees = -face_data.rotation_deg  # Negate
```

### Low frame rate

**Optimize:**

1. Reduce resolution: 640×480 → 320×240
2. Lower quality: 70% → 50%
3. Skip frames: Send every 2nd frame
4. Check server processing time in logs

---

## 📞 Getting Help

**Before asking for help, provide:**

1. **Godot version:** `print(Engine.get_version_info())`
2. **Network setup:** Localhost or LAN? Server IP?
3. **Error logs:** From both Godot and Python server
4. **Test results:** Does `test_client.py` work?
5. **Detection stats:** Frame rate, response rate, processing time

**Common questions:**

- "No detections" → Check network with test client
- "Wrong positions" → Verify coordinate system
- "Low FPS" → Reduce resolution/quality
- "Jittery masks" → Add smoothing/interpolation

---

## ✅ Checklist

**Before going live:**

- [ ] Server running and accessible
- [ ] Test client works (`python test_client.py webcam`)
- [ ] Godot receives detections
- [ ] Masks render at correct positions
- [ ] Rotation works correctly
- [ ] Multi-face support works
- [ ] Error handling implemented
- [ ] Connection timeout handled
- [ ] Performance acceptable (15-30 FPS)
- [ ] Debug mode can be toggled
- [ ] Firewall rules configured (if LAN)

---

## 📚 Additional Resources

- **Server Documentation:** See `README_UDP_SERVER.md`
- **Protocol Details:** See `udp_server/protocol.py`
- **Test Client:** Use `test_client.py` to verify server
- **Godot UDP Docs:** <https://docs.godotengine.org/en/stable/classes/class_packetpeerudp.html>

---

**Questions? Issues?**

- Test with `test_client.py` first
- Check server logs with `--log-level DEBUG`
- Verify network connectivity
- Contact backend team with logs and setup details
