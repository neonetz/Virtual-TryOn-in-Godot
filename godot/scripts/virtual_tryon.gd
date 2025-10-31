extends Control
# VirtualTryOn.gd
# Virtual Try-On demo using UDP for high-performance real-time detection
# Attach this to your TryOn scene root node

@onready var camera_feed: TextureRect = $CameraFeed
@onready var status_label: Label = $StatusLabel
@onready var face_count_label: Label = $FaceCountLabel
@onready var fps_label: Label = $FPSLabel

var detector_client: FaceDetectorClient
var camera: CameraFeed
var camera_texture: CameraTexture

# Performance tracking
var frame_count: int = 0
var fps_timer: float = 0.0
var current_fps: float = 0.0

func _ready():
	# Initialize UDP face detector client
	detector_client = FaceDetectorClient.new()
	add_child(detector_client)
	
	# Connect signals
	detector_client.detection_completed.connect(_on_detection_completed)
	detector_client.detection_failed.connect(_on_detection_failed)
	detector_client.connection_status_changed.connect(_on_connection_status_changed)
	
	# Set detection rate (FPS)
	detector_client.set_send_rate(30.0)  # Send 30 frames per second to server
	
	# Initialize camera
	_initialize_camera()

func _initialize_camera():
	# Get camera feed (feed 0 is usually the default webcam)
	var cameras = CameraServer.get_feed_count()
	
	if cameras == 0:
		status_label.text = "No camera found"
		push_error("No camera available")
		return
	
	camera = CameraServer.get_feed(0)
	camera_texture = CameraTexture.new()
	camera_texture.camera_feed_id = camera.get_id()
	
	# Set to camera feed texture rect
	if camera_feed:
		camera_feed.texture = camera_texture
		camera_feed.expand_mode = TextureRect.EXPAND_FIT_WIDTH
		camera_feed.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT
	
	print("Camera initialized: ", camera.get_name())
	status_label.text = "Camera ready - UDP connected"

func _process(delta):
	# Update FPS counter
	fps_timer += delta
	frame_count += 1
	
	if fps_timer >= 1.0:
		current_fps = frame_count / fps_timer
		if fps_label:
			fps_label.text = "FPS: %.1f" % current_fps
		frame_count = 0
		fps_timer = 0.0
	
	# Process camera frame
	_process_frame()

func _process_frame():
	if camera_texture == null:
		return
	
	# Get current camera image
	var image = camera_texture.get_image()
	
	if image == null:
		return
	
	# Send to UDP detector (throttled internally)
	detector_client.detect_faces(image)

func _on_detection_completed(faces: Array, result_image: Image):
	# Update display with result image
	if camera_feed:
		var texture = ImageTexture.create_from_image(result_image)
		camera_feed.texture = texture
	
	# Update face count
	if face_count_label:
		face_count_label.text = "Faces: %d" % faces.size()
	
	# Print face information (optional - comment out for performance)
	# for i in range(faces.size()):
	# 	var face = faces[i]
	# 	print("Face ", i, ": x=", face.x, " y=", face.y, " w=", face.w, " h=", face.h, " score=", face.score)

func _on_detection_failed(error: String):
	if status_label:
		status_label.text = "Error: " + error
	push_warning("Detection failed: " + error)

func _on_connection_status_changed(connected: bool):
	if connected:
		status_label.text = "UDP Server: Connected"
	else:
		status_label.text = "UDP Server: Disconnected"
		push_warning("UDP server not connected. Start server with: python godot_bridge_udp.py")

func _exit_tree():
	# Cleanup
	if detector_client:
		detector_client.close()
