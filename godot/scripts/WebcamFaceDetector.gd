"""
Webcam Face Detector Controller
Manages webcam capture and face detection with mask overlay
"""

extends Control
class_name WebcamFaceDetector

## UI References
@onready var camera_preview: TextureRect = $CameraPreview
@onready var result_display: TextureRect = $ResultDisplay
@onready var fps_label: Label = $FPSLabel
@onready var stats_label: Label = $StatsLabel

## Face Detector Client
var face_detector: FaceDetectorClient

## Camera
var camera_feed: CameraFeed
var camera_index: int = 0

## Settings
@export var process_every_n_frames: int = 2  ## Skip frames for performance
@export var mask_type: String = "default"
@export var target_width: int = 640
@export var target_height: int = 480

## State
var frame_count: int = 0
var fps_samples: Array[float] = []
var last_frame_time: float = 0.0
var is_processing: bool = false


func _ready() -> void:
	# Initialize face detector client
	face_detector = FaceDetectorClient.new()
	add_child(face_detector)
	
	# Connect signals
	face_detector.faces_detected.connect(_on_faces_detected)
	face_detector.detection_failed.connect(_on_detection_failed)
	face_detector.connection_established.connect(_on_connected)
	
	# Connect to backend
	face_detector.connect_to_server("127.0.0.1", 5555)
	
	# Initialize camera
	_initialize_camera()
	
	print("[WebcamFaceDetector] Ready")


func _initialize_camera() -> void:
	"""Initialize webcam capture"""
	var num_cameras = CameraServer.get_feed_count()
	
	if num_cameras == 0:
		push_error("[WebcamFaceDetector] No cameras found")
		return
	
	print("[WebcamFaceDetector] Found %d camera(s)" % num_cameras)
	
	# Get first camera
	camera_feed = CameraServer.get_feed(camera_index)
	
	if camera_feed:
		camera_feed.feed_is_active = true
		print("[WebcamFaceDetector] Camera %d activated" % camera_index)
	else:
		push_error("[WebcamFaceDetector] Failed to get camera feed")


func _process(delta: float) -> void:
	if not camera_feed or not camera_feed.feed_is_active:
		return
	
	# Calculate FPS
	_update_fps(delta)
	
	# Get camera frame
	var image = camera_feed.get_frame()
	
	if not image:
		return
	
	# Display original camera feed
	if camera_preview:
		camera_preview.texture = ImageTexture.create_from_image(image)
	
	# Process frame (with skipping for performance)
	frame_count += 1
	
	if frame_count % process_every_n_frames == 0 and not is_processing:
		_process_frame(image)
	
	# Update stats display
	_update_stats()


func _process_frame(image: Image) -> void:
	"""Send frame to backend for processing"""
	is_processing = true
	
	# Resize image for faster processing
	if image.get_width() != target_width or image.get_height() != target_height:
		image.resize(target_width, target_height, Image.INTERPOLATE_BILINEAR)
	
	# Send to backend
	face_detector.detect_faces(image, mask_type)


func _on_faces_detected(faces: Array, overlay_image: Image) -> void:
	"""Handle detected faces with mask overlay"""
	is_processing = false
	
	if overlay_image:
		# Display result with mask overlay
		if result_display:
			result_display.texture = ImageTexture.create_from_image(overlay_image)
	
	# Log detection info
	if faces.size() > 0:
		print("[WebcamFaceDetector] Detected %d face(s)" % faces.size())


func _on_detection_failed(error: String) -> void:
	"""Handle detection failure"""
	is_processing = false
	push_warning("[WebcamFaceDetector] Detection failed: %s" % error)


func _on_connected() -> void:
	"""Handle successful connection to backend"""
	print("[WebcamFaceDetector] Connected to backend server")


func _update_fps(delta: float) -> void:
	"""Update FPS counter"""
	if delta > 0:
		var current_fps = 1.0 / delta
		fps_samples.append(current_fps)
		
		# Keep only last 30 samples
		if fps_samples.size() > 30:
			fps_samples.pop_front()
		
		# Calculate average FPS
		var avg_fps = 0.0
		for fps in fps_samples:
			avg_fps += fps
		avg_fps /= fps_samples.size()
		
		# Update label
		if fps_label:
			fps_label.text = "FPS: %.1f" % avg_fps


func _update_stats() -> void:
	"""Update statistics display"""
	if not stats_label:
		return
	
	var stats = face_detector.get_stats()
	
	stats_label.text = """
	Connected: %s
	Requests: %d
	Responses: %d
	Pending: %d
	Avg Latency: %.1f ms
	Success Rate: %.1f%%
	""" % [
		"Yes" if stats.connected else "No",
		stats.total_requests,
		stats.total_responses,
		stats.pending_requests,
		stats.avg_latency_ms,
		stats.success_rate
	]


## Public API

func set_camera_index(index: int) -> void:
	"""Switch to different camera"""
	if index < 0 or index >= CameraServer.get_feed_count():
		push_error("[WebcamFaceDetector] Invalid camera index: %d" % index)
		return
	
	# Deactivate current camera
	if camera_feed:
		camera_feed.feed_is_active = false
	
	# Activate new camera
	camera_index = index
	_initialize_camera()


func set_mask_type(new_mask_type: String) -> void:
	"""Change mask type"""
	mask_type = new_mask_type
	print("[WebcamFaceDetector] Mask type changed to: %s" % mask_type)


func set_process_interval(interval: int) -> void:
	"""Set frame skip interval (1 = every frame, 2 = every other frame, etc.)"""
	process_every_n_frames = max(1, interval)
	print("[WebcamFaceDetector] Processing every %d frame(s)" % process_every_n_frames)


func toggle_camera() -> void:
	"""Toggle camera on/off"""
	if camera_feed:
		camera_feed.feed_is_active = not camera_feed.feed_is_active
		print("[WebcamFaceDetector] Camera %s" % ("ON" if camera_feed.feed_is_active else "OFF"))


func _exit_tree() -> void:
	"""Cleanup on exit"""
	if camera_feed:
		camera_feed.feed_is_active = false
	
	print("[WebcamFaceDetector] Cleanup complete")
