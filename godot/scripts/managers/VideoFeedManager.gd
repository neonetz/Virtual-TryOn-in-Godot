# Video Feed Manager
# Manages video display and updates the UI with frames from UDP client
extends Node

# Reference to TextureRect for displaying video
@onready var video_display: TextureRect

# Current frame data
var current_image: Image = null
var current_texture: ImageTexture = null
var current_bbox: Array = [0, 0, 0, 0]

# Performance statistics
var frame_count: int = 0
var last_fps_time: float = 0.0
var current_fps: float = 0.0

# Signal for other components to react to new frames
signal video_updated(bbox: Array)


func _ready() -> void:
	# Find the TextureRect node for video display
	video_display = get_node_or_null("../VideoDisplay")
	
	if video_display == null:
		push_error("[Video Manager] VideoDisplay node not found")
		return
	
	# Connect to UDP client signals
	var udp_client = get_node_or_null("../UDPClient")
	if udp_client != null:
		udp_client.frame_received.connect(_on_frame_received)
		udp_client.connection_status_changed.connect(_on_connection_status_changed)
		print("[Video Manager] Connected to UDP client")
	else:
		push_error("[Video Manager] UDPClient node not found")


func _on_frame_received(image: Image, bbox: Array) -> void:
	"""
	Called when a new frame is received from UDP client.
	Updates the video display with the new frame.
	"""
	if image == null:
		return
	
	# Store current frame data
	current_image = image
	current_bbox = bbox
	
	# Update texture
	if current_texture == null:
		current_texture = ImageTexture.create_from_image(image)
	else:
		current_texture.update(image)
	
	# Apply texture to display
	video_display.texture = current_texture
	
	# Emit signal for clothing overlay system
	emit_signal("video_updated", bbox)
	
	# Update statistics
	update_fps_counter()


func _on_connection_status_changed(connected: bool) -> void:
	"""
	Called when connection status changes.
	"""
	if connected:
		print("[Video Manager] Video stream ready")
	else:
		print("[Video Manager] Video stream disconnected")


func update_fps_counter() -> void:
	"""
	Updates and displays FPS statistics.
	"""
	frame_count += 1
	var current_time = Time.get_ticks_msec() / 1000.0
	
	if current_time - last_fps_time >= 1.0:
		current_fps = frame_count / (current_time - last_fps_time)
		frame_count = 0
		last_fps_time = current_time


func get_current_bbox() -> Array:
	"""
	Returns the current bounding box coordinates.
	Returns: [x, y, w, h]
	"""
	return current_bbox


func get_fps() -> float:
	"""
	Returns current FPS.
	"""
	return current_fps


func get_video_size() -> Vector2:
	"""
	Returns the size of the current video frame.
	"""
	if current_image != null:
		return Vector2(current_image.get_width(), current_image.get_height())
	return Vector2.ZERO
