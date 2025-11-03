# UDP Video Receiver
# Receives processed video frames from Python backend
# Backend does: Webcam capture → Face detection → Mask overlay → UDP send
# Godot does: UDP receive → Display

extends Node

# UDP connection
var udp_socket: PacketPeerUDP
var listening_port: int = 5555

# Frame display
var texture_rect: TextureRect
var current_texture: ImageTexture

# Statistics
var frames_received: int = 0
var last_frame_id: int = -1
var fps_display: float = 0.0
var faces_detected: int = 0
var processing_time: float = 0.0

# UI labels
var fps_label: Label
var status_label: Label
var faces_label: Label

# State
var _is_connected: bool = false
var was_streaming: bool = false

# Timeout detection
var last_frame_time: float = 0.0
var timeout_seconds: float = 3.0  # 3 seconds without frames = disconnected


func _ready() -> void:
	print("[UDP Video Receiver] Initializing...")
	
	# Create UDP socket
	udp_socket = PacketPeerUDP.new()
	
	# Bind to port
	var err = udp_socket.bind(listening_port)
	if err != OK:
		push_error("[UDP Video Receiver] Failed to bind to port %d: %s" % [listening_port, error_string(err)])
		return
	
	print("[UDP Video Receiver] Listening on port %d" % listening_port)
	
	# Get texture display
	texture_rect = get_node_or_null("../VideoDisplay")
	if texture_rect == null:
		push_error("[UDP Video Receiver] VideoDisplay TextureRect not found!")
		return
	
	# Create initial texture
	current_texture = ImageTexture.new()
	
	# Get UI labels
	fps_label = get_node_or_null("../UI/FPSLabel")
	status_label = get_node_or_null("../UI/StatusLabel")
	faces_label = get_node_or_null("../UI/FacesLabel")
	
	_is_connected = true
	last_frame_time = Time.get_ticks_msec() / 1000.0
	print("[UDP Video Receiver] Ready to receive frames")


func _process(_delta: float) -> void:
	if not _is_connected:
		return
	
	# Check for incoming packets
	while udp_socket.get_available_packet_count() > 0:
		var packet = udp_socket.get_packet()
		_process_packet(packet)
	
	# Check for timeout (server disconnected)
	var current_time = Time.get_ticks_msec() / 1000.0
	if was_streaming and (current_time - last_frame_time) > timeout_seconds:
		_handle_disconnect()
	
	# Update UI
	_update_ui()


func _process_packet(packet: PackedByteArray) -> void:
	"""Process received UDP packet"""
	# Parse JSON
	var json_str = packet.get_string_from_utf8()
	var json_parser = JSON.new()
	var parse_result = json_parser.parse(json_str)
	
	if parse_result != OK:
		push_warning("[UDP Video Receiver] JSON parse error: %s" % json_parser.get_error_message())
		return
	
	var data = json_parser.data
	
	# Extract frame data
	var frame_id = data.get("frame_id", -1)
	var base64_image = data.get("image", "")
	fps_display = data.get("fps", 0.0)
	faces_detected = data.get("faces", 0)
	processing_time = data.get("processing_ms", 0.0)
	
	# Check frame ID
	if frame_id <= last_frame_id:
		print("[UDP Video Receiver] Duplicate/old frame: %d (last: %d)" % [frame_id, last_frame_id])
		return
	
	last_frame_id = frame_id
	frames_received += 1
	
	# Update last frame time
	last_frame_time = Time.get_ticks_msec() / 1000.0
	was_streaming = true
	
	# Decode image
	if base64_image != "":
		_decode_and_display_image(base64_image)
	
	# Debug log every 30 frames
	if frames_received % 30 == 0:
		print("[UDP Video Receiver] Stats: frames=%d, fps=%.1f, faces=%d, process_ms=%.1f" % 
			  [frames_received, fps_display, faces_detected, processing_time])


func _decode_and_display_image(base64_str: String) -> void:
	"""Decode base64 JPEG and display"""
	# Decode base64
	var jpeg_buffer = Marshalls.base64_to_raw(base64_str)
	
	if jpeg_buffer.size() == 0:
		push_warning("[UDP Video Receiver] Failed to decode base64")
		return
	
	# Create image from JPEG buffer
	var image = Image.new()
	var err = image.load_jpg_from_buffer(jpeg_buffer)
	
	if err != OK:
		push_warning("[UDP Video Receiver] Failed to load JPEG: %s" % error_string(err))
		return
	
	# Update texture
	if current_texture == null:
		current_texture = ImageTexture.new()
	
	current_texture = ImageTexture.create_from_image(image)
	
	# Get viewport size for aspect ratio adjustment
	var _viewport_size = get_viewport().get_visible_rect().size
	
	# Display
	if texture_rect != null:
		texture_rect.texture = current_texture


func _update_ui() -> void:
	"""Update UI labels"""
	if fps_label != null:
		fps_label.text = "FPS: %.1f" % fps_display
	
	if status_label != null:
		if not was_streaming:
			status_label.text = "Waiting for frames..."
			status_label.modulate = Color.YELLOW
		else:
			var current_time = Time.get_ticks_msec() / 1000.0
			if (current_time - last_frame_time) > timeout_seconds:
				status_label.text = "Disconnected"
				status_label.modulate = Color.RED
			else:
				status_label.text = "Connected | Streaming"
				status_label.modulate = Color.GREEN
	
	if faces_label != null:
		faces_label.text = "Faces: %d | Process: %.1fms" % [faces_detected, processing_time]


func _handle_disconnect() -> void:
	"""Handle server disconnect"""
	print("[UDP Video Receiver] Server disconnected (timeout)")
	
	# Clear video display
	if texture_rect != null:
		texture_rect.texture = null
	
	# Reset stats
	fps_display = 0.0
	faces_detected = 0
	processing_time = 0.0
	was_streaming = false
	
	# Update last frame time to prevent spam
	last_frame_time = Time.get_ticks_msec() / 1000.0


func _exit_tree() -> void:
	"""Cleanup"""
	if udp_socket != null:
		udp_socket.close()
	
	print("[UDP Video Receiver] Closed (received %d frames)" % frames_received)
