"""
UDP Face Detection Client for Godot Integration
GDScript class to communicate with Python face detection backend
"""

extends Node
class_name FaceDetectorClient

## Signals
signal faces_detected(faces: Array, overlay_image: Image)
signal detection_failed(error: String)
signal connection_established()
signal connection_lost()

## Configuration
@export var server_host: String = "127.0.0.1"
@export var server_port: int = 5555
@export var timeout_ms: int = 1000

## Internal state
var udp: PacketPeerUDP
var is_connected: bool = false
var frame_id: int = 0
var pending_requests: Dictionary = {}  # frame_id -> timestamp

## Statistics
var total_requests: int = 0
var total_responses: int = 0
var avg_latency_ms: float = 0.0


func _ready() -> void:
	udp = PacketPeerUDP.new()
	print("[FaceDetectorClient] Initialized")


func connect_to_server(host: String = "", port: int = 0) -> bool:
	"""
	Connect to Python backend server
	
	Args:
		host: Server IP address (defaults to export var)
		port: Server port (defaults to export var)
	
	Returns:
		true if connection successful
	"""
	if host != "":
		server_host = host
	if port > 0:
		server_port = port
	
	var err = udp.set_dest_address(server_host, server_port)
	
	if err != OK:
		push_error("[FaceDetectorClient] Failed to connect: %s:%d" % [server_host, server_port])
		detection_failed.emit("Connection failed")
		return false
	
	is_connected = true
	connection_established.emit()
	print("[FaceDetectorClient] Connected to %s:%d" % [server_host, server_port])
	return true


func disconnect_from_server() -> void:
	"""Disconnect from server"""
	udp.close()
	is_connected = false
	connection_lost.emit()
	print("[FaceDetectorClient] Disconnected")


func detect_faces(image: Image, mask_type: String = "default") -> int:
	"""
	Send image to backend for face detection
	
	Args:
		image: Image to process
		mask_type: Type of mask to apply
	
	Returns:
		Frame ID for tracking this request
	"""
	if not is_connected:
		push_warning("[FaceDetectorClient] Not connected to server")
		detection_failed.emit("Not connected")
		return -1
	
	# Encode image to JPEG
	var jpeg_buffer = image.save_jpg_to_buffer(85)  # 85% quality
	var base64_image = Marshalls.raw_to_base64(jpeg_buffer)
	
	# Create request
	var request = {
		"frame_id": frame_id,
		"timestamp": Time.get_unix_time_from_system(),
		"image": base64_image,
		"mask_type": mask_type,
		"options": {
			"resolution": [image.get_width(), image.get_height()],
			"quality": 85
		}
	}
	
	# Serialize to JSON
	var json_str = JSON.stringify(request)
	var packet = json_str.to_utf8_buffer()
	
	# Send via UDP
	var err = udp.put_packet(packet)
	
	if err != OK:
		push_error("[FaceDetectorClient] Failed to send packet")
		detection_failed.emit("Send failed")
		return -1
	
	# Track request
	pending_requests[frame_id] = Time.get_ticks_msec()
	total_requests += 1
	
	var current_frame_id = frame_id
	frame_id += 1
	
	return current_frame_id


func _process(delta: float) -> void:
	"""Check for responses from server"""
	if not is_connected:
		return
	
	# Check for incoming packets
	while udp.get_available_packet_count() > 0:
		var packet = udp.get_packet()
		var json_str = packet.get_string_from_utf8()
		
		# Parse JSON
		var json = JSON.new()
		var error = json.parse(json_str)
		
		if error != OK:
			push_error("[FaceDetectorClient] JSON parse error: %s" % json.get_error_message())
			continue
		
		var response: Dictionary = json.data
		_handle_response(response)


func _handle_response(response: Dictionary) -> void:
	"""Process response from server"""
	var response_frame_id = response.get("frame_id", -1)
	var status = response.get("status", "error")
	
	# Calculate latency
	if pending_requests.has(response_frame_id):
		var request_time = pending_requests[response_frame_id]
		var latency = Time.get_ticks_msec() - request_time
		
		# Update average latency
		avg_latency_ms = (avg_latency_ms * total_responses + latency) / (total_responses + 1)
		
		pending_requests.erase(response_frame_id)
		total_responses += 1
	
	# Handle different statuses
	match status:
		"success":
			_handle_success(response)
		"no_face":
			print("[FaceDetectorClient] No face detected in frame %d" % response_frame_id)
			detection_failed.emit("No face detected")
		"error":
			var error_msg = response.get("error", "Unknown error")
			push_error("[FaceDetectorClient] Server error: %s" % error_msg)
			detection_failed.emit(error_msg)
		_:
			push_warning("[FaceDetectorClient] Unknown status: %s" % status)


func _handle_success(response: Dictionary) -> void:
	"""Handle successful detection response"""
	var faces = response.get("faces", [])
	var processing_time = response.get("processing_time_ms", 0.0)
	
	print("[FaceDetectorClient] Detected %d face(s), processing: %.1f ms, latency: %.1f ms" % 
		[faces.size(), processing_time, avg_latency_ms])
	
	# Decode overlay image
	var overlay_image: Image = null
	if response.has("overlay_image"):
		var base64_overlay = response.get("overlay_image", "")
		var jpeg_buffer = Marshalls.base64_to_raw(base64_overlay)
		
		overlay_image = Image.new()
		var err = overlay_image.load_jpg_from_buffer(jpeg_buffer)
		
		if err != OK:
			push_error("[FaceDetectorClient] Failed to decode overlay image")
			overlay_image = null
	
	# Emit signal
	faces_detected.emit(faces, overlay_image)


func get_stats() -> Dictionary:
	"""Get client statistics"""
	return {
		"connected": is_connected,
		"total_requests": total_requests,
		"total_responses": total_responses,
		"pending_requests": pending_requests.size(),
		"avg_latency_ms": avg_latency_ms,
		"success_rate": (float(total_responses) / total_requests * 100.0) if total_requests > 0 else 0.0
	}


func _exit_tree() -> void:
	"""Cleanup on exit"""
	disconnect_from_server()
