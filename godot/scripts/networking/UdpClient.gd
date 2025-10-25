# UDP Client for receiving video stream from Python server
# Handles connection, registration, packet reception, and frame decoding
extends Node

# Network configuration
const SERVER_HOST = "127.0.0.1"
const SERVER_PORT = 9999

# UDP socket for communication
var udp_socket: PacketPeerUDP

# Connection state
var is_connected: bool = false
var is_registered: bool = false

# Packet reception buffer
var packet_buffer: PackedByteArray = PackedByteArray()
var expected_frame_size: int = 0
var bbox_data: PackedInt32Array = PackedInt32Array([0, 0, 0, 0])

# Signals for communication with other components
signal frame_received(image: Image, bbox: Array)
signal connection_status_changed(connected: bool)


func _ready() -> void:
	# Initialize UDP socket
	udp_socket = PacketPeerUDP.new()
	
	# Connect to server
	connect_to_server()


func connect_to_server() -> void:
	"""
	Establishes connection to Python UDP server.
	"""
	var err = udp_socket.bind(0)  # Bind to any available port
	
	if err != OK:
		push_error("[UDP Client] Failed to bind UDP socket: " + str(err))
		return
	
	udp_socket.set_dest_address(SERVER_HOST, SERVER_PORT)
	is_connected = true
	
	print("[UDP Client] Connected to server: " + SERVER_HOST + ":" + str(SERVER_PORT))
	
	# Register with server to start receiving frames
	register_client()
	
	emit_signal("connection_status_changed", true)


func register_client() -> void:
	"""
	Sends REGISTER command to server to begin receiving video stream.
	"""
	var message = "REGISTER".to_utf8_buffer()
	var err = udp_socket.put_packet(message)
	
	if err != OK:
		push_error("[UDP Client] Failed to send REGISTER: " + str(err))
	else:
		is_registered = true
		print("[UDP Client] Registered with server")


func unregister_client() -> void:
	"""
	Sends UNREGISTER command to server to stop receiving video stream.
	"""
	if not is_connected:
		return
	
	var message = "UNREGISTER".to_utf8_buffer()
	var err = udp_socket.put_packet(message)
	
	if err != OK:
		push_error("[UDP Client] Failed to send UNREGISTER: " + str(err))
	else:
		is_registered = false
		print("[UDP Client] Unregistered from server")


func _process(_delta: float) -> void:
	"""
	Processes incoming packets every frame.
	"""
	if not is_connected:
		return
	
	# Check for available packets
	while udp_socket.get_available_packet_count() > 0:
		var packet = udp_socket.get_packet()
		process_packet(packet)


func process_packet(packet: PackedByteArray) -> void:
	"""
	Processes a received packet containing frame data and bounding box.
	Packet structure:
	- 4 bytes: frame size (uint32)
	- N bytes: JPEG frame data
	- 16 bytes: bounding box (4x uint32: x, y, w, h)
	"""
	if packet.size() < 20:  # Minimum: 4 (size) + 0 (empty frame) + 16 (bbox)
		return
	
	# Extract frame size from first 4 bytes
	var frame_size = packet.decode_u32(0)
	
	# Check if packet is complete
	var total_size = 4 + frame_size + 16
	if packet.size() < total_size:
		push_warning("[UDP Client] Incomplete packet received")
		return
	
	# Extract JPEG frame data
	var jpeg_data = packet.slice(4, 4 + frame_size)
	
	# Extract bounding box data (last 16 bytes)
	var bbox_offset = 4 + frame_size
	var bbox_x = packet.decode_u32(bbox_offset)
	var bbox_y = packet.decode_u32(bbox_offset + 4)
	var bbox_w = packet.decode_u32(bbox_offset + 8)
	var bbox_h = packet.decode_u32(bbox_offset + 12)
	
	# Decode JPEG to Image
	var image = decode_jpeg(jpeg_data)
	
	if image != null:
		# Emit signal with decoded frame and bounding box
		var bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
		emit_signal("frame_received", image, bbox)


func decode_jpeg(jpeg_data: PackedByteArray) -> Image:
	"""
	Decodes JPEG data to Image object.
	"""
	if jpeg_data.size() == 0:
		return null
	
	var image = Image.new()
	var err = image.load_jpg_from_buffer(jpeg_data)
	
	if err != OK:
		push_error("[UDP Client] Failed to decode JPEG: " + str(err))
		return null
	
	return image


func disconnect_from_server() -> void:
	"""
	Disconnects from server and cleans up resources.
	"""
	if is_connected:
		unregister_client()
		udp_socket.close()
		is_connected = false
		
		print("[UDP Client] Disconnected from server")
		emit_signal("connection_status_changed", false)


func _exit_tree() -> void:
	"""
	Ensures proper cleanup when node is removed from scene tree.
	"""
	disconnect_from_server()
