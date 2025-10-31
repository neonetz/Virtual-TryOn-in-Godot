extends Node
# FaceDetectorClient.gd
# UDP client for high-performance face detection communication
# Much faster than HTTP for real-time video streaming

class_name FaceDetectorClient

# Server configuration
var server_host: String = "127.0.0.1"
var server_port: int = 8888

# UDP socket
var udp_socket: PacketPeerUDP

# State
var packet_id: int = 1
var is_connected: bool = false
var last_send_time: float = 0.0
var send_interval: float = 0.033  # ~30 FPS (adjust for performance)

# Signals
signal detection_completed(faces: Array, image: Image)
signal detection_failed(error: String)
signal connection_status_changed(connected: bool)

func _ready():
	# Create UDP socket
	udp_socket = PacketPeerUDP.new()
	
	# Try to connect (UDP is connectionless, but we set destination)
	var error = udp_socket.connect_to_host(server_host, server_port)
	
	if error == OK:
		is_connected = true
		connection_status_changed.emit(true)
		print("UDP client initialized: ", server_host, ":", server_port)
	else:
		is_connected = false
		connection_status_changed.emit(false)
		push_error("Failed to initialize UDP client: " + str(error))

func _process(delta):
	# Check for incoming packets
	if udp_socket.get_available_packet_count() > 0:
		_receive_packet()

## Detect faces in an image
## image: Image object to process
func detect_faces(image: Image) -> void:
	if not is_connected:
		detection_failed.emit("UDP not connected")
		return
	
	# Throttle sending (don't send too fast)
	var current_time = Time.get_ticks_msec() / 1000.0
	if current_time - last_send_time < send_interval:
		return
	
	last_send_time = current_time
	
	# Convert image to JPEG bytes
	var image_data = image.save_jpg_to_buffer(0.85)
	
	if image_data.size() == 0:
		detection_failed.emit("Failed to encode image")
		return
	
	# Build packet: [packet_id][data_length][image_data]
	var packet = PackedByteArray()
	
	# Add packet ID (4 bytes, unsigned int)
	packet.append_array(_int_to_bytes(packet_id))
	
	# Add data length (4 bytes, unsigned int)
	packet.append_array(_int_to_bytes(image_data.size()))
	
	# Add image data
	packet.append_array(image_data)
	
	# Send packet
	var error = udp_socket.put_packet(packet)
	
	if error != OK:
		detection_failed.emit("Failed to send UDP packet: " + str(error))
		return
	
	# Increment packet ID
	packet_id += 1
	if packet_id > 999999:
		packet_id = 1

## Receive and parse response packet
func _receive_packet() -> void:
	var packet = udp_socket.get_packet()
	
	if packet.size() < 12:  # Minimum: 4+4+4 = 12 bytes
		detection_failed.emit("Invalid packet: too short")
		return
	
	# Parse header: [packet_id][num_faces][data_length]
	var received_packet_id = _bytes_to_int(packet.slice(0, 4))
	var num_faces = _bytes_to_int(packet.slice(4, 8))
	var data_length = _bytes_to_int(packet.slice(8, 12))
	
	# Check for error response
	if received_packet_id == 0 and num_faces == 0 and data_length == 0:
		detection_failed.emit("Server error")
		return
	
	var offset = 12
	
	# Parse face data: for each face: [x][y][w][h][score]
	var faces = []
	for i in range(num_faces):
		if offset + 20 > packet.size():
			break
		
		var face = {
			"x": _bytes_to_int(packet.slice(offset, offset + 4)),
			"y": _bytes_to_int(packet.slice(offset + 4, offset + 8)),
			"w": _bytes_to_int(packet.slice(offset + 8, offset + 12)),
			"h": _bytes_to_int(packet.slice(offset + 12, offset + 16)),
			"score": _bytes_to_int(packet.slice(offset + 16, offset + 20)) / 1000.0  # Convert back to float
		}
		faces.append(face)
		offset += 20
	
	# Parse image data
	if offset + data_length > packet.size():
		detection_failed.emit("Invalid packet: image data truncated")
		return
	
	var image_data = packet.slice(offset, offset + data_length)
	
	# Decode image
	var result_image = Image.new()
	var error = result_image.load_jpg_from_buffer(image_data)
	
	if error != OK:
		detection_failed.emit("Failed to decode image")
		return
	
	# Emit success signal
	detection_completed.emit(faces, result_image)

## Convert int to 4 bytes (little-endian, unsigned)
func _int_to_bytes(value: int) -> PackedByteArray:
	var bytes = PackedByteArray()
	bytes.resize(4)
	bytes.encode_u32(0, value)
	return bytes

## Convert 4 bytes to int (little-endian, unsigned)
func _bytes_to_int(bytes: PackedByteArray) -> int:
	if bytes.size() < 4:
		return 0
	return bytes.decode_u32(0)

## Set send rate (FPS)
func set_send_rate(fps: float) -> void:
	send_interval = 1.0 / fps if fps > 0 else 0.033

## Close UDP socket
func close():
	if udp_socket:
		udp_socket.close()
		is_connected = false
		connection_status_changed.emit(false)

func _exit_tree():
	close()
