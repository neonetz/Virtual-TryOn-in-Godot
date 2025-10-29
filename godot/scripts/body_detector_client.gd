extends Node
# FaceDetectorClient.gd
# Client for communicating with Python face detection server
# Place this in godot/scripts/

class_name FaceDetectorClient

# Server configuration
var server_url: String = "http://127.0.0.1:5000"
var http_request: HTTPRequest

# Signals
signal detection_completed(faces: Array, image: Image)
signal detection_failed(error: String)
signal health_check_completed(status: bool)

func _ready():
	# Create HTTP request node
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

## Check if server is running
func check_health() -> void:
	var url = server_url + "/health"
	var error = http_request.request(url)
	
	if error != OK:
		push_error("Failed to send health check request")
		health_check_completed.emit(false)

## Detect faces in an image
## image: Image object to process
## overlay_mask: Whether to overlay mask on detected faces
## draw_boxes: Whether to draw bounding boxes
func detect_faces(image: Image, overlay_mask: bool = true, draw_boxes: bool = false) -> void:
	# Convert image to base64
	var image_base64 = _image_to_base64(image)
	
	if image_base64 == "":
		push_error("Failed to encode image")
		detection_failed.emit("Failed to encode image")
		return
	
	# Prepare request body
	var body = {
		"image_base64": image_base64,
		"overlay_mask": overlay_mask,
		"draw_boxes": draw_boxes
	}
	
	var json_body = JSON.stringify(body)
	
	# Send request
	var url = server_url + "/detect"
	var headers = ["Content-Type: application/json"]
	var error = http_request.request(url, headers, HTTPClient.METHOD_POST, json_body)
	
	if error != OK:
		push_error("Failed to send detection request")
		detection_failed.emit("Failed to send request")

## Update the mask template on the server
func update_mask(mask_path: String) -> void:
	var body = {
		"mask_path": mask_path
	}
	
	var json_body = JSON.stringify(body)
	
	var url = server_url + "/update_mask"
	var headers = ["Content-Type: application/json"]
	var error = http_request.request(url, headers, HTTPClient.METHOD_POST, json_body)
	
	if error != OK:
		push_error("Failed to send mask update request")

## Convert Godot Image to base64 string
func _image_to_base64(image: Image) -> String:
	# Save image to buffer as JPG
	var buffer = image.save_jpg_to_buffer(0.9)
	
	if buffer.size() == 0:
		return ""
	
	# Encode to base64
	return Marshalls.raw_to_base64(buffer)

## Convert base64 string to Godot Image
func _base64_to_image(base64_string: String) -> Image:
	# Decode base64
	var buffer = Marshalls.base64_to_raw(base64_string)
	
	if buffer.size() == 0:
		return null
	
	# Load image from buffer
	var image = Image.new()
	var error = image.load_jpg_from_buffer(buffer)
	
	if error != OK:
		# Try PNG if JPG fails
		error = image.load_png_from_buffer(buffer)
		if error != OK:
			return null
	
	return image

## Handle HTTP response
func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS:
		push_error("HTTP request failed: " + str(result))
		detection_failed.emit("HTTP request failed")
		return
	
	if response_code != 200:
		push_error("Server error: " + str(response_code))
		detection_failed.emit("Server error: " + str(response_code))
		return
	
	# Parse JSON response
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	
	if parse_result != OK:
		push_error("Failed to parse JSON response")
		detection_failed.emit("Failed to parse response")
		return
	
	var response = json.data
	
	# Check for health check response
	if response.has("status"):
		if response.status == "ok":
			health_check_completed.emit(true)
		else:
			health_check_completed.emit(false)
		return
	
	# Handle detection response
	if response.has("faces") and response.has("image_base64"):
		var faces = response.faces
		var result_image = _base64_to_image(response.image_base64)
		
		if result_image != null:
			detection_completed.emit(faces, result_image)
		else:
			push_error("Failed to decode result image")
			detection_failed.emit("Failed to decode result image")
	else:
		push_error("Invalid response format")
		detection_failed.emit("Invalid response format")
