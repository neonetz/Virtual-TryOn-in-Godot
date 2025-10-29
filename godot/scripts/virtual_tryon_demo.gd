extends Control
# VirtualTryOnDemo.gd
# Demo script showing how to use the face detector with webcam
# Place this in godot/scripts/

@onready var camera_feed: TextureRect = $CameraFeed
@onready var status_label: Label = $StatusLabel
@onready var face_count_label: Label = $FaceCountLabel

var detector_client: FaceDetectorClient
var camera: CameraFeed
var camera_texture: CameraTexture

func _ready():
	# Initialize face detector client
	detector_client = FaceDetectorClient.new()
	add_child(detector_client)
	
	# Connect signals
	detector_client.detection_completed.connect(_on_detection_completed)
	detector_client.detection_failed.connect(_on_detection_failed)
	detector_client.health_check_completed.connect(_on_health_check_completed)
	
	# Check server health
	status_label.text = "Checking server..."
	detector_client.check_health()
	
	# Try to initialize camera
	_initialize_camera()

func _initialize_camera():
	# Get camera feed (feed 0 is usually the default webcam)
	var cameras = CameraServer.get_feed_count()
	
	if cameras == 0:
		status_label.text = "No camera found"
		return
	
	camera = CameraServer.get_feed(0)
	camera_texture = CameraTexture.new()
	camera_texture.camera_feed_id = camera.get_id()
	
	# Set to camera feed texture rect
	if camera_feed:
		camera_feed.texture = camera_texture
	
	print("Camera initialized: ", camera.get_name())

func _process(delta):
	# Process camera frame every N frames (adjust for performance)
	if Engine.get_frames_drawn() % 10 == 0:  # Process every 10 frames (~6 FPS at 60 FPS)
		_process_frame()

func _process_frame():
	if camera_texture == null:
		return
	
	# Get current camera image
	var image = camera_texture.get_image()
	
	if image == null:
		return
	
	# Send to detector
	detector_client.detect_faces(image, true, false)

func _on_detection_completed(faces: Array, result_image: Image):
	# Update display with result image
	if camera_feed:
		var texture = ImageTexture.create_from_image(result_image)
		camera_feed.texture = texture
	
	# Update face count
	if face_count_label:
		face_count_label.text = "Faces detected: " + str(faces.size())
	
	# Print face information (optional)
	for i in range(faces.size()):
		var face = faces[i]
		print("Face ", i, ": x=", face.x, " y=", face.y, " w=", face.w, " h=", face.h, " score=", face.score)

func _on_detection_failed(error: String):
	if status_label:
		status_label.text = "Detection failed: " + error

func _on_health_check_completed(status: bool):
	if status:
		status_label.text = "Server: Connected"
	else:
		status_label.text = "Server: Not available"
		push_warning("Python server not running. Start with: python godot_bridge.py")

## Change mask (call this from UI button)
func change_mask(mask_path: String):
	detector_client.update_mask(mask_path)
	status_label.text = "Mask changed"
