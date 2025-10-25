# Try-On Scene UI Controller
# Manages the Try-On scene and coordinates all components
extends Node

# Component references
var udp_client: Node
var video_manager: Node
var clothing_manager: Node

# UI elements
var status_label: Label
var fps_label: Label
var controls_label: Label


func _ready() -> void:
	print("[TryOn Controller] Initializing Try-On scene...")
	
	# Get component references
	udp_client = get_node_or_null("UDPClient")
	video_manager = get_node_or_null("VideoFeedManager")
	clothing_manager = get_node_or_null("ClothingOverlayManager")
	
	# Get UI element references
	status_label = get_node_or_null("UI/StatusLabel")
	fps_label = get_node_or_null("UI/FPSLabel")
	controls_label = get_node_or_null("UI/ControlsLabel")
	
	# Setup UI
	setup_ui()
	
	# Verify components
	verify_components()
	
	print("[TryOn Controller] Initialization complete")


func verify_components() -> void:
	"""
	Verifies that all required components are present.
	"""
	var missing_components = []
	
	if udp_client == null:
		missing_components.append("UDPClient")
	if video_manager == null:
		missing_components.append("VideoFeedManager")
	if clothing_manager == null:
		missing_components.append("ClothingOverlayManager")
	
	if missing_components.size() > 0:
		push_error("[TryOn Controller] Missing components: " + str(missing_components))
	else:
		print("[TryOn Controller] All components verified")


func setup_ui() -> void:
	"""
	Sets up UI labels and displays controls information.
	"""
	if status_label != null:
		status_label.text = "Connecting to server..."
	
	if controls_label != null:
		controls_label.text = """
		CONTROLS:
		← → : Switch clothing
		SPACE: Toggle overlay
		M: Manual mode
		↑ ↓: Scale clothing
		W A S D: Adjust position
		ESC: Back to menu
		"""


func _process(_delta: float) -> void:
	"""
	Updates UI elements every frame.
	"""
	update_status()
	update_fps_display()


func update_status() -> void:
	"""
	Updates connection status display.
	"""
	if status_label == null or udp_client == null:
		return
	
	if udp_client.is_connected:
		status_label.text = "Connected | Streaming"
		status_label.modulate = Color.GREEN
	else:
		status_label.text = "Disconnected"
		status_label.modulate = Color.RED


func update_fps_display() -> void:
	"""
	Updates FPS counter display.
	"""
	if fps_label == null or video_manager == null:
		return
	
	var fps = video_manager.get_fps()
	fps_label.text = "FPS: %.1f" % fps


func _input(event: InputEvent) -> void:
	"""
	Handles global input events.
	"""
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ESCAPE:
			go_back_to_menu()


func go_back_to_menu() -> void:
	"""
	Returns to main menu.
	"""
	print("[TryOn Controller] Returning to main menu...")
	get_tree().change_scene_to_file("res://godot/scenes/menus/MainMenu.tscn")


func _exit_tree() -> void:
	"""
	Cleanup when exiting the scene.
	"""
	print("[TryOn Controller] Cleaning up...")
