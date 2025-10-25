# Clothing Overlay Manager
# Manages clothing selection and overlay positioning based on body detection
extends Node

# Reference to the clothing display sprite
@onready var clothing_sprite: Sprite2D

# Available clothing items (add your PNG files here)
var clothing_items: Array[String] = [
	"res://godot/assets/clothes/shirt_01.png",
	"res://godot/assets/clothes/shirt_02.png",
	"res://godot/assets/clothes/jacket_01.png"
]

# Current state
var current_clothing_index: int = 0
var is_overlay_enabled: bool = true

# Positioning parameters
var scale_factor: float = 1.0
var vertical_offset: float = 0.0
var horizontal_offset: float = 0.0

# Manual adjustment mode
var manual_mode: bool = false

# Signals
signal clothing_changed(index: int)


func _ready() -> void:
	# Find the Sprite2D node for clothing display
	clothing_sprite = get_node_or_null("../ClothingOverlay")
	
	if clothing_sprite == null:
		push_warning("[Clothing Manager] ClothingOverlay node not found - creating new one")
		create_clothing_sprite()
	
	# Connect to video manager for bbox updates
	var video_manager = get_node_or_null("../VideoFeedManager")
	if video_manager != null:
		video_manager.video_updated.connect(_on_video_updated)
		print("[Clothing Manager] Connected to video feed manager")
	
	# Load first clothing item
	load_clothing(current_clothing_index)


func create_clothing_sprite() -> void:
	"""
	Creates a Sprite2D node for displaying clothing overlay.
	"""
	clothing_sprite = Sprite2D.new()
	clothing_sprite.name = "ClothingOverlay"
	clothing_sprite.z_index = 10  # Display above video
	get_parent().add_child(clothing_sprite)


func _on_video_updated(bbox: Array) -> void:
	"""
	Called when video frame is updated with new bounding box.
	Automatically positions clothing based on detected body region.
	"""
	if not is_overlay_enabled or manual_mode:
		return
	
	var bbox_x = bbox[0]
	var bbox_y = bbox[1]
	var bbox_w = bbox[2]
	var bbox_h = bbox[3]
	
	# Only update if valid detection exists
	if bbox_w > 0 and bbox_h > 0:
		update_clothing_position(bbox_x, bbox_y, bbox_w, bbox_h)


func update_clothing_position(x: int, y: int, w: int, h: int) -> void:
	"""
	Updates clothing sprite position and scale based on bounding box.
	"""
	if clothing_sprite == null or clothing_sprite.texture == null:
		return
	
	# Calculate center position of bounding box
	var center_x = x + w / 2.0
	var center_y = y + h / 2.0
	
	# Apply offsets
	center_x += horizontal_offset
	center_y += vertical_offset
	
	# Set sprite position
	clothing_sprite.position = Vector2(center_x, center_y)
	
	# Calculate scale based on bounding box width
	var texture_width = clothing_sprite.texture.get_width()
	var desired_width = w * scale_factor
	var scale = desired_width / texture_width
	
	clothing_sprite.scale = Vector2(scale, scale)


func load_clothing(index: int) -> void:
	"""
	Loads a clothing item by index.
	"""
	if index < 0 or index >= clothing_items.size():
		push_warning("[Clothing Manager] Invalid clothing index: " + str(index))
		return
	
	var clothing_path = clothing_items[index]
	
	# Check if file exists
	if not ResourceLoader.exists(clothing_path):
		push_warning("[Clothing Manager] Clothing file not found: " + clothing_path)
		return
	
	# Load texture
	var texture = load(clothing_path) as Texture2D
	
	if texture == null:
		push_error("[Clothing Manager] Failed to load clothing: " + clothing_path)
		return
	
	# Apply to sprite
	clothing_sprite.texture = texture
	current_clothing_index = index
	
	print("[Clothing Manager] Loaded clothing: " + clothing_path)
	emit_signal("clothing_changed", index)


func next_clothing() -> void:
	"""
	Switches to the next clothing item.
	"""
	current_clothing_index = (current_clothing_index + 1) % clothing_items.size()
	load_clothing(current_clothing_index)


func previous_clothing() -> void:
	"""
	Switches to the previous clothing item.
	"""
	current_clothing_index = (current_clothing_index - 1 + clothing_items.size()) % clothing_items.size()
	load_clothing(current_clothing_index)


func toggle_overlay() -> void:
	"""
	Toggles clothing overlay visibility.
	"""
	is_overlay_enabled = not is_overlay_enabled
	
	if clothing_sprite != null:
		clothing_sprite.visible = is_overlay_enabled


func set_scale_factor(factor: float) -> void:
	"""
	Adjusts the scale factor for clothing overlay.
	"""
	scale_factor = clamp(factor, 0.5, 2.0)


func set_vertical_offset(offset: float) -> void:
	"""
	Adjusts vertical position offset.
	"""
	vertical_offset = offset


func set_horizontal_offset(offset: float) -> void:
	"""
	Adjusts horizontal position offset.
	"""
	horizontal_offset = offset


func toggle_manual_mode() -> void:
	"""
	Toggles between automatic and manual positioning mode.
	"""
	manual_mode = not manual_mode
	print("[Clothing Manager] Manual mode: " + str(manual_mode))


func _input(event: InputEvent) -> void:
	"""
	Handles keyboard input for clothing control.
	"""
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_LEFT:
				previous_clothing()
			KEY_RIGHT:
				next_clothing()
			KEY_SPACE:
				toggle_overlay()
			KEY_M:
				toggle_manual_mode()
			KEY_UP:
				set_scale_factor(scale_factor + 0.1)
			KEY_DOWN:
				set_scale_factor(scale_factor - 0.1)
			KEY_W:
				set_vertical_offset(vertical_offset - 10)
			KEY_S:
				set_vertical_offset(vertical_offset + 10)
			KEY_A:
				set_horizontal_offset(horizontal_offset - 10)
			KEY_D:
				set_horizontal_offset(horizontal_offset + 10)
