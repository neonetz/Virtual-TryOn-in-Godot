# Mask Scroll Selector
# Standalone UI component - Attach this script to a CanvasLayer or Control node
# Script will automatically create ScrollContainer with all masks at position (1800, 20)
# 
# USAGE:
# 1. Add a CanvasLayer node to your scene
# 2. Attach this script to that CanvasLayer
# 3. Run the scene
# 4. Mask selector will appear at top-right corner (1800, 20)

extends CanvasLayer

# Selected mask
var selected_mask: String = "mask-1.png"  # Default mask
var mask_buttons: Array = []

# UDP sender for backend
var udp_sender: PacketPeerUDP

# Backend connection
var backend_host: String = "127.0.0.1"
var backend_port: int = 5556  # Different port for mask commands

# UI Components (auto-created)
var scroll_container: ScrollContainer
var button_container: VBoxContainer

# Button sounds
var hover_sound: AudioStreamPlayer
var click_sound: AudioStreamPlayer


func _ready() -> void:
	print("[MaskSelector] Initializing standalone UI...")
	
	# Set layer to render on top
	layer = 10
	
	# Setup UDP sender
	udp_sender = PacketPeerUDP.new()
	print("[MaskSelector] UDP sender created for backend commands")
	
	# Load sounds
	_load_sounds()
	
	# Create scroll container programmatically
	_create_scroll_container()
	
	# Setup scroll container properties
	_setup_scroll_container()
	
	# Create button container
	_create_button_container()
	
	# Load masks from folder
	_load_masks()
	
	print("[MaskSelector] Ready - Default mask: %s" % selected_mask)
	print("[MaskSelector] UI created at position (1800, 20)")
	print("[MaskSelector] Total buttons created: %d" % mask_buttons.size())
	print("[MaskSelector] ScrollContainer visible: %s" % scroll_container.visible)
	print("[MaskSelector] ScrollContainer size: %s" % scroll_container.size)
	print("[MaskSelector] ScrollContainer position: %s" % scroll_container.position)


func _load_sounds() -> void:
	"""Load button sounds"""
	hover_sound = AudioStreamPlayer.new()
	hover_sound.stream = load("res://godot/assets/sounds/HoverButton.mp3")
	hover_sound.volume_db = -5
	add_child(hover_sound)
	
	click_sound = AudioStreamPlayer.new()
	click_sound.stream = load("res://godot/assets/sounds/ClickButton.mp3")
	click_sound.volume_db = -5
	add_child(click_sound)


func _create_scroll_container() -> void:
	"""Create ScrollContainer programmatically"""
	scroll_container = ScrollContainer.new()
	scroll_container.name = "MaskScrollContainer"
	
	# Add to scene tree
	add_child(scroll_container)
	
	print("[MaskSelector] ScrollContainer created")


func _setup_scroll_container() -> void:
	"""Setup scroll container properties"""
	if scroll_container == null:
		push_error("[MaskSelector] ScrollContainer not created!")
		return
	
	# Set position (NO ANCHOR - direct position)
	scroll_container.position = Vector2(1800, 20)
	
	# Set size (smaller - width x height)
	scroll_container.custom_minimum_size = Vector2(120, 700)
	scroll_container.size = Vector2(120, 700)
	
	# Enable vertical scroll only
	scroll_container.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll_container.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	
	# Make visible
	scroll_container.visible = true
	
	print("[MaskSelector] ScrollContainer configured at (1800, 20)")


func _create_button_container() -> void:
	"""Create VBoxContainer for buttons (1 column layout)"""
	if scroll_container == null:
		push_error("[MaskSelector] ScrollContainer not available!")
		return
	
	button_container = VBoxContainer.new()
	button_container.name = "ButtonContainer"
	button_container.custom_minimum_size = Vector2(110, 0)
	button_container.add_theme_constant_override("separation", 10)  # 10px gap between each panel
	
	# Add to scroll container
	scroll_container.add_child(button_container)
	print("[MaskSelector] VBoxContainer created (1 column layout)")


func _load_masks() -> void:
	"""Load all masks from godot/assets/masks/ folder"""
	var mask_dir_path = "res://godot/assets/masks"
	var dir = DirAccess.open(mask_dir_path)
	
	if dir == null:
		push_error("[MaskSelector] Cannot open mask directory: %s" % mask_dir_path)
		return
	
	# Collect all PNG files
	var mask_files: Array = []
	dir.list_dir_begin()
	var file_name = dir.get_next()
	
	while file_name != "":
		if not dir.current_is_dir() and file_name.ends_with(".png"):
			mask_files.append(file_name)
		file_name = dir.get_next()
	
	dir.list_dir_end()
	
	# Sort files (mask-1.png, mask-2.png, etc.)
	mask_files.sort()
	
	print("[MaskSelector] Found %d masks: %s" % [mask_files.size(), mask_files])
	
	# Create buttons for each mask
	for mask_file in mask_files:
		_create_mask_button(mask_file, mask_dir_path)
	
	# Select default mask (mask-1.png)
	if mask_buttons.size() > 0:
		_select_mask_button(0)


func _create_mask_button(mask_filename: String, mask_dir: String) -> void:
	"""Create a panel with mask button inside (1 panel per mask)"""
	var mask_path = mask_dir + "/" + mask_filename
	
	# Load mask texture
	var texture = load(mask_path) as Texture2D
	if texture == null:
		push_warning("[MaskSelector] Cannot load texture: %s" % mask_path)
		return
	
	# Create Panel (this is the clickable box) - FULL WIDTH
	var panel = PanelContainer.new()
	panel.custom_minimum_size = Vector2(110, 110)
	panel.name = "Panel_" + mask_filename
	
	# Panel style (white border box)
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.15, 0.15, 0.15, 1.0)  # Dark background
	style_box.border_color = Color(1, 1, 1, 1)  # White border
	style_box.border_width_left = 2
	style_box.border_width_right = 2
	style_box.border_width_top = 2
	style_box.border_width_bottom = 2
	panel.add_theme_stylebox_override("panel", style_box)
	
	# Create TextureRect to show the image (NOT button yet)
	var texture_rect = TextureRect.new()
	texture_rect.texture = texture
	texture_rect.custom_minimum_size = Vector2(100, 100)
	texture_rect.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	texture_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	texture_rect.name = "TextureRect_" + mask_filename
	
	# Add texture to panel
	panel.add_child(texture_rect)
	
	# Make the PANEL clickable using gui_input
	panel.mouse_filter = Control.MOUSE_FILTER_STOP  # Enable mouse input
	
	# Store references
	panel.set_meta("mask_filename", mask_filename)
	panel.set_meta("button_index", mask_buttons.size())
	panel.set_meta("texture_rect", texture_rect)
	
	# Connect signals to PANEL (not button)
	panel.gui_input.connect(_on_panel_clicked.bind(panel))
	panel.mouse_entered.connect(_on_panel_hover.bind(panel))
	
	# Add panel to container (1 panel = 1 row = 1 mask)
	button_container.add_child(panel)
	mask_buttons.append(panel)  # Store panel instead of button
	
	print("[MaskSelector] Created clickable panel for: %s" % mask_filename)


func _on_panel_hover(panel: PanelContainer) -> void:
	"""Play hover sound when mouse enters panel"""
	if hover_sound and hover_sound.stream:
		hover_sound.play()


func _on_panel_clicked(event: InputEvent, panel: PanelContainer) -> void:
	"""Handle panel click"""
	if event is InputEventMouseButton:
		if event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
			# Play click sound
			if click_sound and click_sound.stream:
				click_sound.play()
			
			# Get panel index and select it
			var panel_index = panel.get_meta("button_index")
			_select_mask_button(panel_index)


func _on_mask_button_hover(button: TextureButton) -> void:
	"""Play hover sound"""
	if hover_sound and hover_sound.stream:
		hover_sound.play()


func _on_mask_button_clicked(button: TextureButton) -> void:
	"""Handle mask button click"""
	if click_sound and click_sound.stream:
		click_sound.play()
	
	var button_index = button.get_meta("button_index")
	_select_mask_button(button_index)


func _select_mask_button(index: int) -> void:
	"""Select a mask button and notify backend"""
	if index < 0 or index >= mask_buttons.size():
		return
	
	# Clear all selections (reset panel border to white)
	for panel in mask_buttons:
		var style_box = StyleBoxFlat.new()
		style_box.bg_color = Color(0.15, 0.15, 0.15, 1.0)  # Dark background
		style_box.border_color = Color(1, 1, 1, 1)  # White border (normal)
		style_box.border_width_left = 2
		style_box.border_width_right = 2
		style_box.border_width_top = 2
		style_box.border_width_bottom = 2
		panel.add_theme_stylebox_override("panel", style_box)
	
	# Highlight selected panel (green border)
	var selected_panel = mask_buttons[index]
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.15, 0.15, 0.15, 1.0)  # Dark background
	style_box.border_color = Color(0.2, 1.0, 0.2, 1.0)  # GREEN border (selected)
	style_box.border_width_left = 4  # Thicker border
	style_box.border_width_right = 4
	style_box.border_width_top = 4
	style_box.border_width_bottom = 4
	selected_panel.add_theme_stylebox_override("panel", style_box)
	
	# Update selected mask
	selected_mask = selected_panel.get_meta("mask_filename")
	
	print("[MaskSelector] Selected mask: %s" % selected_mask)
	
	# Send to backend
	_send_mask_change_to_backend(selected_mask)


func _send_mask_change_to_backend(mask_filename: String) -> void:
	"""Send mask change command to backend via UDP"""
	var command = {
		"command": "change_mask",
		"mask": mask_filename
	}
	
	var json_str = JSON.stringify(command)
	var json_bytes = json_str.to_utf8_buffer()
	
	# Send UDP packet to backend
	var err = udp_sender.set_dest_address(backend_host, backend_port)
	if err != OK:
		push_warning("[MaskSelector] Failed to set UDP destination: %s" % error_string(err))
		return
	
	err = udp_sender.put_packet(json_bytes)
	if err != OK:
		push_warning("[MaskSelector] Failed to send UDP packet: %s" % error_string(err))
		return
	
	print("[MaskSelector] Sent mask change to backend: %s" % mask_filename)


func get_selected_mask() -> String:
	"""Get currently selected mask filename"""
	return selected_mask


func _exit_tree() -> void:
	"""Cleanup"""
	if udp_sender:
		udp_sender.close()
	
	print("[MaskSelector] Cleanup complete")
