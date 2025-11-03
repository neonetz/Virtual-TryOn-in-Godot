extends Node

# Button sounds
var hover_sound: AudioStreamPlayer
var click_sound: AudioStreamPlayer


func _ready() -> void:
	# Load audio resources
	hover_sound = AudioStreamPlayer.new()
	hover_sound.stream = load("res://godot/assets/sounds/HoverButton.mp3")
	hover_sound.volume_db = 0
	add_child(hover_sound)
	
	click_sound = AudioStreamPlayer.new()
	click_sound.stream = load("res://godot/assets/sounds/ClickButton.mp3")
	click_sound.volume_db = 0
	add_child(click_sound)
	
	# Setup button sounds
	setup_button_sounds()


func setup_button_sounds() -> void:
	"""Detect and setup sounds for all TextureButton inside Button_manager"""
	var button_manager = get_node_or_null("Button_manager")
	if button_manager == null:
		print("[TryOn] Button_manager not found!")
		return
	
	# Find all TextureButton children
	var buttons = find_texture_buttons(button_manager)
	
	print("[TryOn] Found %d TextureButton(s) in Button_manager" % buttons.size())
	
	for button in buttons:
		if button is TextureButton:
			connect_button_signals(button)


func find_texture_buttons(node: Node) -> Array:
	"""Recursively find all TextureButton nodes"""
	var buttons: Array = []
	
	for child in node.get_children():
		if child is TextureButton:
			buttons.append(child)
			print("[TryOn] Found TextureButton: %s" % child.name)
		
		# Recursive search in children
		if child.get_child_count() > 0:
			buttons.append_array(find_texture_buttons(child))
	
	return buttons


func connect_button_signals(button: TextureButton) -> void:
	"""Connect hover and click signals to a button"""
	# Connect mouse_entered for hover sound
	if not button.mouse_entered.is_connected(_on_button_hover):
		button.mouse_entered.connect(_on_button_hover.bind(button))
		print("[TryOn] Connected hover sound to: %s" % button.name)
	
	# Connect pressed for click sound
	if not button.pressed.is_connected(_on_button_click):
		button.pressed.connect(_on_button_click.bind(button))
		print("[TryOn] Connected click sound to: %s" % button.name)
	
	# Connect specific button actions
	if button.name == "Back":
		if not button.pressed.is_connected(_on_back_pressed):
			button.pressed.connect(_on_back_pressed)
			print("[TryOn] Connected Back button to MainMenu transition")


func _on_button_hover(button: TextureButton) -> void:
	"""Play hover sound when mouse enters button"""
	print("[TryOn] Hover on button: %s" % button.name)
	if hover_sound and hover_sound.stream:
		hover_sound.play()


func _on_button_click(button: TextureButton) -> void:
	"""Play click sound when button is pressed"""
	print("[TryOn] Click on button: %s" % button.name)
	if click_sound and click_sound.stream:
		click_sound.play()


func _on_back_pressed() -> void:
	"""Handle Back button press - return to MainMenu"""
	print("[TryOn] Back button pressed - returning to MainMenu")
	
	# Play fade in animation (if exists)
	var anim_player = get_node_or_null("Fade_transition/AnimationPlayer")
	if anim_player:
		anim_player.play("fade_in")
		# Wait for animation to finish
		await anim_player.animation_finished
	
	# Change scene to MainMenu
	get_tree().change_scene_to_file("res://godot/scenes/menus/MainMenu.tscn")
