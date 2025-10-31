extends Node

var button_type = null  # "start" | "settings" | "about" | "exit"
var rain_audio: AudioStreamPlayer = null
var click_audio: AudioStreamPlayer = null
var hover_audio: AudioStreamPlayer = null

func _ready() -> void:
	# pastikan layer petir tidak blokir klik
	if has_node("LightningLayer"):
		if $LightningLayer is Control:
			$LightningLayer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	if has_node("LightningLayer/LightningFlash"):
		$LightningLayer/LightningFlash.mouse_filter = Control.MOUSE_FILTER_IGNORE
	
	# Setup and play rain audio
	setup_rain_audio()
	
	# Setup button sound effects
	setup_button_sounds()
	
	# Connect button signals for hover and click sounds
	connect_button_signals()

func _on_start_pressed() -> void:
	play_click_sound()
	button_type = "start"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_settings_pressed() -> void:
	play_click_sound()
	button_type = "settings"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_about_pressed() -> void:
	play_click_sound()
	button_type = "about"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_exit_pressed() -> void:
	play_click_sound()
	button_type = "exit"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_fade_timer_timeout() -> void:
	# Stop rain audio before changing scene
	stop_rain_audio()
	
	match button_type:
		"start":
			get_tree().change_scene_to_file("res://godot/scenes/tryon/TryOn.tscn")
		"settings":
			get_tree().change_scene_to_file("res://godot/scenes/menus/Settings.tscn")
		"about":
			get_tree().change_scene_to_file("res://godot/scenes/menus/About.tscn")
		"exit":
			get_tree().quit()

func _on_spook_timer_timeout() -> void:
	if not has_node("SpookContainer"):
		return
	$SpookContainer.visible = true
	$SpookContainer.modulate = Color(1, 1, 1, 0)
	$SpookContainer/AnimationPlayer.play("spook")

func _on_spook_animation_finished(anim_name: StringName) -> void:
	if anim_name == "spook":
		if has_node("SpookContainer"):
			$SpookContainer.visible = false

func setup_rain_audio() -> void:
	"""Setup and play rain audio (Thunderstorm.mp3)"""
	# Check if AudioStreamPlayer already exists
	if has_node("RainAudio"):
		rain_audio = get_node("RainAudio")
	else:
		# Create new AudioStreamPlayer
		rain_audio = AudioStreamPlayer.new()
		rain_audio.name = "RainAudio"
		add_child(rain_audio)
	
	# Load the rain audio file
	var audio_stream = load("res://godot/assets/sounds/Thunderstorm.mp3")
	if audio_stream:
		rain_audio.stream = audio_stream
		rain_audio.autoplay = false
		rain_audio.volume_db = -5  # Adjust volume (0 = normal, negative = quieter)
		rain_audio.play()
		print("[Main Menu] Rain audio started")
	else:
		push_error("[Main Menu] Failed to load Thunderstorm.mp3")

func stop_rain_audio() -> void:
	"""Stop rain audio when leaving main menu"""
	if rain_audio and rain_audio.playing:
		rain_audio.stop()
		print("[Main Menu] Rain audio stopped")

func setup_button_sounds() -> void:
	"""Setup button click and hover sound effects"""
	# Create click sound player
	if has_node("ClickAudio"):
		click_audio = get_node("ClickAudio")
	else:
		click_audio = AudioStreamPlayer.new()
		click_audio.name = "ClickAudio"
		add_child(click_audio)
	
	# Load click sound
	var click_stream = load("res://godot/assets/sounds/ClickButton.mp3")
	if click_stream:
		click_audio.stream = click_stream
		click_audio.volume_db = 0
		print("[Main Menu] Click sound loaded")
	else:
		push_error("[Main Menu] Failed to load ClickButton.mp3")
	
	# Create hover sound player
	if has_node("HoverAudio"):
		hover_audio = get_node("HoverAudio")
	else:
		hover_audio = AudioStreamPlayer.new()
		hover_audio.name = "HoverAudio"
		add_child(hover_audio)
	
	# Load hover sound
	var hover_stream = load("res://godot/assets/sounds/HoverButton.mp3")
	if hover_stream:
		hover_audio.stream = hover_stream
		hover_audio.volume_db = 0  # Changed from -3 to 0 (louder)
		print("[Main Menu] Hover sound loaded")
	else:
		push_error("[Main Menu] Failed to load HoverButton.mp3")

func connect_button_signals() -> void:
	"""Connect mouse_entered signal to all buttons for hover sound"""
	# Cari semua button di scene tree secara otomatis
	var all_buttons = find_all_buttons(self)
	
	print("[Main Menu] Found " + str(all_buttons.size()) + " buttons")
	
	for button in all_buttons:
		if button is BaseButton:
			# Connect hover event
			if not button.mouse_entered.is_connected(_on_button_hover):
				button.mouse_entered.connect(_on_button_hover)
				print("[Main Menu] Connected hover sound to: " + button.name)

func find_all_buttons(node: Node) -> Array:
	"""Recursively find all Button nodes in the scene tree"""
	var buttons = []
	
	for child in node.get_children():
		if child is BaseButton:
			buttons.append(child)
		# Recursive search in children
		buttons.append_array(find_all_buttons(child))
	
	return buttons

func _on_button_hover() -> void:
	"""Play hover sound when mouse enters any button"""
	print("[Main Menu] Button hovered - playing hover sound")
	play_hover_sound()

func play_click_sound() -> void:
	"""Play button click sound"""
	if click_audio and click_audio.stream:
		click_audio.play()

func play_hover_sound() -> void:
	"""Play button hover sound"""
	if hover_audio and hover_audio.stream:
		print("[Main Menu] Playing hover sound...")
		hover_audio.play()
	else:
		print("[Main Menu] Hover audio not ready!")
