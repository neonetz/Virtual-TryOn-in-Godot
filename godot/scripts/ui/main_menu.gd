# godot/scripts/ui/main_menu.gd
extends Node

var button_type = null  # "start" | "settings" | "about" | "exit"

func _ready() -> void:
	pass

func _on_start_pressed() -> void:
	button_type = "start"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_settings_pressed() -> void:
	button_type = "settings"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_about_pressed() -> void:
	button_type = "about"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

func _on_exit_pressed() -> void:
	button_type = "exit"
	$Fade_transition.show()
	$Fade_transition/Fade_timer.start()
	$Fade_transition/AnimationPlayer.play("fade_in")

# -- dipanggil saat Timer timeout 
func _on_fade_timer_timeout() -> void:
	match button_type:
		"start":
			get_tree().change_scene_to_file("res://godot/scenes/tryon/TryOn.tscn")
		"settings":
			get_tree().change_scene_to_file("res://godot/scenes/menus/Settings.tscn")
		"about":
			get_tree().change_scene_to_file("res://godot/scenes/menus/About.tscn")
		"exit":
			get_tree().quit()
