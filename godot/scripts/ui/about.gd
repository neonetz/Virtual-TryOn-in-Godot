extends Node


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	$Fade_transition/AnimationPlayer.play("fade_out")
