extends Node2D

@export var spawn_area_width: float = 1280.0   # lebar area hujan
@export var min_speed: float = 600.0           # kecepatan jatuh minimum
@export var max_speed: float = 950.0           # kecepatan jatuh maksimum
@export var spawn_rate: float = 0.02           # makin kecil makin deras (interval spawn)
@export var drop_length: float = 45.0          # panjang garis hujan
@export var max_drops: int = 180               # batas jumlah partikel aktif
@export var spawn_y_offset: float = -50.0      # mulai dari sedikit di atas layar

var _timer := 0.0

func _process(delta: float) -> void:
	# spawn hujan
	_timer -= delta
	if _timer <= 0.0:
		_timer = spawn_rate
		if get_child_count() < max_drops:
			_spawn_drop()

	# gerakkan semua hujan turun
	for child in get_children():
		if child is Line2D:
			child.position.y += child.get_meta("speed") * delta
			# kalau sudah lewat bawah layar -> hapus
			if child.position.y > get_viewport_rect().size.y + 100.0:
				child.queue_free()


func _spawn_drop() -> void:
	var viewport_size = get_viewport_rect().size
	var drop = Line2D.new()
	
	# style garis
	drop.width = 1.5
	drop.default_color = Color(0.7, 0.7, 0.9, 0.8)  # kebiruan dikit biar kelihatan
	drop.points = [
		Vector2(0, 0),
		Vector2(0, drop_length)
	]

	# posisi awal random di atas layar
	var rand_x = randf_range(-50.0, spawn_area_width + 50.0)
	drop.position = Vector2(rand_x, spawn_y_offset)

	# kecepatan random
	var speed = randf_range(min_speed, max_speed)
	drop.set_meta("speed", speed)

	add_child(drop)
