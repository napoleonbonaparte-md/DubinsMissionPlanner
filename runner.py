from dubins import DubinsIterator

# Example arguments; replace with actual required parameters
start = (0, 0, 0)  # (x, y, heading)
end = (10, 10, 1.57)  # (x, y, heading)
turning_radius = 1.0
step_size = 1.0

dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)

while dubins_iterator.has_next():
	point = dubins_iterator.get_next_point()
	if point.valid:
		print(f"x={point.x:.2f}, y={point.y:.2f}, theta={point.theta:.2f}, t={point.t:.2f}")