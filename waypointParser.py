from dubins import DubinsIterator
class Waypoint:
    def __init__(self, num, is_Active=0, relativity=3, type=16, params=[0,0,0,0], lat=0.0, long=0.0, alt=0.0, unknown=1):
        self.num = num
        self.is_Active = is_Active
        self.relativity = relativity
        self.type = type
        self.params = params
        self.lat = lat
        self.long = long
        self.alt = alt
        self.unknown = unknown
    
class waypointParser():
    def __init__(self):
        self.waypoints = [] 

    def parse_waypoints(self):
        with open("way.waypoints", "r") as file:
            lines = file.readlines()
            for line in lines[1:]:
                data = line.split('\t')
                num = int(data[0])
                is_Active = int(data[1])
                relativity = int(data[2])
                type = int(data[3])
                params = list(map(float, data[4:8]))
                lat = float(data[8])
                long = float(data[9])
                alt = float(data[10])
                unknown = int(data[11])
                waypoint = Waypoint(num, is_Active, relativity, type, params, lat, long, alt, unknown)
                self.waypoints.append(waypoint) 

def main():
    parser = waypointParser()
    parser.parse_waypoints()
    print(f"First waypoint lat: {parser.waypoints[0].lat}, long: {parser.waypoints[0].long}")
    # Example arguments; replace with actual required parameters
    start = (0, 0, 0)  # (x, y, heading)
    end = (10, 10, 1.57)  # (x, y, heading)
    turning_radius = 1.0
    step_size = 0.1

    dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)

    while dubins_iterator.has_next():
        point = dubins_iterator.get_next_point()
        if point.valid:
            print(f"x={point.x:.2f}, y={point.y:.2f}, theta={point.theta:.2f}, t={point.t:.2f}")


if __name__ == "__main__":
    main()