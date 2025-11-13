from dubins import DubinsIterator
from heading_calc import Waypoint, calculate_heading_xy
import math
import pyproj

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
    def add_waypoints_at_index(self, index, new_waypoints):
        for i, wp in enumerate(new_waypoints):
            self.waypoints.insert(index + i, wp)
    def write_waypoints(self, filename="output.waypoints"):
        with open(filename, "w") as file:
            file.write("QGC WPL 110\n")
            for idx, wp in enumerate(self.waypoints):
                line = f"{idx}\t{wp.is_Active}\t{wp.relativity}\t{wp.type}\t"
                line += "\t".join(f"{param:.8f}" for param in wp.params) + "\t"
                line += f"{wp.lat:.7f}\t{wp.long:.7f}\t{wp.alt:.6f}\t{wp.unknown}\n"
                file.write(line)

# Convert lat/long to x/y (assuming a simple flat Earth approximation for small distances)
# You may want to replace this with a more accurate projection if needed
def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    # Approximate conversion: 1 deg latitude ~ 111km, 1 deg longitude ~ 111km * cos(latitude)
    dx = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
    dy = (lat - ref_lat) * 111000
    return dx, dy

def xy_to_latlon(x, y, ref_lat, ref_lon):
    lat = y / 111000 + ref_lat
    lon = x / (111000 * math.cos(math.radians(ref_lat))) + ref_lon
    return lat, lon

def build_intermeddiate_dubins_path(start, end, turning_radius, step_size, ref_lat, ref_lon):
        dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)
        points = dubins_iterator.get_all_points()
        twenty_point_trial = math.floor(len(points) / 50 )
        idx = 0
        intermeddiate_points = []
        while idx < len(points):
            point = points[idx]
            if point.valid:
                lat, lon = xy_to_latlon(point.x, point.y, ref_lat=ref_lat, ref_lon=ref_lon)
                waypoint = Waypoint(0, 0, 3, 16, [0,10,0,0], lat, lon, 100, 1)
                intermeddiate_points.append(waypoint)
                print(f"Dubins Point: lat={lat:.6f}, long={lon:.6f}, theta={math.degrees(point.theta):.2f}, t={point.t:.2f}")
            idx +=  twenty_point_trial 
        return intermeddiate_points

    
def build_dubins_path(parser: waypointParser, turning_radius: float, step_size: float):
    ref_lat = parser.waypoints[0].lat
    ref_lon = parser.waypoints[0].long
    idx = 0
    while idx < len(parser.waypoints) - 2:
        if parser.waypoints[idx].type == 40:
            print(f"Detected Dubisns start lat {parser.waypoints[idx].lat}, long {parser.waypoints[idx].long}")
            x_pre, y_pre = latlon_to_xy(parser.waypoints[idx - 1].lat, parser.waypoints[idx - 1].long, ref_lat, ref_lon)
            x_start, y_start = latlon_to_xy(parser.waypoints[idx].lat, parser.waypoints[idx].long, ref_lat, ref_lon)
            x_end, y_end = latlon_to_xy(parser.waypoints[idx + 1].lat, parser.waypoints[idx + 1].long, ref_lat, ref_lon)
            x_post, y_post = latlon_to_xy(parser.waypoints[idx + 2].lat, parser.waypoints[idx + 2].long, ref_lat, ref_lon)
            heading_start = calculate_heading_xy(x_pre, y_pre, x_start, y_start)
            heading_end = calculate_heading_xy(x_end, y_end, x_post, y_post)
            print(f"calculated headings: start {math.degrees(heading_start):.2f}, end {math.degrees(heading_end):.2f}")
            start = (x_start, y_start, heading_start)
            end = (x_end, y_end, heading_end)
            waypoints = build_intermeddiate_dubins_path(start, end, turning_radius, step_size, ref_lat, ref_lon)
            parser.add_waypoints_at_index(idx + 1, waypoints)
            idx += len(waypoints) + 1
        else:
            idx += 1
    write_filename = "dubins_output.waypoints"
    parser.write_waypoints(write_filename)
    
def main():
    parser = waypointParser()
    parser.parse_waypoints()
    print(f"First waypoint lat: {parser.waypoints[0].lat}, long: {parser.waypoints[0].long}")
    # Example arguments; replace with actual required parameters
    turning_radius = 300.0
    step_size = 0.1
    build_dubins_path(parser, turning_radius, step_size)

if __name__ == "__main__":
    main()