from typing import List
from dubins import DubinsIterator
from heading_calc import Waypoint, calculate_heading_xy
import math

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


def calculate_loiter_waypoint(x1, y1, x2, y2, ref_lat, ref_lon,
                              turning_radius, turn_right=False):
    """
    Compute a LOITER_TO_ALT waypoint that makes the plane fly an arc of
    given turning_radius between two EN points (x1,y1) and (x2,y2).

    x1, y1, x2, y2  : EN coordinates (meters) in some local frame
    ref_lat, ref_lon: reference lat/lon for xy_to_latlon
    turning_radius  : desired turn radius R (meters)
    turn_right      : True -> right-hand / clockwise, False -> left-hand / CCW

    Returns:
        Waypoint object for MAV_CMD_NAV_LOITER_TO_ALT (31)
    """
    if x1 == x2 and y1 == y2:
        return None
    # Vector from P1 to P2
    dx = x2 - x1
    dy = y2 - y1
    chord_len = math.hypot(dx, dy)  # L

    # Check feasibility: chord length must be <= 2R
    if chord_len > 2.0 * turning_radius:
        raise ValueError(
            f"Cannot form a circle of radius {turning_radius:.1f} m through "
            f"points separated by {chord_len:.1f} m (need L <= 2R)."
        )

    # Midpoint M between P1 and P2
    mid_x = (x1 + x2) * 0.5
    mid_y = (y1 + y2) * 0.5

    # Unit vector along chord (from P1 to P2)
    ux = dx / chord_len
    uy = dy / chord_len

    # Perpendicular unit vectors (normals)
    # Left normal (-uy, ux), Right normal (uy, -ux)
    if turn_right:
        nx, ny = uy, -ux
        loiter_radius_param = +turning_radius  # >0 = clockwise in Plane
    else:
        nx, ny = -uy, ux
        loiter_radius_param = -turning_radius  # <0 = counter-clockwise

    # Distance from midpoint to circle center:
    # h = sqrt(R^2 - (L/2)^2)
    half_L = 0.5 * chord_len
    h = math.sqrt(turning_radius**2 - half_L**2)

    # Circle center in EN frame
    center_x = mid_x + nx * h
    center_y = mid_y + ny * h

    # Convert center EN -> lat/lon
    lat, lon = xy_to_latlon(center_x, center_y, ref_lat, ref_lon)

    # Build LOITER_TO_ALT waypoint
    # MAV_CMD_NAV_LOITER_TO_ALT (31) params for Plane:
    # param1: unused
    # param2: radius (m), sign sets CW/CCW
    # param3: unused
    # param4: XTrack Tangent (0=center, 1=tangent)
    params = [0, loiter_radius_param, 0, 1]

    # Example: altitude 100m, frame=3 (GLOBAL_REL_ALT), autocontinue=1
    waypoint = Waypoint(
        0,          # seq (caller can overwrite)
        0,          # current
        3,          # frame (MAV_FRAME_GLOBAL_RELATIVE_ALT)
        31,         # command (MAV_CMD_NAV_LOITER_TO_ALT)
        params,     # [p1, p2, p3, p4]
        lat,
        lon,
        100,        # alt (m) – adjust as needed
        1           # autocontinue
    )

    return waypoint

def calc_waypoint_spacing(turning_radius: float) -> float:
    degrees_per_waypoint = 17  # or 15 for fewer waypoints
    angle_per_waypoint = math.radians(degrees_per_waypoint)
    step_size = turning_radius * angle_per_waypoint
    return step_size

def build_intermeddiate_dubins_path(start : List[float], end: List[float], turning_radius : float,
                                     step_size: float, ref_lat: float, ref_lon: float) -> List[Waypoint]:
        dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)
        points = dubins_iterator.get_segment_points()
    
        idx = 0
        intermeddiate_points = []
            
        while idx < len(points):
            point = points[idx]
            if point.valid:
                lat, lon = xy_to_latlon(point.x, point.y, ref_lat=ref_lat, ref_lon=ref_lon)
                if idx == 0 or idx == len(points) - 1  or points[idx].segment_idx != points[idx-1].segment_idx or points[idx + 1].segment_idx != points[idx].segment_idx: 
                    waypoint_num = 16 # Moving to next segment wants harder correction.;
                else:
                    waypoint_num = 16  # Segment inner point
                waypoint = Waypoint(0, 0, 3, waypoint_num, [0,100,0,0], lat, lon, 100, 1)
                print("point segment idx:", point.segment_idx)
                intermeddiate_points.append(waypoint)
                print(f"Dubins Point: lat={lat:.6f}, long={lon:.6f}, theta={math.degrees(point.theta):.2f}, t={point.t:.2f}")
            idx +=  1
        return intermeddiate_points

    
def build_dubins_path(parser: waypointParser, turning_radius: float, step_size: float):
    ref_lat = parser.waypoints[0].lat
    ref_lon = parser.waypoints[0].long
    idx = 0
    while idx < len(parser.waypoints) - 2:
        if parser.waypoints[idx].type == 40:
            print(f"Detected Dubisns start lat {parser.waypoints[idx].lat}, long {parser.waypoints[idx].long}")
            parser.waypoints[idx].type = 16  # Change to normal waypoint
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
    WRITE_FILENAME = "dubins_output.waypoints"
    parser.write_waypoints(WRITE_FILENAME)

def constrain_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))

def calc_turn_radius(airspeed_ms: float, bank_angle_rad: float) -> float:
    g = 9.80665
    # avoid tan(0) and crazy bank angles
    min_bank = math.radians(5.0)
    max_bank = math.radians(80.0)

    bank_angle_rad = constrain_float(bank_angle_rad, min_bank, max_bank)

    return (airspeed_ms * airspeed_ms) / (g * math.tan(bank_angle_rad)) 

def main():
    parser = waypointParser()
    parser.parse_waypoints()
    print(f"First waypoint lat: {parser.waypoints[0].lat}, long: {parser.waypoints[0].long}")
    # Need to check what value does bank_angle has?  
    turning_radius = calc_turn_radius(airspeed_ms=55.0, bank_angle_rad=math.radians(25.0)) 
    step_size = calc_waypoint_spacing(turning_radius)
    build_dubins_path(parser, turning_radius, step_size)

if __name__ == "__main__":
    main()