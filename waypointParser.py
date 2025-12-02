from enum import Enum
from typing import List
from dubins import DubinsIterator
from heading_calc import Waypoint, calculate_heading_xy
import math

WAYPOINT_NUM = 16
DUBINS_WAYPOINT_NUM = 39

class SEG_TYPE(Enum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


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


def calc_waypoint_spacing(turning_radius: float) -> float:
    degrees_per_waypoint = 17  # or 15 for fewer waypoints
    angle_per_waypoint = math.radians(degrees_per_waypoint)
    step_size = turning_radius * angle_per_waypoint
    return step_size

def build_transition_zones(points: List, transition_points: int):
     # Pre-calculate segment boundaries and transition zones
    segment_boundaries = []
    transition_zones = set()  # Set of indices that are in transition zones
    
    for idx in range(len(points)):
        if idx == 0:
            continue
        if points[idx].segment_idx != points[idx - 1].segment_idx:
            segment_boundaries.append(idx)
            
            # Check if this is an opposite transition (R↔L or L↔R)
            prev_seg = points[idx - 1].segment_idx
            curr_seg = points[idx].segment_idx
            is_opposite = (
                (prev_seg == 0 and curr_seg == 2) or  # L to R
                (prev_seg == 2 and curr_seg == 0)     # R to L
            )
            
            if is_opposite:
                # Mark transition zone: transition_points before and after boundary
                for offset in range(transition_points):
                    trans_idx = idx + offset
                    if 0 < trans_idx < len(points) - 1:  # Don't mark first/last
                        transition_zones.add(trans_idx)
    return segment_boundaries, transition_zones

def build_intermeddiate_dubins_path(start : List[float], end: List[float], turning_radius : float,
                                     step_size: float, ref_lat: float, ref_lon: float) -> List[Waypoint]:
        dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)
        points = dubins_iterator.get_segment_points()
        max_angle_switch = math.radians(50.0) # radians. 
        transition_points = math.ceil(max_angle_switch * turning_radius / step_size) + 1
        print("Transition points:", transition_points)
        segment_boundaries, transition_zones = build_transition_zones(points, transition_points)
        print(f"Segment boundaries at indices: {segment_boundaries}")
        print(f"Transition zones at indices: {sorted(transition_zones)}")
        # Build waypoints
        intermeddiate_points = []
        for idx, point in enumerate(points):
            if not point.valid:
                continue
                
            lat, lon = xy_to_latlon(point.x, point.y, ref_lat=ref_lat, ref_lon=ref_lon)
            
            # Decide waypoint type
            if idx == 0 or idx == len(points) - 1:
                waypoint_num = DUBINS_WAYPOINT_NUM
            elif idx in segment_boundaries or (len(transition_zones) > 0 and (idx == min(transition_zones) or idx == max(transition_zones))):
                waypoint_num = DUBINS_WAYPOINT_NUM
            else:
                waypoint_num = WAYPOINT_NUM
            
            waypoint = Waypoint(0, 0, 3, waypoint_num, [0, 100, 0, 0], lat, lon, 100, 1)
            intermeddiate_points.append(waypoint)
            print(f"Segment {point.segment_idx}, idx {idx}, Type {waypoint_num}: lat={lat:.6f}, lon={lon:.6f}, theta={math.degrees(point.theta):.2f}")
            
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
    max_bank = math.radians(60.0)

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