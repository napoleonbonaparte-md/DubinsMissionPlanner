from enum import Enum
from typing import List
from Trajectory.TrajectoryIntegrator import PerformanceDatabase, TrajectoryIntegrator
from dubins import DubinsIterator
from heading_calc import Waypoint, calculate_heading_xy
import math
import json

WAYPOINT_NUM = 16
DUBINS_WAYPOINT_NUM = 39
ARC_CONFIGURATION_WAPOINT_NUM = 40
last_alt = 0
last_mach = 0.2
perf_db = PerformanceDatabase("./Trajectory/Performances_Cov412BNoLG.xlsx")
trac_ing = TrajectoryIntegrator(perf_db=perf_db)
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
    degrees_per_waypoint = 10  # or 15 for fewer waypoints
    angle_per_waypoint = math.radians(degrees_per_waypoint)
    step_size = turning_radius * angle_per_waypoint
    return step_size


def export_trajectory_to_json(trac_ing: TrajectoryIntegrator, ref_lat: float, ref_lon: float, output_path: str = "trajectory_export.json"):
    """Export trajectory integrator data to JSON with lat/lon coordinates.
    
    Args:
        trac_ing: TrajectoryIntegrator instance with trajectory data
        ref_lat: Reference latitude for coordinate conversion
        ref_lon: Reference longitude for coordinate conversion
        output_path: Output JSON file path
    """
    # Extract arrays from trajectory integrator
    t_arr = getattr(trac_ing, 't', [])
    x_arr = getattr(trac_ing, 'x', [])
    y_arr = getattr(trac_ing, 'y', [])
    z_arr = getattr(trac_ing, 'z', [])
    velocity_arr = getattr(trac_ing, 'velocity', [])
    mach_arr = getattr(trac_ing, 'mach', [])
    fuel_arr = getattr(trac_ing, 'fuel', [])
    heading_arr = getattr(trac_ing, 'heading', [])
    phase_arr = getattr(trac_ing, 'phase', [])
    Ps_arr = getattr(trac_ing, 'Ps', [])
    gammaV_arr = getattr(trac_ing, 'gammaV', [])
    fuel_flow_arr = getattr(trac_ing, 'fuel_flow', [])
    
    # Build samples list
    n_samples = len(t_arr)
    samples = []
    
    for i in range(n_samples):
        x_val = x_arr[i] if i < len(x_arr) else None
        y_val = y_arr[i] if i < len(y_arr) else None
        
        # Convert x,y to lat/lon
        lat_val = None
        lon_val = None
        if x_val is not None and y_val is not None:
            lat_val, lon_val = xy_to_latlon(x_val, y_val, ref_lat, ref_lon)
        
        sample = {
            'index': i,
            't_s': t_arr[i] if i < len(t_arr) else None,
            'x_m': x_val,
            'y_m': y_val,
            'lat_deg': lat_val,
            'lon_deg': lon_val,
            'alt_m': z_arr[i] if i < len(z_arr) else None,
            'velocity_ms': velocity_arr[i] if i < len(velocity_arr) else None,
            'mach': mach_arr[i] if i < len(mach_arr) else None,
            'fuel_kg': fuel_arr[i] if i < len(fuel_arr) else None,
            'heading_rad': heading_arr[i] if i < len(heading_arr) else None,
            'phase': phase_arr[i] if i < len(phase_arr) else None,
            'Ps_ms': Ps_arr[i] if i < len(Ps_arr) else None,
            'gammaV_deg': gammaV_arr[i] if i < len(gammaV_arr) else None,
            'fuel_flow_kgs': fuel_flow_arr[i] if i < len(fuel_flow_arr) else None,
        }
        samples.append(sample)
    
    # Build output structure
    output_data = {
        'metadata': {
            'ref_lat_deg': ref_lat,
            'ref_lon_deg': ref_lon,
            'num_samples': n_samples
        },
        'samples': samples
    }
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Exported {n_samples} trajectory samples to {output_path}")

def build_transition_zones(points: List, transition_points: int):
     # Pre-calculate segment boundaries and transition zones
    segment_boundaries = []
    transition_zones = set()  # Set of indices that are in transition zones
    
    for idx in range(1, len(points) - 1):
        if points[idx].segment_idx != points[idx - 1].segment_idx or points[idx + 1].segment_idx != points[idx].segment_idx:
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


def calculate_circle_center(p1, p2, p3):
    """Calculate circle center from 3 points using perpendicular bisector method.
    
    Args:
        p1, p2, p3: tuples of (x, y) coordinates
    
    Returns:
        (center_x, center_y) or None if points are collinear
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Midpoints
    mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
    mx2, my2 = (x2 + x3) / 2, (y2 + y3) / 2
    
    # Slopes of chords
    if abs(x2 - x1) < 1e-6:
        slope1 = float('inf')
    else:
        slope1 = (y2 - y1) / (x2 - x1)
    
    if abs(x3 - x2) < 1e-6:
        slope2 = float('inf')
    else:
        slope2 = (y3 - y2) / (x3 - x2)
    
    # Perpendicular slopes
    if abs(slope1) < 1e-6:
        perp_slope1 = float('inf')
    elif slope1 == float('inf'):
        perp_slope1 = 0
    else:
        perp_slope1 = -1 / slope1
    
    if abs(slope2) < 1e-6:
        perp_slope2 = float('inf')
    elif slope2 == float('inf'):
        perp_slope2 = 0
    else:
        perp_slope2 = -1 / slope2
    
    # Find intersection of perpendicular bisectors
    if perp_slope1 == float('inf'):
        center_x = mx1
        center_y = perp_slope2 * (center_x - mx2) + my2
    elif perp_slope2 == float('inf'):
        center_x = mx2
        center_y = perp_slope1 * (center_x - mx1) + my1
    elif abs(perp_slope1 - perp_slope2) < 1e-6:
        return None  # Collinear points
    else:
        center_x = (my2 - my1 + perp_slope1 * mx1 - perp_slope2 * mx2) / (perp_slope1 - perp_slope2)
        center_y = perp_slope1 * (center_x - mx1) + my1
    
    return (center_x, center_y)


def calc_arc_circles(points: List, ref_lat: float, ref_lon: float, turning_radius: float) -> tuple:
    """Calculate circle centers for each L_SEG and R_SEG segment using 3-point method.
    
    Returns:
        tuple: (left_circles, right_circles) where each is a list of (lat, lon, segment_idx)
    """
    left_circle = None
    right_circle = None
    
    # Group points by segment
    segments = {}
    for idx, point in enumerate(points):
        if not point.valid:
            continue
        seg_idx = point.segment_idx
        if seg_idx not in segments:
            segments[seg_idx] = []
        segments[seg_idx].append((idx, point))
    
    # For each segment, calculate circle center if it's a turn
    for seg_idx, seg_points in segments.items():
        if len(seg_points) < 3:
            print("CONTIUED")
            continue
            
        # Get 3 points: start, middle, end of segment
        start_idx = 0
        mid_idx = len(seg_points) // 2
        end_idx = len(seg_points) - 1
        
        p1 = (seg_points[start_idx][1].x, seg_points[start_idx][1].y)
        p2 = (seg_points[mid_idx][1].x, seg_points[mid_idx][1].y)
        p3 = (seg_points[end_idx][1].x, seg_points[end_idx][1].y)
        
        center = calculate_circle_center(p1, p2, p3)
        
        if center:
            center_x, center_y = center
            center_lat, center_lon = xy_to_latlon(center_x, center_y, ref_lat, ref_lon)
            
            # Determine if it's a left or right turn
            if seg_idx == SEG_TYPE.LEFT.value:
                left_circle = Waypoint(0, 0, 3, ARC_CONFIGURATION_WAPOINT_NUM, [seg_idx, turning_radius, 0, 0], center_lat, center_lon, 100, 1)
            elif seg_idx == SEG_TYPE.RIGHT.value:
                right_circle = Waypoint(0, 0, 3, ARC_CONFIGURATION_WAPOINT_NUM, [seg_idx, turning_radius, 0, 0], center_lat, center_lon, 100, 1)
    
    return left_circle, right_circle



def build_intermeddiate_dubins_path(start : List[float], end: List[float], turning_radius : float,
                                     step_size: float, ref_lat: float, ref_lon: float) -> List[Waypoint]:
        global last_alt
        global last_mach
        global trac_ing
        dubins_iterator = DubinsIterator(start, end, turning_radius, step_size)
        points = dubins_iterator.get_segment_points()
        # Build waypoints
        left_circle, right_circle = calc_arc_circles(points, ref_lat, ref_lon, turning_radius)
        print(f"Left arc circles: {left_circle}")
        print(f"Right arc circles: {right_circle}")
        valid_points = [] # PathPoint that we actually use.
        intermeddiate_points = []
        prev_segment = None


        for idx, point in enumerate(points):
            if not point.valid:
                continue
            # Trajectory integration is defined per-leg, so only run it when we have a valid "next" point
            if idx >= len(points) - 1 or not points[idx + 1].valid:
                continue

           
            # Run the trajectory integrator on the current leg
            

            current_segment = point.segment_idx
            is_segment_start = (prev_segment is None or prev_segment != current_segment)
            is_segment_end = (idx == len(points) - 1 or points[idx + 1].segment_idx != current_segment)
            
            lat, lon = xy_to_latlon(point.x, point.y, ref_lat=ref_lat, ref_lon=ref_lon)
            
            # Add arc configuration waypoint at the start of each arc segment
            if is_segment_start:
                if left_circle and current_segment == SEG_TYPE.LEFT.value:
                    intermeddiate_points.append(left_circle)
                    print(f"Inserted left arc configuration waypoint at idx {idx}")
                elif right_circle and current_segment == SEG_TYPE.RIGHT.value:
                    intermeddiate_points.append(right_circle)
                    print(f"Inserted right arc configuration waypoint at idx {idx}")
            
            # For arc segments (LEFT or RIGHT), only add first and last waypoint
            # For straight segments, get_segment_points already returns only first and last
            should_add_waypoint = False
            if current_segment == SEG_TYPE.STRAIGHT.value:
                should_add_waypoint = True  # Add all straight segment points
                waypoint_num =DUBINS_WAYPOINT_NUM 
            elif current_segment == SEG_TYPE.LEFT.value or current_segment == SEG_TYPE.RIGHT.value:
                # Only add first and last point of arc segments
                if is_segment_start or is_segment_end:
                    should_add_waypoint = True
                    waypoint_num = DUBINS_WAYPOINT_NUM
            
            if should_add_waypoint:
                valid_points.append(point)
                # Best-effort extraction of last altitude/velocity from either:
                # 1) a dict returned by _process_leg, or
                # 2) attributes populated on the integrator instance

   
  

                waypoint = Waypoint(0, 0, 3, waypoint_num, [0, 50, 0, 0], lat, lon, 100, 1)

                
                intermeddiate_points.append(waypoint)
                print(f"Segment {current_segment}, idx {idx}, Type {waypoint_num}: lat={lat:.6f}, lon={lon:.6f}, theta={math.degrees(point.theta):.2f}")
            
            prev_segment = current_segment

        second_points = []
        for idx, point in enumerate(valid_points):
            z = None 
            mach = None
            next_point = points[idx + 1]
           
            # Basic leg geometry (useful both for logging and for integrators that rely on leg length / track)
            dx_m = next_point.x - point.x
            dy_m = next_point.y - point.y
            leg_length_m = math.hypot(dx_m, dy_m)
            leg_track_rad = math.atan2(dy_m, dx_m)

            leg_result = trac_ing._process_leg(start_wp=point, end_wp=next_point, leg_idx=0, leg_distance=leg_length_m, current_alt=last_alt,current_mach=last_mach,current_heading=point.theta)
            z = leg_result[0]
            mach = leg_result[1]
                            # (Optional) if you want to use these values later, keep them; for now they are computed and available
            print(f"Leg {idx}->{idx+1}: s={leg_length_m:.2f}m track={math.degrees(leg_track_rad):.1f}deg z={z} v={mach}")
            vel = mach * 343.0  # Convert Mach to m/s (approximate at sea level)
            alt = z
            last_alt = alt 
            last_mach = mach
            do_change_speed  = Waypoint(
                    0,  # seq
                    0,      # current
                    3,      # frame
                    178,    # command (178 = MAV_CMD_DO_CHANGE_SPEED)
                    [0,      # speed_type (0 = airspeed in m/s)
                    vel,    # speed value
                    -1,     # throttle (-1 = no change)
                    0],      # relative (0 = absolute)
                    lat,
                    lon,
                    alt,
                    1       # autocontinue
                )
            
            
            intermeddiate_points[idx].alt = z
            second_points.append(intermeddiate_points[idx]) 
            second_points.append(do_change_speed)
        

        return second_points

    
def build_dubins_path(parser: waypointParser, turning_radius: float, step_size: float):
    global trac_ing
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
    # Export trajectory data to JSON
    export_trajectory_to_json(trac_ing, ref_lat=ref_lat, ref_lon=ref_lon, output_path="trajectory_export.json")

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
    turning_radius = calc_turn_radius(airspeed_ms=55.0, bank_angle_rad=math.radians(45.0)) 
    turning_radius = 1
    print(f"Calculated turning radius: {turning_radius:.2f} m")
    step_size = calc_waypoint_spacing(turning_radius)
    build_dubins_path(parser, turning_radius, step_size)

if __name__ == "__main__":
    main()