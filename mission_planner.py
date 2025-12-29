"""
Mission Planner - Integrated Dubins Path with Performance-Based Trajectory
===========================================================================
Combines Dubins path generation with performance-based altitude and speed profiles.

Features:
- Dubins path waypoints with curved trajectories
- Performance-optimized altitudes and speeds
- DO_CHANGE_SPEED commands at each waypoint
- QGC WPL 110 format output
- Arc configuration waypoints for curved segments

Author: Flight Performance Team
Date: December 2025
"""

import sys
import os
import math
import numpy as np
from typing import List, Tuple

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Trajectory.TrajectoryIntegrator import PerformanceDatabase, TrajectoryIntegrator
from heading_calc import Waypoint, calculate_heading_xy
from waypointParser import (
    latlon_to_xy, xy_to_latlon, calc_waypoint_spacing,
    SEG_TYPE, calculate_circle_center
)

# Waypoint type constants
WAYPOINT_NAV = 16
DO_CHANGE_SPEED = 178
DUBINS_WAYPOINT = 39
ARC_CONFIG_WAYPOINT = 40


class MissionPlanner:
    """Integrate performance trajectory with Dubins waypoint generation"""
    
    def __init__(self, performance_db_file: str):
        """
        Initialize mission planner
        
        Parameters:
        -----------
        performance_db_file : str
            Path to Excel file with performance data
        """
        print(f"Loading performance database...")
        self.perf_db = PerformanceDatabase(performance_db_file)
        self.integrator = TrajectoryIntegrator(self.perf_db)
        
    def plan_mission(self, waypoints_xy: List[Tuple[float, float]], 
                    origin_lat: float, origin_lon: float,
                    cruise_altitude: float, cruise_mach: float,
                    initial_fuel: float,
                    climb_mode='optimal', max_climb_factor=0.9,
                    initial_altitude=0, initial_mach=0.2,
                    loop_mission=False) -> List[Waypoint]:
        """
        Generate complete mission with Dubins paths, altitudes, and speeds
        
        Parameters:
        -----------
        waypoints_xy : List[Tuple[float, float]]
            List of waypoint positions in local XY coordinates [m]
        origin_lat, origin_lon : float
            Mission origin (lat/lon in degrees)
        cruise_altitude : float
            Cruise altitude [m]
        cruise_mach : float
            Cruise Mach number (0 or negative = use optimal from database)
        initial_fuel : float
            Initial fuel mass [kg]
        climb_mode : str
            'optimal' or 'max'
        max_climb_factor : float
            Factor for max climb (0-1)
        initial_altitude : float
            Starting altitude [m]
        initial_mach : float
            Starting Mach number
        loop_mission : bool
            If True, loop back to first waypoint
            
        Returns:
        --------
        waypoints : List[Waypoint]
            Complete mission waypoints with speeds and altitudes
        """
        print(f"\n=== Mission Planning ===")
        print(f"Waypoints: {len(waypoints_xy)}")
        print(f"Origin: ({origin_lat:.6f}, {origin_lon:.6f})")
        print(f"Cruise: {cruise_altitude}m @ Mach {cruise_mach if cruise_mach > 0 else 'optimal'}")
        
        # Calculate trajectory using TrajectoryIntegrator
        trajectory = self.integrator.calculate_trajectory(
            waypoints_xy, cruise_altitude, cruise_mach,
            initial_fuel, climb_mode, max_climb_factor,
            initial_altitude, initial_mach, loop_mission
        )
        
        # Convert trajectory to waypoints
        waypoints = self._trajectory_to_waypoints(
            trajectory, waypoints_xy, origin_lat, origin_lon, loop_mission
        )
        
        print(f"\n=== Mission Generated ===")
        print(f"Total waypoints: {len(waypoints)}")
        print(f"Mission duration: {trajectory['t'][-1]/60:.1f} min")
        print(f"Distance: {self._path_length(trajectory['x'], trajectory['y'])/1000:.1f} km")
        print(f"Fuel: {initial_fuel - trajectory['fuel'][-1]:.1f} kg consumed")
        
        return waypoints
    
    def _trajectory_to_waypoints(self, trajectory: dict, 
                                waypoints_xy: List[Tuple[float, float]],
                                origin_lat: float, origin_lon: float,
                                loop_mission: bool) -> List[Waypoint]:
        """
        Convert trajectory to waypoint list with Dubins arc markers
        
        Returns waypoints in QGC WPL 110 format with:
        - Home position
        - Arc configuration waypoints for curved segments
        - Dubins waypoints (all points for straight, first/last for arcs)
        - DO_CHANGE_SPEED commands
        - Optional loop jump
        
        Follows the pattern from waypointParser.build_intermeddiate_dubins_path
        """
        x_traj = trajectory['x']
        y_traj = trajectory['y']
        z_traj = trajectory['z']
        velocity = trajectory['velocity']
        dubins_paths = trajectory['dubins_paths']
        
        waypoints = []
        
        # Home position
        home = Waypoint(
            num=0, is_Active=1, relativity=0, type=WAYPOINT_NAV,
            params=[0, 0, 0, 0],
            lat=origin_lat, long=origin_lon, alt=0, unknown=1
        )
        waypoints.append(home)
        
        # Track first actual waypoint for loop
        first_wp_index = len(waypoints)
        
        # Process each Dubins path segment between waypoints
        for leg_idx, dubins in enumerate(dubins_paths):
            # Get trajectory indices for this leg
            wp_start_idx = self._find_closest_trajectory_point(
                waypoints_xy[leg_idx][0], waypoints_xy[leg_idx][1],
                x_traj, y_traj
            )
            wp_end_idx = self._find_closest_trajectory_point(
                waypoints_xy[leg_idx + 1][0], waypoints_xy[leg_idx + 1][1],
                x_traj, y_traj
            )
            
            # Get Dubins path data
            dubins_x = dubins['x']
            dubins_y = dubins['y']
            path_type = dubins.get('type', 'UNKNOWN')
            
            # Calculate arc centers for this path
            left_circle, right_circle = self._calc_arc_circles_for_path(
                dubins, origin_lat, origin_lon, z_traj[wp_start_idx]
            )
            
            # Create DubinsIterator-like points with segment info
            dubins_points = self._create_segmented_points(dubins, path_type)
            
            # Process points following build_intermeddiate_dubins_path pattern
            prev_segment = None
            for pt_idx, (seg_type, pt_x, pt_y) in enumerate(dubins_points):
                current_segment = seg_type
                is_segment_start = (prev_segment is None or prev_segment != current_segment)
                is_segment_end = (pt_idx == len(dubins_points) - 1 or 
                                 dubins_points[pt_idx + 1][0] != current_segment)
                
                # Interpolate altitude and velocity along the leg
                leg_progress = pt_idx / max(len(dubins_points) - 1, 1)
                traj_idx = int(wp_start_idx + leg_progress * (wp_end_idx - wp_start_idx))
                traj_idx = min(traj_idx, len(z_traj) - 1)
                
                alt = z_traj[traj_idx]
                vel = velocity[traj_idx]
                lat, lon = xy_to_latlon(pt_x, pt_y, origin_lat, origin_lon)
                
                # Add arc configuration waypoint at segment start
                if is_segment_start:
                    if left_circle and current_segment == SEG_TYPE.LEFT.value:
                        waypoints.append(left_circle)
                    elif right_circle and current_segment == SEG_TYPE.RIGHT.value:
                        waypoints.append(right_circle)
                
                # Decide whether to add this waypoint
                # Only add first and last point of each segment (straight or arc)
                should_add_waypoint = False
                if current_segment == SEG_TYPE.STRAIGHT.value:
                    # Only add first and last point of straight segments
                    if is_segment_start or is_segment_end:
                        should_add_waypoint = True
                elif current_segment in [SEG_TYPE.LEFT.value, SEG_TYPE.RIGHT.value]:
                    # Only add first and last point of arc segments
                    if is_segment_start or is_segment_end:
                        should_add_waypoint = True
                
                if should_add_waypoint:
                    # Navigation waypoint
                    wp = Waypoint(
                        num=0, is_Active=0, relativity=3, type=DUBINS_WAYPOINT,
                        params=[0, 50, 0, 0],
                        lat=lat, long=lon, alt=alt, unknown=1
                    )
                    waypoints.append(wp)
                    
                    # Speed command
                    speed_cmd = Waypoint(
                        num=0, is_Active=0, relativity=3, type=DO_CHANGE_SPEED,
                        params=[0, vel, -1, 0],
                        lat=lat, long=lon, alt=alt, unknown=1
                    )
                    waypoints.append(speed_cmd)
                
                prev_segment = current_segment
        
        # Add loop jump if requested
        if loop_mission:
            # Get cruise speed for loop
            cruise_idx = len(x_traj) // 2
            cruise_vel = velocity[cruise_idx]
            
            # DO_CHANGE_SPEED to constant cruise
            loop_speed = Waypoint(
                num=0, is_Active=0, relativity=3, type=DO_CHANGE_SPEED,
                params=[0, cruise_vel, -1, 0],
                lat=0, long=0, alt=0, unknown=1
            )
            waypoints.append(loop_speed)
            
            # DO_JUMP (type 177)
            jump_cmd = Waypoint(
                num=0, is_Active=0, relativity=3, type=177,
                params=[first_wp_index, -1, 0, 0],  # Jump to first WP, infinite loop
                lat=0, long=0, alt=0, unknown=1
            )
            waypoints.append(jump_cmd)
        
        # Renumber waypoints
        for idx, wp in enumerate(waypoints):
            wp.num = idx
        
        return waypoints
    
    def _calc_arc_circles_for_path(self, dubins: dict, origin_lat: float,
                                   origin_lon: float, altitude: float) -> Tuple:
        """
        Calculate arc circle centers for Dubins path
        Uses the turn_radius from the dubins dict (from performance tables)
        Returns (left_circle, right_circle) waypoints or None
        """
        dubins_x = dubins['x']
        dubins_y = dubins['y']
        path_type = dubins.get('type', 'UNKNOWN')
        turn_radius = dubins.get('turn_radius', 1000)  # Use stored turn radius
        
        left_circle = None
        right_circle = None
        
        # Only process curved paths
        if len(dubins_x) < 10 or path_type in ['STRAIGHT', 'STRAIGHT_FALLBACK']:
            return left_circle, right_circle
        
        # Path types: LSL, RSR, LSR, RSL
        if len(path_type) >= 3:
            first_turn = path_type[0]   # L or R
            second_turn = path_type[2]  # L or R
            
            n_points = len(dubins_x)
            
            # First arc (first third of points)
            if first_turn in ['L', 'R']:
                third = n_points // 3
                if third >= 3:
                    p1 = (dubins_x[0], dubins_y[0])
                    p2 = (dubins_x[third // 2], dubins_y[third // 2])
                    p3 = (dubins_x[third], dubins_y[third])
                    
                    center = calculate_circle_center(p1, p2, p3)
                    if center:
                        cx, cy = center
                        # Use turn_radius from performance tables, not calculated radius
                        lat, lon = xy_to_latlon(cx, cy, origin_lat, origin_lon)
                        
                        seg_type = SEG_TYPE.LEFT.value if first_turn == 'L' else SEG_TYPE.RIGHT.value
                        arc_wp = Waypoint(
                            num=0, is_Active=0, relativity=3, type=ARC_CONFIG_WAYPOINT,
                            params=[seg_type, turn_radius, 0, 0],
                            lat=lat, long=lon, alt=altitude, unknown=1
                        )
                        
                        if first_turn == 'L':
                            left_circle = arc_wp
                        else:
                            right_circle = arc_wp
            
            # Second arc (last third of points)
            if second_turn in ['L', 'R']:
                third = n_points // 3
                if third >= 3:
                    p1 = (dubins_x[2 * third], dubins_y[2 * third])
                    p2 = (dubins_x[2 * third + third // 2], dubins_y[2 * third + third // 2])
                    p3 = (dubins_x[-1], dubins_y[-1])
                    
                    center = calculate_circle_center(p1, p2, p3)
                    if center:
                        cx, cy = center
                        # Use turn_radius from performance tables, not calculated radius
                        lat, lon = xy_to_latlon(cx, cy, origin_lat, origin_lon)
                        
                        seg_type = SEG_TYPE.LEFT.value if second_turn == 'L' else SEG_TYPE.RIGHT.value
                        arc_wp = Waypoint(
                            num=0, is_Active=0, relativity=3, type=ARC_CONFIG_WAYPOINT,
                            params=[seg_type, turn_radius, 0, 0],
                            lat=lat, long=lon, alt=altitude, unknown=1
                        )
                        
                        if second_turn == 'L':
                            left_circle = arc_wp
                        else:
                            right_circle = arc_wp
        
        return left_circle, right_circle
    
    def _create_segmented_points(self, dubins: dict, path_type: str) -> List[Tuple[int, float, float]]:
        """
        Create list of (segment_type, x, y) tuples from Dubins path
        Segments the path into L/S/R sections based on path_type
        
        Returns:
            List of (seg_type, x, y) where seg_type is SEG_TYPE enum value
        """
        dubins_x = dubins['x']
        dubins_y = dubins['y']
        n_points = len(dubins_x)
        
        points = []
        
        # For straight paths, all points are straight
        if path_type in ['STRAIGHT', 'STRAIGHT_FALLBACK']:
            for i in range(n_points):
                points.append((SEG_TYPE.STRAIGHT.value, dubins_x[i], dubins_y[i]))
            return points
        
        # Parse path type (e.g., "LSL", "RSR", "LSR", "RSL")
        if len(path_type) < 3:
            # Unknown path type, treat as straight
            for i in range(n_points):
                points.append((SEG_TYPE.STRAIGHT.value, dubins_x[i], dubins_y[i]))
            return points
        
        first_turn = path_type[0]   # L or R
        second_turn = path_type[2]  # L or R
        
        # Divide points into three segments (approximate)
        third = n_points // 3
        
        # First arc
        first_seg = SEG_TYPE.LEFT.value if first_turn == 'L' else SEG_TYPE.RIGHT.value
        for i in range(third):
            points.append((first_seg, dubins_x[i], dubins_y[i]))
        
        # Straight segment
        for i in range(third, 2 * third):
            points.append((SEG_TYPE.STRAIGHT.value, dubins_x[i], dubins_y[i]))
        
        # Second arc
        second_seg = SEG_TYPE.LEFT.value if second_turn == 'L' else SEG_TYPE.RIGHT.value
        for i in range(2 * third, n_points):
            points.append((second_seg, dubins_x[i], dubins_y[i]))
        
        return points
    
    def _find_closest_trajectory_point(self, x: float, y: float,
                                      x_traj: np.ndarray, 
                                      y_traj: np.ndarray) -> int:
        """Find index of closest trajectory point to given XY coordinate"""
        distances = np.sqrt((x_traj - x)**2 + (y_traj - y)**2)
        return int(np.argmin(distances))
    
    def _path_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate total path length"""
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))
    
    def write_waypoints(self, waypoints: List[Waypoint], filename: str):
        """
        Write waypoints to QGC WPL 110 format file
        
        Parameters:
        -----------
        waypoints : List[Waypoint]
            List of waypoint objects
        filename : str
            Output filename (.waypoints)
        """
        with open(filename, "w") as file:
            file.write("QGC WPL 110\n")
            for wp in waypoints:
                line = f"{wp.num}\t{wp.is_Active}\t{wp.relativity}\t{wp.type}\t"
                line += "\t".join(f"{param:.8f}" for param in wp.params) + "\t"
                line += f"{wp.lat:.7f}\t{wp.long:.7f}\t{wp.alt:.6f}\t{wp.unknown}\n"
                file.write(line)
        
        print(f"\nMission saved to: {filename}")
        print(f"Format: QGC WPL 110")
        print(f"Total items: {len(waypoints)}")


def example_mission():
    """Example mission planning workflow"""
    
    # Configuration
    PERFORMANCE_DB = "Trajectory/Performances_Cov412BNoLG.xlsx"  # Update with your file
    OUTPUT_FILE = "mission_output.waypoints"
    
    # Mission parameters
    origin_lat = 32.0  # degrees
    origin_lon = 35.0  # degrees
    cruise_altitude = 6000.0  # meters
    cruise_mach = 0  # or 0 for optimal
    initial_fuel = 100.0  # kg
    
    # Define mission waypoints (XY in meters from origin)
    waypoints_xy = [
        (0, 0),           # Start
        (5000, 0),        # East
        (10000, 5000),    # Northeast
        (10000, 10000),   # North
        (5000, 10000),    # Northwest
        (0, 5000),        # Back toward start
    ]
    
    # Create planner
    planner = MissionPlanner(PERFORMANCE_DB)
    
    # Plan mission
    waypoints = planner.plan_mission(
        waypoints_xy=waypoints_xy,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        cruise_altitude=cruise_altitude,
        cruise_mach=cruise_mach,
        initial_fuel=initial_fuel,
        climb_mode='optimal',
        initial_altitude=0,
        initial_mach=0.2,
        loop_mission=False  # Set True for continuous loop
    )
    
    # Write to file
    planner.write_waypoints(waypoints, OUTPUT_FILE)
    
    return waypoints


if __name__ == "__main__":
    # Run example mission
    waypoints = example_mission()
    
    print("\n=== Mission Planning Complete ===")
    print("Update PERFORMANCE_DB path in example_mission() to your Excel file")
    print("Modify waypoints_xy to define your mission")
