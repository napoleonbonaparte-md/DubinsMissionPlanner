"""
Trajectory Integration Tool with Performance Data
==================================================
Integrates aircraft trajectory using performance envelope data.
- Horizontal: Dubins path with performance-based turn radii
- Vertical: Climb → Cruise → Descent profile

Features:
- Export to CSV for data analysis
- Export to ArduPilot PLAN format (.waypoints) for SITL testing
  * Includes velocity checkpoints (DO_CHANGE_SPEED commands) to maintain desired Mach
  * User-selectable origin (lat/lon)
  * Configurable waypoint spacing

Author: Flight Performance Team
Date: December 2025
"""

# cspell:ignore dubins DubinsIterator waypointParser calc_waypoint_spacing
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.interpolate import RegularGridInterpolator, interp1d, NearestNDInterpolator
from scipy.integrate import odeint, solve_ivp
import threading
import json
import warnings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dubins import DubinsIterator
warnings.filterwarnings('ignore')


def calc_waypoint_spacing(turning_radius: float) -> float:
    """Return a reasonable Dubins sampling step size [m] for a given turn radius.

    Kept local to avoid circular imports with waypointParser.
    """
    degrees_per_waypoint = 10.0
    angle_per_waypoint = np.deg2rad(degrees_per_waypoint)
    try:
        r = float(turning_radius)
    except Exception:
        r = 0.0
    step_size = r * angle_per_waypoint
    return max(step_size, 1.0)


def isa_atmosphere(altitude):
    """ISA atmosphere model - returns T[K], a[m/s], P[Pa], rho[kg/m3]"""
    T0 = 288.15
    P0 = 101325
    g = 9.80665
    R = 287.05
    gamma = 1.4
    
    if altitude <= 11000:
        lapse_rate = -0.0065
        T = T0 + lapse_rate * altitude
        P = P0 * (T / T0) ** (-g / (lapse_rate * R))
    else:
        T = 216.65
        P = 22632 * np.exp(-g * (altitude - 11000) / (R * T))
    
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    
    return T, a, P, rho


class PerformanceDatabase:
    """Load and interpolate performance envelope data"""
    
    def __init__(self, excel_file): 
        """
        Load performance data from Excel file exported by FlightEnvelops.py
        
        Parameters:
        -----------
        excel_file : str
            Path to Excel file with performance data
        """
        self.excel_file = excel_file
        self._load_data()
        self._build_interpolators()
    
    def _load_data(self):
        """Load all sheets from Excel file"""
        print(f"Loading performance data from: {self.excel_file}")
        
        # Read all sheets
        excel_data = pd.read_excel(self.excel_file, sheet_name=None)
        
        # Load configuration
        if 'Config' not in excel_data:
            raise ValueError("Invalid file format: Missing 'Config' sheet")
        
        config = excel_data['Config']
        
        # Parse vectors
        alt_row = config[config['Parameter'] == 'Altitude Vector [m]']
        self.altitude_vec = np.array([float(x) for x in str(alt_row['Value'].values[0]).split(',')])
        
        mach_row = config[config['Parameter'] == 'Mach Vector']
        self.mach_vec = np.array([float(x) for x in str(mach_row['Value'].values[0]).split(',')])
        
        fuel_row = config[config['Parameter'] == 'Fuel Mass Vector [kg]']
        self.fuel_mass_vec = np.array([float(x) for x in str(fuel_row['Value'].values[0]).split(',')])
        
        # Load aircraft parameters
        self.empty_mass = float(config[config['Parameter'] == 'Empty Mass [kg]']['Value'].values[0])
        
        n_alt = len(self.altitude_vec)
        n_mach = len(self.mach_vec)
        n_fuel = len(self.fuel_mass_vec)
        
        # Initialize data arrays
        self.data = {
            'Ps': np.full((n_alt, n_mach, n_fuel), np.nan),
            'gammaV': np.full((n_alt, n_mach, n_fuel), np.nan),
            'Range2Kg': np.full((n_alt, n_mach, n_fuel), np.nan),
            'TurnRadius': np.full((n_alt, n_mach, n_fuel), np.nan),
            'FuelFlow': np.full((n_alt, n_mach, n_fuel), np.nan),
            'Alpha': np.full((n_alt, n_mach, n_fuel), np.nan),
            'L2D': np.full((n_alt, n_mach, n_fuel), np.nan),
        }
        self.service_ceiling = np.full((n_mach, n_fuel), np.nan)
        
        # Load optimal Mach data (2D: altitude x fuel)
        self.optimal_mach = np.full((n_alt, n_fuel), np.nan)
        if 'OptimalMach' in excel_data:
            print(f"  Found OptimalMach sheet")
            df = excel_data['OptimalMach']
            print(f"  OptimalMach sheet columns: {df.columns.tolist()}")
            print(f"  OptimalMach sheet shape: {df.shape}")
            if 'OptimalMach' in df.columns:
                # Data is in multi-index format (Altitude_m, Fuel_kg)
                values = df['OptimalMach'].values
                print(f"  OptimalMach values shape: {values.shape}")
                print(f"  Expected shape: ({n_alt}, {n_fuel}) = ({n_alt * n_fuel},) flat")
                if values.shape[0] == n_alt * n_fuel:
                    self.optimal_mach = values.reshape(n_alt, n_fuel)
                    valid_count = np.sum(~np.isnan(self.optimal_mach))
                    print(f"  Loaded OptimalMach sheet: {valid_count}/{self.optimal_mach.size} valid points")
                    if valid_count > 0:
                        print(f"  OptimalMach range: {np.nanmin(self.optimal_mach):.3f} - {np.nanmax(self.optimal_mach):.3f}")
                else:
                    print(f"  ERROR: Size mismatch! Got {values.shape[0]}, expected {n_alt * n_fuel}")
            else:
                print(f"  Warning: OptimalMach column not found in OptimalMach sheet")
        else:
            print(f"  Warning: OptimalMach sheet not found in Excel file")
            print(f"  Available sheets: {list(excel_data.keys())}")

        # Load service ceiling data (2D: mach x fuel)
        service_sheet = 'ServiceCieling'
        if service_sheet in excel_data:
            print(f"  Found {service_sheet} sheet")
            df = excel_data[service_sheet]

            service_col = next((col for col in df.columns if 'service' in str(col).lower()), None)
            mach_col = next((col for col in df.columns if 'mach' in str(col).lower()), None)
            fuel_col = next((col for col in df.columns if 'fuel' in str(col).lower()), None)

            if service_col is None:
                print(f"  Warning: {service_sheet} sheet missing service column")
            else:
                service_values = pd.to_numeric(df[service_col], errors='coerce').to_numpy()
                if service_values.shape[0] == n_mach * n_fuel:
                    try:
                        self.service_ceiling = service_values.reshape(n_mach, n_fuel)
                        valid_count = np.sum(~np.isnan(self.service_ceiling))
                        print(f"  Loaded {service_sheet}: {valid_count}/{self.service_ceiling.size} points (flat)")
                    except ValueError as exc:
                        print(f"  Warning: {service_sheet} reshape failed ({exc})")
                elif mach_col is not None and fuel_col is not None:
                    mach_vals = pd.to_numeric(df[mach_col], errors='coerce')
                    fuel_vals = pd.to_numeric(df[fuel_col], errors='coerce')
                    value_df = pd.DataFrame({
                        'mach': mach_vals,
                        'fuel': fuel_vals,
                        'ceiling': service_values
                    }).dropna()
                    if value_df.empty:
                        print(f"  Warning: {service_sheet} sheet has no numeric rows")
                    else:
                        pivot = value_df.pivot_table(
                            index='mach',
                            columns='fuel',
                            values='ceiling',
                            aggfunc='mean'
                        )
                        pivot = pivot.sort_index().sort_index(axis=1)
                        pivot_values = pivot.to_numpy()
                        pivot_mach = pd.to_numeric(pivot.index.to_numpy(), errors='coerce')
                        pivot_fuel = pd.to_numeric(pivot.columns.to_numpy(), errors='coerce')

                        if pivot_values.size == 0:
                            print(f"  Warning: {service_sheet} pivot produced empty grid")
                        else:
                            for m_idx, mach_value in enumerate(self.mach_vec):
                                if np.isnan(pivot_mach).all():
                                    break
                                mach_match_idx = np.nanargmin(np.abs(pivot_mach - mach_value))
                                for f_idx, fuel_value in enumerate(self.fuel_mass_vec):
                                    if np.isnan(pivot_fuel).all():
                                        break
                                    fuel_match_idx = np.nanargmin(np.abs(pivot_fuel - fuel_value))
                                    val = pivot_values[mach_match_idx, fuel_match_idx]
                                    if not np.isnan(val):
                                        self.service_ceiling[m_idx, f_idx] = float(val)

                            valid_count = np.sum(~np.isnan(self.service_ceiling))
                            print(f"  Loaded {service_sheet}: {valid_count}/{self.service_ceiling.size} points (pivot)")
                else:
                    print(f"  Warning: {service_sheet} sheet missing Mach/Fuel columns")
        else:
            print(f"  Warning: {service_sheet} sheet not found in Excel file")
        
        # Load data from fuel sheets
        for t_idx, fuel_mass in enumerate(self.fuel_mass_vec):
            sheet_name = f'Fuel_{fuel_mass:.0f}kg'
            if sheet_name in excel_data:
                df = excel_data[sheet_name]
                for metric in self.data.keys():
                    if metric in df.columns:
                        values = df[metric].values
                        self.data[metric][:, :, t_idx] = values.reshape(n_alt, n_mach)
        
        print(f"Loaded data: {len(self.altitude_vec)} altitudes, {len(self.mach_vec)} Mach, {len(self.fuel_mass_vec)} fuel states")
    
    def _build_interpolators(self):
        """Build 3D interpolators for each performance metric"""
        # Create interpolators for each metric
        self.interpolators = {}
        
        print(f"Building interpolators:")
        print(f"  Altitude range: {self.altitude_vec[0]:.0f} - {self.altitude_vec[-1]:.0f} m")
        print(f"  Mach range: {self.mach_vec[0]:.3f} - {self.mach_vec[-1]:.3f}")
        print(f"    Minimum Mach: {self.mach_vec[0]:.3f}")
        print(f"    Maximum Mach: {self.mach_vec[-1]:.3f}")
        print(f"  Fuel mass range: {self.fuel_mass_vec[0]:.1f} - {self.fuel_mass_vec[-1]:.1f} kg")
        print(f"  Available fuel masses: {self.fuel_mass_vec}")
        
        # Build OptimalMach interpolator (2D: altitude x fuel)
        valid_mask = ~np.isnan(self.optimal_mach)
        if np.sum(valid_mask) > 0:
            self.interpolators['OptimalMach'] = RegularGridInterpolator(
                (self.altitude_vec, self.fuel_mass_vec),
                self.optimal_mach,
                method='linear',
                bounds_error=False,
                fill_value=0.5
            )
            print(f"  OptimalMach: {np.sum(valid_mask)}/{self.optimal_mach.size} valid points")
            optimal_min = np.nanmin(self.optimal_mach)
            optimal_max = np.nanmax(self.optimal_mach)
            print(f"    Optimal Mach range: {optimal_min:.3f} - {optimal_max:.3f}")

        service_mask = ~np.isnan(self.service_ceiling)
        service_count = np.sum(service_mask)
        if service_count > 0:
            self.interpolators['ServiceCieling'] = RegularGridInterpolator(
                (self.mach_vec, self.fuel_mass_vec),
                self.service_ceiling,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            if service_count > 3:
                mach_grid, fuel_grid = np.meshgrid(
                    self.mach_vec, self.fuel_mass_vec, indexing='ij'
                )
                service_points = np.column_stack([
                    mach_grid[service_mask],
                    fuel_grid[service_mask]
                ])
                self.interpolators['ServiceCieling_nearest'] = NearestNDInterpolator(
                    service_points,
                    self.service_ceiling[service_mask]
                )
            service_min = np.nanmin(self.service_ceiling[service_mask])
            service_max = np.nanmax(self.service_ceiling[service_mask])
            print(f"  ServiceCieling: {service_count}/{self.service_ceiling.size} valid points")
            print(f"    Service ceiling range: {service_min:.0f} - {service_max:.0f} m")
        
        for metric, values in self.data.items():
            # Create mask for valid data
            valid_mask = ~np.isnan(values)
            valid_count = np.sum(valid_mask)
            print(f"  {metric}: {valid_count}/{values.size} valid points ({100*valid_count/values.size:.1f}%)")
            
            if valid_count > 10:
                try:
                    # Use nearest neighbor to fill gaps, then linear interpolation
                    # This helps with missing data in certain Mach ranges
                    
                    # Get valid points
                    valid_indices = np.where(valid_mask)
                    valid_points = np.column_stack([
                        self.altitude_vec[valid_indices[0]],
                        self.mach_vec[valid_indices[1]],
                        self.fuel_mass_vec[valid_indices[2]]
                    ])
                    valid_values = values[valid_mask]
                    
                    # Create nearest neighbor interpolator for fallback
                    self.interpolators[f'{metric}_nearest'] = NearestNDInterpolator(
                        valid_points, valid_values
                    )
                    
                    # Try regular grid interpolator (will have NaN gaps)
                    self.interpolators[metric] = RegularGridInterpolator(
                        (self.altitude_vec, self.mach_vec, self.fuel_mass_vec),
                        values,
                        method='linear',
                        bounds_error=False,
                        fill_value=None
                    )
                except Exception as e:
                    print(f"    Warning: Could not create interpolator for {metric}: {e}")
    
    def get_performance(self, altitude, mach, fuel_mass, metric='Ps'):
        """
        Get performance metric at given flight condition
        
        Parameters:
        -----------
        altitude : float - Altitude [m]
        mach : float - Mach number
        fuel_mass : float - Fuel mass [kg]
        metric : str - Performance metric name
        
        Returns:
        --------
        value : float - Interpolated performance value
        """
        if metric not in self.interpolators:
            return np.nan
        
        # Clip to valid ranges
        alt = np.clip(altitude, self.altitude_vec[0], self.altitude_vec[-1])
        m = np.clip(mach, self.mach_vec[0], self.mach_vec[-1])
        fm = np.clip(fuel_mass, self.fuel_mass_vec[0], self.fuel_mass_vec[-1])
        
        try:
            value = self.interpolators[metric]([alt, m, fm])[0]
            
            # If linear interpolation returns NaN (gap in data), use nearest neighbor
            if np.isnan(value) and f'{metric}_nearest' in self.interpolators:
                value = self.interpolators[f'{metric}_nearest']([alt, m, fm])[0]
                if metric == 'Ps':
                    print(f"  Debug: Using nearest neighbor for Ps at alt={alt/1000:.0f}km, mach={m:.3f} -> {value:.2f} m/s")
            
            # Debug print for Ps
            if metric == 'Ps' and (np.isnan(value) or value == 0):
                print(f"  Debug: Ps lookup at alt={alt:.0f}, mach={m:.3f}, fuel={fm:.1f} -> {value}")
                print(f"    Fuel range in DB: {self.fuel_mass_vec[0]:.1f} - {self.fuel_mass_vec[-1]:.1f} kg")
                print(f"    Clipped fuel: {fm:.1f} kg (original: {fuel_mass:.1f} kg)")
            return value if not np.isnan(value) else 0.0
        except Exception as e:
            if metric == 'Ps':
                print(f"  Warning: Interpolation failed for {metric}: {e}")
            return 0.0
    
    def get_turn_radius(self, altitude, mach, fuel_mass):
        """Get minimum turn radius [m]"""
        radius_m = self.get_performance(altitude, mach, fuel_mass, 'TurnRadius')
        if radius_m <= 0 or np.isnan(radius_m):
            # Estimate from load factor: R = V^2 / (g * sqrt(n^2 - 1))
            # Assume n=2 for coordinated turn
            T, a, P, rho = isa_atmosphere(altitude)
            V = mach * a
            radius_m = (V * V) / (9.81 * np.sqrt(3))  # n=2 -> sqrt(n^2-1) = sqrt(3)
        return max(radius_m, 100)  # minimum 100m
    
    def get_fuel_flow(self, altitude, mach, fuel_mass):
        """Get fuel flow [kg/s]"""
        return max(self.get_performance(altitude, mach, fuel_mass, 'FuelFlow'), 0.0)
    
    def get_Ps(self, altitude, mach, fuel_mass):
        """Get specific excess power [m/s]"""
        return self.get_performance(altitude, mach, fuel_mass, 'Ps')
    
    def get_optimal_cruise_mach(self, altitude, fuel_mass):
        """Get optimal cruise Mach from OptimalMach sheet"""
        if 'OptimalMach' not in self.interpolators:
            print(f"  Warning: OptimalMach interpolator not found, using fallback 0.5")
            return 0.5  # Fallback
        
        # Clip to valid ranges
        alt = np.clip(altitude, self.altitude_vec[0], self.altitude_vec[-1])
        fm = np.clip(fuel_mass, self.fuel_mass_vec[0], self.fuel_mass_vec[-1])
        
        try:
            mach = self.interpolators['OptimalMach']([alt, fm])[0]
            result = mach if not np.isnan(mach) else 0.5
            return result
        except Exception as e:
            print(f"  Warning: OptimalMach lookup failed at alt={alt:.0f}, fuel={fm:.1f}: {e}")
            return 0.5

    def get_service_ceiling(self, mach, fuel_mass):
        """Get service ceiling altitude [m] for given Mach and fuel"""
        fallback = None
        if hasattr(self, 'altitude_vec') and len(self.altitude_vec) > 0:
            fallback = float(self.altitude_vec[-1])

        if 'ServiceCieling' not in self.interpolators:
            return fallback

        m = np.clip(mach, self.mach_vec[0], self.mach_vec[-1])
        fm = np.clip(fuel_mass, self.fuel_mass_vec[0], self.fuel_mass_vec[-1])

        try:
            ceiling = self.interpolators['ServiceCieling']([m, fm])[0]
            if (np.isnan(ceiling) or ceiling <= 0) and 'ServiceCieling_nearest' in self.interpolators:
                ceiling = self.interpolators['ServiceCieling_nearest']([m, fm])[0]
        except Exception as exc:
            print(f"  Warning: ServiceCieling lookup failed at mach={m:.3f}, fuel={fm:.1f}: {exc}")
            ceiling = np.nan

        if np.isnan(ceiling) or ceiling <= 0:
            return fallback
        return float(ceiling)


class DubinsPath:
    """Generate Dubins path between waypoints"""
    SEGMENT_MAP = {0: 'L', 1: 'S', 2: 'R'}
    
    @staticmethod
    def calculate_path(start_pos, start_heading, end_pos, end_heading, turn_radius):
        """
        Calculate Dubins path using the shared waypointParser backend.
        Falls back to the analytic implementation if the iterator fails.
        """
        path = DubinsPath._calculate_with_waypoint_parser(
            start_pos, start_heading, end_pos, end_heading, turn_radius
        )
        if path is not None:
            return path
        return DubinsPath._calculate_geometry_path(
            start_pos, start_heading, end_pos, end_heading, turn_radius
        )

    @staticmethod
    def _calculate_with_waypoint_parser(start_pos, start_heading, end_pos, end_heading, turn_radius):
        """Use DubinsIterator from waypointParser for path generation."""
        try:
            if turn_radius <= 0:
                return None
            step_size = max(calc_waypoint_spacing(turn_radius), 1.0)
            start_state = (float(start_pos[0]), float(start_pos[1]), float(start_heading))
            end_state = (float(end_pos[0]), float(end_pos[1]), float(end_heading))
            iterator = DubinsIterator(start_state, end_state, float(turn_radius), float(step_size))
            points = iterator.get_segment_points()
            valid_points = [pt for pt in points if getattr(pt, 'valid', True)]
            if len(valid_points) < 2:
                return None

            x = np.array([float(pt.x) for pt in valid_points], dtype=float)
            y = np.array([float(pt.y) for pt in valid_points], dtype=float)
            heading = np.array([float(pt.theta) for pt in valid_points], dtype=float)

            keep_mask = [True]
            for idx in range(1, len(x)):
                step = np.hypot(x[idx] - x[idx-1], y[idx] - y[idx-1])
                keep_mask.append(step > 1e-6)
            if not all(keep_mask):
                keep_mask = np.array(keep_mask, dtype=bool)
                x = x[keep_mask]
                y = y[keep_mask]
                heading = heading[keep_mask]
            if len(x) < 2:
                return None

            dx = np.diff(x)
            dy = np.diff(y)
            distance = float(np.sum(np.hypot(dx, dy)))
            if distance <= 0:
                return None

            segment_sequence = []
            last_seg = None
            for pt in valid_points:
                seg_char = DubinsPath.SEGMENT_MAP.get(int(pt.segment_idx))
                if seg_char and seg_char != last_seg:
                    segment_sequence.append(seg_char)
                    last_seg = seg_char
            path_type = ''.join(segment_sequence[:3]) if segment_sequence else 'ITER'

            return {
                'x': x,
                'y': y,
                'heading': heading,
                'distance': distance,
                'type': path_type
            }
        except Exception as exc:
            print(f"  Warning: waypointParser Dubins backend failed ({exc}), falling back to analytic solution")
            return None

    @staticmethod
    def _calculate_geometry_path(start_pos, start_heading, end_pos, end_heading, turn_radius):
        """Legacy analytic Dubins solver used as a fallback."""
        # Check if heading change is very small - use straight line
        heading_diff = abs(np.arctan2(np.sin(end_heading - start_heading), 
                                      np.cos(end_heading - start_heading)))
        
        # Also check if we're aligned with the destination
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        direction_to_end = np.arctan2(dy, dx)
        direction_diff = abs(np.arctan2(np.sin(direction_to_end - start_heading),
                                       np.cos(direction_to_end - start_heading)))
        
        # Use straight line if: headings are nearly same AND aligned with destination
        if heading_diff < 0.01 and direction_diff < 0.05:  # ~0.6° and ~3° thresholds
            n_points = 2
            return {
                'x': np.linspace(start_pos[0], end_pos[0], n_points),
                'y': np.linspace(start_pos[1], end_pos[1], n_points),
                'heading': np.full(n_points, start_heading),
                'distance': distance,
                'type': 'STRAIGHT'
            }
        
        # Calculate all four Dubins paths and select shortest
        paths = []
        
        for path_type in ['LSL', 'RSR', 'LSR', 'RSL']:
            try:
                path = DubinsPath._calculate_single_path(
                    start_pos, start_heading, end_pos, end_heading, 
                    turn_radius, path_type
                )
                if path is not None:
                    paths.append(path)
            except:
                continue
        
        if not paths:
            # Fallback: straight line
            print(f"WARNING: No valid Dubins paths found, using straight line")
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            n_points = 2  # Only start and end points
            
            return {
                'x': np.linspace(start_pos[0], end_pos[0], n_points),
                'y': np.linspace(start_pos[1], end_pos[1], n_points),
                'heading': np.full(n_points, np.arctan2(dy, dx)),
                'distance': distance,
                'type': 'STRAIGHT_FALLBACK'
            }
        
        # Return shortest path
        best_path = min(paths, key=lambda p: p['distance'])
        print(f"Selected Dubins path: {best_path['type']}, {len(paths)} options")
        return best_path
    
    @staticmethod
    def _calculate_single_path(start_pos, start_heading, end_pos, end_heading, R, path_type):
        """
        Calculate Dubins path using proper geometry.
        Based on standard Dubins curve formulas.
        """
        x1, y1 = start_pos
        x2, y2 = end_pos
        theta1 = start_heading
        theta2 = end_heading
        
        # Helper function to normalize angle to [-π, π]
        def mod2pi(x):
            return np.arctan2(np.sin(x), np.cos(x))
        
        # Calculate turn centers
        if path_type[0] == 'L':
            cx1 = x1 - R * np.sin(theta1)
            cy1 = y1 + R * np.cos(theta1)
        else:  # 'R'
            cx1 = x1 + R * np.sin(theta1)
            cy1 = y1 - R * np.cos(theta1)
        
        if path_type[2] == 'L':
            cx2 = x2 - R * np.sin(theta2)
            cy2 = y2 + R * np.cos(theta2)
        else:  # 'R'
            cx2 = x2 + R * np.sin(theta2)
            cy2 = y2 - R * np.cos(theta2)
        
        # Use standard Dubins formulation with coordinate transformation
        # The standard formulation works with positions, not centers
        # Calculate distance and angle between START and END positions
        dx = x2 - x1
        dy = y2 - y1
        D = np.sqrt(dx**2 + dy**2)
        d = D / R  # Normalized distance between positions
        
        # Transform to coordinate system where line from start to end is the x-axis
        theta = np.arctan2(dy, dx)  # Angle of line connecting positions
        
        # Relative angles (transformed coordinates)
        alpha = mod2pi(theta1 - theta)  # Start heading relative to line
        beta = mod2pi(theta2 - theta)   # End heading relative to line
        
        # Precompute trig values for efficiency
        sa = np.sin(alpha)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        cb = np.cos(beta)
        c_ab = np.cos(alpha - beta)
        
        if path_type == 'LSL':
            # LSL: Left-Straight-Left
            if D < 0.01:
                return None
            p_sq = 2 + d*d - 2*c_ab + 2*d*(sa - sb)
            if p_sq >= 0:
                tmp = np.arctan2(cb - ca, d + sa - sb)
                angle1 = mod2pi(tmp - alpha)
                angle2 = mod2pi(beta - tmp)
                if angle1 < 0:
                    angle1 += 2 * np.pi
                if angle2 < 0:
                    angle2 += 2 * np.pi
                length_straight = np.sqrt(p_sq) * R
            else:
                return None
                
        elif path_type == 'RSR':
            # RSR: Right-Straight-Right
            if D < 0.01:
                return None
            p_sq = 2 + d*d - 2*c_ab + 2*d*(sb - sa)
            if p_sq >= 0:
                tmp = np.arctan2(ca - cb, d - sa + sb)
                angle1 = mod2pi(alpha - tmp)
                angle2 = mod2pi(tmp - beta)
                if angle1 < 0:
                    angle1 += 2 * np.pi
                if angle2 < 0:
                    angle2 += 2 * np.pi
                length_straight = np.sqrt(p_sq) * R
            else:
                return None
                
        elif path_type == 'LSR':
            # LSR: Left-Straight-Right (external tangent)
            if d < 2.0:
                return None
            p_sq = -2 + d*d + 2*c_ab + 2*d*(sa + sb)
            if p_sq >= 0:
                p = np.sqrt(p_sq)
                tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
                angle1 = mod2pi(tmp - alpha)
                angle2 = mod2pi(tmp - beta)
                if angle1 < 0:
                    angle1 += 2 * np.pi
                if angle2 < 0:
                    angle2 += 2 * np.pi
                length_straight = p * R
            else:
                return None
                
        elif path_type == 'RSL':
            # RSL: Right-Straight-Left (external tangent)
            if d < 2.0:
                return None
            p_sq = -2 + d*d + 2*c_ab - 2*d*(sa + sb)
            if p_sq >= 0:
                p = np.sqrt(p_sq)
                tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
                angle1 = mod2pi(alpha - tmp)
                angle2 = mod2pi(beta - tmp)
                if angle1 < 0:
                    angle1 += 2 * np.pi
                if angle2 < 0:
                    angle2 += 2 * np.pi
                length_straight = p * R
            else:
                return None
        else:
            return None
        
        # Generate path points using standard Dubins segment transformation
        # Use higher resolution for smooth paths
        n_arc = 200
        # For straight segment, use normalized length (length_straight is in world units, so divide by R)
        p_norm = length_straight / R
        n_straight = max(2, int(p_norm * 20))  # Scale n_straight by normalized length
        
        # The analytical solution gives us angles in the normalized/transformed space
        # where the start is at origin heading at angle alpha, all in units of rho=1
        # We generate segments in this normalized space, then transform back
        
        # Segment 1 (first arc)
        t1_vals = np.linspace(0, angle1, n_arc)
        seg1_x = []
        seg1_y = []
        seg1_h = []
        
        for t in t1_vals:
            if path_type[0] == 'L':
                # Left turn from (0,0) heading at angle alpha
                seg1_x.append(np.sin(alpha + t) - np.sin(alpha))
                seg1_y.append(-np.cos(alpha + t) + np.cos(alpha))
                seg1_h.append(alpha + t)
            else:  # R
                # Right turn from (0,0) heading at angle alpha
                seg1_x.append(-np.sin(alpha - t) + np.sin(alpha))
                seg1_y.append(np.cos(alpha - t) - np.cos(alpha))
                seg1_h.append(alpha - t)
        
        # End configuration of segment 1
        q1_x = seg1_x[-1]
        q1_y = seg1_y[-1]
        q1_h = seg1_h[-1]
        
        # Segment 2 (straight)
        # In normalized space (rho=1)
        t2_vals = np.linspace(0, p_norm, n_straight)
        seg2_x = []
        seg2_y = []
        seg2_h = []
        
        for t in t2_vals:
            # Straight segment in direction of q1_h, in normalized space
            seg2_x.append(q1_x + np.cos(q1_h) * t)
            seg2_y.append(q1_y + np.sin(q1_h) * t)
            seg2_h.append(q1_h)
        
        # End configuration of segment 2
        q2_x = seg2_x[-1]
        q2_y = seg2_y[-1]
        q2_h = seg2_h[-1]
        
        # Segment 3 (second arc)
        t3_vals = np.linspace(0, angle2, n_arc)
        seg3_x = []
        seg3_y = []
        seg3_h = []
        
        for t in t3_vals:
            if path_type[2] == 'L':
                # Left turn from q2 heading at angle q2_h
                seg3_x.append(q2_x + np.sin(q2_h + t) - np.sin(q2_h))
                seg3_y.append(q2_y - np.cos(q2_h + t) + np.cos(q2_h))
                seg3_h.append(q2_h + t)
            else:  # R
                # Right turn from q2 heading at angle q2_h
                seg3_x.append(q2_x - np.sin(q2_h - t) + np.sin(q2_h))
                seg3_y.append(q2_y + np.cos(q2_h - t) - np.cos(q2_h))
                seg3_h.append(q2_h - t)
        
        # Combine all segments (in normalized space, rho=1)
        # Remove first points of seg2 and seg3 to avoid duplication at boundaries
        x_norm = np.concatenate([seg1_x, seg2_x[1:], seg3_x[1:]])
        y_norm = np.concatenate([seg1_y, seg2_y[1:], seg3_y[1:]])
        heading_norm = np.concatenate([seg1_h, seg2_h[1:], seg3_h[1:]])
        
        # Transform from normalized coordinates back to world coordinates
        # Scale by R, rotate by theta, and translate to actual start position
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x = x1 + R * (x_norm * cos_theta - y_norm * sin_theta)
        y = y1 + R * (x_norm * sin_theta + y_norm * cos_theta)
        heading = heading_norm + theta
        
        total_distance = R * angle1 + length_straight + R * angle2
        
        return {
            'x': x,
            'y': y,
            'heading': heading,
            'distance': total_distance,
            'type': path_type
        }
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    @staticmethod
    def _normalize_angle_diff(angle, turn_direction):
        """Normalize angle difference for turn direction (1=left/CCW, -1=right/CW)"""
        # Normalize to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        if turn_direction > 0:  # Left turn (counter-clockwise)
            if angle < 0:
                angle += 2 * np.pi
        else:  # Right turn (clockwise)
            if angle > 0:
                angle -= 2 * np.pi
        
        return angle


class TrajectoryIntegrator:
    """Integrate 3D trajectory with performance data"""
    
    def __init__(self, perf_db):
        """
        Initialize trajectory integrator
        
        Parameters:
        -----------
        perf_db : PerformanceDatabase
            Performance database instance
        """
        self.perf_db = perf_db
        self.g = 9.80665  # Gravity [m/s²]
        
        # Avoid mutable default args
        self.t = [] 
        self.x = [] 
        self.y = [] 
        self.z = [] 
        self.heading = [] 
        self.velocity = [] 
        self.mach = [] 
        self.fuel = [] 
        self.fuel_flow = [] 
        self.Ps = [] 
        self.gammaV = [] 
        self.phase = [] 

    def get_optimal_mach(self, altitude, fuel_mass):
        """
        Get optimal Mach number from OptimalMach table
        
        Parameters:
        -----------
        altitude : float - Altitude [m]
        fuel_mass : float - Fuel mass [kg]
        
        Returns:
        --------
        optimal_mach : float - Optimal Mach number from database
        """
        return self.perf_db.get_optimal_cruise_mach(altitude, fuel_mass)
    
    def _process_leg(
        self,
        leg_idx,
        total_legs=1,
        start_wp=None,
        end_wp=None,
        current_heading=0.0,
        end_heading=0.0,
        current_alt=0.0,
        current_mach=0.2,
        current_fuel=100.0,
        cruise_altitude=3000.0,
        cruise_mach=0.0,
        accumulated_distance=0.0,
        cruise_start_distance=10000.0,
        cruise_end_distance=0.0,
        total_distance=0.0,
        use_optimal_cruise=True,
        climb_mode="optimal",
        max_climb_factor=0.9,
        resolve_envelope_ceiling=None,
        leg_distance=0.0,
    ):
        """
        Process a single leg of the trajectory.

        Notes:
        - All inputs have defaults so callers can omit them when convenient.
        - List-like trajectory histories default to empty lists (created per-call).
        - `resolve_envelope_ceiling` defaults to a function returning None.
        """
        if start_wp is None or end_wp is None:
            raise ValueError("start_wp and end_wp are required")


        if resolve_envelope_ceiling is None:
            def resolve_envelope_ceiling(_mach, _fuel):
                return None

        # Get turn radius at cruise conditions for planning
        planning_mach = cruise_mach if cruise_mach > 0 else self.get_optimal_mach(current_alt, current_fuel)
        turn_radius = self.perf_db.get_turn_radius(current_alt, planning_mach, current_fuel)
        print(f"  Turn radius: {turn_radius:.0f} m (at {current_alt:.0f}m, Mach {planning_mach:.3f})")



      
        # Calculate leg distance
        leg_start_distance = accumulated_distance
        leg_end_distance = accumulated_distance + leg_distance

        print(f"  Distance tracking:")
        print(f"    Leg distance: {leg_distance/1000:.2f} km")
        print(f"    Accumulated before: {leg_start_distance/1000:.1f} km")
        print(f"    Accumulated after: {leg_end_distance/1000:.1f} km")
        print(f"    Climb ends at: {cruise_start_distance/1000:.1f} km")
        print(f"    Descent starts at: {cruise_end_distance/1000:.1f} km")   

        # Integrate along this leg
        dt = 1.0  # Time step [s]

        # Setup state and parameters for integration
        state = {
            'alt': current_alt,
            'mach': current_mach,
            'fuel': current_fuel,
            'leg_distance': 0.0,  # Initialize at start of leg
            'x': self.x[-1] if len(self.x) > 0 else 0.0,
            'y': self.y[-1] if len(self.y) > 0 else 0.0,
            'heading': self.heading[-1] if len(self.heading) > 0 else 0.0
        }

        leg_params = {
            'dubins': 0,
            'dubins_distances': 0,
            'leg_distance': leg_distance,
            'leg_start_distance': leg_start_distance,
            'leg_idx': leg_idx,
            'total_legs': total_legs, 
            'heading': current_heading
        }

        mission_params = {
            'cruise_altitude': cruise_altitude,
            'cruise_mach': cruise_mach,
            'use_optimal_cruise': use_optimal_cruise,
            'climb_mode': climb_mode,
            'max_climb_factor': max_climb_factor,
            'cruise_start_distance': cruise_start_distance,
            'cruise_end_distance': cruise_end_distance,
            'total_distance': total_distance,
            'resolve_envelope_ceiling': resolve_envelope_ceiling
        }

        while state['leg_distance'] < leg_distance:
            new_state, phase_info = self._integrate_time_step(state, leg_params, mission_params, dt)

            self.t.append(self.t[-1] if len(self.t) > 0 else 0 + dt)
            self.x.append(new_state['x'])
            self.y.append(new_state['y'])
            self.z.append(new_state['alt'])
            self.heading.append(new_state['heading'])
            self.mach.append(new_state['mach'])
            self.fuel.append(new_state['fuel'])
            self.fuel_flow.append(phase_info['ff'])
            self.Ps.append(phase_info['Ps'])
            self.gammaV.append(phase_info['gamma_deg'])
            self.phase.append(phase_info['phase'])
            self.velocity.append(new_state['velocity'])

            state = new_state
            current_alt = new_state['alt']
            current_mach = new_state['mach']
            current_fuel = new_state['fuel']

            if current_fuel <= 0:
                print("  WARNING: Out of fuel!")
                break

        # Ensure we end exactly at the waypoint
        self.x.append(end_wp.x)
        self.y.append(end_wp.y)

        accumulated_distance = leg_end_distance

        return current_alt, current_mach, current_fuel, end_heading, accumulated_distance

    def _integrate_time_step(self, current_state,  leg_params, mission_params, dt=1.0):
        """
        Integrate one time step along the trajectory
        
        Parameters:
        -----------
        current_state : dict
            Current aircraft state with keys: alt, mach, fuel, leg_distance, x, y, heading
        leg_params : dict
            Leg parameters: dubins, dubins_distances, leg_distance, leg_start_distance, 
                           leg_idx, total_legs
        mission_params : dict
            Mission parameters: cruise_altitude, cruise_mach, use_optimal_cruise, climb_mode,
                              max_climb_factor, cruise_start_distance, cruise_end_distance,
                              total_distance, resolve_envelope_ceiling
        dt : float
            Time step [s]
        
        Returns:
        --------
        new_state : dict
            Updated state after time step
        phase_info : dict
            Phase information: phase, climb_rate, V, gamma_deg, Ps, ff
        """
        # Unpack current state
        current_alt = current_state['alt']
        current_mach = current_state['mach']
        current_fuel = current_state['fuel']
        current_leg_distance = current_state['leg_distance']
        
        # Unpack leg parameters
        dubins = leg_params['dubins']
        dubins_distances = leg_params['dubins_distances']
        leg_distance = leg_params['leg_distance']
        leg_start_distance = leg_params['leg_start_distance']
        leg_idx = leg_params['leg_idx']
        total_legs = leg_params['total_legs']
        
        # Unpack mission parameters
        cruise_altitude = mission_params['cruise_altitude']
        cruise_mach = mission_params['cruise_mach']
        use_optimal_cruise = mission_params['use_optimal_cruise']
        climb_mode = mission_params['climb_mode']
        max_climb_factor = mission_params['max_climb_factor']
        cruise_start_distance = mission_params['cruise_start_distance']
        cruise_end_distance = mission_params['cruise_end_distance']
        total_distance = mission_params['total_distance']
        resolve_envelope_ceiling = mission_params['resolve_envelope_ceiling']
        
        # Current total distance
        current_distance = leg_start_distance + leg_distance
        
        # Determine current phase based on accumulated distance
        if current_distance < cruise_start_distance:
            current_phase = 'CLIMB'
        elif current_distance < cruise_end_distance:
            current_phase = 'CRUISE'
        else:
            current_phase = 'DESCENT'
        
        # Get performance at current state
        current_Ps = self.perf_db.get_Ps(current_alt, current_mach, current_fuel)
        current_ff = self.perf_db.get_fuel_flow(current_alt, current_mach, current_fuel)
        
        # Atmosphere
        T, a, P, rho = self._atmosphere(current_alt)
        V = current_mach * a
        
        # Track available Ps for energy management in max climb
        available_Ps_for_accel = 0
        
        # Vertical rate
        if current_phase == 'CLIMB':
            if climb_mode == 'max':
                # For max climb, first find the Mach that gives maximum Ps
                best_mach = 0.2
                best_Ps = -999
                for test_mach in np.linspace(0.2, 0.7, 15):
                    test_Ps = self.perf_db.get_Ps(current_alt, test_mach, current_fuel)
                    if test_Ps > best_Ps:
                        best_Ps = test_Ps
                        best_mach = test_mach
                
                # Use Ps at the best Mach, scaled by factor
                if best_Ps > 0:
                    climb_rate = best_Ps * max_climb_factor
                    available_Ps_for_accel = best_Ps * (1.0 - max_climb_factor)
                else:
                    climb_rate = 0
                    available_Ps_for_accel = 0
            else:  # 'optimal'
                optimal_mach = self.get_optimal_mach(current_alt, current_fuel)
                optimal_Ps = self.perf_db.get_Ps(current_alt, optimal_mach, current_fuel)
                climb_rate = optimal_Ps if optimal_Ps > 0 else 0
                available_Ps_for_accel = 0
            
            climb_rate = min(climb_rate, 20)  # Max 20 m/s
            # Smoothly approach cruise altitude
            altitude_error = cruise_altitude - current_alt
            if altitude_error < 100:
                climb_rate = min(climb_rate, altitude_error / dt)
            if current_alt >= cruise_altitude:
                current_alt = cruise_altitude
                climb_rate = 0
        elif current_phase == 'DESCENT':
            # Calculate descent rate to reach ground at end of descent phase
            distance_remaining_in_descent = total_distance - current_distance
            if distance_remaining_in_descent > 100 and current_alt > 10:
                time_remaining = distance_remaining_in_descent / V
                required_descent_rate = current_alt / time_remaining
                
                optimal_descent_rate = -self.get_optimal_descent_rate(
                    current_alt, current_mach, current_fuel, method='energy'
                )
                
                climb_rate = min(-required_descent_rate, optimal_descent_rate)
            elif current_alt > 5:
                climb_rate = -10
            else:
                current_alt = 0
                climb_rate = 0
        else:  # CRUISE
            altitude_error = cruise_altitude - current_alt
            if abs(altitude_error) > 1:
                climb_rate = np.clip(altitude_error / dt, -5, 5)
            else:
                climb_rate = 0
        
        # Update altitude while respecting envelope ceiling
        current_envelope_ceiling = resolve_envelope_ceiling(current_mach, current_fuel)
        next_alt = current_alt + climb_rate * dt
        if (climb_rate > 0 and current_envelope_ceiling is not None and 
            next_alt >= current_envelope_ceiling):
            next_alt = current_alt
            climb_rate = 0
        current_alt = max(0, next_alt)
        
        # Determine target Mach based on phase
        if current_phase == 'CLIMB':
            if climb_mode == 'optimal':
                target_mach = self.get_optimal_mach(current_alt, current_fuel)
                mach_rate = (target_mach - current_mach) * 0.05
            else:  # 'max'
                best_mach_target = 0.2
                best_Ps_check = -999
                for test_mach in np.linspace(0.2, 0.7, 15):
                    test_Ps = self.perf_db.get_Ps(current_alt, test_mach, current_fuel)
                    if test_Ps > best_Ps_check:
                        best_Ps_check = test_Ps
                        best_mach_target = test_mach
                
                if V > 1 and available_Ps_for_accel > 0:
                    dV_dt = available_Ps_for_accel
                    mach_accel = dV_dt / a
                    mach_rate = mach_accel + (best_mach_target - current_mach) * 0.05
                else:
                    mach_rate = (best_mach_target - current_mach) * 0.1
        elif current_phase == 'CRUISE':
            if use_optimal_cruise:
                target_mach = self.get_optimal_mach(current_alt, current_fuel)
            else:
                target_mach = cruise_mach
            mach_rate = (target_mach - current_mach) * 0.1
        else:  # DESCENT
            if current_alt < 1000:
                target_mach = 0.2
            else:
                target_mach = self.get_optimal_mach(current_alt, current_fuel)
            mach_rate = (target_mach - current_mach) * 0.05
        
        current_mach += mach_rate * dt
        current_mach = np.clip(current_mach, 0.15, 0.9)
        
        # Update fuel
        current_fuel -= current_ff * dt
        current_fuel = max(current_fuel, 0)
        
        current_x = current_state['x'] + V*np.sin(leg_params['heading'])*dt
        current_y = current_state['y'] + V*np.cos(leg_params['heading'])*dt

        # Interpolate position from Dubins path
        # idx = np.searchsorted(dubins_distances, current_leg_distance)
        # if idx >= len(dubins['x']):
        #     idx = len(dubins['x']) - 1
        
        # if idx == 0:
        #     current_x = dubins['x'][0]
        #     current_y = dubins['y'][0]
        #     current_heading = dubins['heading'][0]
        # elif idx >= len(dubins['x']):
        #     current_x = dubins['x'][-1]
        #     current_y = dubins['y'][-1]
        #     current_heading = dubins['heading'][-1]
        # else:
        #     dist_frac = dubins_distances[idx] - dubins_distances[idx-1]
        #     if dist_frac > 0:
        #         t_interp = (current_leg_distance - dubins_distances[idx-1]) / dist_frac
        #         t_interp = np.clip(t_interp, 0, 1)
        #         current_x = dubins['x'][idx-1] + t_interp * (dubins['x'][idx] - dubins['x'][idx-1])
        #         current_y = dubins['y'][idx-1] + t_interp * (dubins['y'][idx] - dubins['y'][idx-1])
        #         current_heading = dubins['heading'][idx-1] + t_interp * (dubins['heading'][idx] - dubins['heading'][idx-1])
        #     else:
        #         current_x = dubins['x'][idx]
        #         current_y = dubins['y'][idx]
        #         current_heading = dubins['heading'][idx]
        
        # Calculate flight path angle
        if V > 0.1:
            gamma_rad = np.arctan2(climb_rate, V)
            gamma_deg = np.degrees(gamma_rad)
        else:
            gamma_deg = 0
        
        # Update distance traveled
        V_horizontal = np.sqrt(max(V**2 - climb_rate**2, 0))
        current_leg_distance += V_horizontal * dt
        
        # Package new state
        new_state = {
            'alt': current_alt,
            'mach': current_mach,
            'velocity': V,
            'fuel': current_fuel,
            'leg_distance': current_leg_distance,
            'x': current_x,
            'y': current_y,
            'heading': leg_params['heading']
        }
        
        # Package phase info
        phase_info = {
            'phase': current_phase,
            'climb_rate': climb_rate,
            'gamma_deg': gamma_deg,
            'Ps': current_Ps,
            'ff': current_ff
        }
        
        return new_state, phase_info

    def get_optimal_descent_rate(self, altitude, mach, fuel_mass, method='energy'):
        """
        Calculate optimal descent rate for fuel-efficient descent
        
        Parameters:
        -----------
        altitude : float - Altitude [m]
        mach : float - Current Mach number
        fuel_mass : float - Fuel mass [kg]
        method : str - 'energy' (energy management) or 'L/D' (geometric)
        
        Returns:
        --------
        descent_rate : float - Optimal descent rate [m/s] (negative = descending)
        """
        T, a, P, rho = self._atmosphere(altitude)
        V = mach * a
        
        if method == 'energy':
            # Energy Management Approach (BETTER - physically accurate)
            # Maintain OptimalMach during descent using gravity
            # This accounts for energy state (potential + kinetic)
            
            # Get optimal Mach at current altitude
            optimal_mach = self.get_optimal_mach(altitude, fuel_mass)
            
            # Calculate required flight path angle to maintain optimal Mach
            # Energy equation: dE/dt = -D*V + T*V - m*g*V_z
            # For steady descent at optimal Mach: dE/dt ≈ 0 (energy neutral)
            # This means: D ≈ T - m*g*sin(gamma)
            
            # Get L/D at optimal Mach (for drag calculation)
            L_D_optimal = self.perf_db.get_performance(altitude, optimal_mach, fuel_mass, 'L2D')
            
            # At optimal Mach with minimal thrust (idle descent):
            # Flight path angle: sin(gamma) ≈ 1/L_D
            # Descent rate: V_z = V * sin(gamma)
            if L_D_optimal > 0:
                V_optimal = optimal_mach * a
                sin_gamma = 1.0 / L_D_optimal
                optimal_rate = -V_optimal * sin_gamma  # Negative = descending
            else:
                optimal_rate = -10  # Fallback
        
        else:  # method == 'L/D'
            # Geometric L/D Approach (simpler, less accurate)
            # Uses current Mach, not optimal
            L_D = self.perf_db.get_performance(altitude, mach, fuel_mass, 'L2D')
            
            # Best glide angle: tan(gamma) = 1 / L_D
            # Descent rate: V_z = V * sin(gamma) ≈ V / L_D for small angles
            if L_D > 0:
                optimal_rate = -V / L_D  # Negative = descending
            else:
                optimal_rate = -10  # Fallback
        
        # Limit to reasonable descent rates (-30 to -3 m/s)
        return max(min(optimal_rate, -3), -30)
    
    def calculate_climb_distance(self, target_altitude, cruise_mach, fuel_mass):
        """
        Calculate horizontal distance required to climb to target altitude
        
        Parameters:
        -----------
        target_altitude : float - Target altitude [m]
        cruise_mach : float - Target cruise Mach
        fuel_mass : float - Current fuel mass [kg]
        
        Returns:
        --------
        distance : float - Horizontal distance [m]
        """
        current_alt = 0
        current_mach = 0.2
        distance = 0
        dt = 1.0  # Time step [s]
        
        while current_alt < target_altitude - 5:
            # Find optimal Mach for current altitude from OptimalMach table
            optimal_mach = self.get_optimal_mach(current_alt, fuel_mass)
            
            # Get climb performance
            Ps = self.perf_db.get_Ps(current_alt, current_mach, fuel_mass)
            climb_rate = max(min(Ps, 20), 0)  # Limit to 20 m/s max
            
            if climb_rate < 0.1:
                # Can't climb anymore
                break
            
            # Atmosphere
            T, a, P, rho = self._atmosphere(current_alt)
            V = current_mach * a
            
            # Update altitude and distance
            current_alt += climb_rate * dt
            distance += V * dt
            
            # Accelerate towards optimal Mach for climb
            mach_rate = (optimal_mach - current_mach) * 0.05
            current_mach += mach_rate * dt
            current_mach = np.clip(current_mach, 0.15, 0.9)
        
        return distance
    
    def calculate_descent_distance(self, start_altitude, cruise_mach, fuel_mass):
        """
        Calculate horizontal distance required to descend from altitude
        
        Parameters:
        -----------
        start_altitude : float - Starting altitude [m]
        cruise_mach : float - Current Mach
        fuel_mass : float - Current fuel mass [kg]
        
        Returns:
        --------
        distance : float - Horizontal distance [m]
        """
        current_alt = start_altitude
        current_mach = cruise_mach
        distance = 0
        dt = 1.0  # Time step [s]
        descent_rate = 10  # m/s
        
        while current_alt > 5:
            # Atmosphere
            T, a, P, rho = self._atmosphere(current_alt)
            V = current_mach * a
            
            # Update altitude and distance
            current_alt -= descent_rate * dt
            current_alt = max(current_alt, 0)
            distance += V * dt
            
            # Decelerate gradually
            if current_alt < 1000:
                target_mach = 0.2
            else:
                target_mach = cruise_mach
            
            mach_rate = (target_mach - current_mach) * 0.05
            current_mach += mach_rate * dt
            current_mach = np.clip(current_mach, 0.15, 0.9)
        
        return distance
    
    def calculate_trajectory(self, waypoints, cruise_altitude, cruise_mach, 
                           initial_fuel, climb_mode='optimal', max_climb_factor=0.9,
                           initial_altitude=0, initial_mach=0.2, loop_mission=False):
        """
        Calculate 3D trajectory through waypoints
        
        Parameters:
        -----------
        waypoints : list of [x, y] - Waypoint positions [m]
        cruise_altitude : float - Cruise altitude [m]
        cruise_mach : float - Cruise Mach number (0 or negative = use optimal from database)
        initial_fuel : float - Initial fuel mass [kg]
        climb_mode : str - 'optimal' (climb at optimal Mach) or 'max' (maximum climb rate)
        max_climb_factor : float - Factor to apply to max climb rate (default 0.9 = 90%)
        initial_altitude : float - Starting altitude [m] (default 0)
        initial_mach : float - Starting Mach number (default 0.2)
        loop_mission : bool - If True, skip descent (stay at cruise altitude for loop)
        
        Returns:
        --------
        trajectory : dict with time histories of all states
        """
        # Initialize trajectory
        t = [0]
        x = [waypoints[0][0]]
        y = [waypoints[0][1]]
        z = [initial_altitude]  # Start at specified altitude
        heading = [0]
        
        # Calculate initial velocity from initial Mach and altitude
        T_init, a_init, P_init, rho_init = self._atmosphere(initial_altitude)
        v_init = initial_mach * a_init
        
        velocity = [v_init]
        mach = [initial_mach]
        fuel = [initial_fuel]
        fuel_flow = [0]
        Ps = [0]
        gammaV = [0]  # Flight path angle [deg]
        
        # Determine initial phase based on starting altitude
        if initial_altitude >= cruise_altitude * 0.95:
            initial_phase = 'CRUISE'
        elif initial_altitude > 0:
            initial_phase = 'CLIMB'
        else:
            initial_phase = 'TAKEOFF'
        
        phase = [initial_phase]
        
        current_fuel = initial_fuel
        current_alt = initial_altitude
        current_mach = initial_mach  # Start Mach

        # Guard against leaving the performance envelope (stay 100 m below ceiling)
        envelope_ceiling_margin = 100.0
        baseline_envelope_ceiling = None
        ceiling_guard_triggered = False
        if hasattr(self.perf_db, 'altitude_vec') and len(self.perf_db.altitude_vec) > 0:
            baseline_envelope_ceiling = float(self.perf_db.altitude_vec[-1]) - envelope_ceiling_margin

        def resolve_envelope_ceiling(mach_value, fuel_value):
            """Return the current envelope ceiling based on service data."""
            service_ceiling = None
            if hasattr(self.perf_db, 'get_service_ceiling'):
                service_ceiling = self.perf_db.get_service_ceiling(mach_value, fuel_value)
            if service_ceiling is not None and not np.isnan(service_ceiling):
                return max(0.0, service_ceiling - envelope_ceiling_margin)
            return baseline_envelope_ceiling
        
        # Determine if using optimal cruise Mach from database
        use_optimal_cruise = (cruise_mach <= 0)
        
        print(f"\n=== Trajectory Calculation ===")
        print(f"Waypoints: {len(waypoints)}")
        print(f"Cruise: {cruise_altitude}m, Mach {'optimal (from database)' if use_optimal_cruise else cruise_mach}")
        print(f"Initial state: altitude={initial_altitude}m, Mach={initial_mach}")
        print(f"Initial fuel: {initial_fuel} kg")
        
        # Calculate total horizontal distance INCLUDING turns (Dubins paths)
        print(f"\nPre-calculating total path distance...")
        total_distance = 0
        temp_headings = []
        
        # Calculate headings at waypoints 
        # Strategy: Use incoming direction to arrive tangent to each leg
        # This minimizes turning and creates optimal straight-then-turn patterns
        for i in range(len(waypoints)):
            if i == 0:
                # First waypoint: use direction to next waypoint  
                dx = waypoints[i+1][0] - waypoints[i][0]
                dy = waypoints[i+1][1] - waypoints[i][1]
                temp_headings.append(np.arctan2(dy, dx))
            else:
                # All others: use incoming direction (maintain tangency)
                dx = waypoints[i][0] - waypoints[i-1][0]
                dy = waypoints[i][1] - waypoints[i-1][1]
                temp_headings.append(np.arctan2(dy, dx))
        
        # Now calculate actual Dubins path distances
        turn_radius = 500  # Estimate for planning, actual will vary
        for i in range(len(waypoints) - 1):
            dubins_estimate = DubinsPath.calculate_path(
                waypoints[i], temp_headings[i],
                waypoints[i+1], temp_headings[i+1], turn_radius
            )
            total_distance += dubins_estimate['distance']
        
        print(f"Total path distance: {total_distance/1000:.1f} km")
        
        # Calculate climb and descent distances from performance data
        print(f"Calculating vertical profile from performance data...")
        climb_distance = self.calculate_climb_distance(cruise_altitude, cruise_mach, initial_fuel)
        
        # Descent handling: skip if loop mission
        if loop_mission:
            descent_distance = 0
            cruise_start_distance = climb_distance
            cruise_end_distance = total_distance
            print(f"Loop mission: No descent, staying at cruise altitude")
        else:
            # Descent should only happen near the final waypoint
            # Calculate distance needed for descent (cruise altitude / descent rate * cruise speed)
            T, a, P, rho = self._atmosphere(cruise_altitude)
            # Use optimal Mach if cruise_mach not specified
            if use_optimal_cruise:
                effective_cruise_mach = self.get_optimal_mach(cruise_altitude, initial_fuel)
                print(f"  Using optimal cruise Mach for descent calculation: {effective_cruise_mach:.3f}")
            else:
                effective_cruise_mach = cruise_mach
            V_cruise = effective_cruise_mach * a
            descent_rate = 10  # m/s
            descent_time = cruise_altitude / descent_rate
            descent_distance = V_cruise * descent_time  # Distance covered during descent
            
            # Make sure descent doesn't start too early
            descent_distance = min(descent_distance, total_distance * 0.2)  # Max 20% of path for descent
            
            # Verify we have enough distance
            if climb_distance + descent_distance > total_distance * 0.9:
                # Adjust if climb+descent would take too much of the path
                scale_factor = (total_distance * 0.8) / (climb_distance + descent_distance)
                climb_distance *= scale_factor
                descent_distance *= scale_factor
                print(f"  Warning: Adjusted vertical profile to fit horizontal path")
            
            # Define vertical waypoints based on calculated distances
            cruise_start_distance = climb_distance
            cruise_end_distance = total_distance - descent_distance
        
        print(f"Vertical profile (from tables):")
        print(f"  Climb: 0 - {climb_distance/1000:.1f} km (calculated from Ps)")
        print(f"  Cruise: {cruise_start_distance/1000:.1f} - {cruise_end_distance/1000:.1f} km")
        if not loop_mission:
            print(f"  Descent: {cruise_end_distance/1000:.1f} - {total_distance/1000:.1f} km ({descent_rate} m/s, distance: {descent_distance/1000:.1f} km)")
        
        accumulated_distance = 0
        
        # Store Dubins paths for visualization
        dubins_paths = []
        
        # Use the pre-calculated headings for the actual flight
        waypoint_headings = temp_headings
        
        print(f"Waypoint headings (deg): {[np.degrees(h) for h in waypoint_headings]}")
        
        print(f"\nProcessing {len(waypoints)-1} legs:")
        for i, wp in enumerate(waypoints):
            print(f"  WP{i+1}: ({wp[0]:.0f}, {wp[1]:.0f}) m, heading: {np.degrees(waypoint_headings[i]):.1f}°")
        
        # Process each leg
        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            end_wp = waypoints[i + 1]
            
            # Use pre-calculated headings at waypoints
            if i == 0:
                current_heading = waypoint_headings[0]
            else:
                # Use the ending heading from previous leg
                pass  # current_heading already updated from _process_leg
            
            # Target heading at destination waypoint
            end_heading = waypoint_headings[i + 1]
            
            # Process this leg
            current_alt, current_mach, current_fuel, current_heading, accumulated_distance = \
                self._process_leg(
                    i, len(waypoints) - 1, start_wp, end_wp,
                    current_heading, end_heading, current_alt, current_mach, current_fuel,
                    cruise_altitude, cruise_mach, accumulated_distance,
                    cruise_start_distance, cruise_end_distance, total_distance,
                    use_optimal_cruise, climb_mode, max_climb_factor,
                    resolve_envelope_ceiling, t, x, y, z, heading, velocity,
                    mach, fuel, fuel_flow, Ps, gammaV, phase, dubins_paths
                )
            
            if current_fuel <= 0:
                break
        
        # Convert to arrays and package results
        trajectory = {
            't': np.array(t),
            'x': np.array(x),
            'y': np.array(y),
            'z': np.array(z),
            'heading': np.array(heading),
            'velocity': np.array(velocity),
            'mach': np.array(mach),
            'fuel': np.array(fuel),
            'fuel_flow': np.array(fuel_flow),
            'Ps': np.array(Ps),
            'gammaV': np.array(gammaV),
            'phase': phase,
            'dubins_paths': dubins_paths
        }
        
        print(f"\n=== Trajectory Complete ===")
        print(f"Total time: {trajectory['t'][-1]/60:.1f} min")
        print(f"Total distance: {self._path_length(trajectory['x'], trajectory['y'])/1000:.1f} km")
        print(f"Fuel consumed: {initial_fuel - trajectory['fuel'][-1]:.1f} kg")
        
        return trajectory
    
    def _atmosphere(self, altitude):
        """ISA atmosphere model"""
        T0 = 288.15
        P0 = 101325
        g = 9.80665
        R = 287.05
        gamma = 1.4
        
        if altitude <= 11000:
            lapse_rate = -0.0065
            T = T0 + lapse_rate * altitude
            P = P0 * (T / T0) ** (-g / (lapse_rate * R))
        else:
            T = 216.65
            P = 22632 * np.exp(-g * (altitude - 11000) / (R * T))
        
        rho = P / (R * T)
        a = np.sqrt(gamma * R * T)
        
        return T, a, P, rho
    
    def _path_length(self, x, y):
        """Calculate path length"""
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx**2 + dy**2))
    
    def export_to_ardupilot_plan(self, trajectory, origin_lat, origin_lon, filename, 
                                  loop_mission=False, original_waypoints=None):
        """
        Export trajectory to ArduPilot PLAN format (.waypoints file)
        
        Parameters:
        -----------
        trajectory : dict
            Trajectory dictionary from integrate_trajectory()
        origin_lat : float
            Origin latitude in degrees
        origin_lon : float
            Origin longitude in degrees
        filename : str
            Output filename for .waypoints file
        loop_mission : bool
            If True, add jump command to loop back to first waypoint (horse-race pattern)
        original_waypoints : list
            List of original waypoints [[x1, y1], [x2, y2], ...] to include in plan
        """
        # Earth radius in meters
        R = 6371000.0
        
        # Convert local XY to lat/lon
        def xy_to_latlon(x, y, origin_lat, origin_lon):
            """Convert local X,Y (meters) to lat/lon using simple equirectangular projection"""
            lat = origin_lat + (y / R) * (180.0 / np.pi)
            lon = origin_lon + (x / R) * (180.0 / np.pi) / np.cos(origin_lat * np.pi / 180.0)
            return lat, lon
        
        # Get trajectory data
        x = trajectory['x']
        y = trajectory['y']
        z = trajectory['z']
        velocity = trajectory['velocity']
        mach = trajectory['mach']
        
        # Convert to lat/lon
        waypoint_commands = []
        
        # Mission item 0: Home position (MAV_CMD_NAV_WAYPOINT = 16)
        home_lat = origin_lat
        home_lon = origin_lon
        home_alt = 0.0
        waypoint_commands.append([
            0,      # seq
            1,      # current (1 for home position)
            0,      # frame (0 = MAV_FRAME_GLOBAL)
            16,     # command (16 = MAV_CMD_NAV_WAYPOINT)
            0, 0, 0, 0,  # params 1-4 (not used for waypoint)
            f"{home_lat:.7f}",
            f"{home_lon:.7f}",
            home_alt,
            1       # autocontinue
        ])
        
        # Track first waypoint for jump (if loop_mission)
        first_wp_seq = len(waypoint_commands)
        
        # Add ORIGINAL WAYPOINTS ONLY (must-have)
        # No need for intermediate sampled waypoints - just use the defined waypoints
        if original_waypoints and len(original_waypoints) > 0:
            for wp_idx, wp in enumerate(original_waypoints):
                lat, lon = xy_to_latlon(wp[0], wp[1], origin_lat, origin_lon)
                
                # Find altitude and velocity at this waypoint from trajectory
                distances_to_wp = np.sqrt((x - wp[0])**2 + (y - wp[1])**2)
                closest_idx = np.argmin(distances_to_wp)
                alt = z[closest_idx]
                vel = velocity[closest_idx]
                
                # Waypoint command (MAV_CMD_NAV_WAYPOINT = 16)
                waypoint_commands.append([
                    len(waypoint_commands),  # seq
                    0,      # current (0 for normal waypoint)
                    3,      # frame (3 = MAV_FRAME_GLOBAL_RELATIVE_ALT)
                    16,     # command (16 = MAV_CMD_NAV_WAYPOINT)
                    0, 0, 0, 0,  # params: hold_time, accept_radius, pass_radius, yaw
                    f"{lat:.7f}",
                    f"{lon:.7f}",
                    alt,
                    1       # autocontinue
                ])
                
                # Add DO_CHANGE_SPEED command for this waypoint
                waypoint_commands.append([
                    len(waypoint_commands),  # seq
                    0,      # current
                    3,      # frame
                    178,    # command (178 = MAV_CMD_DO_CHANGE_SPEED)
                    0,      # speed_type (0 = airspeed in m/s)
                    vel,    # speed value
                    -1,     # throttle (-1 = no change)
                    0,      # relative (0 = absolute)
                    f"{lat:.7f}",
                    f"{lon:.7f}",
                    alt,
                    1       # autocontinue
                ])
        
        # Add loop jump command if requested (horse-race pattern)
        if loop_mission:
            # Set constant cruise velocity for the loop
            # Use the velocity from the last waypoint (cruise speed)
            if original_waypoints and len(original_waypoints) > 0:
                last_wp = original_waypoints[-1]
                distances_to_wp = np.sqrt((x - last_wp[0])**2 + (y - last_wp[1])**2)
                closest_idx = np.argmin(distances_to_wp)
                cruise_vel = velocity[closest_idx]
                
                # Add constant speed command before jump
                waypoint_commands.append([
                    len(waypoint_commands),  # seq
                    0,      # current
                    3,      # frame
                    178,    # command (178 = MAV_CMD_DO_CHANGE_SPEED)
                    0,      # speed_type (0 = airspeed in m/s)
                    cruise_vel,  # constant cruise speed for loop
                    -1,     # throttle (-1 = no change)
                    0,      # relative (0 = absolute)
                    0, 0, 0,  # lat, lon, alt (unused for DO_CHANGE_SPEED)
                    1       # autocontinue
                ])
            
            # MAV_CMD_DO_JUMP (177): Jump to specified waypoint
            # Params: target_seq (waypoint to jump to), repeat_count (-1 for infinite)
            waypoint_commands.append([
                len(waypoint_commands),  # seq
                0,      # current
                3,      # frame
                177,    # command (177 = MAV_CMD_DO_JUMP)
                first_wp_seq,  # param1: target waypoint sequence number
                -1,     # param2: repeat count (-1 = infinite loop)
                0, 0,   # params 3-4 (unused)
                0, 0, 0,  # lat, lon, alt (unused for DO_JUMP)
                1       # autocontinue
            ])
        
        # Renumber sequences to be sequential
        for idx, cmd in enumerate(waypoint_commands):
            cmd[0] = idx
        
        # Write to .waypoints file (QGC WPL 110 format)
        with open(filename, 'w') as f:
            f.write("QGC WPL 110\n")
            for cmd in waypoint_commands:
                line_items = [
                    cmd[0],   # seq
                    cmd[1],   # current
                    cmd[2],   # frame
                    cmd[3],   # command
                    cmd[4], cmd[5], cmd[6], cmd[7],  # params 1-4
                    cmd[8],   # lat
                    cmd[9],   # lon
                    cmd[10],  # alt
                    cmd[11]   # autocontinue
                ]
                f.write("\t".join(str(x) for x in line_items) + "\n")
        
        print(f"\nArduPilot mission exported to: {filename}")
        print(f"Waypoints: {len(original_waypoints) if original_waypoints else 0}")
        print(f"Total mission items: {len(waypoint_commands)}")
        print(f"Origin: ({origin_lat:.6f}, {origin_lon:.6f})")
        print(f"Loop mission: {loop_mission}")
        if loop_mission:
            print(f"Jump target: waypoint #{first_wp_seq} (infinite loop)")
        
        # Get velocity stats from original waypoints
        if original_waypoints and len(original_waypoints) > 0:
            wp_velocities = []
            for wp in original_waypoints:
                distances_to_wp = np.sqrt((x - wp[0])**2 + (y - wp[1])**2)
                closest_idx = np.argmin(distances_to_wp)
                wp_velocities.append(velocity[closest_idx])
            print(f"Velocity range: {min(wp_velocities):.1f} - {max(wp_velocities):.1f} m/s")


class TrajectoryApp:
    """GUI Application for trajectory planning"""
    
    def __init__(self, master):
        self.master = master
        master.title("Aircraft Trajectory Integrator")
        master.geometry("1600x900")
        
        self.perf_db = None
        self.trajectory = None
        self.waypoints = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # === LEFT PANEL: Controls ===
        control_frame = ttk.LabelFrame(main_frame, text="Mission Parameters", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        row = 0
        
        # Load performance data
        ttk.Button(control_frame, text="Load Performance Data", 
                  command=self._load_performance).grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        row += 1
        
        self.perf_status = ttk.Label(control_frame, text="No data loaded", foreground="red")
        self.perf_status.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        row += 1
        
        # Flight parameters
        ttk.Label(control_frame, text="Cruise Altitude [m]:").grid(row=row, column=0, sticky=tk.W)
        self.cruise_alt = tk.DoubleVar(value=3000)
        ttk.Entry(control_frame, textvariable=self.cruise_alt, width=15).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Cruise Mach:").grid(row=row, column=0, sticky=tk.W)
        self.cruise_mach = tk.DoubleVar(value=0)
        cruise_entry = ttk.Entry(control_frame, textvariable=self.cruise_mach, width=15)
        cruise_entry.grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(control_frame, text="  (0 = use optimal)", font=('Arial', 8)).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Initial Fuel [kg]:").grid(row=row, column=0, sticky=tk.W)
        self.initial_fuel = tk.DoubleVar(value=50)
        ttk.Entry(control_frame, textvariable=self.initial_fuel, width=15).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Initial Altitude [m]:").grid(row=row, column=0, sticky=tk.W)
        self.initial_altitude = tk.DoubleVar(value=0)
        ttk.Entry(control_frame, textvariable=self.initial_altitude, width=15).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Initial Mach:").grid(row=row, column=0, sticky=tk.W)
        self.initial_mach = tk.DoubleVar(value=0.2)
        ttk.Entry(control_frame, textvariable=self.initial_mach, width=15).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Climb Mode:").grid(row=row, column=0, sticky=tk.W)
        self.climb_mode = tk.StringVar(value='optimal')
        climb_frame = ttk.Frame(control_frame)
        climb_frame.grid(row=row, column=1, sticky=tk.W)
        ttk.Radiobutton(climb_frame, text="Optimal", variable=self.climb_mode, value='optimal').pack(side=tk.LEFT)
        ttk.Radiobutton(climb_frame, text="Max", variable=self.climb_mode, value='max').pack(side=tk.LEFT, padx=(10,0))
        row += 1
        
        ttk.Label(control_frame, text="Max Climb Factor:").grid(row=row, column=0, sticky=tk.W)
        self.max_climb_factor = tk.DoubleVar(value=0.9)
        ttk.Entry(control_frame, textvariable=self.max_climb_factor, width=15).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(control_frame, text="  (fraction of max Ps)", font=('Arial', 8)).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        row += 1
        
        # Waypoint entry
        ttk.Label(control_frame, text="Waypoints (X, Y) [m]:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        # Waypoint list with scrollbar
        wp_frame = ttk.Frame(control_frame)
        wp_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        scrollbar = ttk.Scrollbar(wp_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.waypoint_list = tk.Listbox(wp_frame, height=10, yscrollcommand=scrollbar.set)
        self.waypoint_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.waypoint_list.yview)
        row += 1
        
        # Waypoint input
        wp_input_frame = ttk.Frame(control_frame)
        wp_input_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E)
        
        ttk.Label(wp_input_frame, text="X:").pack(side=tk.LEFT)
        self.wp_x = tk.DoubleVar(value=0)
        ttk.Entry(wp_input_frame, textvariable=self.wp_x, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(wp_input_frame, text="Y:").pack(side=tk.LEFT, padx=(10,0))
        self.wp_y = tk.DoubleVar(value=0)
        ttk.Entry(wp_input_frame, textvariable=self.wp_y, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(wp_input_frame, text="Add", command=self._add_waypoint).pack(side=tk.LEFT, padx=5)
        ttk.Button(wp_input_frame, text="Remove", command=self._remove_waypoint).pack(side=tk.LEFT)
        row += 1
        
        # Preset patterns
        ttk.Label(control_frame, text="Quick patterns:").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        pattern_frame = ttk.Frame(control_frame)
        pattern_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Button(pattern_frame, text="Square", command=lambda: self._load_pattern('square')).pack(side=tk.LEFT, padx=2)
        ttk.Button(pattern_frame, text="Triangle", command=lambda: self._load_pattern('triangle')).pack(side=tk.LEFT, padx=2)
        ttk.Button(pattern_frame, text="Clear", command=self._clear_waypoints).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Loop mission option
        ttk.Label(control_frame, text="Mission Type:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        self.loop_mission = tk.BooleanVar(value=False)
        loop_check = ttk.Checkbutton(control_frame, text="Loop mission (horse-race)", variable=self.loop_mission)
        loop_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        ttk.Label(control_frame, text="  Jump back to first WP for continuous loop", font=('Arial', 8)).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        row += 1
        
        # Plot X-axis selector
        ttk.Label(control_frame, text="Time Histories X-axis:").grid(row=row, column=0, sticky=tk.W)
        self.xaxis_mode = tk.StringVar(value='time')
        xaxis_frame = ttk.Frame(control_frame)
        xaxis_frame.grid(row=row, column=1, sticky=tk.W)
        ttk.Radiobutton(xaxis_frame, text="Time", variable=self.xaxis_mode, value='time', 
                       command=self._update_plots).pack(side=tk.LEFT)
        ttk.Radiobutton(xaxis_frame, text="Path Length", variable=self.xaxis_mode, value='path', 
                       command=self._update_plots).pack(side=tk.LEFT, padx=(10,0))
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        row += 1
        
        # Calculate button
        self.calc_button = ttk.Button(control_frame, text="Calculate Trajectory", 
                                      command=self._calculate_trajectory, state='disabled')
        self.calc_button.grid(row=row, column=0, columnspan=2, pady=10, sticky=tk.W+tk.E)
        row += 1
        
        # Export button
        self.export_button = ttk.Button(control_frame, text="Export Results (CSV)", 
                                       command=self._export_results, state='disabled')
        self.export_button.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        row += 1
        
        # Export to ArduPilot button
        self.export_ardupilot_button = ttk.Button(control_frame, text="Export to ArduPilot PLAN", 
                                       command=self._export_to_ardupilot, state='disabled')
        self.export_ardupilot_button.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        row += 1
        
        # Launch Flight Envelope Tool button
        self.launch_envelope_button = ttk.Button(control_frame, text="Launch Flight Envelope Tool", 
                                                command=self._launch_flight_envelope)
        self.launch_envelope_button.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        row += 1
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=row, column=0, columnspan=2, pady=10)
        
        # === RIGHT PANEL: Plots ===
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder
        placeholder = ttk.Frame(self.notebook)
        self.notebook.add(placeholder, text="Trajectory Plot")
        ttk.Label(placeholder, text="Load performance data and calculate trajectory to view plots",
                 font=('Arial', 12)).pack(expand=True)
    
    def _load_performance(self):
        """Load performance database"""
        filename = filedialog.askopenfilename(
            title="Load Performance Data",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        # Update status and disable button during loading
        self.perf_status.config(text="Loading...", foreground="orange")
        self.calc_button.config(state='disabled')
        
        # Load in background thread
        def load_thread():
            try:
                perf_db = PerformanceDatabase(filename)
                
                # Update UI on main thread
                self.master.after(0, lambda: self._on_load_success(perf_db, filename))
                
            except Exception as e:
                # Update UI on main thread
                error_msg = str(e)
                self.master.after(0, lambda: self._on_load_error(error_msg))
                import traceback
                traceback.print_exc()
        
        # Start loading thread
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def _on_load_success(self, perf_db, filename):
        """Called on main thread when load succeeds"""
        self.perf_db = perf_db
        
        # Build status message with Mach ranges
        base_name = filename.split('/')[-1].split('\\')[-1]
        status_text = f"Loaded: {base_name}"
        self.perf_status.config(text=status_text, foreground="green")
        self.calc_button.config(state='normal')
        
        # Show detailed info in message box
        mach_min = perf_db.mach_vec[0]
        mach_max = perf_db.mach_vec[-1]
        
        info_msg = f"Performance data loaded successfully!\n\n"
        info_msg += f"Altitude range: {perf_db.altitude_vec[0]:.0f} - {perf_db.altitude_vec[-1]:.0f} m\n"
        info_msg += f"Mach range: {mach_min:.3f} - {mach_max:.3f}\n"
        info_msg += f"  • Minimum Mach: {mach_min:.3f}\n"
        info_msg += f"  • Maximum Mach: {mach_max:.3f}\n"
        
        # Add optimal Mach range if available
        if hasattr(perf_db, 'optimal_mach') and not np.all(np.isnan(perf_db.optimal_mach)):
            opt_min = np.nanmin(perf_db.optimal_mach)
            opt_max = np.nanmax(perf_db.optimal_mach)
            info_msg += f"  • Optimal Mach range: {opt_min:.3f} - {opt_max:.3f}\n"
        
        info_msg += f"Fuel range: {perf_db.fuel_mass_vec[0]:.1f} - {perf_db.fuel_mass_vec[-1]:.1f} kg"
        
        messagebox.showinfo("Performance Data Loaded", info_msg)
    
    def _on_load_error(self, error_msg):
        """Called on main thread when load fails"""
        self.perf_status.config(text="Load failed", foreground="red")
        messagebox.showerror("Error", f"Failed to load performance data:\n{error_msg}")
    
    def _add_waypoint(self):
        """Add waypoint to list"""
        x = self.wp_x.get()
        y = self.wp_y.get()
        
        self.waypoints.append([x, y])
        self.waypoint_list.insert(tk.END, f"WP{len(self.waypoints)}: ({x:.0f}, {y:.0f})")
        
        # Increment for next point
        self.wp_x.set(x + 5000)
    
    def _remove_waypoint(self):
        """Remove selected waypoint"""
        selection = self.waypoint_list.curselection()
        if selection:
            idx = selection[0]
            self.waypoint_list.delete(idx)
            self.waypoints.pop(idx)
    
    def _clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints = []
        self.waypoint_list.delete(0, tk.END)
    
    def _load_pattern(self, pattern):
        """Load preset waypoint pattern"""
        self._clear_waypoints()
        
        if pattern == 'square':
            points = [[0, 0], [10000, 0], [10000, 10000], [0, 10000], [0, 0]]
        elif pattern == 'triangle':
            points = [[0, 0], [10000, 0], [5000, 8660], [0, 0]]
        else:
            return
        
        for x, y in points:
            self.waypoints.append([x, y])
            self.waypoint_list.insert(tk.END, f"WP{len(self.waypoints)}: ({x:.0f}, {y:.0f})")
    
    def _calculate_trajectory(self):
        """Calculate trajectory"""
        if self.perf_db is None:
            messagebox.showwarning("Warning", "Please load performance data first")
            return
        
        if len(self.waypoints) < 2:
            messagebox.showwarning("Warning", "Please add at least 2 waypoints")
            return
        
        try:
            self.status_label.config(text="Calculating trajectory...", foreground="orange")
            self.master.update()
            
            # Create integrator
            integrator = TrajectoryIntegrator(self.perf_db)
            
            # Calculate trajectory with loop_mission flag
            self.trajectory = integrator.calculate_trajectory(
                self.waypoints,
                self.cruise_alt.get(),
                self.cruise_mach.get(),
                self.initial_fuel.get(),
                self.climb_mode.get(),
                self.max_climb_factor.get(),
                self.initial_altitude.get(),
                self.initial_mach.get(),
                loop_mission=self.loop_mission.get()
            )
            
            # Create plots
            self._create_plots()
            
            self.status_label.config(text="Trajectory calculated!", foreground="green")
            self.export_button.config(state='normal')
            self.export_ardupilot_button.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Trajectory calculation failed:\n{str(e)}")
            self.status_label.config(text="Calculation failed", foreground="red")
            import traceback
            traceback.print_exc()
    
    def _update_plots(self):
        """Update plots when X-axis mode changes"""
        if self.trajectory is not None:
            self._create_plots()
    
    def _create_plots(self):
        """Create trajectory visualization plots"""
        # Clear existing tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        traj = self.trajectory
        
        # Calculate cumulative path length
        dx = np.diff(traj['x'], prepend=traj['x'][0])
        dy = np.diff(traj['y'], prepend=traj['y'][0])
        path_length = np.cumsum(np.sqrt(dx**2 + dy**2)) / 1000  # km
        
        # Determine X-axis data and label
        if self.xaxis_mode.get() == 'time':
            xdata = traj['t'] / 60
            xlabel = 'Time [min]'
        else:
            xdata = path_length
            xlabel = 'Path Length [km]'
        
        # === 3D Trajectory Plot ===
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="3D Trajectory")
        
        fig1 = Figure(figsize=(12, 8), dpi=100)
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Plot trajectory (time-based integration)
        ax1.plot(traj['x']/1000, traj['y']/1000, traj['z'], 'b-', linewidth=1.5, label='Trajectory', alpha=0.7)
        
        # Plot Dubins paths on ground (altitude = 0) with distinct color
        for idx, dubins in enumerate(traj['dubins_paths']):
            label = 'Dubins Path' if idx == 0 else None
            ax1.plot(dubins['x']/1000, dubins['y']/1000, 0, 'r-', linewidth=3, alpha=0.9, label=label, zorder=5)
        
        # Plot waypoints
        wp_array = np.array(self.waypoints)
        ax1.scatter(wp_array[:, 0]/1000, wp_array[:, 1]/1000, 0, 
                   c='red', s=100, marker='o', label='Waypoints', zorder=10)
        
        # Label waypoints
        for i, wp in enumerate(self.waypoints):
            ax1.text(wp[0]/1000, wp[1]/1000, 0, f'  WP{i+1}', fontsize=9)
        
        ax1.set_xlabel('X [km]')
        ax1.set_ylabel('Y [km]')
        ax1.set_zlabel('Altitude [m]')
        ax1.set_title('3D Flight Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create frame for proper toolbar positioning
        toolbar_frame1 = ttk.Frame(tab1)
        toolbar_frame1.pack(side=tk.TOP, fill=tk.X)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar1 = NavigationToolbar2Tk(canvas1, toolbar_frame1)
        toolbar1.update()
        
        # === Time Histories ===
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Time Histories")
        
        fig2 = Figure(figsize=(12, 10), dpi=100)
        
        # Altitude (top left) - this is the master subplot for x-axis sharing
        ax2_1 = fig2.add_subplot(3, 2, 1)
        ax2_1.plot(xdata, traj['z'], 'b-', linewidth=2)
        ax2_1.set_ylabel('Altitude [m]')
        ax2_1.set_title('Altitude Profile')
        ax2_1.grid(True, alpha=0.3)
        ax2_1.tick_params(labelbottom=False)
        
        # Mach (top right) - share x-axis with ax2_1
        ax2_2 = fig2.add_subplot(3, 2, 2, sharex=ax2_1)
        ax2_2.plot(xdata, traj['mach'], 'g-', linewidth=2)
        ax2_2.set_title('Mach Number')
        ax2_2.grid(True, alpha=0.3)
        ax2_2.tick_params(labelbottom=False)
        ax2_2.yaxis.tick_right()
        ax2_2.yaxis.set_label_position('right')
        ax2_2.set_ylabel('Mach')
        
        # Fuel (middle left) - share x-axis with ax2_1
        ax2_3 = fig2.add_subplot(3, 2, 3, sharex=ax2_1)
        ax2_3.plot(xdata, traj['fuel'], 'r-', linewidth=2)
        ax2_3.set_ylabel('Fuel [kg]')
        ax2_3.set_title('Fuel Remaining')
        ax2_3.grid(True, alpha=0.3)
        ax2_3.tick_params(labelbottom=False)
        
        # Fuel Flow (middle right) - share x-axis with ax2_1
        ax2_4 = fig2.add_subplot(3, 2, 4, sharex=ax2_1)
        ax2_4.plot(xdata, traj['fuel_flow'], 'm-', linewidth=2)
        ax2_4.set_title('Fuel Flow Rate')
        ax2_4.grid(True, alpha=0.3)
        ax2_4.tick_params(labelbottom=False)
        ax2_4.yaxis.tick_right()
        ax2_4.yaxis.set_label_position('right')
        ax2_4.set_ylabel('Fuel Flow [kg/s]')
        
        # Specific Excess Power (bottom left) - share x-axis with ax2_1
        ax2_5 = fig2.add_subplot(3, 2, 5, sharex=ax2_1)
        ax2_5.plot(xdata, traj['Ps'], 'c-', linewidth=2)
        ax2_5.set_xlabel(xlabel)
        ax2_5.set_ylabel('Ps [m/s]')
        ax2_5.set_title('Specific Excess Power')
        ax2_5.grid(True, alpha=0.3)
        
        # Flight Path Angle (bottom right) - share x-axis with ax2_1
        ax2_6 = fig2.add_subplot(3, 2, 6, sharex=ax2_1)
        ax2_6.plot(xdata, traj['gammaV'], 'b-', linewidth=2)
        ax2_6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Color-code by phase
        phase_colors = {'CLIMB': 'green', 'CRUISE': 'blue', 'DESCENT': 'orange', 'TAKEOFF': 'purple'}
        for phase_name, color in phase_colors.items():
            phase_mask = [p == phase_name for p in traj['phase']]
            if any(phase_mask):
                phase_xdata = xdata[phase_mask]
                phase_gamma = traj['gammaV'][phase_mask]
                ax2_6.scatter(phase_xdata, phase_gamma, c=color, s=10, alpha=0.5, label=phase_name)
        
        ax2_6.set_xlabel(xlabel)
        ax2_6.set_title('Flight Path Angle')
        ax2_6.legend()
        ax2_6.grid(True, alpha=0.3)
        ax2_6.yaxis.tick_right()
        ax2_6.yaxis.set_label_position('right')
        ax2_6.set_ylabel('Flight Path Angle γ (deg)')
        
        fig2.tight_layout()
        
        # Create frame for proper toolbar positioning
        toolbar_frame = ttk.Frame(tab2)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar2 = NavigationToolbar2Tk(canvas2, toolbar_frame)
        toolbar2.update()
        
        # Add hover data tips for time histories - create annotation for each subplot
        annots = {}
        for ax in [ax2_1, ax2_2, ax2_3, ax2_4, ax2_5, ax2_6]:
            annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                               arrowprops=dict(arrowstyle="->"),
                               fontsize=9, zorder=100)
            annot.set_visible(False)
            annots[ax] = annot
        
        def update_annot(ax, annot, line, ind, x_label, y_label):
            """Update annotation with data values and smart positioning"""
            x, y = line.get_data()
            idx = ind["ind"][0]
            annot.xy = (x[idx], y[idx])
            text = f"{x_label}: {x[idx]:.2f}\n{y_label}: {y[idx]:.2f}"
            annot.set_text(text)
            
            # Smart positioning: choose quadrant based on position in plot
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # Determine position relative to plot center
            x_pos = (x[idx] - xlim[0]) / x_range  # 0 to 1
            y_pos = (y[idx] - ylim[0]) / y_range  # 0 to 1
            
            # Choose offset direction to keep annotation visible
            if x_pos > 0.7:  # Right side
                x_offset = -80
            else:  # Left or center
                x_offset = 20
            
            if y_pos > 0.7:  # Top
                y_offset = -40
            else:  # Bottom or center
                y_offset = 20
            
            annot.set_position((x_offset, y_offset))
        
        def hover2(event):
            """Show data tip on hover"""
            if event.inaxes:
                # Map axes to their line and label info
                ax_info = {
                    ax2_1: (ax2_1.lines[0], 'Altitude [m]'),
                    ax2_2: (ax2_2.lines[0], 'Mach'),
                    ax2_3: (ax2_3.lines[0], 'Fuel [kg]'),
                    ax2_4: (ax2_4.lines[0], 'Fuel Flow [kg/s]'),
                    ax2_5: (ax2_5.lines[0], 'Ps [m/s]'),
                    ax2_6: (ax2_6.lines[0], 'γ [deg]')
                }
                
                ax = event.inaxes
                if ax in ax_info:
                    line, y_label = ax_info[ax]
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(ax, annots[ax], line, ind, xlabel, y_label)
                        annots[ax].set_visible(True)
                        fig2.canvas.draw_idle()
                        return
            
            # Hide all annotations if not hovering
            for annot in annots.values():
                if annot.get_visible():
                    annot.set_visible(False)
                    fig2.canvas.draw_idle()
        
        fig2.canvas.mpl_connect("motion_notify_event", hover2)
    
    def _export_results(self):
        """Export trajectory results to CSV"""
        if self.trajectory is None:
            messagebox.showwarning("Warning", "No trajectory to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Trajectory Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'Time_s': self.trajectory['t'],
                'X_m': self.trajectory['x'],
                'Y_m': self.trajectory['y'],
                'Altitude_m': self.trajectory['z'],
                'Heading_rad': self.trajectory['heading'],
                'Velocity_ms': self.trajectory['velocity'],
                'Mach': self.trajectory['mach'],
                'Fuel_kg': self.trajectory['fuel'],
                'FuelFlow_kgs': self.trajectory['fuel_flow'],
                'Ps_ms': self.trajectory['Ps'],
                'Phase': self.trajectory['phase']
            })
            
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Trajectory exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _export_to_ardupilot(self):
        """Export trajectory to ArduPilot PLAN format"""
        if self.trajectory is None:
            messagebox.showwarning("Warning", "No trajectory to export")
            return
        
        if self.perf_db is None:
            messagebox.showwarning("Warning", "No performance database loaded")
            return
        
        # Create dialog for origin selection
        dialog = tk.Toplevel(self.master)
        dialog.title("ArduPilot PLAN Export")
        dialog.geometry("400x200")
        dialog.transient(self.master)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Set Mission Origin", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0,15))
        
        # Latitude input
        ttk.Label(frame, text="Origin Latitude (deg):").grid(row=1, column=0, sticky=tk.W, pady=5)
        lat_var = tk.DoubleVar(value=32.0)
        lat_entry = ttk.Entry(frame, textvariable=lat_var, width=20)
        lat_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Longitude input
        ttk.Label(frame, text="Origin Longitude (deg):").grid(row=2, column=0, sticky=tk.W, pady=5)
        lon_var = tk.DoubleVar(value=35.0)
        lon_entry = ttk.Entry(frame, textvariable=lon_var, width=20)
        lon_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        result = {'confirmed': False}
        
        def on_ok():
            result['confirmed'] = True
            result['lat'] = lat_var.get()
            result['lon'] = lon_var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20,0))
        
        ttk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        self.master.wait_window(dialog)
        
        if not result['confirmed']:
            return
        
        # Get file location
        filename = filedialog.asksaveasfilename(
            title="Save ArduPilot Mission",
            defaultextension=".waypoints",
            filetypes=[("Waypoint files", "*.waypoints"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Create trajectory integrator instance to access export method
            integrator = TrajectoryIntegrator(perf_db=self.perf_db)
            
            # Export to ArduPilot format
            integrator.export_to_ardupilot_plan(
                self.trajectory,
                result['lat'],
                result['lon'],
                filename,
                loop_mission=self.loop_mission.get(),
                original_waypoints=self.waypoints
            )
            
            messagebox.showinfo(
                "Success", 
                f"Mission exported to:\n{filename}\n\n"
                f"Origin: ({result['lat']:.6f}, {result['lon']:.6f})\n"
                f"You can now load this mission in Mission Planner or QGroundControl\n"
                f"and test it in ArduPilot SITL."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _launch_flight_envelope(self):
        """Launch the Flight Envelope tool"""
        import subprocess
        import sys
        import os
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'FlightEnvelops.py')
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
                self.status_label.config(text="Launched Flight Envelope Tool", foreground="green")
            else:
                messagebox.showerror("Error", f"FlightEnvelops.py not found at:\n{script_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Flight Envelope Tool:\n{str(e)}")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
