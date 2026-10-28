# MyCobot Maze Navigation - Configuration
# Author: Rhutvik

"""Central configuration for the maze navigation pipeline."""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class CameraConfig:
    """Camera settings for maze capture."""
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class MazeConfig:
    """Maze solving parameters."""
    # Maze logical dimensions (grid cells)
    grid_width: int = 6
    grid_height: int = 6
    
    # Waypoint extraction
    max_waypoints: int = 50
    min_path_pixels: int = 50
    
    # Image processing
    gaussian_blur_ksize: Tuple[int, int] = (5, 5)
    morph_kernel_size: int = 10
    morph_dilate_iterations: int = 11
    morph_erode_iterations: int = 1


@dataclass
class RobotConfig:
    """MyCobot Pro 600 parameters."""
    # Workspace calibration: 4 corners in robot coordinates (mm)
    # Order: [top-left, top-right, bottom-left, bottom-right]
    # These should be measured for your specific setup
    calibration_corners: List[Tuple[float, float, float]] = None
    
    # Robot base offset from maze coordinate origin (mm)
    base_offset_x: float = 0.0
    base_offset_y: float = 0.0
    base_offset_z: float = 0.0
    
    # End-effector offset from flange (mm)
    ee_offset_z: float = 100.0  # Tool length
    
    # Joint limits (degrees) - MyCobot Pro 600
    joint_limits_deg: List[Tuple[float, float]] = None
    
    # Home position (degrees)
    home_position_deg: List[float] = None
    
    def __post_init__(self):
        if self.calibration_corners is None:
            # Default example corners (must be calibrated for real use)
            self.calibration_corners = [
                (409.0, 270.0, 230.0),  # top-left
                (267.0, 270.0, 230.0),  # top-right
                (409.0, 414.0, 230.0),  # bottom-left
                (267.0, 414.0, 230.0),  # bottom-right
            ]
        
        if self.joint_limits_deg is None:
            self.joint_limits_deg = [
                (-175, 175),   # J1
                (-110, 110),   # J2
                (-170, 170),   # J3
                (-185, 185),   # J4
                (-120, 120),   # J5
                (-360, 360),   # J6
            ]
        
        if self.home_position_deg is None:
            self.home_position_deg = [0.0, -45.0, 90.0, -45.0, 90.0, 0.0]


@dataclass
class IKConfig:
    """Inverse kinematics solver parameters."""
    # Position/orientation weights for IK
    # [x, y, z, rx, ry, rz] - prioritize position over orientation
    weights: List[float] = None
    
    # Solver settings
    max_iterations: int = 100
    position_tolerance: float = 1e-4  # meters
    orientation_tolerance: float = 1e-3  # radians
    
    # Joint velocity limits (deg/s) for smoothing
    max_joint_vel_deg_s: float = 60.0
    
    # Offset adjustments (matching original MATLAB code)
    joint_offsets_deg: List[float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [0.25, 0.25, 0.25, 1.0, 1.0, 1.0]
        
        if self.joint_offsets_deg is None:
            # Original MATLAB adjustments:
            # J1: +180, J2: -90, J4: -90, J5: *(-1)
            self.joint_offsets_deg = [180.0, -90.0, 0.0, -90.0, -1.0, 0.0]


@dataclass
class TCPConfig:
    """TCP communication settings (for real robot)."""
    server_ip: str = "192.168.1.159"
    server_port: int = 5001
    timeout: float = 5.0
    command_delay: float = 0.5  # seconds between commands


@dataclass
class SimConfig:
    """Simulation settings."""
    dt: float = 0.01  # Simulation timestep
    record_video: bool = False
    video_fps: int = 30
    output_dir: str = "output"
    log_dir: str = "logs"


@dataclass
class VizConfig:
    """Visualization settings."""
    show_maze_overlay: bool = True
    show_robot_animation: bool = True
    show_coordinate_frames: bool = True
    trajectory_point_size: int = 50
    robot_color: str = 'blue'
    path_color: str = 'green'
    waypoint_color: str = 'red'


# Global instances
camera = CameraConfig()
maze = MazeConfig()
robot = RobotConfig()
ik = IKConfig()
tcp = TCPConfig()
sim = SimConfig()
viz = VizConfig()