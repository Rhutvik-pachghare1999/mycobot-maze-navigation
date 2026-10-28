# Stage 2: Coordinate Mapper
# Author: Rhutvik

"""Maps maze waypoints to robot workspace coordinates using bilinear interpolation.

Based on original map_coordinates_to_robot function.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import robot, maze


@dataclass
class CoordinateMapping:
    """Result of coordinate mapping."""
    maze_waypoints: List[Tuple[float, float]]  # Normalized maze coords
    robot_waypoints: List[Tuple[float, float, float]]  # Robot coords (mm)
    calibration_matrix: np.ndarray  # Transformation matrix


class CoordinateMapper:
    """Maps normalized maze coordinates to robot workspace coordinates."""
    
    def __init__(self, corners: List[Tuple[float, float, float]] = None,
                 maze_dims: Tuple[int, int] = None):
        """
        Args:
            corners: 4 robot coordinates [top-left, top-right, bottom-left, bottom-right] in mm
            maze_dims: (width, height) of logical maze grid
        """
        if corners is None:
            corners = robot.calibration_corners
        if maze_dims is None:
            maze_dims = (maze.grid_width, maze.grid_height)
        
        self.corners = np.array(corners)  # (4, 3)
        self.maze_dims = maze_dims
        
        # Validate corners
        if self.corners.shape != (4, 3):
            raise ValueError(f"Expected 4 corners with 3 coords each, got {self.corners.shape}")
        
        # Precompute bilinear interpolation coefficients
        self._compute_interpolation_coeffs()
    
    def _compute_interpolation_coeffs(self):
        """Compute bilinear interpolation coefficients."""
        # Corners order: [top-left, top-right, bottom-left, bottom-right]
        tl, tr, bl, br = self.corners
        
        # For bilinear interpolation:
        # x_robot = a0 + a1*u + a2*v + a3*u*v
        # y_robot = b0 + b1*u + b2*v + b3*u*v
        # z_robot = c0 + c1*u + c2*v + c3*u*v
        # where u = x_maze / (W-1), v = y_maze / (H-1)
        
        W, H = self.maze_dims
        u_coords = np.array([0, 1, 0, 1])  # tl, tr, bl, br
        v_coords = np.array([0, 0, 1, 1])
        
        # Solve for coefficients using 4 corner points
        A = np.column_stack([np.ones(4), u_coords, v_coords, u_coords * v_coords])
        
        self.coeffs_x = np.linalg.solve(A, self.corners[:, 0])
        self.coeffs_y = np.linalg.solve(A, self.corners[:, 1])
        self.coeffs_z = np.linalg.solve(A, self.corners[:, 2])
    
    def map_point(self, maze_point: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Map single maze point to robot coordinates.
        
        Args:
            maze_point: (x, y) in maze grid coordinates (0 to W, 0 to H)
            
        Returns:
            (x, y, z) in robot coordinates (mm)
        """
        x_maze, y_maze = maze_point
        W, H = self.maze_dims
        
        # Normalize to [0, 1]
        u = x_maze / max(W - 1, 1)
        v = y_maze / max(H - 1, 1)
        
        # Bilinear interpolation
        x_robot = (self.coeffs_x[0] + self.coeffs_x[1]*u + 
                   self.coeffs_x[2]*v + self.coeffs_x[3]*u*v)
        y_robot = (self.coeffs_y[0] + self.coeffs_y[1]*u + 
                   self.coeffs_y[2]*v + self.coeffs_y[3]*u*v)
        z_robot = (self.coeffs_z[0] + self.coeffs_z[1]*u + 
                   self.coeffs_z[2]*v + self.coeffs_z[3]*u*v)
        
        return (float(x_robot), float(y_robot), float(z_robot))
    
    def map_waypoints(self, waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """Map list of maze waypoints to robot coordinates."""
        return [self.map_point(wp) for wp in waypoints]
    
    def create_mapping(self, waypoints: List[Tuple[float, float]]) -> CoordinateMapping:
        """Create full mapping result."""
        robot_wps = self.map_waypoints(waypoints)
        
        # Create transformation matrix for reference
        W, H = self.maze_dims
        A = np.column_stack([np.ones(4), [0,1,0,1], [0,0,1,1], [0,0,0,1]])
        T = np.column_stack([self.coeffs_x, self.coeffs_y, self.coeffs_z])  # (4, 3)
        
        return CoordinateMapping(
            maze_waypoints=waypoints,
            robot_waypoints=robot_wps,
            calibration_matrix=T
        )
    
    def save_trajectory_csv(self, robot_waypoints: List[Tuple[float, float, float]], 
                            filename: str = None):
        """Save robot trajectory to CSV (x,y,z in mm)."""
        if filename is None:
            filename = f"{robot.__dict__.get('output_dir', 'output')}/robot_trajectory.csv"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        np.savetxt(filename, robot_waypoints, delimiter=',', fmt='%.3f')
        print(f"Saved {len(robot_waypoints)} waypoints to {filename}")
        
        return filename
    
    def visualize_mapping(self, maze_waypoints: List[Tuple[float, float]], 
                          robot_waypoints: List[Tuple[float, float, float]]):
        """Visualize the coordinate mapping."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Maze coordinates
        maze_pts = np.array(maze_waypoints)
        axes[0].plot(maze_pts[:, 0], maze_pts[:, 1], 'b.-', markersize=4)
        axes[0].scatter(maze_pts[0, 0], maze_pts[0, 1], c='green', s=100, marker='o', label='Start')
        axes[0].scatter(maze_pts[-1, 0], maze_pts[-1, 1], c='red', s=100, marker='*', label='End')
        axes[0].set_title('Maze Coordinates (Normalized)')
        axes[0].set_xlabel('X (grid)')
        axes[0].set_ylabel('Y (grid)')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_xlim(-0.5, self.maze_dims[0] + 0.5)
        axes[0].set_ylim(-0.5, self.maze_dims[1] + 0.5)
        
        # Robot coordinates
        robot_pts = np.array(robot_waypoints)
        axes[1].plot(robot_pts[:, 0], robot_pts[:, 1], 'r.-', markersize=4)
        axes[1].scatter(robot_pts[0, 0], robot_pts[0, 1], c='green', s=100, marker='o', label='Start')
        axes[1].scatter(robot_pts[-1, 0], robot_pts[-1, 1], c='red', s=100, marker='*', label='End')
        
        # Plot calibration corners
        axes[1].scatter(self.corners[:, 0], self.corners[:, 1], c='blue', s=150, marker='s', label='Calibration')
        for i, (x, y, z) in enumerate(self.corners):
            axes[1].annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        axes[1].set_title('Robot Workspace Coordinates (mm)')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Y (mm)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{robot.__dict__.get('output_dir', 'output')}/coordinate_mapping.png", dpi=150)
        plt.show()
        
        return fig