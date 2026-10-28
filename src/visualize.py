# Visualization for MyCobot Maze Navigation
# Author: Rhutvik

"""3D visualization of robot executing maze trajectory."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Optional
import os
from config import viz, sim, robot
from robot_model import create_robot_model


class RobotVisualizer:
    """3D visualization of MyCobot Pro 600."""
    
    def __init__(self):
        self.robot = create_robot_model()
        self.fig = None
        self.ax = None
        self.robot_links = []
        self.trajectory_line = None
        self.waypoint_scatter = None
        self.ee_marker = None
        
        # Link colors
        self.link_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    
    def create_figure(self):
        """Create 3D figure."""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('MyCobot Pro 600 - Maze Navigation')
        
        # Set equal aspect (approximate)
        self.ax.set_box_aspect([1, 1, 0.8])
        
        return self.fig
    
    def draw_robot(self, joint_angles_deg: np.ndarray, alpha: float = 0.8):
        """Draw robot at given joint configuration."""
        # Clear previous robot
        for link in self.robot_links:
            link.remove()
        self.robot_links = []
        
        if self.ee_marker:
            self.ee_marker.remove()
        
        # Convert to radians
        q = np.deg2rad(joint_angles_deg)
        
        # Get link positions via forward kinematics
        # Compute transform for each joint
        T = np.eye(4)
        positions = [T[:3, 3] * 1000]  # Base at origin (mm)
        
        for i, (theta, d, a, alpha_dh) in enumerate(self.robot.dh):
            q_i = q[i] + np.deg2rad(theta)
            T_i = self.robot.dh_transform(q_i, d, a, np.deg2rad(alpha_dh))
            T = T @ T_i
            positions.append(T[:3, 3] * 1000)
        
        # Add end-effector offset
        ee_T = np.eye(4)
        ee_T[2, 3] = robot.ee_offset_z
        T_ee = T @ ee_T
        positions.append(T_ee[:3, 3] * 1000)
        
        positions = np.array(positions)
        
        # Draw links
        for i in range(len(positions) - 1):
            link = self.ax.plot(
                [positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                [positions[i, 2], positions[i+1, 2]],
                color=self.link_colors[i % len(self.link_colors)],
                linewidth=4, alpha=alpha
            )[0]
            self.robot_links.append(link)
        
        # Draw joints
        joints = self.ax.scatter(
            positions[:-1, 0], positions[:-1, 1], positions[:-1, 2],
            c='black', s=60, alpha=alpha, depthshade=True
        )
        self.robot_links.append(joints)
        
        # Draw end-effector
        self.ee_marker = self.ax.scatter(
            [positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
            c='red', s=150, marker='*', alpha=1.0, depthshade=True,
            label='End-Effector'
        )
        
        return positions[-1]
    
    def draw_trajectory(self, trajectory_mm: np.ndarray, 
                       waypoints_mm: np.ndarray = None):
        """Draw planned trajectory and waypoints."""
        if trajectory_mm is not None and len(trajectory_mm) > 0:
            traj = trajectory_mm
            self.trajectory_line = self.ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                'g--', alpha=0.5, linewidth=2, label='Planned Path'
            )[0]
        
        if waypoints_mm is not None and len(waypoints_mm) > 0:
            self.waypoint_scatter = self.ax.scatter(
                waypoints_mm[:, 0], waypoints_mm[:, 1], waypoints_mm[:, 2],
                c='red', s=viz.trajectory_point_size, marker='o', alpha=0.8,
                label='Waypoints', depthshade=True
            )
            
            # Mark start and end
            self.ax.scatter([waypoints_mm[0, 0]], [waypoints_mm[0, 1]], [waypoints_mm[0, 2]],
                           c='green', s=200, marker='o', label='Start')
            self.ax.scatter([waypoints_mm[-1, 0]], [waypoints_mm[-1, 1]], [waypoints_mm[-1, 2]],
                           c='red', s=200, marker='*', label='End')
        
        # Draw calibration corners
        if hasattr(robot, 'calibration_corners') and robot.calibration_corners:
            corners = np.array(robot.calibration_corners)
            self.ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
                           c='blue', s=150, marker='s', alpha=0.6, label='Calibration')
            for i, (x, y, z) in enumerate(corners):
                self.ax.text(x, y, z, f'  C{i}', fontsize=8)
    
    def draw_workspace_bounds(self):
        """Draw robot workspace bounds."""
        # Approximate workspace as sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        r = 600  # Approximate reach in mm
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, alpha=0.03, color='gray', rstride=2, cstride=2)
    
    def set_limits(self, trajectory_mm: np.ndarray = None):
        """Set axis limits based on trajectory."""
        if trajectory_mm is not None and len(trajectory_mm) > 0:
            margin = 100
            self.ax.set_xlim(trajectory_mm[:, 0].min() - margin, trajectory_mm[:, 0].max() + margin)
            self.ax.set_ylim(trajectory_mm[:, 1].min() - margin, trajectory_mm[:, 1].max() + margin)
            self.ax.set_zlim(0, trajectory_mm[:, 2].max() + margin)
        else:
            self.ax.set_xlim(-600, 600)
            self.ax.set_ylim(-600, 600)
            self.ax.set_zlim(0, 600)
    
    def animate_trajectory(self, joint_trajectory_deg: np.ndarray,
                          trajectory_mm: np.ndarray = None,
                          waypoints_mm: np.ndarray = None,
                          interval: int = 200,
                          save_path: str = None):
        """Animate robot executing trajectory."""
        self.create_figure()
        
        # Draw static elements
        if trajectory_mm is not None:
            self.draw_trajectory(trajectory_mm, waypoints_mm)
        self.draw_workspace_bounds()
        self.set_limits(trajectory_mm)
        self.ax.legend()
        
        def animate(frame):
            if frame < len(joint_trajectory_deg):
                ee_pos = self.draw_robot(joint_trajectory_deg[frame])
                self.fig.suptitle(f'Waypoint {frame+1}/{len(joint_trajectory_deg)} | '
                                 f'EE: ({ee_pos[0]:.0f}, {ee_pos[1]:.0f}, {ee_pos[2]:.0f}) mm',
                                 fontsize=12)
        
        anim = animation.FuncAnimation(
            self.fig, animate, frames=len(joint_trajectory_deg),
            interval=interval, repeat=False, blit=False
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            print("Done!")
        else:
            plt.show()
        
        return anim
    
    def plot_static_comparison(self, planned_mm: np.ndarray, 
                               executed_mm: np.ndarray = None):
        """Plot planned vs executed trajectory."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
        
        for ax, title, color in zip(axes, 
                                    ['Planned', 'Executed', 'Comparison'],
                                    ['green', 'blue', 'black']):
            if title == 'Planned' or title == 'Comparison':
                ax.plot(planned_mm[:, 0], planned_mm[:, 1], planned_mm[:, 2],
                       'g--', alpha=0.5, linewidth=2, label='Planned')
                ax.scatter(planned_mm[:, 0], planned_mm[:, 1], planned_mm[:, 2],
                          c='green', s=20, alpha=0.5)
            
            if title == 'Executed' or title == 'Comparison':
                if executed_mm is not None:
                    ax.plot(executed_mm[:, 0], executed_mm[:, 1], executed_mm[:, 2],
                           'b-', alpha=0.7, linewidth=2, label='Executed')
                    ax.scatter(executed_mm[:, 0], executed_mm[:, 1], executed_mm[:, 2],
                              c='blue', s=20, alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{sim.output_dir}/trajectory_comparison.png", dpi=150)
        plt.show()


def create_sample_maze_image(output_path: str = "assets/sample_maze.png"):
    """Create a sample maze image for testing."""
    import cv2
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create 6x6 grid maze
    cell_size = 80
    grid_w, grid_h = 6, 6
    img_w, img_h = grid_w * cell_size, grid_h * cell_size
    
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Draw grid lines
    for i in range(grid_w + 1):
        cv2.line(img, (i * cell_size, 0), (i * cell_size, img_h), (200, 200, 200), 2)
    for j in range(grid_h + 1):
        cv2.line(img, (0, j * cell_size), (img_w, j * cell_size), (200, 200, 200), 2)
    
    # Draw maze walls (black) - simple maze pattern
    walls = [
        # Vertical walls
        ((1, 0), (1, 2)), ((2, 2), (2, 4)), ((3, 1), (3, 5)),
        ((4, 0), (4, 3)), ((5, 3), (5, 6)),
        # Horizontal walls
        ((0, 1), (2, 1)), ((2, 3), (4, 3)), ((3, 4), (5, 4)),
        ((1, 5), (4, 5)),
    ]
    
    for (x1, y1), (x2, y2) in walls:
        if x1 == x2:  # Vertical
            cv2.line(img, (x1 * cell_size, y1 * cell_size), 
                    (x1 * cell_size, y2 * cell_size), (0, 0, 0), 6)
        else:  # Horizontal
            cv2.line(img, (x1 * cell_size, y1 * cell_size), 
                    (x2 * cell_size, y1 * cell_size), (0, 0, 0), 6)
    
    # Add path (solution) - lighter gray for visibility
    path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), 
            (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
    
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        cx1, cy1 = x1 * cell_size + cell_size//2, y1 * cell_size + cell_size//2
        cx2, cy2 = x2 * cell_size + cell_size//2, y2 * cell_size + cell_size//2
        cv2.line(img, (cx1, cy1), (cx2, cy2), (150, 150, 150), 8)
    
    cv2.imwrite(output_path, img)
    print(f"Sample maze saved to {output_path}")