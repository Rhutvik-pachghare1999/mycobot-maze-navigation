# Stage 3: Inverse Kinematics Solver
# Author: Rhutvik

"""Solves IK for each robot waypoint and exports joint angles to CSV.

Replicates the original Matlab_Code.m pipeline:
1. Load robot_trajectory.csv (XYZ waypoints in mm)
2. For each waypoint, solve IK
3. Apply joint offsets
4. Save joint_angles.csv
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import csv
import os
from config import robot, ik, sim
from robot_model import MyCobotKinematics, create_robot_model


@dataclass
class IKResult:
    """Result of IK solving for all waypoints."""
    waypoints: np.ndarray  # (N, 3) robot coordinates in mm
    joint_angles_deg: np.ndarray  # (N, 6) joint angles in degrees
    success_flags: np.ndarray  # (N,) boolean
    solver_info: List[str]


class IKSolver:
    """Inverse kinematics solver for robot trajectory."""
    
    def __init__(self):
        self.robot = create_robot_model()
        self.initial_guess = self.robot.home_pos.copy()
    
    def load_trajectory(self, filepath: str) -> np.ndarray:
        """Load robot trajectory from CSV (mm)."""
        data = np.loadtxt(filepath, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data  # (N, 3) in mm
    
    def solve_trajectory(self, trajectory_mm: np.ndarray) -> IKResult:
        """
        Solve IK for all waypoints in trajectory.
        
        Args:
            trajectory_mm: (N, 3) array of XYZ positions in mm
            
        Returns:
            IKResult with joint angles in degrees
        """
        num_wp = trajectory_mm.shape[0]
        joint_angles = np.zeros((num_wp, 6))
        success_flags = np.zeros(num_wp, dtype=bool)
        solver_info = []
        
        print(f"Solving IK for {num_wp} waypoints...")
        
        for i in range(num_wp):
            # Convert mm to meters
            target_pos = trajectory_mm[i] / 1000.0
            
            # Default orientation: pointing down
            target_ori = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            
            # Solve IK
            solution = self.robot.inverse_kinematics(
                target_pos, target_ori, self.initial_guess
            )
            
            if solution is not None:
                # Apply joint offsets (matching MATLAB)
                solution_adj = self.robot.apply_joint_offsets(solution)
                
                # Convert to degrees
                joint_deg = np.rad2deg(solution_adj)
                joint_angles[i] = joint_deg
                success_flags[i] = True
                solver_info.append(f"Waypoint {i+1}: SUCCESS")
                
                # Update initial guess for next waypoint (continuity)
                self.initial_guess = solution.copy()
            else:
                joint_angles[i] = np.nan
                success_flags[i] = False
                solver_info.append(f"Waypoint {i+1}: FAILED - No IK solution")
                print(f"  Warning: No IK solution for waypoint {i+1}: {target_pos * 1000}")
        
        success_count = np.sum(success_flags)
        print(f"IK complete: {success_count}/{num_wp} successful")
        
        return IKResult(
            waypoints=trajectory_mm,
            joint_angles_deg=joint_angles,
            success_flags=success_flags,
            solver_info=solver_info
        )
    
    def save_joint_angles(self, result: IKResult, filename: str = None):
        """Save joint angles to CSV (degrees)."""
        if filename is None:
            filename = f"{sim.output_dir}/joint_angles.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save only successful waypoints
        valid = result.joint_angles_deg[result.success_flags]
        np.savetxt(filename, valid, delimiter=',', fmt='%.2f')
        print(f"Saved {len(valid)} joint angle sets to {filename}")
        
        return filename
    
    def load_joint_angles(self, filename: str) -> np.ndarray:
        """Load joint angles from CSV."""
        data = np.loadtxt(filename, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data  # (N, 6) in degrees
    
    def validate_trajectory(self, joint_angles_deg: np.ndarray) -> dict:
        """Validate joint trajectory for limits and smoothness."""
        joint_rad = np.deg2rad(joint_angles_deg)
        
        # Check limits
        limits_ok = True
        violations = []
        for i, (lo, hi) in enumerate(robot.joint_limits_deg):
            joint_vals = joint_angles_deg[:, i]
            if np.any(joint_vals < lo) or np.any(joint_vals > hi):
                limits_ok = False
                violations.append(f"Joint {i+1}: range [{joint_vals.min():.1f}, {joint_vals.max():.1f}], limit [{lo}, {hi}]")
        
        # Check smoothness (max joint velocity)
        if len(joint_angles_deg) > 1:
            joint_vel = np.abs(np.diff(joint_angles_deg, axis=0)) / sim.dt  # deg/s
            max_vel = np.max(joint_vel, axis=0)
            vel_ok = np.all(max_vel <= ik.max_joint_vel_deg_s)
            vel_violations = []
            for i, (mv, limit) in enumerate(zip(max_vel, [ik.max_joint_vel_deg_s]*6)):
                if mv > limit:
                    vel_violations.append(f"Joint {i+1}: max vel {mv:.1f} deg/s > limit {limit}")
        else:
            vel_ok = True
            vel_violations = []
            max_vel = np.zeros(6)
        
        return {
            'limits_ok': limits_ok,
            'limit_violations': violations,
            'velocity_ok': vel_ok,
            'velocity_violations': vel_violations,
            'max_joint_velocities': max_vel
        }
    
    def run_pipeline(self, trajectory_file: str = None) -> IKResult:
        """Run complete IK pipeline."""
        if trajectory_file is None:
            trajectory_file = f"{sim.output_dir}/robot_trajectory.csv"
        
        # Load trajectory
        trajectory = self.load_trajectory(trajectory_file)
        print(f"Loaded trajectory: {trajectory.shape[0]} waypoints")
        print(f"Range: X[{trajectory[:,0].min():.1f}, {trajectory[:,0].max():.1f}], "
              f"Y[{trajectory[:,1].min():.1f}, {trajectory[:,1].max():.1f}], "
              f"Z[{trajectory[:,2].min():.1f}, {trajectory[:,2].max():.1f}] mm")
        
        # Solve IK
        result = self.solve_trajectory(trajectory)
        
        # Save results
        self.save_joint_angles(result)
        
        # Validate
        if np.any(result.success_flags):
            valid_angles = result.joint_angles_deg[result.success_flags]
            validation = self.validate_trajectory(valid_angles)
            print("\nValidation:")
            print(f"  Joint limits: {'PASS' if validation['limits_ok'] else 'FAIL'}")
            if not validation['limits_ok']:
                for v in validation['limit_violations']:
                    print(f"    - {v}")
            print(f"  Velocity limits: {'PASS' if validation['velocity_ok'] else 'FAIL'}")
            if not validation['velocity_ok']:
                for v in validation['velocity_violations']:
                    print(f"    - {v}")
        
        return result


def run_ik_pipeline(trajectory_file: str = None):
    """Convenience function to run IK pipeline."""
    solver = IKSolver()
    return solver.run_pipeline(trajectory_file)