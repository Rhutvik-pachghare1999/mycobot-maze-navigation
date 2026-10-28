# Robot Model - MyCobot Pro 600 Kinematics
# Author: Rhutvik

"""MyCobot Pro 600 forward/inverse kinematics using DH parameters.

Simplified model for simulation. For production, use URDF + proper IK solver.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from config import robot, ik


@dataclass
class DHParams:
    """Denavit-Hartenberg parameters for MyCobot Pro 600."""
    # Approximate DH parameters (mm, deg) - adjust for accuracy
    # [theta_offset, d, a, alpha] for each joint
    params = [
        [0,     130,    0,     -90],   # J1: base rotation
        [0,     0,      425,   0],     # J2: shoulder
        [0,     0,      392,   0],     # J3: elbow
        [0,     160,    0,     -90],   # J4: wrist 1
        [0,     0,      0,     90],    # J5: wrist 2
        [0,     80,     0,     0],     # J6: wrist 3
    ]
    
    @classmethod
    def get_params(cls):
        return cls.params


class MyCobotKinematics:
    """Forward and inverse kinematics for MyCobot Pro 600."""
    
    def __init__(self):
        self.dh = DHParams.get_params()
        self.num_joints = 6
        self.joint_limits = np.deg2rad([lim for lim in robot.joint_limits_deg])
        self.home_pos = np.deg2rad(robot.home_position_deg)
    
    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Compute DH transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,     d],
            [0,   0,      0,      1]
        ])
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose from joint angles.
        
        Args:
            joint_angles: 6 joint angles in radians
            
        Returns:
            4x4 transformation matrix (end-effector in base frame)
        """
        T = np.eye(4)
        for i, (theta, d, a, alpha) in enumerate(self.dh):
            # Apply joint offset
            q = joint_angles[i] + np.deg2rad(self.dh[i][0])
            T_i = self.dh_transform(q, d, a, np.deg2rad(alpha))
            T = T @ T_i
        
        # Apply end-effector offset
        ee_offset = np.eye(4)
        ee_offset[2, 3] = robot.ee_offset_z / 1000.0  # Convert to meters
        T = T @ ee_offset
        
        return T
    
    def get_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end-effector position (x, y, z) in meters."""
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3]
    
    def get_orientation(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end-effector orientation as rotation matrix."""
        T = self.forward_kinematics(joint_angles)
        return T[:3, :3]
    
    def inverse_kinematics(self, target_pos: np.ndarray, 
                          target_ori: np.ndarray = None,
                          initial_guess: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using numerical optimization.
        
        Args:
            target_pos: Target position (x, y, z) in meters
            target_ori: Target orientation as 3x3 rotation matrix (optional)
            initial_guess: Initial joint angles in radians
            
        Returns:
            Joint angles in radians, or None if failed
        """
        if target_ori is None:
            # Default: end-effector pointing down
            target_ori = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
        
        if initial_guess is None:
            initial_guess = self.home_pos.copy()
        
        def objective(q):
            """Cost function: position + orientation error."""
            T = self.forward_kinematics(q)
            pos = T[:3, 3]
            ori = T[:3, :3]
            
            # Position error
            pos_err = np.sum((pos - target_pos) ** 2)
            
            # Orientation error (Frobenius norm of difference)
            ori_err = np.sum((ori - target_ori) ** 2)
            
            # Weighted sum
            w = ik.weights
            return w[0]*pos_err + w[1]*pos_err + w[2]*pos_err + w[3]*ori_err + w[4]*ori_err + w[5]*ori_err
        
        # Joint limits as bounds
        bounds = [(float(lo), float(hi)) for lo, hi in self.joint_limits]
        
        # Optimize
        result = minimize(
            objective, initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': ik.max_iterations, 'ftol': 1e-6}
        )
        
        if result.success:
            return result.x
        else:
            # Try with different initial guesses
            for guess in [self.home_pos, np.zeros(6), np.deg2rad([0, -90, 90, -90, 90, 0])]:
                result = minimize(objective, guess, method='SLSQP', bounds=bounds,
                                options={'maxiter': ik.max_iterations})
                if result.success:
                    return result.x
            
            return None
    
    def apply_joint_offsets(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """Apply joint offset adjustments (matching original MATLAB)."""
        offsets = np.deg2rad(ik.joint_offsets_deg)
        adjusted = joint_angles_rad.copy()
        
        # J1: +180 deg
        adjusted[0] += offsets[0]
        # J2: -90 deg
        adjusted[1] += offsets[1]
        # J3: no offset
        # J4: -90 deg
        adjusted[3] += offsets[3]
        # J5: multiply by -1
        adjusted[4] *= offsets[4]  # -1.0
        # J6: no offset
        
        return adjusted
    
    def check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within limits."""
        return np.all(joint_angles >= self.joint_limits[:, 0]) and \
               np.all(joint_angles <= self.joint_limits[:, 1])


def create_robot_model() -> MyCobotKinematics:
    """Factory function."""
    return MyCobotKinematics()