# Stage 4: TCP Client (Simulated Execution)
# Author: Rhutvik

"""Simulated TCP client for sending joint angles to robot.

In simulation mode, visualizes robot executing trajectory.
In real mode, connects to robot controller via TCP.
"""

import socket
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from config import tcp, sim, robot


@dataclass
class TCPResult:
    """Result of TCP execution."""
    commands_sent: int
    commands_successful: int
    errors: List[str]
    total_time: float


class TCPClient:
    """TCP client for robot command execution."""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to robot controller."""
        if self.simulation_mode:
            print("Simulation mode: No real TCP connection")
            self.connected = True
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(tcp.timeout)
            self.socket.connect((tcp.server_ip, tcp.server_port))
            self.connected = True
            print(f"Connected to robot at {tcp.server_ip}:{tcp.server_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command and receive response."""
        if not self.connected:
            return None
        
        if self.simulation_mode:
            # Simulate response
            time.sleep(0.01)
            return "OK"
        
        try:
            self.socket.sendall(command.encode('utf-8'))
            response = self.socket.recv(1024).decode('utf-8')
            return response
        except Exception as e:
            print(f"Send failed: {e}")
            return None
    
    def send_joint_angles(self, angles_deg: np.ndarray, speed: int = 400) -> bool:
        """Send joint angles command."""
        # Format: set_angles(j1, j2, j3, j4, j5, j6, speed)
        rounded = [round(a, 2) for a in angles_deg]
        cmd = f"set_angles({', '.join(map(str, rounded))}, {speed})"
        response = self.send_command(cmd)
        return response is not None and "OK" in response.upper()
    
    def execute_trajectory(self, joint_angles_file: str) -> TCPResult:
        """Execute full trajectory from joint angles CSV."""
        import os
        data = np.loadtxt(joint_angles_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        print(f"Executing {len(data)} waypoints...")
        
        errors = []
        successful = 0
        start_time = time.time()
        
        for i, angles in enumerate(data):
            if self.simulation_mode:
                # In simulation, just log
                print(f"  [{i+1}/{len(data)}] Angles: {angles}")
                successful += 1
                time.sleep(tcp.command_delay * 0.1)  # Fast sim
            else:
                if self.send_joint_angles(angles):
                    successful += 1
                else:
                    errors.append(f"Waypoint {i+1}: Failed to send")
                time.sleep(tcp.command_delay)
        
        total_time = time.time() - start_time
        
        return TCPResult(
            commands_sent=len(data),
            commands_successful=successful,
            errors=errors,
            total_time=total_time
        )


class SimulatedRobotExecutor:
    """Simulates robot execution with visualization."""
    
    def __init__(self):
        from robot_model import create_robot_model
        self.robot = create_robot_model()
        self.current_angles = np.deg2rad(robot.home_position_deg)
    
    def execute_trajectory(self, joint_angles_deg: np.ndarray, 
                          visualize: bool = True) -> List[np.ndarray]:
        """
        Simulate robot executing joint trajectory.
        
        Returns:
            List of end-effector positions at each step
        """
        positions = []
        
        print(f"Simulating robot execution: {len(joint_angles_deg)} waypoints")
        
        for i, angles_deg in enumerate(joint_angles_deg):
            angles_rad = np.deg2rad(angles_deg)
            
            # Interpolate from current to target (smooth motion)
            steps = 10
            for step in range(steps + 1):
                alpha = step / steps
                interp_angles = self.current_angles + alpha * (angles_rad - self.current_angles)
                
                # Get end-effector position
                pos = self.robot.get_position(interp_angles)
                positions.append(pos.copy())
                
                if visualize and step == steps:
                    print(f"  [{i+1}/{len(joint_angles_deg)}] Pos: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm, "
                          f"Angles: {np.rad2deg(interp_angles)}")
            
            self.current_angles = angles_rad.copy()
        
        return positions
    
    def get_current_pose(self) -> np.ndarray:
        """Get current end-effector pose."""
        return self.robot.forward_kinematics(self.current_angles)


def run_execution_pipeline(joint_angles_file: str = None, 
                          simulation: bool = True) -> TCPResult:
    """Run execution pipeline."""
    if joint_angles_file is None:
        joint_angles_file = f"{sim.output_dir}/joint_angles.csv"
    
    # Load joint angles
    data = np.loadtxt(joint_angles_file, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if simulation:
        executor = SimulatedRobotExecutor()
        positions = executor.execute_trajectory(data, visualize=True)
        
        # Save executed trajectory
        os.makedirs(sim.output_dir, exist_ok=True)
        np.savetxt(f"{sim.output_dir}/executed_trajectory.csv", 
                   np.array(positions) * 1000, delimiter=',', fmt='%.3f')
        print(f"Executed trajectory saved to {sim.output_dir}/executed_trajectory.csv")
        
        return TCPResult(
            commands_sent=len(data),
            commands_successful=len(data),
            errors=[],
            total_time=len(data) * tcp.command_delay
        )
    else:
        client = TCPClient(simulation_mode=False)
        if client.connect():
            try:
                return client.execute_trajectory(joint_angles_file)
            finally:
                client.disconnect()
        else:
            return TCPResult(0, 0, ["Failed to connect"], 0)


import os