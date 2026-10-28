#!/usr/bin/env python3
# MyCobot Maze Navigation - Main Entry Point
# Author: Rhutvik

"""Main pipeline for MyCobot maze navigation replica.

Pipeline stages:
1. Maze solving (OpenCV)
2. Coordinate mapping (Maze -> Robot)
3. Inverse kinematics
4. Execution (simulated or real TCP)

Run with:
    python src/main.py                    # Full pipeline with visualization
    python src/main.py --stage 1          # Only maze solving
    python src/main.py --stage 2          # Only coordinate mapping
    python src/main.py --stage 3          # Only IK
    python src/main.py --stage 4          # Only execution
    python src/main.py --record           # Record animation
    python src/main.py --create-sample    # Create sample maze
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import sim, robot, maze
from maze_solver import MazeSolver
from coordinate_mapper import CoordinateMapper
from ik_solver import IKSolver, run_ik_pipeline
from tcp_client import run_execution_pipeline, SimulatedRobotExecutor
from visualize import RobotVisualizer, create_sample_maze_image


def ensure_dirs():
    """Create output directories."""
    os.makedirs(sim.output_dir, exist_ok=True)
    os.makedirs(sim.log_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)


def run_stage1_maze_solver(image_path: str = None, headless: bool = True):
    """Stage 1: Solve maze from image."""
    print("\n" + "="*50)
    print("STAGE 1: MAZE SOLVING")
    print("="*50)
    
    solver = MazeSolver()
    
    if image_path is None:
        image_path = "assets/sample_maze.png"
        if not os.path.exists(image_path):
            print("No maze image found. Creating sample...")
            create_sample_maze_image(image_path)
    
    solution = solver.solve(image_source="file", image_path=image_path, headless=headless)
    
    print(f"Found {len(solution.waypoints)} raw waypoints")
    print(f"Limited to {len(solution.waypoints_limited)} waypoints")
    
    # Save waypoints for next stage
    np.savetxt(f"{sim.output_dir}/maze_waypoints.csv", 
               solution.waypoints_limited, delimiter=',', fmt='%.4f')
    
    # Visualize
    solver.visualize_solution(solution)
    
    return solution


def run_stage2_coordinate_mapping():
    """Stage 2: Map maze waypoints to robot coordinates."""
    print("\n" + "="*50)
    print("STAGE 2: COORDINATE MAPPING")
    print("="*50)
    
    # Load maze waypoints
    waypoint_file = f"{sim.output_dir}/maze_waypoints.csv"
    if not os.path.exists(waypoint_file):
        print(f"Waypoint file not found: {waypoint_file}")
        print("Run stage 1 first.")
        return None
    
    maze_waypoints = np.loadtxt(waypoint_file, delimiter=',')
    if maze_waypoints.ndim == 1:
        maze_waypoints = maze_waypoints.reshape(1, -1)
    
    print(f"Loaded {len(maze_waypoints)} maze waypoints")
    
    # Create mapper with calibration corners
    mapper = CoordinateMapper(
        corners=robot.calibration_corners,
        maze_dims=(maze.grid_width, maze.grid_height)
    )
    
    # Map to robot coordinates
    mapping = mapper.create_mapping(maze_waypoints)
    
    print(f"Mapped to {len(mapping.robot_waypoints)} robot waypoints")
    for i, (m, r) in enumerate(zip(mapping.maze_waypoints[:5], mapping.robot_waypoints[:5])):
        print(f"  Maze {m} -> Robot ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) mm")
    if len(mapping.robot_waypoints) > 5:
        print(f"  ... and {len(mapping.robot_waypoints) - 5} more")
    
    # Save robot trajectory
    mapper.save_trajectory_csv(mapping.robot_waypoints)
    
    # Visualize mapping
    mapper.visualize_mapping(mapping.maze_waypoints, mapping.robot_waypoints)
    
    return mapping


def run_stage3_ik():
    """Stage 3: Inverse kinematics."""
    print("\n" + "="*50)
    print("STAGE 3: INVERSE KINEMATICS")
    print("="*50)
    
    result = run_ik_pipeline()
    return result


def run_stage4_execution(simulation: bool = True, record: bool = False):
    """Stage 4: Execute trajectory."""
    print("\n" + "="*50)
    print("STAGE 4: TRAJECTORY EXECUTION")
    print("="*50)
    
    # Load joint angles
    joint_file = f"{sim.output_dir}/joint_angles.csv"
    if not os.path.exists(joint_file):
        print(f"Joint angles file not found: {joint_file}")
        print("Run stage 3 first.")
        return None
    
    joint_angles = np.loadtxt(joint_file, delimiter=',')
    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)
    
    print(f"Loaded {len(joint_angles)} joint configurations")
    
    if simulation:
        # Simulated execution with visualization
        executor = SimulatedRobotExecutor()
        positions = executor.execute_trajectory(joint_angles, visualize=True)
        
        # Save executed trajectory
        np.savetxt(f"{sim.output_dir}/executed_trajectory.csv", 
                   np.array(positions) * 1000, delimiter=',', fmt='%.3f')
        
        # Visualize robot animation
        if viz.show_robot_animation:
            visualizer = RobotVisualizer()
            
            # Load planned trajectory for comparison
            planned_traj = None
            traj_file = f"{sim.output_dir}/robot_trajectory.csv"
            if os.path.exists(traj_file):
                planned_traj = np.loadtxt(traj_file, delimiter=',')
            
            visualizer.animate_trajectory(
                joint_trajectory_deg=joint_angles,
                trajectory_mm=planned_traj,
                interval=300,
                save_path=f"{sim.output_dir}/simulation.mp4" if record else None
            )
            
            # Comparison plot
            if planned_traj is not None:
                executed = np.loadtxt(f"{sim.output_dir}/executed_trajectory.csv", delimiter=',')
                visualizer.plot_static_comparison(planned_traj, executed)
    else:
        # Real TCP execution
        result = run_execution_pipeline(joint_file, simulation=False)
        print(f"Execution result: {result.commands_successful}/{result.commands_sent} successful")
    
    return True


def run_full_pipeline(record: bool = False):
    """Run complete pipeline."""
    print("\n" + "#"*60)
    print("# MYCOBOT MAZE NAVIGATION - FULL PIPELINE")
    print("# Author: Rhutvik")
    print("#"*60)
    
    ensure_dirs()
    
    # Stage 1
    solution = run_stage1_maze_solver(headless=True)
    
    # Stage 2
    mapping = run_stage2_coordinate_mapping()
    
    # Stage 3
    ik_result = run_stage3_ik()
    
    # Stage 4
    run_stage4_execution(simulation=True, record=record)
    
    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    print(f"Outputs in {sim.output_dir}/")
    print("  - maze_solution.png")
    print("  - robot_trajectory.csv")
    print("  - joint_angles.csv")
    print("  - executed_trajectory.csv")
    if record:
        print("  - simulation.mp4")


def main():
    parser = argparse.ArgumentParser(description='MyCobot Maze Navigation Pipeline')
    parser.add_argument('--stage', type=int, choices=[1,2,3,4], default=0,
                       help='Run specific stage (0 = all)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to maze image (stage 1)')
    parser.add_argument('--record', action='store_true',
                       help='Record animation to video')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample maze image')
    parser.add_argument('--real', action='store_true',
                       help='Use real robot via TCP (stage 4)')
    args = parser.parse_args()
    
    ensure_dirs()
    
    if args.create_sample:
        create_sample_maze_image()
        return
    
    if args.stage == 1:
        run_stage1_maze_solver(args.image)
    elif args.stage == 2:
        run_stage2_coordinate_mapping()
    elif args.stage == 3:
        run_stage3_ik()
    elif args.stage == 4:
        run_stage4_execution(simulation=not args.real, record=args.record)
    else:
        run_full_pipeline(record=args.record)


if __name__ == '__main__':
    main()