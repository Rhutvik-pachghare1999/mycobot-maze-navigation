# Stage 1: Maze Solver (Computer Vision)
# Author: Rhutvik

"""Maze solving using OpenCV - replicates the original Code.py pipeline.

Pipeline:
1. Capture/load maze image
2. User selects ROI (maze region)
3. Grayscale + OTSU thresholding
4. Contour detection for path
5. Morphological operations to extract solution path
6. Waypoint extraction and normalization
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from config import maze, camera, sim


@dataclass
class MazeSolution:
    """Result of maze solving."""
    original_image: np.ndarray
    cropped_image: np.ndarray
    binary_mask: np.ndarray
    solution_image: np.ndarray  # Green path overlay
    waypoints: List[Tuple[float, float]]  # Normalized maze coordinates (0-1)
    waypoints_limited: List[Tuple[float, float]]  # Limited to max_waypoints


class MazeSolver:
    """Solves maze from image using OpenCV."""
    
    def __init__(self):
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (maze.morph_kernel_size, maze.morph_kernel_size)
        )
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load maze image from file."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return img
    
    def capture_from_camera(self) -> np.ndarray:
        """Capture single frame from camera (for real hardware)."""
        cap = cv2.VideoCapture(camera.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        print("Press 'c' to capture, 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.imshow('Camera - Press c to capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                captured = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("User quit")
        
        cap.release()
        cv2.destroyAllWindows()
        return captured
    
    def select_roi(self, image: np.ndarray, headless: bool = False) -> np.ndarray:
        """Interactive ROI selection for maze region."""
        if headless:
            # Use full image in headless mode
            print("Headless mode: using full image")
            return image
        
        print("Draw rectangle around maze, then press ENTER or SPACE")
        r = cv2.selectROI("Select Maze Region", image, False)
        cv2.destroyWindow("Select Maze Region")
        
        if r[2] == 0 or r[3] == 0:
            raise ValueError("Invalid ROI selected")
        
        x, y, w, h = map(int, r)
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def solve_maze(self, cropped_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve maze using thresholding and morphology.
        
        Returns:
            (binary_mask, solution_image_with_green_path)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        gray = cv2.GaussianBlur(gray, maze.gaussian_blur_ksize, 0)
        
        # OTSU thresholding (inverted: path becomes white)
        _, binary = cv2.threshold(gray, 0, 255, 
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (external only)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            raise ValueError("No contours found in maze")
        
        # Find largest contour (assumed to be maze walls)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create path mask by drawing contour thick
        path_mask = np.zeros_like(binary)
        cv2.drawContours(path_mask, [largest_contour], -1, 255, thickness=10)
        
        # Morphological operations to extract solution path
        # Dilate to connect path segments
        dilated = cv2.dilate(path_mask, self.kernel, iterations=maze.morph_dilate_iterations)
        # Erode slightly
        eroded = cv2.erode(dilated, self.kernel, iterations=maze.morph_erode_iterations)
        # Difference gives the path skeleton
        path_skeleton = cv2.absdiff(eroded, dilated)
        
        # Create green overlay on original
        solution_img = cropped_img.copy()
        solution_img[path_skeleton == 255] = (0, 255, 0)  # Green in BGR
        
        return path_skeleton, solution_img
    
    def extract_waypoints(self, solution_img: np.ndarray, 
                          maze_dims: Tuple[int, int] = None) -> List[Tuple[float, float]]:
        """
        Extract waypoints from green path.
        
        Args:
            solution_img: Image with green path (BGR)
            maze_dims: (width, height) of logical maze grid
            
        Returns:
            List of (x, y) waypoints normalized to maze dimensions
        """
        if maze_dims is None:
            maze_dims = (maze.grid_width, maze.grid_height)
        
        # Detect green pixels (path)
        green_mask = ((solution_img[:, :, 0] == 0) & 
                      (solution_img[:, :, 1] == 255) & 
                      (solution_img[:, :, 2] == 0))
        
        # Get pixel coordinates
        path_pixels = np.column_stack(np.where(green_mask))
        
        if len(path_pixels) == 0:
            return []
        
        # Normalize to maze dimensions
        # path_pixels are (row, col) = (y, x)
        h, w = solution_img.shape[:2]
        normalized = [
            ((x / w) * maze_dims[0], (y / h) * maze_dims[1])
            for y, x in path_pixels
        ]
        
        return normalized
    
    def limit_waypoints(self, waypoints: List[Tuple[float, float]], 
                        max_points: int = None) -> List[Tuple[float, float]]:
        """Downsample waypoints to maximum count."""
        if max_points is None:
            max_points = maze.max_waypoints
        
        if len(waypoints) <= max_points:
            return waypoints
        
        # Evenly sample
        step = len(waypoints) // max_points
        limited = waypoints[::step][:max_points]
        
        # Ensure last point is included
        if limited[-1] != waypoints[-1]:
            limited.append(waypoints[-1])
        
        return limited
    
    def add_landing_point(self, waypoints: List[Tuple[float, float]], 
                          maze_dims: Tuple[int, int] = None) -> List[Tuple[float, float]]:
        """Add final landing point at bottom of maze."""
        if maze_dims is None:
            maze_dims = (maze.grid_width, maze.grid_height)
        
        if not waypoints:
            return []
        
        last_wp = waypoints[-1]
        # Add point at same x, bottom of maze (y = max)
        landing = (last_wp[0], maze_dims[1])
        return waypoints + [landing]
    
    def solve(self, image_source: str = "file", image_path: str = None, headless: bool = False) -> MazeSolution:
        """
        Run complete maze solving pipeline.
        
        Args:
            image_source: "file" or "camera"
            image_path: Path to maze image (if file)
            headless: If True, skip interactive ROI selection
            
        Returns:
            MazeSolution with all intermediate results
        """
        # Load image
        if image_source == "camera":
            original = self.capture_from_camera()
        else:
            if image_path is None:
                image_path = "assets/sample_maze.png"
            original = self.load_image(image_path)
        
        # Select ROI
        cropped = self.select_roi(original, headless=headless)
        
        # Solve
        binary_mask, solution_img = self.solve_maze(cropped)
        
        # Extract waypoints
        waypoints = self.extract_waypoints(solution_img)
        waypoints_limited = self.limit_waypoints(waypoints)
        waypoints_final = self.add_landing_point(waypoints_limited)
        
        # Save intermediate results
        os.makedirs(sim.output_dir, exist_ok=True)
        cv2.imwrite(f"{sim.output_dir}/cropped_maze.png", cropped)
        cv2.imwrite(f"{sim.output_dir}/binary_mask.png", binary_mask)
        cv2.imwrite(f"{sim.output_dir}/maze_solution.png", solution_img)
        
        return MazeSolution(
            original_image=original,
            cropped_image=cropped,
            binary_mask=binary_mask,
            solution_image=solution_img,
            waypoints=waypoints,
            waypoints_limited=waypoints_final
        )
    
    def visualize_solution(self, solution: MazeSolution):
        """Display solution with waypoints."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].imshow(cv2.cvtColor(solution.original_image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(cv2.cvtColor(solution.cropped_image, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Cropped Maze')
        axes[0,1].axis('off')
        
        axes[1,0].imshow(solution.binary_mask, cmap='gray')
        axes[1,0].set_title('Binary Mask')
        axes[1,0].axis('off')
        
        # Solution with waypoints
        sol_vis = solution.solution_image.copy()
        h, w = sol_vis.shape[:2]
        for x_norm, y_norm in solution.waypoints_limited:
            px = int((x_norm / maze.grid_width) * w)
            py = int((y_norm / maze.grid_height) * h)
            cv2.circle(sol_vis, (px, py), 5, (255, 0, 0), -1)  # Red waypoints
        
        axes[1,1].imshow(cv2.cvtColor(sol_vis, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title(f'Solution + Waypoints ({len(solution.waypoints_limited)} pts)')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{sim.output_dir}/maze_solution_visualization.png", dpi=150)
        plt.show()
        
        return fig


import os