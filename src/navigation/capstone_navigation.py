"""
Navigation stack for the capstone project in the Physical AI & Humanoid Robotics course.
This module integrates path planning, localization, and navigation execution for the complete system.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import heapq
import math
from enum import Enum
from datetime import datetime


class NavigationStatus(Enum):
    """Status of navigation"""
    IDLE = "idle"
    PLANNING_PATH = "planning_path"
    FOLLOWING_PATH = "following_path"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    REACHED_GOAL = "reached_goal"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PathPlannerType(Enum):
    """Type of path planner to use"""
    A_STAR = "a_star"
    RRT = "rrt"
    D_STAR_LITE = "d_star_lite"
    NAV_FN = "nav_fn"


@dataclass
class RobotPose:
    """Represents the pose of the robot"""
    x: float
    y: float
    theta: float  # Orientation in radians
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Path:
    """Represents a planned path"""
    waypoints: List[Tuple[float, float]]  # List of (x, y) coordinates
    planner_type: PathPlannerType
    computed_at: datetime = None
    length: float = 0.0  # Total length of the path
    clearance: float = 0.0  # Minimum clearance from obstacles
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
        
        # Calculate path length
        if len(self.waypoints) > 1:
            for i in range(1, len(self.waypoints)):
                dx = self.waypoints[i][0] - self.waypoints[i-1][0]
                dy = self.waypoints[i][1] - self.waypoints[i-1][1]
                self.length += math.sqrt(dx*dx + dy*dy)


class AStarPlanner:
    """A* path planner implementation"""
    
    def __init__(self, resolution: float = 0.05):
        self.resolution = resolution
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], 
                 occupancy_grid: np.ndarray) -> Optional[Path]:
        """Plan a path using A* algorithm"""
        height, width = occupancy_grid.shape
        
        # Convert world coordinates to grid coordinates
        start_grid = (int(start[1] / self.resolution + height/2), 
                      int(start[0] / self.resolution + width/2))
        goal_grid = (int(goal[1] / self.resolution + height/2), 
                     int(goal[0] / self.resolution + width/2))
        
        # Check bounds
        if (not (0 <= start_grid[0] < height and 0 <= start_grid[1] < width) or
            not (0 <= goal_grid[0] < height and 0 <= goal_grid[1] < width)):
            return None
        
        # Check if start or goal is occupied
        if occupancy_grid[start_grid] > 50 or occupancy_grid[goal_grid] > 50:
            return None
        
        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                
                # Convert grid path back to world coordinates
                world_path = [
                    (point[1] * self.resolution - width * self.resolution / 2,
                     point[0] * self.resolution - height * self.resolution / 2)
                    for point in path
                ]
                
                return Path(
                    waypoints=world_path,
                    planner_type=PathPlannerType.A_STAR
                )
            
            for neighbor in self._get_neighbors(current, occupancy_grid):
                tentative_g_score = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic (Manhattan distance) between two grid positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two grid positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_neighbors(self, pos: Tuple[int, int], occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Get traversable neighbors of a position"""
        neighbors = []
        height, width = occupancy_grid.shape
        
        # 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if (0 <= new_pos[0] < height and 
                    0 <= new_pos[1] < width and 
                    occupancy_grid[new_pos] <= 50):  # Free space threshold
                    neighbors.append(new_pos)
        
        return neighbors
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path by following came_from links"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path


class RRTPlanner:
    """Rapidly-exploring Random Tree path planner"""
    
    def __init__(self, resolution: float = 0.05, max_iterations: int = 1000):
        self.resolution = resolution
        self.max_iterations = max_iterations
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], 
                 occupancy_grid: np.ndarray) -> Optional[Path]:
        """Plan a path using RRT algorithm"""
        # For simplicity in this implementation, we'll return None
        # A full RRT implementation would be extensive
        return None


class PathFollower:
    """Follows a planned path with obstacle avoidance"""
    
    def __init__(self, linear_vel: float = 0.5, angular_vel: float = 0.5, 
                 tolerance: float = 0.1):
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.tolerance = tolerance
        self.current_path: Optional[Path] = None
        self.current_waypoint_idx = 0
        self.is_active = False
    
    def follow_path(self, path: Path, current_pose: RobotPose, 
                   local_map: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate velocity commands to follow the path"""
        if not path.waypoints or self.current_waypoint_idx >= len(path.waypoints):
            return 0.0, 0.0  # Stop when path is complete
        
        # Get current target waypoint
        target = path.waypoints[self.current_waypoint_idx]
        
        # Calculate distance to target
        dx = target[0] - current_pose.x
        dy = target[1] - current_pose.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if we've reached the current waypoint
        if distance < self.tolerance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(path.waypoints):
                return 0.0, 0.0  # Path complete
        
        # Calculate angle to target
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - current_pose.theta
        
        # Normalize angle difference to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Basic proportional control
        angular_cmd = max(-self.angular_vel, min(self.angular_vel, 2.0 * angle_diff))
        linear_cmd = min(self.linear_vel, 0.5 * distance)  # Slow down as we approach
        
        # If there's a local map and we detect obstacles ahead, reduce speed
        if local_map is not None:
            safe_distance = self._check_obstacle_ahead(current_pose, local_map)
            if safe_distance < 0.5:  # Less than 0.5m to obstacle
                linear_cmd *= safe_distance / 0.5  # Scale down speed proportionally
        
        return linear_cmd, angular_cmd
    
    def _check_obstacle_ahead(self, pose: RobotPose, local_map: np.ndarray) -> float:
        """Check for obstacles in the immediate vicinity"""
        # For simplicity, just return a safe distance
        # In a real implementation, this would analyze the local_map
        return 1.0  # 1 meter to nearest obstacle


class CapstoneNavigator:
    """
    Comprehensive navigation system for the capstone project that integrates
    path planning, localization, and path following with obstacle avoidance.
    """
    
    def __init__(self):
        self.path_planner = AStarPlanner()
        self.path_follower = PathFollower()
        self.status = NavigationStatus.IDLE
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_pose: Optional[RobotPose] = None
        self.global_map: Optional[np.ndarray] = None
        self.local_map: Optional[np.ndarray] = None
        self.last_command_time = datetime.now()
        self.replan_threshold = 1.0  # Replan if we deviate more than 1m from path
    
    def set_global_map(self, occupancy_grid: np.ndarray):
        """Set the global occupancy grid map"""
        self.global_map = occupancy_grid
    
    def set_local_map(self, occupancy_grid: np.ndarray):
        """Set the local occupancy grid map"""
        self.local_map = occupancy_grid
    
    def set_pose(self, x: float, y: float, theta: float):
        """Update the robot's current pose"""
        self.current_pose = RobotPose(x, y, theta)
    
    def navigate_to(self, goal_x: float, goal_y: float) -> bool:
        """Start navigation to a goal position"""
        if self.global_map is None or self.current_pose is None:
            return False
        
        self.current_goal = (goal_x, goal_y)
        self.status = NavigationStatus.PLANNING_PATH
        
        # Plan path to goal
        path = self.path_planner.plan_path(
            (self.current_pose.x, self.current_pose.y),
            (goal_x, goal_y),
            self.global_map
        )
        
        if path is None:
            self.status = NavigationStatus.FAILED
            return False
        
        # Start following the path
        self.path_follower.current_path = path
        self.path_follower.current_waypoint_idx = 0
        self.status = NavigationStatus.FOLLOWING_PATH
        
        return True
    
    def get_navigation_command(self) -> Tuple[float, float]:
        """Get the current navigation command (linear velocity, angular velocity)"""
        if self.status != NavigationStatus.FOLLOWING_PATH:
            return 0.0, 0.0
        
        if self.current_pose is None or self.path_follower.current_path is None:
            return 0.0, 0.0
        
        # Check if we need to replan (e.g., due to obstacles)
        if self._should_replan():
            self._replan_path()
        
        # Get command from path follower
        cmd = self.path_follower.follow_path(
            self.path_follower.current_path,
            self.current_pose,
            self.local_map
        )
        
        # Update status if we've reached the goal
        if (self.path_follower.current_waypoint_idx >= 
            len(self.path_follower.current_path.waypoints)):
            self.status = NavigationStatus.REACHED_GOAL
        
        self.last_command_time = datetime.now()
        return cmd
    
    def _should_replan(self) -> bool:
        """Check if we should replan the path"""
        if (self.current_pose is None or 
            self.path_follower.current_path is None or
            self.path_follower.current_waypoint_idx >= 
            len(self.path_follower.current_path.waypoints)):
            return False
        
        # Check if we're too far from the planned path
        current_waypoint = self.path_follower.current_path.waypoints[
            min(self.path_follower.current_waypoint_idx, 
                len(self.path_follower.current_path.waypoints) - 1)
        ]
        
        dx = self.current_pose.x - current_waypoint[0]
        dy = self.current_pose.y - current_waypoint[1]
        deviation = math.sqrt(dx*dx + dy*dy)
        
        return deviation > self.replan_threshold
    
    def _replan_path(self):
        """Replan the path to the goal if needed"""
        if (self.current_goal is None or 
            self.global_map is None or
            self.current_pose is None):
            return
        
        # Plan new path from current position to goal
        new_path = self.path_planner.plan_path(
            (self.current_pose.x, self.current_pose.y),
            self.current_goal,
            self.global_map
        )
        
        if new_path is not None:
            # Update the path follower with the new path
            self.path_follower.current_path = new_path
            self.path_follower.current_waypoint_idx = 0
            self.status = NavigationStatus.FOLLOWING_PATH
    
    def cancel_navigation(self):
        """Cancel current navigation"""
        self.status = NavigationStatus.CANCELLED
        self.path_follower.current_path = None
        self.current_goal = None
    
    def is_at_goal(self) -> bool:
        """Check if the robot has reached the goal"""
        return self.status == NavigationStatus.REACHED_GOAL
    
    def get_distance_to_goal(self) -> Optional[float]:
        """Get the distance to the current goal"""
        if self.current_goal is None or self.current_pose is None:
            return None
        
        dx = self.current_goal[0] - self.current_pose.x
        dy = self.current_goal[1] - self.current_pose.y
        return math.sqrt(dx*dx + dy*dy)
    
    def get_navigation_status(self) -> NavigationStatus:
        """Get the current navigation status"""
        return self.status
    
    def get_remaining_path_waypoints(self) -> List[Tuple[float, float]]:
        """Get the remaining waypoints in the current path"""
        if self.path_follower.current_path is None:
            return []
        
        remaining_waypoints = []
        start_idx = self.path_follower.current_waypoint_idx
        if start_idx < len(self.path_follower.current_path.waypoints):
            remaining_waypoints = self.path_follower.current_path.waypoints[start_idx:]
        
        return remaining_waypoints


class NavigationSafetyManager:
    """
    Ensures safe navigation with collision avoidance and emergency stops
    """
    
    def __init__(self, min_distance_to_obstacle: float = 0.3):
        self.min_distance_to_obstacle = min_distance_to_obstacle
        self.is_safe = True
        self.last_safe_check = datetime.now()
        
    def check_safety(self, pose: RobotPose, local_map: np.ndarray,
                    command: Tuple[float, float]) -> Tuple[float, float]:
        """
        Check if the navigation command is safe and modify if necessary
        Returns a safe (linear_vel, angular_vel) command
        """
        linear_vel, angular_vel = command
        
        # Check for obstacles in the direction of movement
        safe_linear = self._check_forward_safety(pose, local_map, linear_vel)
        
        # If not safe to move forward, consider stopping or avoiding
        if not safe_linear and abs(linear_vel) > 0.01:
            # Reduce forward speed significantly or stop
            if abs(linear_vel) > 0.1:
                linear_vel *= 0.1  # Reduce to 10% of desired speed
            else:
                linear_vel = 0.0  # Stop if already slow
        
        # Check if we can make turns safely
        if not self._check_rotation_safety(pose, local_map) and abs(angular_vel) > 0.1:
            angular_vel = 0.0  # Don't rotate if it's not safe
        
        self.is_safe = (abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01)
        self.last_safe_check = datetime.now()
        
        return linear_vel, angular_vel
    
    def _check_forward_safety(self, pose: RobotPose, local_map: np.ndarray, linear_vel: float) -> bool:
        """Check if it's safe to move forward"""
        if linear_vel <= 0:  # Not moving forward
            return True
        
        # Estimate where the robot will be in 0.5 seconds
        dt = 0.5
        future_x = pose.x + linear_vel * dt * math.cos(pose.theta)
        future_y = pose.y + linear_vel * dt * math.sin(pose.theta)
        
        # Check if this position is free in the local map
        # (simplified check - in reality would be more complex)
        height, width = local_map.shape
        resolution = 0.05  # Default resolution
        
        grid_x = int(future_x / resolution + width / 2)
        grid_y = int(future_y / resolution + height / 2)
        
        if (0 <= grid_x < width and 0 <= grid_y < height and 
            local_map[grid_y, grid_x] < 50):  # Free space
            return True
        
        return False
    
    def _check_rotation_safety(self, pose: RobotPose, local_map: np.ndarray) -> bool:
        """Check if it's safe to rotate"""
        # Check potential collision points around the robot
        # This is a simplified check - in reality would check robot's bounding box
        for angle_offset in [math.pi/4, math.pi/2, 3*math.pi/4]:  # Check multiple angles
            check_angle = pose.theta + angle_offset
            check_x = pose.x + 0.3 * math.cos(check_angle)  # Check 30cm away
            check_y = pose.y + 0.3 * math.sin(check_angle)
            
            # Convert to grid coordinates
            height, width = local_map.shape
            resolution = 0.05
            
            grid_x = int(check_x / resolution + width / 2)
            grid_y = int(check_y / resolution + height / 2)
            
            if not (0 <= grid_x < width and 0 <= grid_y < height and 
                   local_map[grid_y, grid_x] < 50):  # Occupied space
                return False
        
        return True
    
    def is_navigation_safe(self) -> bool:
        """Check if navigation is currently safe"""
        return self.is_safe


class NavigationManager:
    """
    Main navigation manager that integrates all navigation components
    """
    
    def __init__(self):
        self.navigator = CapstoneNavigator()
        self.safety_manager = NavigationSafetyManager()
        self.is_active = False
    
    def start_navigation(self, goal_x: float, goal_y: float) -> bool:
        """Start navigation to a goal position"""
        self.is_active = True
        return self.navigator.navigate_to(goal_x, goal_y)
    
    def update_sensor_data(self, occupancy_grid: np.ndarray, 
                          x: float, y: float, theta: float):
        """Update with new sensor data and robot pose"""
        # Update maps
        self.navigator.set_global_map(occupancy_grid)
        self.navigator.set_local_map(occupancy_grid)
        
        # Update pose
        self.navigator.set_pose(x, y, theta)
    
    def get_navigation_command(self) -> Tuple[float, float]:
        """Get safe navigation commands"""
        if not self.is_active:
            return 0.0, 0.0
        
        # Get raw navigation command
        raw_cmd = self.navigator.get_navigation_command()
        
        # Apply safety checks
        safe_cmd = self.safety_manager.check_safety(
            self.navigator.current_pose,
            self.navigator.local_map if self.navigator.local_map is not None else np.zeros((100, 100)),
            raw_cmd
        )
        
        return safe_cmd
    
    def stop_navigation(self):
        """Stop current navigation"""
        self.navigator.cancel_navigation()
        self.is_active = False
    
    def is_at_goal(self) -> bool:
        """Check if the robot has reached its goal"""
        return self.navigator.is_at_goal()
    
    def get_navigation_status(self) -> NavigationStatus:
        """Get current navigation status"""
        return self.navigator.get_navigation_status()
    
    def get_distance_to_goal(self) -> Optional[float]:
        """Get distance to current goal"""
        return self.navigator.get_distance_to_goal()