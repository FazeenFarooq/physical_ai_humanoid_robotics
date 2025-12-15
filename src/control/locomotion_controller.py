"""
Humanoid locomotion controller for the Physical AI & Humanoid Robotics course.
This module implements stable locomotion algorithms for humanoid robots.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class LocomotionCommand:
    """Command for humanoid locomotion"""
    target_velocity: Tuple[float, float, float]  # linear x, y, angular z
    gait_type: str  # 'walk', 'trot', 'crawl', etc.
    step_height: float  # maximum step height in meters
    step_length: float  # step length in meters
    step_time: float  # time for each step in seconds


@dataclass
class FootStep:
    """Represents a single foot step in locomotion"""
    foot_id: str  # 'left', 'right', 'front_left', etc.
    position: Tuple[float, float, float]  # x, y, z in world coordinates
    orientation: Tuple[float, float, float, float]  # quaternion (x, y, z, w)
    time: float  # time when foot should reach this position
    phase: float  # 0.0 to 1.0, where 0.0 is lift and 1.0 is plant


class LocomotionController:
    """
    Controller for humanoid locomotion implementing stable walking patterns.
    Based on principles of dynamic balance and gait planning.
    """
    
    def __init__(self, robot_model: str):
        self.robot_model = robot_model
        self.is_active = False
        self.current_velocity = (0.0, 0.0, 0.0)  # x, y, angular
        self.balance_threshold = 0.1  # maximum CoM deviation before adjustment
        self.current_gait = "walk"
        self.step_plan = []  # Planned foot steps
        self.support_polygon = []  # Current support polygon vertices
    
    def start_locomotion(self):
        """Initialize locomotion control"""
        self.is_active = True
        print(f"Locomotion controller started for {self.robot_model}")
    
    def stop_locomotion(self):
        """Stop locomotion control and return to stable pose"""
        self.is_active = False
        self.current_velocity = (0.0, 0.0, 0.0)
        self.step_plan = []
        print(f"Locomotion controller stopped for {self.robot_model}")
    
    def set_velocity(self, command: LocomotionCommand):
        """Set target velocity for the robot"""
        if not self.is_active:
            raise RuntimeError("Locomotion controller not active")
        
        self.current_velocity = command.target_velocity
        self.current_gait = command.gait_type
        
        # Plan steps based on target velocity
        self.step_plan = self._plan_steps(command)
        
        # Update support polygon based on stance feet
        self._update_support_polygon()
    
    def _plan_steps(self, command: LocomotionCommand) -> List[FootStep]:
        """Plan upcoming footsteps based on velocity command"""
        # Calculate step frequency based on velocity
        step_frequency = max(0.5, min(2.0, math.sqrt(abs(command.target_velocity[0]) * 2)))
        
        # Calculate step duration
        step_duration = 1.0 / step_frequency
        
        # Calculate step positions based on desired velocity
        steps = []
        
        # For a simple walk, we alternate feet
        # This is a simplified implementation - real controller would be more complex
        current_time = 0.0
        base_x = 0.0
        base_y = 0.0
        
        # Generate a sequence of steps
        for i in range(10):  # Plan 10 steps ahead
            step_time = current_time + step_duration
            foot_id = "left" if i % 2 == 0 else "right"
            
            # Calculate step location based on desired velocity
            step_x = base_x + command.target_velocity[0] * step_time
            step_y = base_y + command.target_velocity[1] * step_time + (-0.1 if foot_id == "left" else 0.1)  # Offset for walking
            step_z = 0.05  # Lift foot slightly
            
            step = FootStep(
                foot_id=foot_id,
                position=(step_x, step_y, step_z),
                orientation=(0.0, 0.0, 0.0, 1.0),  # No rotation
                time=step_time,
                phase=0.0
            )
            
            steps.append(step)
            current_time = step_time
        
        return steps
    
    def _update_support_polygon(self):
        """Calculate the current support polygon based on stance feet"""
        # For a bipedal walker, the support polygon is the convex hull
        # of the contact points of the stance feet
        # This is a simplified implementation
        self.support_polygon = [
            (-0.1, -0.05),  # Left foot position
            (-0.1, 0.05),   # Left foot position
            (0.1, 0.05),    # Right foot position
            (0.1, -0.05)    # Right foot position
        ]
    
    def is_balance_stable(self) -> bool:
        """Check if the robot's center of mass is within the support polygon"""
        # Simplified balance check
        # In practice, this would involve more complex CoM calculation
        # and support polygon checking
        
        # For now, return True if we have a stable support polygon
        return len(self.support_polygon) >= 3
    
    def adjust_balance(self) -> bool:
        """Make adjustments to maintain balance during locomotion"""
        if not self.is_balance_stable():
            print("Adjusting balance...")
            # Implement balance control strategies
            # - Adjust foot placement
            # - Shift CoM
            # - Use arm movements for balance
            return True
        return False
    
    def get_next_footstep(self) -> Optional[FootStep]:
        """Get the next planned footstep"""
        if self.step_plan:
            return self.step_plan[0]
        return None
    
    def execute_step(self, footstep: FootStep) -> bool:
        """Execute a single footstep"""
        try:
            # Move the specified foot to the target position
            # This would interface with the robot's joint controllers
            print(f"Executing step for {footstep.foot_id} to {footstep.position}")
            
            # Update the step plan
            if self.step_plan:
                self.step_plan.pop(0)
            
            # Update support polygon after step
            self._update_support_polygon()
            
            return True
        except Exception as e:
            print(f"Error executing step: {e}")
            return False
    
    def get_gait_parameters(self, gait_type: str) -> dict:
        """Get parameters for a specific gait type"""
        gait_params = {
            "walk": {
                "step_height": 0.05,
                "step_length": 0.3,
                "step_time": 0.8,
                "stance_phase": 0.6
            },
            "trot": {
                "step_height": 0.07,
                "step_length": 0.4,
                "step_time": 0.6,
                "stance_phase": 0.5
            },
            "crawl": {
                "step_height": 0.03,
                "step_length": 0.2,
                "step_time": 1.0,
                "stance_phase": 0.7
            }
        }
        
        return gait_params.get(gait_type, gait_params["walk"])