"""
Manipulation planning algorithms for the Physical AI & Humanoid Robotics course.
This module implements algorithms for planning robot manipulation tasks.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from src.control.kinematics import KinematicChain, HumanoidKinematicModel


@dataclass
class ObjectInfo:
    """Information about an object to be manipulated"""
    id: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # Quaternion
    dimensions: Tuple[float, float, float]  # width, height, depth
    mass: float
    grasp_points: List[Tuple[float, float, float]]  # Potential grasp points


@dataclass
class ManipulationTask:
    """Definition of a manipulation task"""
    id: str
    task_type: str  # 'grasp', 'place', 'move', 'assemble', etc.
    target_object: ObjectInfo
    target_position: Optional[Tuple[float, float, float]] = None
    target_orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion
    gripper_configuration: str = "default"  # How to configure the gripper


class ManipulationPlanner:
    """
    Planner for robot manipulation tasks including grasp planning,
    trajectory generation, and collision avoidance.
    """
    
    def __init__(self, robot_model: HumanoidKinematicModel):
        self.robot_model = robot_model
        self.workspace_limits = {
            'min': (-1.0, -1.0, 0.0),   # x, y, z minimums
            'max': (1.0, 1.0, 2.0)      # x, y, z maximums
        }
        self.collision_threshold = 0.05  # Minimum distance to avoid collisions
    
    def plan_grasp(self, obj_info: ObjectInfo, arm_chain: str) -> Optional[List[float]]:
        """
        Plan a grasp for the given object using the specified arm
        Returns joint angles for the arm to achieve a suitable grasp
        """
        # Find the best grasp point on the object
        best_grasp_point = self._select_best_grasp_point(obj_info)
        if best_grasp_point is None:
            return None
        
        # Get the kinematic chain for the specified arm
        arm = self.robot_model.get_chain(arm_chain)
        if arm is None:
            return None
        
        # Plan inverse kinematics to reach the grasp point
        # Approach from a safe angle above the object
        approach_pos = (
            best_grasp_point[0],
            best_grasp_point[1],
            best_grasp_point[2] + 0.1  # 10cm above object for safe approach
        )
        
        # Calculate joint angles to reach approach position
        joint_angles_approach = arm.inverse_kinematics(approach_pos)
        
        # Calculate joint angles to reach grasp position
        joint_angles_grasp = arm.inverse_kinematics(best_grasp_point)
        
        # Return the approach angles as the first step
        return joint_angles_approach
    
    def _select_best_grasp_point(self, obj_info: ObjectInfo) -> Optional[Tuple[float, float, float]]:
        """
        Select the best grasp point on an object
        """
        if not obj_info.grasp_points:
            return None
        
        # For now, return the first grasp point
        # In a real implementation, this would consider factors like:
        # - accessibility from the robot's current position
        # - stability of the grasp
        # - orientation needed for the task
        return obj_info.grasp_points[0]
    
    def plan_trajectory(self, start_config: List[float], end_config: List[float], 
                       arm_chain: str, num_waypoints: int = 10) -> List[List[float]]:
        """
        Plan a smooth trajectory between two joint configurations
        """
        # Get the arm kinematic chain
        arm = self.robot_model.get_chain(arm_chain)
        if arm is None:
            return []
        
        # Linear interpolation in joint space
        trajectory = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints  # Interpolation parameter [0, 1]
            
            waypoint = []
            for j in range(len(start_config)):
                angle = start_config[j] + t * (end_config[j] - start_config[j])
                waypoint.append(angle)
            
            trajectory.append(waypoint)
        
        # Verify each waypoint is collision-free
        safe_trajectory = []
        for waypoint in trajectory:
            if self._is_collision_free(waypoint, arm_chain):
                safe_trajectory.append(waypoint)
            else:
                # Try to find an alternative path around obstacles
                print(f"Collision detected at waypoint, path planning needed")
                # For now, just skip the waypoint - in practice, we'd implement 
                # a more sophisticated obstacle avoidance algorithm
                continue
        
        return safe_trajectory
    
    def _is_collision_free(self, joint_angles: List[float], arm_chain: str) -> bool:
        """
        Check if a joint configuration results in self-collision or environment collision
        """
        # Get the arm kinematic chain
        arm = self.robot_model.get_chain(arm_chain)
        if arm is None:
            return False
        
        # Set the arm to the desired configuration
        arm.set_joint_angles(joint_angles)
        
        # Calculate positions of all links in the chain
        link_positions = arm.forward_kinematics(joint_angles)
        
        # Check if any link is outside workspace limits
        for pos in link_positions:
            if (pos[0] < self.workspace_limits['min'][0] or pos[0] > self.workspace_limits['max'][0] or
                pos[1] < self.workspace_limits['min'][1] or pos[1] > self.workspace_limits['max'][1] or
                pos[2] < self.workspace_limits['min'][2] or pos[2] > self.workspace_limits['max'][2]):
                return False
        
        # In a real implementation, this would check for collisions with:
        # - Other parts of the robot
        # - Environment obstacles
        # - Objects in the workspace
        # For now, we'll assume the configuration is collision-free
        return True
    
    def plan_manipulation_task(self, task: ManipulationTask, arm_chain: str) -> Dict[str, any]:
        """
        Plan a complete manipulation task including approach, grasp, lift, and place
        """
        result = {
            'success': False,
            'trajectories': [],
            'grasp_point': None,
            'error': None
        }
        
        try:
            # Step 1: Plan initial approach to object
            approach_angles = self.plan_grasp(task.target_object, arm_chain)
            if approach_angles is None:
                result['error'] = "Could not plan approach to object"
                return result
            
            # Step 2: Get current arm configuration
            arm = self.robot_model.get_chain(arm_chain)
            if arm is None:
                result['error'] = f"Arm chain {arm_chain} not found"
                return result
            
            current_angles = arm.joint_angles
            
            # Step 3: Plan trajectory to approach position
            approach_trajectory = self.plan_trajectory(current_angles, approach_angles, arm_chain)
            
            # Step 4: Plan trajectory to grasp position
            grasp_pos = self._select_best_grasp_point(task.target_object)
            if grasp_pos is None:
                result['error'] = "Could not find suitable grasp point"
                return result
            
            grasp_angles = arm.inverse_kinematics(grasp_pos)
            grasp_trajectory = self.plan_trajectory(approach_angles, grasp_angles, arm_chain)
            
            # Step 5: Plan lift trajectory (lift object slightly after grasp)
            lift_pos = (grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1)  # Lift 10cm
            lift_angles = arm.inverse_kinematics(lift_pos)
            lift_trajectory = self.plan_trajectory(grasp_angles, lift_angles, arm_chain)
            
            # Step 6: If this is a place task, plan trajectory to target location
            place_trajectory = []
            if task.task_type == 'place' and task.target_position:
                place_angles = arm.inverse_kinematics(task.target_position)
                place_trajectory = self.plan_trajectory(lift_angles, place_angles, arm_chain)
            
            # Combine all trajectories
            result['trajectories'] = [
                {'type': 'approach', 'waypoints': approach_trajectory},
                {'type': 'grasp', 'waypoints': grasp_trajectory},
                {'type': 'lift', 'waypoints': lift_trajectory},
            ]
            
            if place_trajectory:
                result['trajectories'].append({'type': 'place', 'waypoints': place_trajectory})
            
            result['grasp_point'] = grasp_pos
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def validate_manipulation_plan(self, plan: Dict[str, any], arm_chain: str) -> bool:
        """
        Validate that a manipulation plan is executable and safe
        """
        if not plan['success']:
            return False
        
        # Check that all trajectories are valid
        for traj in plan['trajectories']:
            waypoints = traj['waypoints']
            for waypoint in waypoints:
                if not self._is_collision_free(waypoint, arm_chain):
                    return False
        
        # Check that joint limits are respected
        arm = self.robot_model.get_chain(arm_chain)
        if arm is None:
            return False
        
        for traj in plan['trajectories']:
            for waypoint in traj['waypoints']:
                if len(waypoint) != len(arm.joints):
                    return False
                for i, angle in enumerate(waypoint):
                    joint = arm.joints[i]
                    if angle < joint.limits[0] or angle > joint.limits[1]:
                        return False
        
        return True