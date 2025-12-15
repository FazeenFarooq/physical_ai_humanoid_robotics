"""
Grasping pipeline for the Physical AI & Humanoid Robotics course.
This module implements the complete pipeline for robot grasping tasks.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math
from enum import Enum


class GraspType(Enum):
    """Types of grasps a robot can perform"""
    PALM_GRASP = "palm_grasp"
    PINCH_GRASP = "pinch_grasp"
    LATERAL_GRASP = "lateral_grasp"
    CYLINDRICAL_GRASP = "cylindrical_grasp"
    SPHERICAL_GRASP = "spherical_grasp"
    FINGERTIP_GRASP = "fingertip_grasp"


@dataclass
class GraspPose:
    """Defines a grasp pose with position, orientation, and grasp type"""
    position: Tuple[float, float, float]  # Position in 3D space
    orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    grasp_type: GraspType
    grasp_width: float  # Required gripper width in meters
    approach_direction: Tuple[float, float, float]  # Approach direction as unit vector
    lift_direction: Tuple[float, float, float]  # Lift direction as unit vector


@dataclass
class ObjectProperties:
    """Properties of an object to be grasped"""
    id: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # Quaternion
    dimensions: Tuple[float, float, float]  # width, height, depth
    mass: float
    surface_type: str  # smooth, rough, soft, etc.
    center_of_mass: Tuple[float, float, float]  # Offset from geometric center


class GraspQualityEvaluator:
    """Evaluates the quality of potential grasps"""
    
    def __init__(self):
        self.quality_weights = {
            'force_closure': 0.3,
            'stability': 0.25,
            'accessibility': 0.2,
            'object_compatibility': 0.15,
            'task_compatibility': 0.1
        }
    
    def evaluate_grasp(self, grasp_pose: GraspPose, obj_props: ObjectProperties) -> float:
        """Evaluate the quality of a grasp (0.0 to 1.0)"""
        quality = 0.0
        
        # Evaluate force closure (ability to resist external forces)
        force_closure_score = self._evaluate_force_closure(grasp_pose, obj_props)
        
        # Evaluate stability (resistance to object slippage)
        stability_score = self._evaluate_stability(grasp_pose, obj_props)
        
        # Evaluate accessibility (feasibility of reaching the grasp pose)
        accessibility_score = self._evaluate_accessibility(grasp_pose)
        
        # Evaluate object compatibility (grasp type matches object)
        object_compatibility_score = self._evaluate_object_compatibility(grasp_pose, obj_props)
        
        # Evaluate task compatibility (grasp is appropriate for subsequent task)
        task_compatibility_score = self._evaluate_task_compatibility(grasp_pose)
        
        # Weighted combination of all factors
        quality = (
            self.quality_weights['force_closure'] * force_closure_score +
            self.quality_weights['stability'] * stability_score +
            self.quality_weights['accessibility'] * accessibility_score +
            self.quality_weights['object_compatibility'] * object_compatibility_score +
            self.quality_weights['task_compatibility'] * task_compatibility_score
        )
        
        return min(1.0, max(0.0, quality))
    
    def _evaluate_force_closure(self, grasp_pose: GraspPose, obj_props: ObjectProperties) -> float:
        """Evaluate if the grasp provides force closure"""
        # Simplified evaluation - in reality, this would involve complex physics
        # calculations to determine if the grasp can resist external forces
        # For now, we'll return a score based on grasp type and object properties
        if grasp_pose.grasp_type in [GraspType.CYLINDRICAL_GRASP, GraspType.SPHERICAL_GRASP]:
            return 0.8
        elif grasp_pose.grasp_type in [GraspType.PALM_GRASP, GraspType.PINCH_GRASP]:
            return 0.6
        else:
            return 0.4
    
    def _evaluate_stability(self, grasp_pose: GraspPose, obj_props: ObjectProperties) -> float:
        """Evaluate the stability of the grasp"""
        # Consider the object's center of mass and how it aligns with the grasp
        # For now, return a simplified score
        return 0.7  # Default stability score
    
    def _evaluate_accessibility(self, grasp_pose: GraspPose) -> float:
        """Evaluate if the grasp pose is accessible to the robot"""
        # In a real implementation, this would check if the robot can physically
        # reach the grasp pose without joint limits or collisions
        # For now, return a high score assuming it's accessible
        return 0.9
    
    def _evaluate_object_compatibility(self, grasp_pose: GraspPose, obj_props: ObjectProperties) -> float:
        """Evaluate if the grasp type is compatible with the object"""
        # Check if grasp width is appropriate for object size
        obj_size = max(obj_props.dimensions)
        if abs(obj_props.dimensions[0] - grasp_pose.grasp_width) < 0.02:  # 2cm tolerance
            width_compatible = 1.0
        else:
            width_compatible = max(0.0, 1.0 - abs(obj_props.dimensions[0] - grasp_pose.grasp_width) / obj_size)
        
        # Match grasp type to object shape
        if obj_props.dimensions[0] > obj_props.dimensions[1] and obj_props.dimensions[0] > obj_props.dimensions[2]:
            # Elongated object - good for cylindrical grasp
            shape_compatible = 1.0 if grasp_pose.grasp_type == GraspType.CYLINDRICAL_GRASP else 0.6
        elif obj_props.dimensions[0] == obj_props.dimensions[1] == obj_props.dimensions[2]:
            # Cubical object - good for palm grasp
            shape_compatible = 1.0 if grasp_pose.grasp_type == GraspType.PALM_GRASP else 0.7
        else:
            # Other shapes - medium compatibility
            shape_compatible = 0.8
        
        return (width_compatible + shape_compatible) / 2.0
    
    def _evaluate_task_compatibility(self, grasp_pose: GraspPose) -> float:
        """Evaluate if the grasp is compatible with the intended task"""
        # For now, return a default score
        return 0.8


class GraspPlanner:
    """
    Planner for robot grasping tasks including grasp selection,
    approach planning, and execution monitoring.
    """
    
    def __init__(self):
        self.evaluator = GraspQualityEvaluator()
        self.min_quality_threshold = 0.5
    
    def generate_grasp_candidates(self, obj_props: ObjectProperties) -> List[GraspPose]:
        """Generate possible grasp poses for an object"""
        candidates = []
        
        # Generate grasp poses based on object dimensions
        width, height, depth = obj_props.dimensions
        
        # Center grasp on top surface
        top_grasp = GraspPose(
            position=(
                obj_props.position[0],
                obj_props.position[1],
                obj_props.position[2] + height/2 + 0.02  # Slightly above top
            ),
            orientation=(0.0, 0.0, 0.0, 1.0),  # Default orientation
            grasp_type=GraspType.PALM_GRASP,
            grasp_width=min(width, depth) * 0.8,  # 80% of smaller dimension
            approach_direction=(0.0, 0.0, -1.0),  # Approach from above
            lift_direction=(0.0, 0.0, 1.0)  # Lift upward
        )
        candidates.append(top_grasp)
        
        # Side grasps for cylindrical objects
        side_grasp_1 = GraspPose(
            position=(
                obj_props.position[0] + width/2 + 0.02,
                obj_props.position[1],
                obj_props.position[2]
            ),
            orientation=(0.0, 0.707, 0.0, 0.707),  # Rotated 90 degrees
            grasp_type=GraspType.CYLINDRICAL_GRASP,
            grasp_width=height * 0.8,
            approach_direction=(-1.0, 0.0, 0.0),  # Approach from +X direction
            lift_direction=(0.0, 0.0, 1.0)  # Lift upward
        )
        candidates.append(side_grasp_1)
        
        side_grasp_2 = GraspPose(
            position=(
                obj_props.position[0] - width/2 - 0.02,
                obj_props.position[1],
                obj_props.position[2]
            ),
            orientation=(0.0, 0.707, 0.0, 0.707),  # Rotated 90 degrees
            grasp_type=GraspType.CYLINDRICAL_GRASP,
            grasp_width=height * 0.8,
            approach_direction=(1.0, 0.0, 0.0),  # Approach from -X direction
            lift_direction=(0.0, 0.0, 1.0)  # Lift upward
        )
        candidates.append(side_grasp_2)
        
        # Pinch grasp if object is small enough
        if min(width, height, depth) < 0.1:  # 10cm threshold
            pinch_grasp = GraspPose(
                position=(
                    obj_props.position[0],
                    obj_props.position[1] + depth/2 + 0.02,
                    obj_props.position[2]
                ),
                orientation=(0.707, 0.0, 0.0, 0.707),  # Rotated 90 degrees around X
                grasp_type=GraspType.PINCH_GRASP,
                grasp_width=min(width, height) * 0.8,
                approach_direction=(0.0, -1.0, 0.0),  # Approach from +Y direction
                lift_direction=(0.0, 0.0, 1.0)  # Lift upward
            )
            candidates.append(pinch_grasp)
        
        return candidates
    
    def select_best_grasp(self, obj_props: ObjectProperties) -> Optional[GraspPose]:
        """Select the best grasp for an object based on quality evaluation"""
        candidates = self.generate_grasp_candidates(obj_props)
        
        best_grasp = None
        best_quality = 0.0
        
        for candidate in candidates:
            quality = self.evaluator.evaluate_grasp(candidate, obj_props)
            if quality > best_quality:
                best_quality = quality
                best_grasp = candidate
        
        if best_grasp and best_quality >= self.min_quality_threshold:
            return best_grasp
        else:
            return None  # No acceptable grasp found
    
    def plan_approach_trajectory(self, grasp_pose: GraspPose, 
                                approach_distance: float = 0.1) -> List[Tuple[float, float, float]]:
        """Plan the approach trajectory to the grasp point"""
        # Calculate approach start position
        start_pos = (
            grasp_pose.position[0] + grasp_pose.approach_direction[0] * approach_distance,
            grasp_pose.position[1] + grasp_pose.approach_direction[1] * approach_distance,
            grasp_pose.position[2] + grasp_pose.approach_direction[2] * approach_distance
        )
        
        # Create trajectory with intermediate waypoints
        # For simplicity, we'll create a linear trajectory
        trajectory = []
        
        # Add intermediate waypoints
        for i in range(5):  # 5 waypoints
            t = i / 4.0  # Parameter from 0 to 1
            pos = (
                start_pos[0] + t * (grasp_pose.position[0] - start_pos[0]),
                start_pos[1] + t * (grasp_pose.position[1] - start_pos[1]),
                start_pos[2] + t * (grasp_pose.position[2] - start_pos[2])
            )
            trajectory.append(pos)
        
        return trajectory


class GraspExecutionMonitor:
    """Monitors the execution of a grasp to detect failures"""
    
    def __init__(self):
        self.is_grasping = False
        self.current_grasp = None
    
    def start_grasp(self, grasp_pose: GraspPose):
        """Begin monitoring a grasp execution"""
        self.is_grasping = True
        self.current_grasp = grasp_pose
    
    def check_grasp_success(self, force_sensors: List[float], 
                           position_error: float, 
                           slip_detected: bool) -> Tuple[bool, str]:
        """Check if the grasp was successful based on sensor feedback"""
        if not self.is_grasping:
            return False, "No active grasp"
        
        # Check if sufficient force is detected
        sufficient_force = any(force > 5.0 for force in force_sensors)  # 5.0N threshold
        
        # Check if position error is within tolerance
        position_ok = position_error < 0.01  # 1cm tolerance
        
        # Check if slip is detected
        no_slip = not slip_detected
        
        success = sufficient_force and position_ok and no_slip
        
        if success:
            status = "Grasp successful"
        elif not sufficient_force:
            status = "Insufficient grip force"
        elif not position_ok:
            status = "Position error too large"
        elif slip_detected:
            status = "Slip detected during grasp"
        else:
            status = "Unknown failure"
        
        return success, status
    
    def release_grasp(self):
        """End the current grasp monitoring"""
        self.is_grasping = False
        self.current_grasp = None


class GraspingPipeline:
    """The complete grasping pipeline orchestrating perception, planning, and execution"""
    
    def __init__(self):
        self.planner = GraspPlanner()
        self.monitor = GraspExecutionMonitor()
    
    def execute_grasp(self, obj_props: ObjectProperties) -> Dict[str, any]:
        """Execute the complete grasping pipeline"""
        result = {
            'success': False,
            'grasp_pose': None,
            'approach_trajectory': [],
            'status': '',
            'execution_log': []
        }
        
        try:
            # Step 1: Select best grasp
            best_grasp = self.planner.select_best_grasp(obj_props)
            if best_grasp is None:
                result['status'] = "No suitable grasp found"
                return result
            
            result['grasp_pose'] = best_grasp
            result['execution_log'].append(f"Selected grasp: {best_grasp.grasp_type.value}")
            
            # Step 2: Plan approach trajectory
            approach_trajectory = self.planner.plan_approach_trajectory(best_grasp)
            result['approach_trajectory'] = approach_trajectory
            result['execution_log'].append(f"Planned approach with {len(approach_trajectory)} waypoints")
            
            # Step 3: Simulate approach and grasp execution
            # (In a real system, this would interface with robot controllers)
            result['execution_log'].append("Executing approach trajectory")
            result['execution_log'].append("At grasp position")
            result['execution_log'].append("Closing gripper")
            
            # Step 4: Monitor grasp success
            # Simulate sensor readings
            force_sensors = [8.0, 7.5]  # Simulated force sensor readings
            position_error = 0.005  # 5mm error
            slip_detected = False  # No slip detected
            
            success, status = self.monitor.check_grasp_success(force_sensors, position_error, slip_detected)
            result['success'] = success
            result['status'] = status
            result['execution_log'].append(f"Grasp result: {status}")
            
        except Exception as e:
            result['status'] = f"Error during grasp execution: {str(e)}"
        
        return result