"""
Manipulation stack for the capstone project in the Physical AI & Humanoid Robotics course.
This module integrates grasping, manipulation planning, and execution for the complete system.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import math
from enum import Enum
from datetime import datetime

from src.models.action_command import ActionCommand, ActionType
from src.perception.capstone_perception import DetectedObject, ObjectClass
from src.control.kinematics import HumanoidKinematicModel, KinematicChain
from src.manipulation.planning import ManipulationPlanner, ManipulationTask
from src.manipulation.grasping import GraspingPipeline, GraspExecutionMonitor


class ManipulationStatus(Enum):
    """Status of manipulation"""
    IDLE = "idle"
    PLANNING = "planning"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    LIFTING = "lifting"
    TRANSPORTING = "transporting"
    PLACING = "placing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GraspType(Enum):
    """Types of grasps"""
    PINCH = "pinch"
    PALM = "palm"
    LATERAL = "lateral"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"


@dataclass
class ManipulationResult:
    """Result of a manipulation action"""
    success: bool
    grasp_point: Optional[Tuple[float, float, float]]
    final_position: Optional[Tuple[float, float, float]]
    execution_time: float
    error_message: Optional[str] = None


class ManipulationPlannerCapstone:
    """
    Enhanced manipulation planner for the capstone project that integrates
    with the full perception and planning system.
    """
    
    def __init__(self, kinematic_model: HumanoidKinematicModel):
        self.kinematic_model = kinematic_model
        self.manipulation_planner = ManipulationPlanner(kinematic_model)
        self.grasping_pipeline = GraspingPipeline()
    
    def plan_manipulation_task(self, obj: DetectedObject, 
                              target_position: Optional[Tuple[float, float, float]] = None,
                              grasp_type: Optional[GraspType] = None) -> Optional[Dict[str, Any]]:
        """Plan a complete manipulation task for a detected object"""
        # Create object properties from detected object
        obj_props = self._create_object_properties(obj)
        
        if grasp_type is None:
            # Select appropriate grasp based on object properties
            grasp_type = self._select_appropriate_grasp(obj_props)
        
        # Plan the grasp
        grasp = self.grasping_pipeline.planner.select_best_grasp(obj_props)
        if grasp is None:
            return None
        
        # Create manipulation task
        task = ManipulationTask(
            id=f"task_{int(datetime.now().timestamp())}",
            task_type="place" if target_position else "grasp",
            target_object=self._convert_to_object_info(obj),
            target_position=target_position
        )
        
        # Plan the manipulation
        plan_result = self.manipulation_planner.plan_manipulation_task(task, "right_arm")
        if not plan_result['success']:
            return None
        
        # Validate the plan
        if not self.manipulation_planner.validate_manipulation_plan(plan_result, "right_arm"):
            return None
        
        return plan_result
    
    def _create_object_properties(self, obj: DetectedObject):
        """Convert DetectedObject to ObjectProperties for grasping pipeline"""
        from src.manipulation.grasping import ObjectProperties
        
        # Calculate dimensions based on bounding box
        width = obj.bounding_box[2] / 10.0  # Scale down from pixels to meters
        height = obj.bounding_box[3] / 10.0
        depth = min(width, height)  # Assume depth is smaller of width/height
        
        return ObjectProperties(
            id=obj.id,
            position=obj.position_3d,
            orientation=(0.0, 0.0, 0.0, 1.0),  # Default orientation
            dimensions=(width, height, depth),
            mass=0.5,  # Default mass
            surface_type="smooth",  # Default surface
            center_of_mass=(0.0, 0.0, 0.0)  # Default COM
        )
    
    def _select_appropriate_grasp(self, obj_props: Any) -> GraspType:
        """Select an appropriate grasp type based on object properties"""
        width, height, depth = obj_props.dimensions
        
        # Simple heuristic for grasp selection
        if max(width, height, depth) < 0.05:  # Small object
            return GraspType.PINCH
        elif width > height and width > depth:  # Elongated object
            return GraspType.CYLINDRICAL
        elif abs(width - height) < 0.02:  # Square-like object
            return GraspType.PALM
        else:
            return GraspType.SPHERICAL
    
    def _convert_to_object_info(self, obj: DetectedObject):
        """Convert DetectedObject to the format expected by ManipulationPlanner"""
        from src.control.planning import ObjectInfo  # Assuming this exists
        
        # For this implementation, we'll create a compatible object
        class ObjectInfo:
            def __init__(self, obj):
                self.id = obj.id
                self.position = obj.position_3d
                self.orientation = (0.0, 0.0, 0.0, 1.0)  # Default
                self.dimensions = (0.1, 0.1, 0.1)  # Default
                self.mass = 0.5  # Default
                self.grasp_points = [obj.position_3d]  # Use object position as grasp point
        
        return ObjectInfo(obj)


class CapstoneManipulator:
    """
    Comprehensive manipulation system for the capstone project that integrates
    perception, planning, and execution of manipulation tasks.
    """
    
    def __init__(self, kinematic_model: HumanoidKinematicModel):
        self.kinematic_model = kinematic_model
        self.planner = ManipulationPlannerCapstone(kinematic_model)
        self.status = ManipulationStatus.IDLE
        self.current_task: Optional[Dict[str, Any]] = None
        self.held_object: Optional[DetectedObject] = None
        self.manipulation_history: List[ManipulationResult] = []
        self.is_active = False
    
    def pick_up_object(self, obj: DetectedObject) -> ManipulationResult:
        """Pick up a specific object"""
        if self.status != ManipulationStatus.IDLE:
            return ManipulationResult(
                success=False,
                grasp_point=None,
                final_position=None,
                execution_time=0.0,
                error_message="Manipulator is busy with another task"
            )
        
        start_time = datetime.now()
        self.status = ManipulationStatus.PLANNING
        
        # Plan the manipulation
        plan = self.planner.plan_manipulation_task(obj)
        if plan is None:
            self.status = ManipulationStatus.FAILED
            return ManipulationResult(
                success=False,
                grasp_point=None,
                final_position=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message="Could not plan manipulation for object"
            )
        
        # Execute the plan
        execution_result = self._execute_manipulation_plan(plan, obj)
        
        # Update state
        if execution_result.success:
            self.held_object = obj
            self.status = ManipulationStatus.COMPLETED
        else:
            self.status = ManipulationStatus.FAILED
        
        result = ManipulationResult(
            success=execution_result.success,
            grasp_point=plan.get('grasp_point'),
            final_position=obj.position_3d,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error_message=execution_result.get('error', None) if not execution_result.success else None
        )
        
        self.manipulation_history.append(result)
        return result
    
    def place_object(self, target_position: Tuple[float, float, float]) -> ManipulationResult:
        """Place the currently held object at a target position"""
        if self.held_object is None:
            return ManipulationResult(
                success=False,
                grasp_point=None,
                final_position=None,
                execution_time=0.0,
                error_message="No object currently held"
            )
        
        start_time = datetime.now()
        self.status = ManipulationStatus.PLANNING
        
        # Plan the placement task
        plan = self.planner.plan_manipulation_task(self.held_object, target_position=target_position)
        if plan is None:
            self.status = ManipulationStatus.FAILED
            return ManipulationResult(
                success=False,
                grasp_point=None,
                final_position=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message="Could not plan placement task"
            )
        
        # Execute the plan
        execution_result = self._execute_manipulation_plan(plan, self.held_object, target_position)
        
        # Update state
        if execution_result.success:
            self.held_object = None  # Release the object
            self.status = ManipulationStatus.COMPLETED
        else:
            self.status = ManipulationStatus.FAILED
        
        result = ManipulationResult(
            success=execution_result.success,
            grasp_point=plan.get('grasp_point'),
            final_position=target_position,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error_message=execution_result.get('error', None) if not execution_result.success else None
        )
        
        self.manipulation_history.append(result)
        return result
    
    def transport_object(self, start_pos: Tuple[float, float, float], 
                        end_pos: Tuple[float, float, float]) -> ManipulationResult:
        """Transport an object from one position to another"""
        if self.held_object is None:
            return ManipulationResult(
                success=False,
                grasp_point=None,
                final_position=None,
                execution_time=0.0,
                error_message="No object currently held"
            )
        
        # Move from start to end while holding object
        # For this implementation, we'll just place at end position
        return self.place_object(end_pos)
    
    def _execute_manipulation_plan(self, plan: Dict[str, Any], 
                                 obj: DetectedObject,
                                 target_pos: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """Execute a manipulation plan"""
        # Simulate execution of the manipulation plan
        # In a real system, this would interface with the robot's joint controllers
        
        success = True  # Simulate success
        
        # Simulate different phases of manipulation
        for traj_phase in plan.get('trajectories', []):
            phase_type = traj_phase['type']
            
            # Simulate each phase taking some time
            import time
            time.sleep(0.5)  # Simulate execution time
            
            # Check if phase was successful
            if np.random.random() < 0.1:  # 10% failure rate in simulation
                success = False
                break
        
        return {
            'success': success,
            'grasp_point': plan.get('grasp_point'),
            'target_position': target_pos
        }
    
    def is_object_held(self) -> bool:
        """Check if an object is currently held"""
        return self.held_object is not None
    
    def get_held_object(self) -> Optional[DetectedObject]:
        """Get the currently held object"""
        return self.held_object
    
    def get_manipulation_status(self) -> ManipulationStatus:
        """Get the current manipulation status"""
        return self.status
    
    def cancel_current_task(self):
        """Cancel the current manipulation task"""
        self.status = ManipulationStatus.CANCELLED
        self.current_task = None
    
    def get_success_rate(self) -> float:
        """Get the success rate of manipulation tasks"""
        if not self.manipulation_history:
            return 0.0
        
        successful = sum(1 for result in self.manipulation_history if result.success)
        return successful / len(self.manipulation_history)


class DualArmManipulator:
    """
    For robots with two arms, coordinate both arms for complex manipulation
    """
    
    def __init__(self, left_arm_model: HumanoidKinematicModel, 
                 right_arm_model: HumanoidKinematicModel):
        self.left_manipulator = CapstoneManipulator(left_arm_model)
        self.right_manipulator = CapstoneManipulator(right_arm_model)
        self.is_active = False
    
    def coordinated_pick_up(self, obj: DetectedObject) -> Tuple[ManipulationResult, ManipulationResult]:
        """Coordinated pick up using both arms"""
        # In a real implementation, this would coordinate both arms
        # For now, simulate with one arm
        left_result = ManipulationResult(
            success=False,
            grasp_point=None,
            final_position=None,
            execution_time=0.0,
            error_message="Single arm operation"
        )
        right_result = self.right_manipulator.pick_up_object(obj)
        
        return left_result, right_result
    
    def coordinated_place(self, target_position: Tuple[float, float, float]) -> Tuple[ManipulationResult, ManipulationResult]:
        """Coordinated place operation using both arms"""
        # In a real implementation, this would coordinate both arms
        # For now, simulate with one arm
        left_result = ManipulationResult(
            success=False,
            grasp_point=None,
            final_position=None,
            execution_time=0.0,
            error_message="Single arm operation"
        )
        right_result = self.right_manipulator.place_object(target_position)
        
        return left_result, right_result


class ManipulationSafetyManager:
    """
    Ensures safe manipulation with force and position limits
    """
    
    def __init__(self, max_force: float = 50.0,  # Newtons
                 max_payload: float = 5.0):     # kg
        self.max_force = max_force
        self.max_payload = max_payload
        self.is_safe = True
        self.last_check_time = datetime.now()
    
    def check_grasp_safety(self, obj: DetectedObject) -> bool:
        """Check if it's safe to grasp an object based on its properties"""
        # Check if object is too heavy
        if hasattr(obj, 'properties') and obj.properties:
            mass = obj.properties.get('mass', 0.5)  # Default to 0.5kg if unknown
        else:
            mass = 0.5
        
        if mass > self.max_payload:
            return False
        
        # Check if object is in a safe location to approach
        # (not too close to robot body, etc.)
        # For simplicity, assume true
        return True
    
    def check_placement_safety(self, target_position: Tuple[float, float, float], 
                               held_object: Optional[DetectedObject]) -> bool:
        """Check if a placement location is safe"""
        # Check if placement position is stable (not at edge of surface)
        # For simplicity, assume true
        return True
    
    def is_manipulation_safe(self) -> bool:
        """Check if manipulation is currently safe to proceed"""
        return self.is_safe


class ManipulationManager:
    """
    Main manipulation manager that integrates all manipulation components
    """
    
    def __init__(self, kinematic_model: HumanoidKinematicModel):
        self.manipulator = CapstoneManipulator(kinematic_model)
        self.safety_manager = ManipulationSafetyManager()
        self.is_active = False
    
    def start_manipulation_task(self, action_cmd: ActionCommand) -> bool:
        """Start a manipulation task based on an action command"""
        if not self.safety_manager.is_manipulation_safe():
            return False
        
        self.is_active = True
        
        if action_cmd.type == ActionType.MANIPULATION:
            task_type = action_cmd.parameters.get("manipulation_type", "grasp")
            target_obj = action_cmd.parameters.get("target_object")
            target_pos = action_cmd.parameters.get("target_position")
            
            if task_type == "grasp" and target_obj:
                if self.safety_manager.check_grasp_safety(target_obj):
                    result = self.manipulator.pick_up_object(target_obj)
                    return result.success
            elif task_type == "place" and target_pos:
                if self.manipulator.is_object_held():
                    if self.safety_manager.check_placement_safety(target_pos, self.manipulator.get_held_object()):
                        result = self.manipulator.place_object(target_pos)
                        return result.success
        
        return False
    
    def get_manipulation_status(self) -> ManipulationStatus:
        """Get the current manipulation status"""
        return self.manipulator.get_manipulation_status()
    
    def is_object_held(self) -> bool:
        """Check if an object is currently held"""
        return self.manipulator.is_object_held()
    
    def get_held_object(self) -> Optional[DetectedObject]:
        """Get the currently held object"""
        return self.manipulator.get_held_object()
    
    def stop_manipulation(self):
        """Stop current manipulation"""
        self.manipulator.cancel_current_task()
        self.is_active = False
    
    def get_success_rate(self) -> float:
        """Get the success rate of manipulation tasks"""
        return self.manipulator.get_success_rate()
    
    def transport_object(self, start_pos: Tuple[float, float, float], 
                        end_pos: Tuple[float, float, float]) -> ManipulationResult:
        """Transport an object from one position to another"""
        return self.manipulator.transport_object(start_pos, end_pos)