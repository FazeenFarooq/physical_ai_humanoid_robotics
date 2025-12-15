"""
Planning stack for the capstone project in the Physical AI & Humanoid Robotics course.
This module integrates task planning, path planning, and action planning for the complete system.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
from datetime import datetime
import heapq
import math

from src.models.task_plan import TaskPlan, TaskStep
from src.models.action_command import ActionCommand, ActionType
from src.perception.capstone_perception import EnvironmentalMap, DetectedObject, ObjectClass
from src.navigation.capstone_navigation import PathPlanner
from src.manipulation.capstone_manipulation import ManipulationPlanner
from src.conversation.capstone_conversation import ConversationManager


class PlanStatus(Enum):
    """Status of a plan"""
    PENDING = "pending"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanType(Enum):
    """Type of plan"""
    TASK = "task"
    PATH = "path"
    ACTION = "action"
    COMPOSITE = "composite"


@dataclass
class PlanningContext:
    """Context information for planning"""
    environmental_map: EnvironmentalMap
    detected_objects: List[DetectedObject]
    robot_pose: Tuple[float, float, float]  # x, y, theta
    robot_capabilities: Dict[str, Any]
    task_requirements: Dict[str, Any]
    time_limit: Optional[float] = None
    energy_limit: Optional[float] = None


@dataclass
class PlanStep:
    """A step in a plan"""
    id: str
    action: ActionCommand
    precondition: Optional[str] = None  # Condition that must be true to execute
    postcondition: Optional[str] = None  # Condition that will be true after execution
    cost: float = 0.0
    duration_estimate: float = 0.0


@dataclass
class Plan:
    """A complete plan"""
    id: str
    plan_type: PlanType
    steps: List[PlanStep]
    context: PlanningContext
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = None
    validated_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    total_cost: float = 0.0
    estimated_duration: float = 0.0
    priority: int = 5
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Calculate total cost and duration
        self.total_cost = sum(step.cost for step in self.steps)
        self.estimated_duration = sum(step.duration_estimate for step in self.steps)


class CapstonePlanner:
    """
    Main planning system for the capstone project that integrates 
    task, path, and action planning capabilities.
    """
    
    def __init__(self):
        self.path_planner = PathPlanner()
        self.manipulation_planner = ManipulationPlanner()
        self.conversation_manager = ConversationManager()
        self.plans: Dict[str, Plan] = {}
        self.active_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
        
        # Capabilities of the robot
        self.robot_capabilities = {
            "navigation": True,
            "manipulation": True,
            "conversation": True,
            "perception": True,
            "max_speed": 1.0,  # m/s
            "max_payload": 2.0,  # kg
            "reach_distance": 1.0,  # m
            "manipulation_dexterity": "high"
        }
    
    def create_task_plan(self, task_description: str, context: PlanningContext) -> Optional[Plan]:
        """Create a high-level task plan from natural language description"""
        # Parse the task description to identify required actions
        action_sequence = self._parse_task_description(task_description)
        
        if not action_sequence:
            return None
        
        # Create plan steps based on the action sequence
        plan_steps = []
        for i, action_desc in enumerate(action_sequence):
            action_id = f"action_{i}_{int(datetime.now().timestamp() * 1000)}"
            
            # Create an appropriate ActionCommand based on the description
            action_cmd = self._create_action_command(action_desc, context)
            
            # Create plan step
            step = PlanStep(
                id=f"step_{i}",
                action=action_cmd,
                precondition=None,  # Determine based on action
                postcondition=None,  # Determine based on action
                cost=self._estimate_action_cost(action_cmd),
                duration_estimate=self._estimate_action_duration(action_cmd)
            )
            
            plan_steps.append(step)
        
        # Create the plan
        plan_id = f"plan_{int(datetime.now().timestamp() * 1000)}"
        plan = Plan(
            id=plan_id,
            plan_type=PlanType.TASK,
            steps=plan_steps,
            context=context,
            priority=5
        )
        
        # Validate the plan
        if self.validate_plan(plan):
            plan.status = PlanStatus.VALID
            self.plans[plan_id] = plan
            return plan
        else:
            plan.status = PlanStatus.INVALID
            return None
    
    def create_composite_plan(self, task_plan: TaskPlan, context: PlanningContext) -> Optional[Plan]:
        """Create a detailed execution plan for a TaskPlan"""
        plan_steps = []
        
        for i, task_step in enumerate(task_plan.steps):
            # Convert the task step into specific action commands
            action_cmds = self._create_actions_for_task_step(task_step, context)
            
            for j, action_cmd in enumerate(action_cmds):
                step = PlanStep(
                    id=f"step_{i}_{j}",
                    action=action_cmd,
                    cost=self._estimate_action_cost(action_cmd),
                    duration_estimate=self._estimate_action_duration(action_cmd)
                )
                
                plan_steps.append(step)
        
        # Create the plan
        plan_id = f"composite_{task_plan.id}_{int(datetime.now().timestamp() * 1000)}"
        plan = Plan(
            id=plan_id,
            plan_type=PlanType.COMPOSITE,
            steps=plan_steps,
            context=context,
            priority=task_plan.priority.value if hasattr(task_plan.priority, 'value') else 5
        )
        
        # Validate the plan
        if self.validate_plan(plan):
            plan.status = PlanStatus.VALID
            self.plans[plan_id] = plan
            return plan
        else:
            plan.status = PlanStatus.INVALID
            return None
    
    def validate_plan(self, plan: Plan) -> bool:
        """Validate that a plan is executable given current context"""
        # Check if robot has required capabilities
        for step in plan.steps:
            if not self._has_capability_for_action(step.action):
                return False
        
        # Check resource constraints
        if plan.context.energy_limit:
            total_energy_cost = sum(self._estimate_energy_cost(step.action) for step in plan.steps)
            if total_energy_cost > plan.context.energy_limit:
                return False
        
        # Check time constraints
        if plan.context.time_limit and plan.estimated_duration > plan.context.time_limit:
            return False
        
        # Validate each step can be executed
        for step in plan.steps:
            if not self._validate_action(step.action, plan.context):
                return False
        
        plan.validated_at = datetime.now()
        return True
    
    def execute_plan(self, plan_id: str) -> bool:
        """Execute a plan by its ID"""
        if plan_id not in self.plans:
            return False
        
        plan = self.plans[plan_id]
        
        if plan.status != PlanStatus.VALID:
            return False
        
        self.active_plan = plan
        plan.status = PlanStatus.EXECUTING
        plan.executed_at = datetime.now()
        
        # Execute each step in sequence
        for step in plan.steps:
            if not self._execute_plan_step(step, plan.context):
                plan.status = PlanStatus.FAILED
                self.active_plan = None
                return False
        
        plan.status = PlanStatus.COMPLETED
        self.plan_history.append(plan)
        self.active_plan = None
        
        return True
    
    def replan(self, failed_plan_id: str, new_context: PlanningContext) -> Optional[Plan]:
        """Create a new plan when the current one fails"""
        if failed_plan_id not in self.plans:
            return None
        
        failed_plan = self.plans[failed_plan_id]
        
        # Create a new plan with the updated context
        # This is a simplified replanning - in a real system, you'd analyze what went wrong
        new_plan = Plan(
            id=f"replan_{failed_plan_id}_{int(datetime.now().timestamp() * 1000)}",
            plan_type=failed_plan.plan_type,
            steps=failed_plan.steps[:],  # Copy steps
            context=new_context,
            priority=failed_plan.priority + 1  # Increase priority since original failed
        )
        
        # Modify steps as needed based on new context
        for step in new_plan.steps:
            # Adjust action parameters based on new context
            self._adjust_action_for_context(step.action, new_context)
        
        # Revalidate the plan
        if self.validate_plan(new_plan):
            new_plan.status = PlanStatus.VALID
            self.plans[new_plan.id] = new_plan
            return new_plan
        else:
            new_plan.status = PlanStatus.INVALID
            return None
    
    def _parse_task_description(self, description: str) -> List[str]:
        """Parse natural language task description into actionable steps"""
        # Simplified parsing - in a real system, this would use NLP techniques
        description_lower = description.lower()
        
        # Keywords for different types of actions
        navigation_keywords = ["go to", "move to", "navigate", "walk", "go"]
        manipulation_keywords = ["pick", "grasp", "take", "put", "place", "lift", "move object"]
        conversation_keywords = ["say", "speak", "tell", "ask", "greet"]
        
        actions = []
        
        # Identify navigation actions
        for keyword in navigation_keywords:
            if keyword in description_lower:
                # Extract location if mentioned
                import re
                location_match = re.search(rf"{keyword}\s+(?:the\s+)?(\w+)", description_lower)
                if location_match:
                    location = location_match.group(1)
                    actions.append(f"navigate_to_{location}")
        
        # Identify manipulation actions
        for keyword in manipulation_keywords:
            if keyword in description_lower:
                # Extract object if mentioned
                import re
                obj_match = re.search(rf"(\w+)\s+{keyword.split()[0] if ' ' in keyword else keyword}", description_lower)
                if obj_match:
                    obj = obj_match.group(1)
                    actions.append(f"manipulate_{obj}")
        
        # Identify conversation actions
        for keyword in conversation_keywords:
            if keyword in description_lower:
                actions.append(f"communicate_{keyword}")
        
        return actions
    
    def _create_action_command(self, action_desc: str, context: PlanningContext) -> ActionCommand:
        """Create an ActionCommand based on action description and context"""
        action_id = f"cmd_{int(datetime.now().timestamp() * 1000)}"
        
        if "navigate" in action_desc:
            # Navigation action
            location = action_desc.split("navigate_to_")[-1] if "navigate_to_" in action_desc else "default"
            
            # Find location in context
            target_position = self._find_position_for_location(location, context)
            
            action_cmd = ActionCommand(
                id=action_id,
                type=ActionType.NAVIGATION,
                parameters={
                    "target_position": target_position,
                    "path_planning_algorithm": "a_star"
                },
                priority=7  # Navigation is usually high priority
            )
        elif "manipulate" in action_desc:
            # Manipulation action
            obj = action_desc.split("manipulate_")[-1] if "manipulate_" in action_desc else "object"
            
            # Find object in context
            target_object = self._find_object_by_name(obj, context)
            
            action_cmd = ActionCommand(
                id=action_id,
                type=ActionType.MANIPULATION,
                parameters={
                    "target_object": target_object,
                    "manipulation_type": "grasp"
                },
                priority=8  # Manipulation is usually high priority
            )
        elif "communicate" in action_desc:
            # Conversation action
            action_cmd = ActionCommand(
                id=action_id,
                type=ActionType.CONVERSATION,
                parameters={
                    "message": f"Performing {action_desc.split('communicate_')[-1]} action",
                    "voice_type": "neutral"
                },
                priority=5
            )
        else:
            # Default to system control
            action_cmd = ActionCommand(
                id=action_id,
                type=ActionType.SYSTEM_CONTROL,
                parameters={"command": action_desc},
                priority=5
            )
        
        return action_cmd
    
    def _find_position_for_location(self, location: str, context: PlanningContext) -> Tuple[float, float, float]:
        """Find the position for a named location in the context"""
        # In a real system, this would look up a map of named locations
        # For simulation, return a default position
        location_positions = {
            "kitchen": (2.0, 1.0, 0.0),
            "living_room": (-1.0, 2.0, 0.0),
            "bedroom": (-2.0, -1.0, 0.0),
            "office": (1.0, -2.0, 0.0),
            "default": (0.0, 0.0, 0.0)
        }
        
        return location_positions.get(location, (0.0, 0.0, 0.0))
    
    def _find_object_by_name(self, obj_name: str, context: PlanningContext) -> Optional[DetectedObject]:
        """Find a detected object by name in the context"""
        for obj in context.detected_objects:
            if obj_name.lower() in obj.class_type.value.lower():
                return obj
        return None
    
    def _estimate_action_cost(self, action: ActionCommand) -> float:
        """Estimate the cost of executing an action"""
        base_costs = {
            ActionType.NAVIGATION: 1.0,
            ActionType.MANIPULATION: 2.0,
            ActionType.CONVERSATION: 0.5,
            ActionType.SYSTEM_CONTROL: 0.1,
            ActionType.DATA_COLLECTION: 0.8
        }
        
        cost = base_costs.get(action.type, 1.0)
        
        # Adjust based on parameters
        if action.type == ActionType.NAVIGATION:
            # Cost increases with distance
            target = action.parameters.get("target_position", (0, 0, 0))
            # Estimate from current position - for now, assume current position is (0,0,0)
            distance = math.sqrt(target[0]**2 + target[1]**2)
            cost += distance * 0.1  # 0.1 cost per meter
        
        return cost
    
    def _estimate_action_duration(self, action: ActionCommand) -> float:
        """Estimate the duration of an action"""
        base_durations = {
            ActionType.NAVIGATION: 10.0,  # seconds
            ActionType.MANIPULATION: 15.0,
            ActionType.CONVERSATION: 5.0,
            ActionType.SYSTEM_CONTROL: 2.0,
            ActionType.DATA_COLLECTION: 8.0
        }
        
        duration = base_durations.get(action.type, 5.0)
        
        # Adjust based on parameters
        if action.type == ActionType.NAVIGATION:
            # Duration increases with distance
            target = action.parameters.get("target_position", (0, 0, 0))
            # Estimate from current position - for now, assume current position is (0,0,0)
            distance = math.sqrt(target[0]**2 + target[1]**2)
            max_speed = self.robot_capabilities.get("max_speed", 1.0)
            duration = max(duration, distance / max_speed)
        
        return duration
    
    def _estimate_energy_cost(self, action: ActionCommand) -> float:
        """Estimate the energy cost of an action"""
        # Simplified energy cost estimation
        energy_costs = {
            ActionType.NAVIGATION: 0.1,  # Units of energy per meter
            ActionType.MANIPULATION: 0.05,  # Units per manipulation
            ActionType.CONVERSATION: 0.001,  # Very low for conversation
            ActionType.SYSTEM_CONTROL: 0.01,
            ActionType.DATA_COLLECTION: 0.02
        }
        
        energy = energy_costs.get(action.type, 0.01)
        
        if action.type == ActionType.NAVIGATION:
            # Energy increases with distance
            target = action.parameters.get("target_position", (0, 0, 0))
            distance = math.sqrt(target[0]**2 + target[1]**2)
            energy += distance * 0.1  # 0.1 energy units per meter
        
        return energy
    
    def _has_capability_for_action(self, action: ActionCommand) -> bool:
        """Check if robot has capability to execute an action"""
        capability_requirements = {
            ActionType.NAVIGATION: "navigation",
            ActionType.MANIPULATION: "manipulation",
            ActionType.CONVERSATION: "conversation",
            ActionType.PERCEPTION: "perception"
        }
        
        required_capability = capability_requirements.get(action.type)
        if required_capability:
            return self.robot_capabilities.get(required_capability, False)
        
        # For other action types, assume capability exists
        return True
    
    def _validate_action(self, action: ActionCommand, context: PlanningContext) -> bool:
        """Validate if an action can be executed in the current context"""
        if action.type == ActionType.NAVIGATION:
            # Check if target location is accessible
            target_pos = action.parameters.get("target_position", (0, 0, 0))
            # Check if path is clear (use path planner to check)
            path = self.path_planner.plan_path(context.robot_pose, target_pos[:2], context.environmental_map)
            return path is not None and len(path) > 0
        
        elif action.type == ActionType.MANIPULATION:
            # Check if target object is reachable
            target_obj = action.parameters.get("target_object")
            if target_obj:
                # Check if object is within reach
                obj_pos = target_obj.position_3d
                robot_pos = context.robot_pose
                distance = math.sqrt((obj_pos[0]-robot_pos[0])**2 + (obj_pos[1]-robot_pos[1])**2)
                reach_distance = self.robot_capabilities.get("reach_distance", 1.0)
                return distance <= reach_distance
        
        # For other action types, assume validation passes
        return True
    
    def _execute_plan_step(self, step: PlanStep, context: PlanningContext) -> bool:
        """Execute a single step of a plan"""
        # In a real system, this would communicate with the appropriate subsystems
        # For simulation, we'll just return success
        
        # Update action status
        step.action.start_execution(executor_name="CapstonePlanner")
        
        # Simulate execution
        import time
        time.sleep(min(0.1, step.duration_estimate))  # Simulate execution time
        
        # Complete the action
        step.action.complete_execution(success=True, result={"status": "completed"})
        
        return True
    
    def _adjust_action_for_context(self, action: ActionCommand, context: PlanningContext):
        """Adjust action parameters based on new context"""
        # This would modify action parameters based on changes in the context
        # For example, if a path is blocked, adjust navigation action to go around
        pass
    
    def get_active_plan_status(self) -> Optional[str]:
        """Get the status of the currently active plan"""
        if self.active_plan:
            return self.active_plan.status.value
        return None
    
    def interrupt_active_plan(self) -> bool:
        """Interrupt the currently active plan"""
        if self.active_plan:
            self.active_plan.status = PlanStatus.CANCELLED
            self.active_plan = None
            return True
        return False
    
    def get_plan_by_id(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by its ID"""
        return self.plans.get(plan_id)


class HierarchicalTaskNetwork:
    """
    Hierarchical Task Network (HTN) planner for complex task decomposition
    """
    
    def __init__(self):
        self.decomposition_rules = {}
        self.primitive_actions = set()
        
        # Define some basic decomposition rules
        self._initialize_decomposition_rules()
    
    def _initialize_decomposition_rules(self):
        """Initialize basic task decomposition rules"""
        # Rule: "fetch object" decomposes into navigation + manipulation
        self.decomposition_rules["fetch_object"] = [
            "navigate_to_object",
            "grasp_object"
        ]
        
        # Rule: "deliver object" decomposes into fetch + navigation + place
        self.decomposition_rules["deliver_object"] = [
            "fetch_object",
            "navigate_to_location",
            "place_object"
        ]
        
        # Define primitive actions (actions that cannot be decomposed)
        self.primitive_actions = {
            "navigate_to_object",
            "navigate_to_location",
            "grasp_object",
            "place_object",
            "say_something"
        }
    
    def decompose_task(self, task_name: str, params: Dict[str, Any] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Decompose a high-level task into primitive actions"""
        if params is None:
            params = {}
        
        if task_name in self.primitive_actions:
            # This is already a primitive action
            return [(task_name, params)]
        
        if task_name not in self.decomposition_rules:
            # No rule for this task
            return []
        
        subtasks = []
        for subtask in self.decomposition_rules[task_name]:
            if subtask in self.primitive_actions:
                # Direct primitive action
                subtasks.append((subtask, params))
            else:
                # Recursively decompose
                subtasks.extend(self.decompose_task(subtask, params))
        
        return subtasks