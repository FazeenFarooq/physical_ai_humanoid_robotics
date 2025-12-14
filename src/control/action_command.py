"""
Action Command Generation for the Physical AI & Humanoid Robotics Course.

This module provides tools for generating robot action commands based on natural language
instructions and perception data. It bridges the gap between high-level goals and
low-level control commands.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math
import json
import logging
from enum import Enum


class ActionType(Enum):
    """Type of action command."""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    INTERACTION = "interaction"
    PERCEPTION = "perception"
    CONTROL = "control"


@dataclass
class ActionCommand:
    """Represents a command to be executed by the robot."""
    id: str
    command_type: ActionType
    parameters: Dict[str, Any]
    priority: int = 1  # 1-5, with 5 being highest
    timeout: float = 30.0  # seconds
    dependencies: List[str] = None  # Other commands this command depends on
    success_conditions: List[Dict[str, Any]] = None  # Conditions for success
    failure_conditions: List[Dict[str, Any]] = None  # Conditions for failure
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.success_conditions is None:
            self.success_conditions = []
        if self.failure_conditions is None:
            self.failure_conditions = []


class ActionGenerator:
    """Generates robot action commands from high-level instructions."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.command_id_counter = 0
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the action generator."""
        logger = logging.getLogger("ActionGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def generate_command_id(self) -> str:
        """Generate a unique command ID."""
        cmd_id = f"cmd_{self.command_id_counter:06d}"
        self.command_id_counter += 1
        return cmd_id
    
    def generate_navigation_command(
        self, 
        target_position: Tuple[float, float, float], 
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        speed: float = 0.5,
        approach_type: str = "direct"
    ) -> ActionCommand:
        """Generate a navigation command."""
        params = {
            "target_position": target_position,
            "speed": speed
        }
        
        if target_orientation:
            params["target_orientation"] = target_orientation
        
        if approach_type == "safe":
            params["safe_approach"] = True
        elif approach_type == "fast":
            params["fast_approach"] = True
        
        return ActionCommand(
            id=self.generate_command_id(),
            command_type=ActionType.NAVIGATION,
            parameters=params,
            priority=2,
            timeout=60.0,
            success_conditions=[
                {
                    "type": "position_reached",
                    "threshold": 0.1  # Within 10cm of target
                }
            ],
            failure_conditions=[
                {
                    "type": "obstacle_detected",
                    "threshold": 0.2  # Obstacle within 20cm
                },
                {
                    "type": "timeout",
                    "threshold": 60.0
                }
            ]
        )
    
    def generate_manipulation_command(
        self,
        object_name: str,
        object_position: Optional[Tuple[float, float, float]] = None,
        grasp_type: str = "top_grasp",
        approach_angle: float = 0.0,
        lift_height: float = 0.1
    ) -> ActionCommand:
        """Generate a manipulation command for grasping an object."""
        params = {
            "object_name": object_name,
            "grasp_type": grasp_type,
            "approach_angle": approach_angle,
            "lift_height": lift_height
        }
        
        if object_position:
            params["object_position"] = object_position
        
        return ActionCommand(
            id=self.generate_command_id(),
            command_type=ActionType.MANIPULATION,
            parameters=params,
            priority=3,
            timeout=45.0,
            success_conditions=[
                {
                    "type": "object_grasped",
                    "object": object_name
                }
            ],
            failure_conditions=[
                {
                    "type": "grasp_failed",
                    "object": object_name
                },
                {
                    "type": "object_not_found",
                    "object": object_name
                },
                {
                    "type": "timeout",
                    "threshold": 45.0
                }
            ]
        )
    
    def generate_locomotion_command(
        self,
        movement_type: str,
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # (linear_x, linear_y, angular_z)
        duration: float = 1.0,
        step_height: float = 0.05
    ) -> ActionCommand:
        """Generate a locomotion command for walking/stepping."""
        params = {
            "movement_type": movement_type,
            "velocity": velocity,
            "duration": duration
        }
        
        if movement_type == "walking":
            params["step_height"] = step_height
        
        return ActionCommand(
            id=self.generate_command_id(),
            command_type=ActionType.LOCOMOTION,
            parameters=params,
            priority=4,
            timeout=duration + 5.0,
            success_conditions=[
                {
                    "type": "movement_executed",
                    "threshold": duration
                }
            ],
            failure_conditions=[
                {
                    "type": "balance_lost"
                },
                {
                    "type": "timeout",
                    "threshold": duration + 5.0
                }
            ]
        )
    
    def generate_interaction_command(
        self,
        action: str,  # "greet", "handover", "take", etc.
        person_id: Optional[str] = None,
        object_name: Optional[str] = None,
        duration: float = 3.0
    ) -> ActionCommand:
        """Generate an interaction command for human-robot interaction."""
        params = {
            "action": action,
            "duration": duration
        }
        
        if person_id:
            params["person_id"] = person_id
        if object_name:
            params["object_name"] = object_name
        
        return ActionCommand(
            id=self.generate_command_id(),
            command_type=ActionType.INTERACTION,
            parameters=params,
            priority=3,
            timeout=duration + 10.0,
            success_conditions=[
                {
                    "type": "interaction_completed",
                    "action": action
                }
            ],
            failure_conditions=[
                {
                    "type": "person_not_detected",
                    "person_id": person_id
                },
                {
                    "type": "timeout",
                    "threshold": duration + 10.0
                }
            ]
        )
    
    def generate_perception_command(
        self,
        task: str,  # "look_at", "scan", "identify", etc.
        target_position: Optional[Tuple[float, float, float]] = None,
        scan_range: float = 1.0,
        duration: float = 5.0
    ) -> ActionCommand:
        """Generate a perception command for sensing environment."""
        params = {
            "task": task,
            "duration": duration
        }
        
        if target_position:
            params["target_position"] = target_position
        if scan_range:
            params["scan_range"] = scan_range
        
        return ActionCommand(
            id=self.generate_command_id(),
            command_type=ActionType.PERCEPTION,
            parameters=params,
            priority=1,
            timeout=duration + 5.0,
            success_conditions=[
                {
                    "type": "perception_completed",
                    "task": task
                }
            ],
            failure_conditions=[
                {
                    "type": "timeout",
                    "threshold": duration + 5.0
                }
            ]
        )


class ActionPlanner:
    """Plans sequences of actions to achieve complex goals."""
    
    def __init__(self):
        self.generator = ActionGenerator()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the action planner."""
        logger = logging.getLogger("ActionPlanner")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def plan_navigation_sequence(
        self, 
        waypoints: List[Tuple[float, float, float]], 
        speeds: Optional[List[float]] = None
    ) -> List[ActionCommand]:
        """Plan a sequence of navigation commands to visit multiple waypoints."""
        commands = []
        
        if not waypoints:
            return commands
        
        if speeds is None:
            speeds = [0.5] * len(waypoints)
        elif len(speeds) != len(waypoints):
            speeds = speeds[:len(waypoints)] + [0.5] * max(0, len(waypoints) - len(speeds))
        
        for i, waypoint in enumerate(waypoints):
            speed = speeds[i]
            approach_type = "safe" if i < len(waypoints) - 1 else "direct"  # Safe approach until last
            
            command = self.generator.generate_navigation_command(
                target_position=waypoint,
                speed=speed,
                approach_type=approach_type
            )
            
            # Add dependency if not the first command
            if i > 0:
                command.dependencies.append(commands[i-1].id)
            
            commands.append(command)
        
        return commands
    
    def plan_pick_and_place(
        self,
        object_name: str,
        object_position: Tuple[float, float, float],
        target_position: Tuple[float, float, float],
        lift_height: float = 0.1
    ) -> List[ActionCommand]:
        """Plan a sequence for picking up an object and placing it elsewhere."""
        commands = []
        
        # 1. Navigate to object
        nav_command = self.generator.generate_navigation_command(
            target_position=(object_position[0], object_position[1], object_position[2]),
            speed=0.3,
            approach_type="safe"
        )
        commands.append(nav_command)
        
        # 2. Look at object
        look_command = self.generator.generate_perception_command(
            task="look_at",
            target_position=object_position,
            duration=2.0
        )
        look_command.dependencies = [nav_command.id]
        commands.append(look_command)
        
        # 3. Grasp object
        grasp_command = self.generator.generate_manipulation_command(
            object_name=object_name,
            object_position=object_position,
            grasp_type="top_grasp",
            lift_height=lift_height
        )
        grasp_command.dependencies = [look_command.id]
        commands.append(grasp_command)
        
        # 4. Navigate to target location
        transport_nav = self.generator.generate_navigation_command(
            target_position=target_position,
            speed=0.2,  # Slower with object
            approach_type="safe"
        )
        transport_nav.dependencies = [grasp_command.id]
        commands.append(transport_nav)
        
        # 5. Place object (manipulation command with placement parameters)
        place_command = ActionCommand(
            id=self.generator.generate_command_id(),
            command_type=ActionType.MANIPULATION,
            parameters={
                "task": "place",
                "object_name": object_name,
                "target_position": target_position,
                "placement_type": "set_down"
            },
            priority=3,
            timeout=30.0,
            dependencies=[transport_nav.id]
        )
        commands.append(place_command)
        
        return commands
    
    def plan_interaction_sequence(
        self,
        person_id: str,
        interaction_type: str,  # "greeting", "handover", "guidance"
        object_name: Optional[str] = None
    ) -> List[ActionCommand]:
        """Plan a sequence for interacting with a person."""
        commands = []
        
        # 1. Navigate toward person
        nav_command = self.generator.generate_navigation_command(
            target_position=(0.5, 0.0, 0.0),  # Relative to person's position
            speed=0.3,
            approach_type="safe"
        )
        commands.append(nav_command)
        
        if interaction_type == "greeting":
            # 2. Greet the person
            greet_command = self.generator.generate_interaction_command(
                action="greet",
                person_id=person_id,
                duration=3.0
            )
            greet_command.dependencies = [nav_command.id]
            commands.append(greet_command)
        
        elif interaction_type == "handover" and object_name:
            # 2. Hold object (assumes already grasped)
            hold_command = ActionCommand(
                id=self.generator.generate_command_id(),
                command_type=ActionType.CONTROL,
                parameters={
                    "task": "hold_object",
                    "object_name": object_name
                },
                priority=4,
                timeout=10.0,
                dependencies=[nav_command.id]
            )
            commands.append(hold_command)
            
            # 3. Hand over object
            handover_command = self.generator.generate_interaction_command(
                action="handover",
                person_id=person_id,
                object_name=object_name,
                duration=5.0
            )
            handover_command.dependencies = [hold_command.id]
            commands.append(handover_command)
        
        elif interaction_type == "guidance":
            # 2. Explain the guidance
            explain_command = self.generator.generate_interaction_command(
                action="explain",
                person_id=person_id,
                duration=5.0
            )
            explain_command.dependencies = [nav_command.id]
            commands.append(explain_command)
            
            # 3. Demonstrate movement
            demo_command = self.generator.generate_locomotion_command(
                movement_type="gesture",
                velocity=(0.0, 0.0, 0.5),
                duration=3.0
            )
            demo_command.dependencies = [explain_command.id]
            commands.append(demo_command)
        
        return commands


class ActionCommandExecutor:
    """Executes action commands on the robot."""
    
    def __init__(self):
        self.planner = ActionPlanner()
        self.active_commands: Dict[str, ActionCommand] = {}
        self.completed_commands: Dict[str, ActionCommand] = {}
        self.failed_commands: Dict[str, ActionCommand] = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the executor."""
        logger = logging.getLogger("ActionCommandExecutor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def execute_command(self, command: ActionCommand) -> bool:
        """Execute a single action command."""
        self.logger.info(f"Executing command {command.id}: {command.command_type.value} with params {command.parameters}")
        
        # Check dependencies
        if not self._check_dependencies_met(command):
            self.logger.warning(f"Dependencies not met for command {command.id}")
            return False
        
        # Add to active commands
        self.active_commands[command.id] = command
        
        try:
            # Execute based on command type
            if command.command_type == ActionType.NAVIGATION:
                success = self._execute_navigation(command)
            elif command.command_type == ActionType.MANIPULATION:
                success = self._execute_manipulation(command)
            elif command.command_type == ActionType.LOCOMOTION:
                success = self._execute_locomotion(command)
            elif command.command_type == ActionType.INTERACTION:
                success = self._execute_interaction(command)
            elif command.command_type == ActionType.PERCEPTION:
                success = self._execute_perception(command)
            else:
                self.logger.error(f"Unknown command type: {command.command_type}")
                success = False
        
        except Exception as e:
            self.logger.error(f"Error executing command {command.id}: {e}")
            success = False
        
        # Update command status
        if success:
            self.completed_commands[command.id] = self.active_commands.pop(command.id)
            self.logger.info(f"Command {command.id} completed successfully")
        else:
            self.failed_commands[command.id] = self.active_commands.pop(command.id)
            self.logger.error(f"Command {command.id} failed")
        
        return success
    
    def _check_dependencies_met(self, command: ActionCommand) -> bool:
        """Check if all dependencies for a command are met."""
        for dep_id in command.dependencies:
            if dep_id not in self.completed_commands:
                return False
        return True
    
    def _execute_navigation(self, command: ActionCommand) -> bool:
        """Execute a navigation command."""
        try:
            target_pos = command.parameters.get("target_position")
            speed = command.parameters.get("speed", 0.5)
            
            if target_pos is None:
                self.logger.error("No target position specified for navigation command")
                return False
            
            # In a real robot, this would call the navigation stack
            self.logger.info(f"Moving towards {target_pos} at speed {speed}")
            
            # Simulate execution
            import time
            time.sleep(0.1 * (1/speed))  # Simulate time based on inverse speed
            
            # Check success conditions
            # For simulation, assume success
            return True
        except Exception as e:
            self.logger.error(f"Navigation command failed: {e}")
            return False
    
    def _execute_manipulation(self, command: ActionCommand) -> bool:
        """Execute a manipulation command."""
        try:
            object_name = command.parameters.get("object_name")
            grasp_type = command.parameters.get("grasp_type", "top_grasp")
            
            if object_name is None:
                self.logger.error("No object name specified for manipulation command")
                return False
            
            self.logger.info(f"Attempting to {grasp_type} the {object_name}")
            
            # Simulate execution
            import time
            time.sleep(2.0)  # Simulate manipulation time
            
            # Check success conditions
            # For simulation, assume success
            return True
        except Exception as e:
            self.logger.error(f"Manipulation command failed: {e}")
            return False
    
    def _execute_locomotion(self, command: ActionCommand) -> bool:
        """Execute a locomotion command."""
        try:
            movement_type = command.parameters.get("movement_type", "walk")
            duration = command.parameters.get("duration", 1.0)
            
            self.logger.info(f"Performing {movement_type} movement")
            
            # Simulate execution
            import time
            time.sleep(duration)
            
            # Check success conditions
            # For simulation, assume success
            return True
        except Exception as e:
            self.logger.error(f"Locomotion command failed: {e}")
            return False
    
    def _execute_interaction(self, command: ActionCommand) -> bool:
        """Execute an interaction command."""
        try:
            action = command.parameters.get("action", "idle")
            duration = command.parameters.get("duration", 1.0)
            
            self.logger.info(f"Performing social interaction: {action}")
            
            # Simulate execution
            import time
            time.sleep(duration)
            
            # Check success conditions
            # For simulation, assume success
            return True
        except Exception as e:
            self.logger.error(f"Interaction command failed: {e}")
            return False
    
    def _execute_perception(self, command: ActionCommand) -> bool:
        """Execute a perception command."""
        try:
            task = command.parameters.get("task", "idle")
            duration = command.parameters.get("duration", 1.0)
            
            self.logger.info(f"Performing perception task: {task}")
            
            # Simulate execution
            import time
            time.sleep(duration)
            
            # Check success conditions
            # For simulation, assume success
            return True
        except Exception as e:
            self.logger.error(f"Perception command failed: {e}")
            return False
    
    def execute_command_sequence(self, commands: List[ActionCommand]) -> Dict[str, Any]:
        """Execute a sequence of commands and return results."""
        results = {
            "executed_commands": [],
            "successful_commands": [],
            "failed_commands": [],
            "overall_success": True
        }
        
        for command in commands:
            success = self.execute_command(command)
            results["executed_commands"].append(command.id)
            
            if success:
                results["successful_commands"].append(command.id)
            else:
                results["failed_commands"].append(command.id)
                results["overall_success"] = False  # If any command fails, sequence fails
        
        return results


def example_usage():
    """Example of how to use the action command generation system."""
    print("Action Command Generation Example")
    print("=" * 40)
    
    # Create action planner
    planner = ActionPlanner()
    
    # Example 1: Simple navigation command
    print("\n1. Generating simple navigation command:")
    nav_cmd = planner.generator.generate_navigation_command(
        target_position=(1.0, 2.0, 0.0),
        speed=0.4
    )
    print(f"   Command: {nav_cmd.command_type.value}")
    print(f"   Params: {nav_cmd.parameters}")
    print(f"   ID: {nav_cmd.id}")
    
    # Example 2: Manipulation command
    print("\n2. Generating manipulation command:")
    manip_cmd = planner.generator.generate_manipulation_command(
        object_name="red_cup",
        object_position=(0.8, 0.5, 0.1),
        grasp_type="side_grasp"
    )
    print(f"   Command: {manip_cmd.command_type.value}")
    print(f"   Params: {manip_cmd.parameters}")
    print(f"   ID: {manip_cmd.id}")
    
    # Example 3: Navigation sequence
    print("\n3. Generating navigation sequence:")
    waypoints = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0)
    ]
    nav_sequence = planner.plan_navigation_sequence(waypoints)
    print(f"   Generated {len(nav_sequence)} navigation commands")
    for i, cmd in enumerate(nav_sequence):
        print(f"   Command {i+1}: {cmd.parameters['target_position']}")
    
    # Example 4: Pick and place sequence
    print("\n4. Generating pick-and-place sequence:")
    pick_place_seq = planner.plan_pick_and_place(
        object_name="book",
        object_position=(0.8, 0.5, 0.1),
        target_position=(1.2, 0.3, 0.1)
    )
    print(f"   Generated {len(pick_place_seq)} commands for pick-and-place task")
    for i, cmd in enumerate(pick_place_seq):
        print(f"   Command {i+1}: {cmd.command_type.value}")
    
    # Example 5: Interaction sequence
    print("\n5. Generating interaction sequence (greeting):")
    greeting_seq = planner.plan_interaction_sequence(
        person_id="person_001",
        interaction_type="greeting"
    )
    print(f"   Generated {len(greeting_seq)} commands for greeting")
    
    # Example 6: Execute a simple command
    print("\n6. Executing a command sequence:")
    executor = ActionCommandExecutor()
    results = executor.execute_command_sequence(nav_sequence[:2])  # Execute first 2 commands
    print(f"   Execution results: {results}")
    
    print("\nAction command generation system initialized successfully!")


if __name__ == "__main__":
    example_usage()