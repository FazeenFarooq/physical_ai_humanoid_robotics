"""
System integration layer for the capstone project in the Physical AI & Humanoid Robotics course.
This module integrates all subsystems (perception, planning, navigation, manipulation, 
conversation, and control) into a cohesive robotic system.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import time
import queue
from datetime import datetime
import logging

from src.perception.capstone_perception import CapstonePerceptionStack, MultiModalPerceptionFusion
from src.planning.capstone_planning import CapstonePlanner, HierarchicalTaskNetwork
from src.navigation.capstone_navigation import NavigationManager
from src.manipulation.capstone_manipulation import ManipulationManager
from src.conversation.capstone_conversation import ConversationalSystemManager
from src.control.failure_recovery import RecoveryManager
from src.models.task_plan import TaskPlan
from src.models.action_command import ActionCommand, ActionType
from src.control.kinematics import HumanoidKinematicModel


@dataclass
class SystemState:
    """Current state of the integrated system"""
    timestamp: datetime
    robot_pose: Tuple[float, float, float]  # x, y, theta
    battery_level: float
    active_tasks: List[str]
    system_status: str
    safety_status: str
    performance_metrics: Dict[str, float]


class SystemCoordinator:
    """
    Coordinates all subsystems to work together as a unified system.
    """
    
    def __init__(self, kinematic_model: HumanoidKinematicModel):
        # Initialize all subsystems
        self.perception_system = CapstonePerceptionStack()
        self.planning_system = CapstonePlanner()
        self.navigation_system = NavigationManager()
        self.manipulation_system = ManipulationManager(kinematic_model)
        self.conversation_system = ConversationalSystemManager()
        self.failure_recovery_system = RecoveryManager()
        
        # HTN for complex task decomposition
        self.htn_planner = HierarchicalTaskNetwork()
        
        # System state tracking
        self.current_state = SystemState(
            timestamp=datetime.now(),
            robot_pose=(0.0, 0.0, 0.0),
            battery_level=1.0,
            active_tasks=[],
            system_status="idle",
            safety_status="safe",
            performance_metrics={}
        )
        
        # Task execution tracking
        self.active_task_queue = queue.Queue()
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Callbacks for system events
        self.state_change_callbacks: List[Callable[[SystemState], None]] = []
        self.task_completion_callbacks: List[Callable[[str, bool], None]] = []
        
        # Threading components
        self.main_thread_running = False
        self.perception_thread = None
        self.control_thread = None
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
    
    def initialize_system(self):
        """Initialize all subsystems and start threads"""
        # Initialize failure recovery first
        self.failure_recovery_system.initialize()
        
        # Initialize all subsystems
        self.perception_system.start_perception_systems()
        self.conversation_system.start_conversation_system("system", mode="standalone")
        
        # Start system threads
        self.main_thread_running = True
        self.perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        
        self.perception_thread.start()
        self.control_thread.start()
        
        # Register components with failure recovery
        self.failure_recovery_system.monitor_component("perception", self._check_perception_health)
        self.failure_recovery_system.monitor_component("planning", self._check_planning_health)
        self.failure_recovery_system.monitor_component("navigation", self._check_navigation_health)
        self.failure_recovery_system.monitor_component("manipulation", self._check_manipulation_health)
        self.failure_recovery_system.monitor_component("conversation", self._check_conversation_health)
        
        logging.info("Capstone System Integration Layer initialized")
    
    def shutdown_system(self):
        """Safely shut down all subsystems"""
        self.main_thread_running = False
        
        # Stop all subsystems
        self.perception_system.stop_perception()
        self.conversation_system.stop_conversation_system()
        self.failure_recovery_system.shutdown()
        
        logging.info("Capstone System Integration Layer shut down")
    
    def execute_task(self, task_plan: TaskPlan) -> bool:
        """Execute a high-level task plan"""
        try:
            # Add task to execution queue
            self.active_task_queue.put(task_plan)
            
            # Update system state
            self.current_state.active_tasks.append(task_plan.id)
            self.current_state.system_status = "executing_task"
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                callback(self.current_state)
            
            logging.info(f"Started execution of task: {task_plan.id}")
            return True
        except Exception as e:
            logging.error(f"Failed to start task execution: {e}")
            self._handle_system_failure("task_execution_failed", str(e))
            return False
    
    def execute_command(self, action_command: ActionCommand) -> bool:
        """Execute a specific action command"""
        try:
            if action_command.type == ActionType.NAVIGATION:
                # Execute navigation command
                target = action_command.parameters.get("target_position", (0, 0))
                success = self.navigation_system.start_navigation(target[0], target[1])
                
                if not success:
                    raise Exception("Navigation system failed to start")
                    
            elif action_command.type == ActionType.MANIPULATION:
                # Execute manipulation command
                success = self.manipulation_system.start_manipulation_task(action_command)
                
                if not success:
                    raise Exception("Manipulation system failed to start")
            
            elif action_command.type == ActionType.CONVERSATION:
                # Execute conversation command
                message = action_command.parameters.get("message", "")
                response = self.conversation_system.get_response(message)
                logging.info(f"Conversation response: {response}")
            
            logging.info(f"Executed command: {action_command.id}")
            return True
        except Exception as e:
            logging.error(f"Failed to execute command: {e}")
            self._handle_system_failure("command_execution_failed", str(e))
            return False
    
    def get_system_state(self) -> SystemState:
        """Get the current system state"""
        return self.current_state
    
    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update the robot's current pose"""
        self.current_state.robot_pose = (x, y, theta)
        self.current_state.timestamp = datetime.now()
    
    def update_battery_level(self, level: float):
        """Update the robot's battery level"""
        self.current_state.battery_level = max(0.0, min(1.0, level))
    
    def add_state_change_callback(self, callback: Callable[[SystemState], None]):
        """Add a callback for system state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_task_completion_callback(self, callback: Callable[[str, bool], None]):
        """Add a callback for task completion events"""
        self.task_completion_callbacks.append(callback)
    
    def _perception_loop(self):
        """Main perception processing loop"""
        while self.main_thread_running:
            try:
                # Get perception data
                perception_data = self.perception_system.get_perception_data_queue()
                
                # Process perception data
                fused_data = self.perception_system.fuse_multimodal_input()
                
                # Update environmental understanding
                if "occupancy_grid" in fused_data:
                    occupancy_grid = np.array(fused_data["occupancy_grid"])
                    self.navigation_system.update_sensor_data(
                        occupancy_grid, 
                        *self.current_state.robot_pose
                    )
                
                # Update system state with new information
                self.current_state.timestamp = datetime.now()
                
                time.sleep(0.1)  # Process perception at 10Hz
            except Exception as e:
                logging.error(f"Error in perception loop: {e}")
                self._handle_system_failure("perception_loop_error", str(e))
    
    def _control_loop(self):
        """Main control and task execution loop"""
        while self.main_thread_running:
            try:
                # Process navigation commands if system is in navigation mode
                if self.navigation_system.is_system_active():
                    linear_vel, angular_vel = self.navigation_system.get_navigation_command()
                    
                    # In a real system, this would send commands to the robot's base controller
                    # For simulation, just log the command
                    logging.debug(f"Navigation command: linear={linear_vel}, angular={angular_vel}")
                
                # Check for tasks in the queue
                try:
                    task = self.active_task_queue.get_nowait()
                    self._execute_single_task(task)
                except queue.Empty:
                    pass  # No tasks to execute
                
                # Update system state
                self._update_system_state()
                
                time.sleep(0.05)  # Control loop at 20Hz
            except Exception as e:
                logging.error(f"Error in control loop: {e}")
                self._handle_system_failure("control_loop_error", str(e))
    
    def _execute_single_task(self, task_plan: TaskPlan):
        """Execute a single task plan"""
        try:
            # Update system status
            self.current_state.system_status = f"executing_{task_plan.id}"
            
            # Decompose complex tasks using HTN
            primitive_tasks = self.htn_planner.decompose_task(task_plan.name, task_plan.requirements)
            
            # Execute each primitive task
            for task_name, params in primitive_tasks:
                # Convert to action command and execute
                action_cmd = self._create_action_command_for_task(task_name, params)
                success = self.execute_command(action_cmd)
                
                if not success:
                    # Task failed
                    self.failed_tasks.append(task_plan.id)
                    self.current_state.active_tasks.remove(task_plan.id)
                    
                    # Trigger failure recovery
                    self._handle_system_failure("task_failed", f"Task {task_plan.id} failed at subtask {task_name}")
                    return
            
            # Task completed successfully
            self.completed_tasks.append(task_plan.id)
            if task_plan.id in self.current_state.active_tasks:
                self.current_state.active_tasks.remove(task_plan.id)
            
            # If no more active tasks, set system to idle
            if not self.current_state.active_tasks:
                self.current_state.system_status = "idle"
            
            # Notify completion callbacks
            for callback in self.task_completion_callbacks:
                callback(task_plan.id, True)
                
            logging.info(f"Completed task: {task_plan.id}")
            
        except Exception as e:
            logging.error(f"Error executing task {task_plan.id}: {e}")
            self.failed_tasks.append(task_plan.id)
            if task_plan.id in self.current_state.active_tasks:
                self.current_state.active_tasks.remove(task_plan.id)
            
            # Trigger failure recovery
            self._handle_system_failure("task_execution_error", str(e))
    
    def _create_action_command_for_task(self, task_name: str, params: Dict[str, Any]) -> ActionCommand:
        """Create an action command for a primitive task"""
        command_id = f"cmd_{task_name}_{int(datetime.now().timestamp() * 1000)}"
        
        if "navigate" in task_name:
            action_type = ActionType.NAVIGATION
        elif "grasp" in task_name or "manipulate" in task_name:
            action_type = ActionType.MANIPULATION
        elif "speak" in task_name or "say" in task_name:
            action_type = ActionType.CONVERSATION
        else:
            action_type = ActionType.SYSTEM_CONTROL
        
        return ActionCommand(
            id=command_id,
            type=action_type,
            parameters=params,
            priority=5
        )
    
    def _update_system_state(self):
        """Update the system state based on subsystem statuses"""
        self.current_state.timestamp = datetime.now()
        
        # Update based on subsystem statuses
        nav_status = self.navigation_system.get_navigation_status().value if self.navigation_system.is_system_active() else "idle"
        manip_status = self.manipulation_system.get_manipulation_status().value if self.manipulation_system.is_active else "idle"
        
        # Determine overall system status
        if self.current_state.active_tasks:
            self.current_state.system_status = "executing_task"
        elif nav_status != "idle":
            self.current_state.system_status = f"navigation_{nav_status}"
        elif manip_status != "idle":
            self.current_state.system_status = f"manipulation_{man_status}"
        else:
            self.current_state.system_status = "idle"
        
        # Update performance metrics
        self.current_state.performance_metrics = {
            "navigation_success": self.navigation_system.get_success_rate() if hasattr(self.navigation_system, 'get_success_rate') else 0.0,
            "manipulation_success": self.manipulation_system.get_success_rate(),
            "conversation_success": 0.95,  # Placeholder
            "system_uptime": (datetime.now() - self.current_state.timestamp).total_seconds() if hasattr(self.current_state, '_init_time') else 0
        }
        
        # Check safety status
        # In a real system, this would check various safety systems
        self.current_state.safety_status = "safe"
    
    def _handle_system_failure(self, failure_type: str, description: str):
        """Handle system failures and trigger recovery"""
        logging.error(f"System failure: {failure_type} - {description}")
        
        # Report to failure recovery system
        from src.control.failure_recovery import FailureType, FailureSeverity
        failure_type_map = {
            "task_execution_failed": FailureType.PLANNING_FAILURE,
            "command_execution_failed": FailureType.SOFTWARE_ERROR,
            "perception_loop_error": FailureType.PERCEPTION_FAILURE,
            "control_loop_error": FailureType.SOFTWARE_ERROR,
            "task_failed": FailureType.PLANNING_FAILURE
        }
        
        mapped_type = failure_type_map.get(failure_type, FailureType.SOFTWARE_ERROR)
        
        self.failure_recovery_system.handle_detected_failure(
            mapped_type,
            "system_coordinator",
            description,
            {"failure_type": failure_type},
            FailureSeverity.HIGH
        )
    
    def _check_perception_health(self) -> bool:
        """Health check for perception system"""
        # In a real implementation, this would check if perception is operating correctly
        return self.perception_system is not None
    
    def _check_planning_health(self) -> bool:
        """Health check for planning system"""
        # In a real implementation, this would check if planning is operating correctly
        return self.planning_system is not None
    
    def _check_navigation_health(self) -> bool:
        """Health check for navigation system"""
        # In a real implementation, this would check if navigation is operating correctly
        return self.navigation_system.is_system_active() if self.navigation_system else False
    
    def _check_manipulation_health(self) -> bool:
        """Health check for manipulation system"""
        # In a real implementation, this would check if manipulation is operating correctly
        return self.manipulation_system.is_active if self.manipulation_system else False
    
    def _check_conversation_health(self) -> bool:
        """Health check for conversation system"""
        # In a real implementation, this would check if conversation is operating correctly
        return self.conversation_system.is_system_active() if self.conversation_system else False


class CapstoneSystemOrchestrator:
    """
    High-level orchestrator that manages the complete capstone system.
    """
    
    def __init__(self, kinematic_model: HumanoidKinematicModel):
        self.system_coordinator = SystemCoordinator(kinematic_model)
        self.is_running = False
    
    def start_system(self):
        """Start the complete integrated system"""
        self.system_coordinator.initialize_system()
        self.is_running = True
        logging.info("Capstone System Orchestrator started")
    
    def stop_system(self):
        """Stop the complete integrated system"""
        self.system_coordinator.shutdown_system()
        self.is_running = False
        logging.info("Capstone System Orchestrator stopped")
    
    def execute_behavior(self, behavior_name: str, parameters: Dict[str, Any] = None) -> bool:
        """Execute a high-level behavior"""
        if parameters is None:
            parameters = {}
        
        # Map behavior names to complex tasks
        behavior_tasks = {
            "patrol": ["navigate_to_waypoint1", "navigate_to_waypoint2", "navigate_to_waypoint3", "return_to_base"],
            "fetch_object": ["localize_object", "navigate_to_object", "grasp_object", "return_with_object", "place_object"],
            "greet_visitor": ["detect_person", "approach_person", "initiate_conversation", "provide_information"],
            "perform_inspection": ["navigate_to_inspection_points", "capture_sensor_data", "analyze_environment", "report_findings"]
        }
        
        if behavior_name not in behavior_tasks:
            logging.error(f"Unknown behavior: {behavior_name}")
            return False
        
        # Create a task plan for the behavior
        # In a real implementation, this would create actual TaskPlan objects
        logging.info(f"Executing behavior: {behavior_name} with parameters: {parameters}")
        
        # For now, just return success
        return True
    
    def execute_task_plan(self, task_plan: TaskPlan) -> bool:
        """Execute a task plan through the integrated system"""
        return self.system_coordinator.execute_task(task_plan)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the complete system status"""
        system_state = self.system_coordinator.get_system_state()
        
        # Get status from each subsystem
        recovery_status = self.system_coordinator.failure_recovery_system.get_system_health_status()
        
        return {
            "system_state": {
                "timestamp": system_state.timestamp.isoformat(),
                "robot_pose": system_state.robot_pose,
                "battery_level": system_state.battery_level,
                "active_tasks": system_state.active_tasks,
                "system_status": system_state.system_status,
                "safety_status": system_state.safety_status,
                "performance_metrics": system_state.performance_metrics
            },
            "recovery_status": recovery_status,
            "is_operational": self.is_running
        }
    
    def emergency_stop(self):
        """Trigger emergency stop across all systems"""
        logging.critical("EMERGENCY STOP TRIGGERED")
        
        # Stop navigation
        self.system_coordinator.navigation_system.stop_navigation()
        
        # Stop manipulation
        self.system_coordinator.manipulation_system.stop_manipulation()
        
        # Clear task queue
        while not self.system_coordinator.active_task_queue.empty():
            try:
                self.system_coordinator.active_task_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update system state
        self.system_coordinator.current_state.system_status = "emergency_stop"
        self.system_coordinator.current_state.safety_status = "unsafe"
        
        # Report to failure recovery as a critical safety violation
        self.system_coordinator._handle_system_failure("emergency_stop", "Emergency stop activated by operator")
    
    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update the robot's pose in the system"""
        self.system_coordinator.update_robot_pose(x, y, theta)
    
    def update_battery_level(self, level: float):
        """Update the battery level in the system"""
        self.system_coordinator.update_battery_level(level)


# Import numpy at the top level since it's used in the code
import numpy as np