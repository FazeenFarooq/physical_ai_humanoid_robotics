"""
Capstone demonstration framework for the Physical AI & Humanoid Robotics course.
This module provides a structured framework for demonstrating the complete 
integrated system capabilities during the final demonstration.
Based on the requirements in /specs/001-physical-ai-course/data-model.md
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import logging
from datetime import datetime
import threading
import queue

from src.capstone.system_integration import CapstoneSystemOrchestrator
from src.models.task_plan import TaskPlan, TaskStep
from src.models.action_command import ActionCommand, ActionType
from src.control.kinematics import HumanoidKinematicModel


@dataclass
class DemoScenario:
    """Definition of a demonstration scenario"""
    id: str
    name: str
    description: str
    task_sequence: List[TaskPlan]
    success_criteria: List[str]
    estimated_duration: float  # in minutes
    difficulty_level: str  # beginner, intermediate, advanced
    required_equipment: List[str]
    safety_considerations: List[str]


@dataclass
class DemoStep:
    """A step within a demonstration"""
    id: str
    name: str
    description: str
    action_commands: List[ActionCommand]
    expected_outcome: str
    timeout: float  # seconds
    success_criteria: List[str]


@dataclass
class DemoExecutionResult:
    """Result of a demonstration execution"""
    demo_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    completed_steps: List[str]
    failed_steps: List[str]
    performance_metrics: Dict[str, float]
    issues_encountered: List[str]
    total_duration: float


class CapstoneDemoFramework:
    """
    Framework for structuring and executing capstone demonstrations.
    """
    
    def __init__(self, orchestrator: CapstoneSystemOrchestrator):
        self.orchestrator = orchestrator
        self.demo_scenarios: Dict[str, DemoScenario] = {}
        self.active_demo: Optional[DemoScenario] = None
        self.demo_execution_queue = queue.Queue()
        self.demo_results: List[DemoExecutionResult] = []
        self.is_demo_running = False
        self.demo_thread = None
        
        # Demo metrics
        self.metrics = {
            "navigation_accuracy": 0.0,
            "manipulation_success_rate": 0.0,
            "conversation_coherence": 0.0,
            "task_completion_time": 0.0,
            "safety_incidents": 0
        }
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
    
    def register_demo_scenario(self, scenario: DemoScenario):
        """Register a new demonstration scenario"""
        self.demo_scenarios[scenario.id] = scenario
        logging.info(f"Registered demo scenario: {scenario.name}")
    
    def get_available_scenarios(self) -> List[DemoScenario]:
        """Get list of available demonstration scenarios"""
        return list(self.demo_scenarios.values())
    
    def execute_demo(self, demo_id: str, parameters: Dict[str, Any] = None) -> DemoExecutionResult:
        """Execute a demonstration by ID"""
        if demo_id not in self.demo_scenarios:
            raise ValueError(f"Demo scenario {demo_id} not found")
        
        scenario = self.demo_scenarios[demo_id]
        return self._execute_scenario(scenario, parameters or {})
    
    def _execute_scenario(self, scenario: DemoScenario, parameters: Dict[str, Any]) -> DemoExecutionResult:
        """Execute a specific scenario"""
        start_time = datetime.now()
        self.active_demo = scenario
        self.is_demo_running = True
        
        completed_steps = []
        failed_steps = []
        issues = []
        
        try:
            logging.info(f"Starting demonstration: {scenario.name}")
            
            # Initialize metrics for this demo
            demo_metrics = self.metrics.copy()
            
            # Execute each task in the sequence
            for i, task_plan in enumerate(scenario.task_sequence):
                logging.info(f"Executing task {i+1}/{len(scenario.task_sequence)}: {task_plan.name}")
                
                # Execute the task
                success = self.orchestrator.execute_task_plan(task_plan)
                
                if success:
                    completed_steps.append(task_plan.id)
                    logging.info(f"Task completed: {task_plan.name}")
                else:
                    failed_steps.append(task_plan.id)
                    issues.append(f"Failed to complete task: {task_plan.name}")
                    logging.warning(f"Task failed: {task_plan.name}")
                    
                    # If critical task fails, potentially abort the demo
                    if i < len(scenario.task_sequence) - 1:  # Not the last task
                        # For this implementation, continue with other tasks
                        continue
            
            # Calculate demo metrics
            total_tasks = len(scenario.task_sequence)
            completed_count = len(completed_steps)
            success_rate = completed_count / total_tasks if total_tasks > 0 else 0
            
            # Update performance metrics
            demo_metrics["task_completion_rate"] = success_rate
            demo_metrics["total_tasks"] = total_tasks
            demo_metrics["completed_tasks"] = completed_count
            demo_metrics["failed_tasks"] = len(failed_steps)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = DemoExecutionResult(
                demo_id=scenario.id,
                start_time=start_time,
                end_time=end_time,
                success=len(failed_steps) == 0 and len(completed_steps) > 0,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                performance_metrics=demo_metrics,
                issues_encountered=issues,
                total_duration=duration
            )
            
            self.demo_results.append(result)
            
            # Log demo completion
            logging.info(f"Demo completed - Success: {result.success}, "
                        f"Duration: {duration:.2f}s, "
                        f"Tasks: {completed_count}/{total_tasks}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error during demo execution: {e}")
            end_time = datetime.now()
            
            result = DemoExecutionResult(
                demo_id=scenario.id,
                start_time=start_time,
                end_time=end_time,
                success=False,
                completed_steps=completed_steps,
                failed_steps=failed_steps + ["demo_error"],
                performance_metrics=self.metrics,
                issues_encountered=[str(e)],
                total_duration=(end_time - start_time).total_seconds()
            )
            
            self.demo_results.append(result)
            return result
        finally:
            self.is_demo_running = False
            self.active_demo = None
    
    def start_autonomous_demo(self, demo_id: str, loop: bool = False):
        """Start a demo in autonomous mode (possibly looping)"""
        def demo_loop():
            while True:
                self._execute_scenario(self.demo_scenarios[demo_id], {})
                
                if not loop:
                    break
                
                time.sleep(5.0)  # Wait 5 seconds between demos
        
        self.demo_thread = threading.Thread(target=demo_loop, daemon=True)
        self.demo_thread.start()
    
    def stop_demo(self):
        """Stop the currently running demo"""
        self.is_demo_running = False
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=1.0)
        
        logging.info("Demo stopped by user")
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get statistics about demo execution"""
        if not self.demo_results:
            return {"message": "No demos executed yet"}
        
        total_demos = len(self.demo_results)
        successful_demos = sum(1 for result in self.demo_results if result.success)
        avg_duration = sum(result.total_duration for result in self.demo_results) / total_demos
        
        # Calculate average metrics
        avg_task_completion = sum(
            result.performance_metrics.get("task_completion_rate", 0) 
            for result in self.demo_results
        ) / total_demos
        
        return {
            "total_demos": total_demos,
            "successful_demos": successful_demos,
            "success_rate": successful_demos / total_demos if total_demos > 0 else 0,
            "average_duration": avg_duration,
            "average_task_completion_rate": avg_task_completion,
            "recent_demo_results": self.demo_results[-5:]  # Last 5 results
        }
    
    def emergency_stop_demo(self):
        """Emergency stop for demo situations"""
        logging.critical("EMERGENCY STOP during demo")
        self.orchestrator.emergency_stop()
        self.stop_demo()


# Predefined demonstration scenarios
class DefaultDemoScenarios:
    """Collection of default demonstration scenarios for the capstone project"""
    
    @staticmethod
    def get_voice_to_intent_demo() -> DemoScenario:
        """Demo for Milestone 1: Voice-to-Intent"""
        # Create action commands for voice interaction
        listen_command = ActionCommand(
            id="listen_cmd",
            type=ActionType.CONVERSATION,
            parameters={"mode": "listening"},
            priority=10
        )
        
        respond_command = ActionCommand(
            id="respond_cmd",
            type=ActionType.CONVERSATION,
            parameters={"message": "I understand your request and will help you with that"},
            priority=10
        )
        
        # Create a simple task plan
        from src.models.task_plan import TaskStep
        task_steps = [
            TaskStep(
                id="step1",
                description="Listen for voice command",
                action_type="conversation",
                parameters={"mode": "listen"},
                estimated_duration=5.0,
                dependencies=[],
                success_criteria=["voice_command_received"],
                failure_recovery="retry_listening"
            ),
            TaskStep(
                id="step2",
                description="Process natural language input",
                action_type="conversation",
                parameters={"process": "nlp"},
                estimated_duration=2.0,
                dependencies=["step1"],
                success_criteria=["intent_classified"],
                failure_recovery="request_repetition"
            ),
            TaskStep(
                id="step3",
                description="Respond to user",
                action_type="conversation",
                parameters={"respond": "acknowledgment"},
                estimated_duration=3.0,
                dependencies=["step2"],
                success_criteria=["response_delivered"],
                failure_recovery="emergency_stop"
            )
        ]
        
        task_plan = TaskPlan(
            id="voice_intent_task",
            name="Voice to Intent Processing",
            description="Demonstrate voice command recognition and intent processing",
            steps=task_steps,
            requirements=["microphone", "speech_recognition", "dialogue_manager"],
            constraints=["indoor_environment"],
            success_criteria=["intent_correctly_classified", "appropriate_response_given"],
            fallback_behaviors=["request_repeat", "switch_to_text_input"],
            estimated_time=10.0
        )
        
        return DemoScenario(
            id="voice_to_intent",
            name="Voice-to-Intent Milestone Demo",
            description="Demonstrate the robot's ability to understand spoken commands and map them to appropriate intents",
            task_sequence=[task_plan],
            success_criteria=[
                "Correctly recognizes spoken command",
                "Appropriately classifies user intent",
                "Provides relevant response"
            ],
            estimated_duration=1.0,  # 1 minute
            difficulty_level="intermediate",
            required_equipment=["microphone", "speakers"],
            safety_considerations=["ensure quiet environment for speech recognition"]
        )
    
    @staticmethod
    def get_perception_mapping_demo() -> DemoScenario:
        """Demo for Milestone 2: Perception & Mapping"""
        # Create task plan for perception and mapping
        task_steps = [
            TaskStep(
                id="perception_scan",
                description="Scan environment using sensors",
                action_type="perception",
                parameters={"scan_type": "360_degree"},
                estimated_duration=10.0,
                dependencies=[],
                success_criteria=["environment_mapped", "obstacles_detected"],
                failure_recovery="increase_sensor_sensitivity"
            ),
            TaskStep(
                id="object_identification",
                description="Identify objects in environment",
                action_type="perception",
                parameters={"identify": "common_objects"},
                estimated_duration=5.0,
                dependencies=["perception_scan"],
                success_criteria=["objects_classified", "positions_determined"],
                failure_recovery="adjust_lighting_conditions"
            ),
            TaskStep(
                id="semantic_mapping",
                description="Create semantic map of environment",
                action_type="perception",
                parameters={"mapping": "semantic"},
                estimated_duration=8.0,
                dependencies=["object_identification"],
                success_criteria=["semantic_map_created", "navigable_areas_identified"],
                failure_recovery="use_basic_occupancy_grid"
            )
        ]
        
        task_plan = TaskPlan(
            id="perception_mapping_task",
            name="Perception and Mapping",
            description="Demonstrate environment perception and mapping capabilities",
            steps=task_steps,
            requirements=["cameras", "lidar", "mapping_algorithms"],
            constraints=["well_lit_environment"],
            success_criteria=["accurate_environment_map", "object_recognition_accuracy > 80%"],
            fallback_behaviors=["use_predefined_map", "request_human_assistance"],
            estimated_time=23.0
        )
        
        return DemoScenario(
            id="perception_mapping",
            name="Perception & Mapping Milestone Demo",
            description="Demonstrate the robot's ability to perceive and create a map of its environment",
            task_sequence=[task_plan],
            success_criteria=[
                "Environment accurately mapped",
                "Objects correctly identified",
                "Safe navigation paths determined"
            ],
            estimated_duration=2.0,  # 2 minutes
            difficulty_level="intermediate",
            required_equipment=["cameras", "lidar", "mapping_software"],
            safety_considerations=["ensure clear scanning area", "avoid reflective surfaces"]
        )
    
    @staticmethod
    def get_navigation_demo() -> DemoScenario:
        """Demo for Milestone 3: Navigation & Obstacle Avoidance"""
        task_steps = [
            TaskStep(
                id="path_planning",
                description="Plan navigation path to destination",
                action_type="planning",
                parameters={"destination": "waypoint_1"},
                estimated_duration=3.0,
                dependencies=[],
                success_criteria=["valid_path_found"],
                failure_recovery="use_alternative_route"
            ),
            TaskStep(
                id="navigation_execute",
                description="Execute navigation to destination",
                action_type="navigation",
                parameters={"target": "waypoint_1"},
                estimated_duration=15.0,
                dependencies=["path_planning"],
                success_criteria=["destination_reached", "obstacles_avoided"],
                failure_recovery="emergency_stop"
            ),
            TaskStep(
                id="obstacle_detection",
                description="Detect and avoid dynamic obstacles",
                action_type="perception",
                parameters={"monitor": "surroundings"},
                estimated_duration=10.0,
                dependencies=["navigation_execute"],
                success_criteria=["obstacles_detected", "path_adjusted"],
                failure_recovery="stop_and_replan"
            ),
            TaskStep(
                id="return_navigation",
                description="Return to starting position",
                action_type="navigation",
                parameters={"target": "start_position"},
                estimated_duration=15.0,
                dependencies=["obstacle_detection"],
                success_criteria=["start_position_reached"],
                failure_recovery="use_safe_return_protocol"
            )
        ]
        
        task_plan = TaskPlan(
            id="navigation_task",
            name="Navigation and Obstacle Avoidance",
            description="Demonstrate navigation and dynamic obstacle avoidance",
            steps=task_steps,
            requirements=["navigation_stack", "obstacle_detection", "path_planning"],
            constraints=["indoor_environment", "obstacle_course_setup"],
            success_criteria=["navigates_course", "avoids_all_obstacles", "completes_circuit"],
            fallback_behaviors=["reduce_speed", "request_path_clearance"],
            estimated_time=43.0
        )
        
        return DemoScenario(
            id="navigation_avoidance",
            name="Navigation & Obstacle Avoidance Demo",
            description="Demonstrate the robot's ability to navigate to destinations while avoiding obstacles",
            task_sequence=[task_plan],
            success_criteria=[
                "Successfully navigates to destination",
                "Avoids all obstacles",
                "Returns to start position"
            ],
            estimated_duration=3.0,  # 3 minutes
            difficulty_level="advanced",
            required_equipment=["navigation_system", "obstacle_setup"],
            safety_considerations=["clear navigation path", "supervise near obstacles"]
        )
    
    @staticmethod
    def get_manipulation_demo() -> DemoScenario:
        """Demo for Milestone 4: Object Identification & Manipulation"""
        task_steps = [
            TaskStep(
                id="object_scan",
                description="Scan for graspable objects",
                action_type="perception",
                parameters={"scan": "graspable_objects"},
                estimated_duration=5.0,
                dependencies=[],
                success_criteria=["graspable_object_found"],
                failure_recovery="expand_search_area"
            ),
            TaskStep(
                id="approach_object",
                description="Navigate to object location",
                action_type="navigation",
                parameters={"target": "object_location"},
                estimated_duration=8.0,
                dependencies=["object_scan"],
                success_criteria=["at_object_location"],
                failure_recovery="recalculate_approach"
            ),
            TaskStep(
                id="grasp_object",
                description="Grasp identified object",
                action_type="manipulation",
                parameters={"action": "grasp", "object": "target_object"},
                estimated_duration=10.0,
                dependencies=["approach_object"],
                success_criteria=["object_grasped"],
                failure_recovery="try_alternative_grasp"
            ),
            TaskStep(
                id="transport_object",
                description="Transport object to destination",
                action_type="manipulation",
                parameters={"action": "transport", "destination": "delivery_location"},
                estimated_duration=12.0,
                dependencies=["grasp_object"],
                success_criteria=["object_transported"],
                failure_recovery="abort_and_return"
            ),
            TaskStep(
                id="place_object",
                description="Place object at target location",
                action_type="manipulation",
                parameters={"action": "place", "location": "delivery_location"},
                estimated_duration=8.0,
                dependencies=["transport_object"],
                success_criteria=["object_placed"],
                failure_recovery="manual_intervention"
            )
        ]
        
        task_plan = TaskPlan(
            id="manipulation_task",
            name="Object Identification and Manipulation",
            description="Demonstrate object identification and manipulation",
            steps=task_steps,
            requirements=["perception_stack", "manipulation_planning", "grasping_system"],
            constraints=["graspable_object_available", "reachable_workspace"],
            success_criteria=["object_identified", "object_grasped", "object_placed_correctly"],
            fallback_behaviors=["use_alternative_grasp", "request_human_help"],
            estimated_time=43.0
        )
        
        return DemoScenario(
            id="object_manipulation",
            name="Object Identification & Manipulation Demo",
            description="Demonstrate the robot's ability to identify, grasp, and manipulate objects",
            task_sequence=[task_plan],
            success_criteria=[
                "Object correctly identified",
                "Successful grasp achieved",
                "Object placed at target location"
            ],
            estimated_duration=3.0,  # 3 minutes
            difficulty_level="advanced",
            required_equipment=["manipulator_arm", "graspable_objects", "target_location"],
            safety_considerations=["ensure safe manipulation space", "lightweight_objects_only"]
        )
    
    @staticmethod
    def get_integration_demo() -> DemoScenario:
        """Full integration demonstration combining all capabilities"""
        # This would combine elements from all previous demos
        task_steps = [
            TaskStep(
                id="voice_command",
                description="Receive voice command from user",
                action_type="conversation",
                parameters={"mode": "listen"},
                estimated_duration=5.0,
                dependencies=[],
                success_criteria=["command_received"],
                failure_recovery="request_repeat"
            ),
            TaskStep(
                id="environment_scan",
                description="Scan environment to understand context",
                action_type="perception",
                parameters={"mode": "scan"},
                estimated_duration=10.0,
                dependencies=["voice_command"],
                success_criteria=["environment_understood"],
                failure_recovery="use_default_map"
            ),
            TaskStep(
                id="task_planning",
                description="Plan sequence of actions to fulfill command",
                action_type="planning",
                parameters={"plan": "command_implementation"},
                estimated_duration=7.0,
                dependencies=["environment_scan"],
                success_criteria=["valid_plan_created"],
                failure_recovery="simplify_task"
            ),
            TaskStep(
                id="navigation_phase",
                description="Navigate to required location",
                action_type="navigation",
                parameters={"target": "task_location"},
                estimated_duration=15.0,
                dependencies=["task_planning"],
                success_criteria=["at_location", "obstacles_avoided"],
                failure_recovery="reroute"
            ),
            TaskStep(
                id="manipulation_phase",
                description="Perform required manipulation",
                action_type="manipulation",
                parameters={"action": "task_specific"},
                estimated_duration=20.0,
                dependencies=["navigation_phase"],
                success_criteria=["manipulation_completed"],
                failure_recovery="try_alternative_approach"
            ),
            TaskStep(
                id="confirmation",
                description="Confirm task completion to user",
                action_type="conversation",
                parameters={"message": "task_completed_confirmation"},
                estimated_duration=5.0,
                dependencies=["manipulation_phase"],
                success_criteria=["user_confirmed_completion"],
                failure_recovery="provide_detailed_explanation"
            )
        ]
        
        task_plan = TaskPlan(
            id="integration_task",
            name="Full System Integration",
            description="Demonstrate complete system integration with all capabilities",
            steps=task_steps,
            requirements=["all_systems_operational"],
            constraints=["well-prepared_environment"],
            success_criteria=["voice_command_understood", "task_completed", "user_satisfied"],
            fallback_behaviors=["degrade_performance", "request_assistance"],
            estimated_time=62.0
        )
        
        return DemoScenario(
            id="full_integration",
            name="Full System Integration Demo",
            description="Demonstrate complete integration of all system capabilities",
            task_sequence=[task_plan],
            success_criteria=[
                "Voice command correctly interpreted",
                "Environment properly understood",
                "Task successfully completed",
                "User interaction successful"
            ],
            estimated_duration=5.0,  # 5 minutes
            difficulty_level="advanced",
            required_equipment=["all_systems"],
            safety_considerations=["comprehensive_safety_check", "supervised_operation"]
        )


class DemoExecutionManager:
    """
    Manager for executing demonstrations with scheduling and reporting capabilities.
    """
    
    def __init__(self, demo_framework: CapstoneDemoFramework):
        self.demo_framework = demo_framework
        self.schedule = []
        self.active_demo = None
    
    def schedule_demo(self, demo_id: str, start_time: datetime, parameters: Dict[str, Any] = None):
        """Schedule a demo to run at a specific time"""
        scheduled_item = {
            "demo_id": demo_id,
            "start_time": start_time,
            "parameters": parameters or {},
            "status": "scheduled"
        }
        
        self.schedule.append(scheduled_item)
        # Sort schedule by time
        self.schedule.sort(key=lambda x: x["start_time"])
    
    def run_scheduled_demos(self):
        """Run all scheduled demos in chronological order"""
        current_time = datetime.now()
        
        for item in self.schedule:
            if item["start_time"] <= current_time and item["status"] == "scheduled":
                logging.info(f"Running scheduled demo: {item['demo_id']}")
                
                try:
                    result = self.demo_framework.execute_demo(item["demo_id"], item["parameters"])
                    item["status"] = "completed"
                    item["result"] = result
                    logging.info(f"Demo {item['demo_id']} completed with success: {result.success}")
                except Exception as e:
                    item["status"] = "failed"
                    item["error"] = str(e)
                    logging.error(f"Demo {item['demo_id']} failed: {e}")
    
    def get_demo_report(self, demo_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a report on demo execution"""
        if demo_id:
            results = [r for r in self.demo_framework.demo_results if r.demo_id == demo_id]
        else:
            results = self.demo_framework.demo_results
        
        if not results:
            return {"message": "No demo results available"}
        
        # Calculate aggregate metrics
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        avg_duration = sum(r.total_duration for r in results) / total_results
        
        # Calculate performance metrics
        avg_task_completion = sum(
            r.performance_metrics.get("task_completion_rate", 0) for r in results
        ) / total_results if results else 0
        
        return {
            "demo_id": demo_id,
            "total_executions": total_results,
            "successful_executions": successful_results,
            "success_rate": successful_results / total_results if total_results > 0 else 0,
            "average_duration": avg_duration,
            "average_task_completion_rate": avg_task_completion,
            "recent_results": [r.__dict__ for r in results[-5:]]  # Last 5 results
        }
    
    def cancel_scheduled_demo(self, demo_id: str):
        """Cancel a scheduled demo"""
        for item in self.schedule:
            if item["demo_id"] == demo_id and item["status"] == "scheduled":
                item["status"] = "cancelled"
                return True
        return False