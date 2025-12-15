"""
Robot environment entity model for the Physical AI & Humanoid Robotics course.
This model represents different environments where robots can operate.
Based on the data model specification in /specs/001-physical-ai-course/data-model.md
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class EnvironmentType(Enum):
    SIMULATION_GAZEBO = "simulation_gazebo"
    SIMULATION_ISAAC = "simulation_isaac"
    SIMULATION_UNITY = "simulation_unity"
    PHYSICAL = "physical"


@dataclass
class RobotEnvironment:
    """
    RobotEnvironment entity representing different operational environments 
    for robots, including simulation and physical spaces.
    """
    id: str
    name: str
    description: str
    type: EnvironmentType
    models: List[str]  # List of robot model IDs supported in this environment
    obstacles: List[Dict[str, any]]  # Static and dynamic objects in the environment
    sensors: List[str]  # Sensor configurations available
    physics_parameters: Dict[str, float]  # Physics engine settings
    tasks: List[str]  # List of tasks that can be performed in this environment
    location: Optional[str] = None  # Physical location if applicable
    
    def is_suitable_for_robot(self, robot_model_id: str) -> bool:
        """Check if a robot model is supported in this environment"""
        return robot_model_id in self.models
    
    def get_sensor_config(self, sensor_type: str) -> Optional[Dict[str, any]]:
        """Get configuration for a specific sensor type in this environment"""
        # This would typically query the sensors list to find configuration
        # For now, returning a basic example
        for sensor in self.sensors:
            if sensor_type in sensor:
                return sensor
        return None
    
    def get_available_tasks(self) -> List[str]:
        """Get list of tasks that can be performed in this environment"""
        return self.tasks[:]
    
    def add_task(self, task_id: str):
        """Add a new task to the environment's available tasks"""
        if task_id not in self.tasks:
            self.tasks.append(task_id)
    
    def remove_task(self, task_id: str):
        """Remove a task from the environment's available tasks"""
        if task_id in self.tasks:
            self.tasks.remove(task_id)