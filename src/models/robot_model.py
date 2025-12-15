"""
Robot model entity model for the Physical AI & Humanoid Robotics course.
This model represents different robot platforms and their specifications.
Based on the data model specification in /specs/001-physical-ai-course/data-model.md
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class RobotActuatorType(Enum):
    SERVO_MOTOR = "servo_motor"
    STEPPER_MOTOR = "stepper_motor"
    HYDRAULIC_ACTUATOR = "hydraulic_actuator"
    PNEUMATIC_ACTUATOR = "pneumatic_actuator"


class SensorType(Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    IMU = "imu"
    GYROSCOPE = "gyroscope"
    ACCELEROMETER = "accelerometer"
    MICROPHONE = "microphone"
    GPS = "gps"


@dataclass
class RobotActuator:
    """Represents an actuator in the robot model"""
    id: str
    type: RobotActuatorType
    name: str
    joint_name: str
    min_position: float
    max_position: float
    max_velocity: float
    max_effort: float


@dataclass
class RobotSensor:
    """Represents a sensor in the robot model"""
    id: str
    type: SensorType
    name: str
    mount_position: List[float]  # [x, y, z] coordinates
    mount_orientation: List[float]  # [roll, pitch, yaw] in radians
    specifications: Dict[str, any]  # Specific sensor specs


@dataclass
class RobotModel:
    """
    RobotModel entity representing physical robot specifications
    and capabilities for use in simulation and real-world deployment.
    """
    id: str
    name: str
    description: str
    kinematic_model: str  # Path to URDF/XACRO file
    actuators: List[RobotActuator]  # Joint motors and their specifications
    sensors: List[RobotSensor]  # Sensor configurations
    computational_resources: Dict[str, any]  # Processing power, memory
    battery_life: float  # Operational time in hours
    workspace: Dict[str, float]  # Reachable area and volume {"x_range": [min, max], ...}
    max_linear_velocity: float  # Maximum linear velocity in m/s
    max_angular_velocity: float  # Maximum angular velocity in rad/s
    payload_capacity: float  # Maximum payload in kg
    dimensions: Dict[str, float]  # Physical dimensions {"length":, "width":, "height":}
    
    def get_actuator_by_joint_name(self, joint_name: str) -> Optional[RobotActuator]:
        """Get an actuator by its joint name"""
        for actuator in self.actuators:
            if actuator.joint_name == joint_name:
                return actuator
        return None
    
    def get_sensor_by_type(self, sensor_type: SensorType) -> List[RobotSensor]:
        """Get all sensors of a specific type"""
        return [sensor for sensor in self.sensors if sensor.type == sensor_type]
    
    def get_actuator_count(self) -> int:
        """Get the total number of actuators"""
        return len(self.actuators)
    
    def get_sensor_count(self) -> int:
        """Get the total number of sensors"""
        return len(self.sensors)
    
    def is_capable_of_task(self, required_actuators: List[str], required_sensors: List[SensorType]) -> bool:
        """
        Check if the robot model is capable of performing a task
        based on required actuators and sensors
        """
        # Check for required actuators
        actuator_names = [act.joint_name for act in self.actuators]
        for req_act in required_actuators:
            if req_act not in actuator_names:
                return False
        
        # Check for required sensor types
        sensor_types = [sensor.type for sensor in self.sensors]
        for req_sensor in required_sensors:
            if req_sensor not in sensor_types:
                return False
        
        return True