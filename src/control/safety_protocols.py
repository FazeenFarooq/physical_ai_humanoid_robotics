"""
Safety protocols for physical robot operation in the Physical AI & Humanoid Robotics Course

This module implements safety measures and protocols to ensure safe operation
of physical robots during course activities.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import threading
import time
import logging

# Configure logging for safety protocols
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Enumeration of safety levels"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


class SafetyZone(Enum):
    """Enumeration of safety zones around the robot"""
    NO_GO = "no_go"  # Completely forbidden
    RESTRICTED = "restricted"  # Limited access
    CAUTION = "caution"  # Requires attention
    FREE = "free"  # Normal operation


class RobotSafetyController:
    """
    Main safety controller for physical robot operation.
    
    The safety controller implements multiple layers of safety protocols:
    - Emergency stop functionality
    - Zone-based safety management
    - Velocity and position limiting
    - Collision avoidance
    - System monitoring
    """
    
    def __init__(self):
        """Initialize the safety controller"""
        self.safety_level = SafetyLevel.SAFE
        self.emergency_stop_activated = False
        self.safety_zones: Dict[str, SafetyZone] = {}
        self.velocity_limits = {
            'linear': {'max': 0.5, 'current': 0.0},  # m/s
            'angular': {'max': 0.5, 'current': 0.0}  # rad/s
        }
        self.position_limits = {
            'x_range': (-5.0, 5.0),
            'y_range': (-5.0, 5.0),
            'z_range': (0.0, 2.0)
        }
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_sensor_readings: Dict[str, Any] = {}
        self.safety_log: List[Dict[str, Any]] = []
        
    def activate_emergency_stop(self) -> bool:
        """Activate the emergency stop, stopping all robot motion"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.emergency_stop_activated = True
        self.safety_level = SafetyLevel.EMERGENCY
        self.log_safety_event("EMERGENCY_STOP", "Emergency stop activated by safety system")
        
        # Stop all robot motion (this would interface with the actual robot control)
        self._stop_robot_motion()
        
        return True
    
    def deactivate_emergency_stop(self) -> bool:
        """Deactivate the emergency stop, allowing robot motion"""
        logger.info("EMERGENCY STOP DEACTIVATED")
        self.emergency_stop_activated = False
        self.safety_level = SafetyLevel.SAFE
        self.log_safety_event("EMERGENCY_STOP", "Emergency stop deactivated")
        
        return True
    
    def _stop_robot_motion(self) -> None:
        """Internal method to stop all robot motion"""
        # This would send stop commands to the robot's actuators
        # In a real implementation, this would interface with the motor controllers
        logger.debug("Stopping all robot motion")
    
    def set_safety_zone(self, zone_id: str, zone_type: SafetyZone, boundary: Dict[str, float]) -> bool:
        """Define a safety zone around the robot"""
        self.safety_zones[zone_id] = zone_type
        # In a real implementation, this would define geometric boundaries
        logger.info(f"Set safety zone {zone_id} as {zone_type.value}")
        return True
    
    def update_sensor_readings(self, sensor_data: Dict[str, Any]) -> None:
        """Update the safety controller with new sensor readings"""
        self.last_sensor_readings.update(sensor_data)
        
        # Perform safety checks based on sensor data
        self._perform_safety_checks()
        
    def _perform_safety_checks(self) -> None:
        """Perform safety checks based on current sensor data and state"""
        # Check for obstacles in safety zones
        obstacles = self.last_sensor_readings.get('obstacles', [])
        
        # Check if any obstacles are in no-go or restricted zones
        for obstacle in obstacles:
            distance = obstacle.get('distance', float('inf'))
            if distance < 0.5:  # Less than 0.5m away
                self._trigger_safety_response(SafetyLevel.WARNING, 
                                            f"Obstacle detected at {distance}m", 
                                            "OBSTACLE_TOO_CLOSE")
            elif distance < 0.2:  # Really close
                self._trigger_safety_response(SafetyLevel.DANGER, 
                                            f"Obstacle dangerously close at {distance}m", 
                                            "OBSTACLE_DANGEROUSLY_CLOSE")
        
        # Check velocity limits
        current_linear_vel = self.last_sensor_readings.get('linear_velocity', 0.0)
        current_angular_vel = self.last_sensor_readings.get('angular_velocity', 0.0)
        
        if abs(current_linear_vel) > self.velocity_limits['linear']['max']:
            self._trigger_safety_response(SafetyLevel.WARNING, 
                                        f"Linear velocity exceeded limit: {current_linear_vel} > {self.velocity_limits['linear']['max']}", 
                                        "VELOCITY_EXCEEDED")
        
        if abs(current_angular_vel) > self.velocity_limits['angular']['max']:
            self._trigger_safety_response(SafetyLevel.WARNING, 
                                        f"Angular velocity exceeded limit: {current_angular_vel} > {self.velocity_limits['angular']['max']}", 
                                        "VELOCITY_EXCEEDED")
        
        # Check position limits
        current_pos = self.last_sensor_readings.get('position', {'x': 0, 'y': 0, 'z': 0})
        x, y, z = current_pos['x'], current_pos['y'], current_pos['z']
        
        x_min, x_max = self.position_limits['x_range']
        y_min, y_max = self.position_limits['y_range']
        z_min, z_max = self.position_limits['z_range']
        
        if not (x_min <= x <= x_max) or not (y_min <= y <= y_max) or not (z_min <= z <= z_max):
            self._trigger_safety_response(SafetyLevel.WARNING, 
                                        f"Position limits exceeded: ({x}, {y}, {z})", 
                                        "POSITION_LIMIT_EXCEEDED")
    
    def _trigger_safety_response(self, level: SafetyLevel, message: str, event_type: str) -> None:
        """Trigger appropriate safety response based on safety level"""
        if level.value > self.safety_level.value:
            self.safety_level = level
            self.log_safety_event(event_type, message)
            
            if level == SafetyLevel.EMERGENCY:
                self.activate_emergency_stop()
            elif level == SafetyLevel.DANGER:
                # Slow down robot significantly
                logger.warning(f"DANGER: {message}")
            elif level == SafetyLevel.WARNING:
                # Log warning but continue operation
                logger.warning(f"WARNING: {message}")
    
    def validate_command(self, cmd_type: str, cmd_params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a robot command against safety constraints"""
        if self.emergency_stop_activated:
            return False, "Emergency stop is activated"
        
        if cmd_type == "move":
            # Check if movement command is safe
            target_pos = cmd_params.get('position', {})
            target_vel = cmd_params.get('velocity', {})
            
            # Check position limits
            x = target_pos.get('x', 0)
            y = target_pos.get('y', 0)
            z = target_pos.get('z', 0)
            
            x_min, x_max = self.position_limits['x_range']
            y_min, y_max = self.position_limits['y_range']
            z_min, z_max = self.position_limits['z_range']
            
            if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                return False, f"Movement would exceed position limits: ({x}, {y}, {z})"
            
            # Check velocity limits
            linear_vel = target_vel.get('linear', 0.0)
            angular_vel = target_vel.get('angular', 0.0)
            
            if abs(linear_vel) > self.velocity_limits['linear']['max']:
                return False, f"Linear velocity {linear_vel} exceeds limit {self.velocity_limits['linear']['max']}"
            
            if abs(angular_vel) > self.velocity_limits['angular']['max']:
                return False, f"Angular velocity {angular_vel} exceeds limit {self.velocity_limits['angular']['max']}"
        
        return True, "Command is safe"
    
    def start_monitoring(self) -> None:
        """Start the safety monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Safety monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the safety monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread"""
        while self.monitoring_active:
            # Perform periodic safety checks
            try:
                self._perform_system_health_check()
            except Exception as e:
                logger.error(f"Error in safety monitoring: {e}")
            
            # Sleep for a short period
            time.sleep(0.1)
    
    def _perform_system_health_check(self) -> None:
        """Perform system health checks"""
        # Check system temperatures
        cpu_temp = self.last_sensor_readings.get('cpu_temp', 0)
        gpu_temp = self.last_sensor_readings.get('gpu_temp', 0)
        
        if cpu_temp > 85 or gpu_temp > 95:
            self._trigger_safety_response(SafetyLevel.WARNING, 
                                        f"High temperature: CPU={cpu_temp}°C, GPU={gpu_temp}°C", 
                                        "HIGH_TEMPERATURE")
    
    def log_safety_event(self, event_type: str, description: str) -> None:
        """Log a safety event"""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'description': description,
            'safety_level': self.safety_level.value
        }
        self.safety_log.append(event)
        
        # Keep only the last 1000 events to prevent memory issues
        if len(self.safety_log) > 1000:
            self.safety_log = self.safety_log[-1000:]
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            'safety_level': self.safety_level.value,
            'emergency_stop': self.emergency_stop_activated,
            'velocity_limits': self.velocity_limits,
            'position_limits': self.position_limits,
            'safety_zones': {k: v.value for k, v in self.safety_zones.items()},
            'last_safety_event': self.safety_log[-1] if self.safety_log else None
        }


def initialize_safety_system() -> RobotSafetyController:
    """
    Initialize the safety system for the Physical AI course.
    
    This function creates and configures the safety controller with
    appropriate defaults for the course environment.
    """
    controller = RobotSafetyController()
    
    # Set appropriate velocity limits for course environment
    controller.velocity_limits['linear']['max'] = 0.5  # m/s
    controller.velocity_limits['angular']['max'] = 0.5  # rad/s
    
    # Set position limits for course environment (if applicable)
    controller.position_limits['x_range'] = (-5.0, 5.0)
    controller.position_limits['y_range'] = (-5.0, 5.0)
    controller.position_limits['z_range'] = (0.0, 2.0)
    
    # Start monitoring
    controller.start_monitoring()
    
    logger.info("Safety system initialized and monitoring started")
    return controller