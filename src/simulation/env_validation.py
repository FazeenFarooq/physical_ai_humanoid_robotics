"""
Environment Validation Tools for the Physical AI & Humanoid Robotics Course.

This module provides tools for validating simulation environments, ensuring they meet 
the requirements for sim-to-real transfer and are suitable for the course objectives.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path


class ValidationResult(Enum):
    """Enumeration for validation result states."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationIssue:
    """Represents an issue found during validation."""
    category: str  # e.g., "physics", "geometry", "sensors"
    severity: ValidationResult
    description: str
    location: Optional[str] = None  # Specific object or area in environment
    suggested_fix: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for validation parameters."""
    physics_accuracy_threshold: float = 0.05  # 5% error tolerance
    sim_to_real_transfer_threshold: float = 0.15  # 15% performance degradation allowed
    minimum_object_size: float = 0.01  # 1cm minimum for reliable simulation
    maximum_object_size: float = 10.0  # 10m maximum for computational feasibility
    minimum_light_intensity: float = 50  # Minimum for vision tasks
    maximum_light_intensity: float = 2000  # Maximum to avoid saturation
    gravity_tolerance: float = 0.1  # Tolerance for gravity setting


class SimulationEnvironmentValidator:
    """Validator for simulation environments."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.issues: List[ValidationIssue] = []
    
    def validate_environment(
        self, 
        env_description: Dict[str, Any], 
        robot_config: Dict[str, Any] = None
    ) -> Tuple[ValidationResult, List[ValidationIssue]]:
        """
        Validate an entire simulation environment.
        
        Args:
            env_description: Dictionary describing the environment
            robot_config: Optional robot configuration to check against environment
            
        Returns:
            Tuple of overall result and list of issues found
        """
        self.issues = []  # Reset issues list
        
        # Validate physics parameters
        self._validate_physics(env_description.get('physics', {}))
        
        # Validate geometry and objects
        objects = env_description.get('objects', [])
        self._validate_geometry(objects)
        
        # Validate lighting
        lighting = env_description.get('lighting', {})
        self._validate_lighting(lighting)
        
        # Validate sensors (if robot config is provided)
        if robot_config:
            self._validate_sensors(robot_config)
        
        # Validate sim-to-real transfer parameters
        self._validate_sim_to_real_params(env_description)
        
        # Determine overall result
        if any(issue.severity == ValidationResult.FAIL for issue in self.issues):
            overall_result = ValidationResult.FAIL
        elif any(issue.severity == ValidationResult.WARNING for issue in self.issues):
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.PASS
            
        return overall_result, self.issues[:]
    
    def _validate_physics(self, physics_config: Dict[str, Any]):
        """Validate physics parameters."""
        gravity = physics_config.get('gravity', [-9.81, 0, 0])
        gravity_magnitude = math.sqrt(sum(c*c for c in gravity))
        
        if abs(gravity_magnitude - 9.81) > self.config.gravity_tolerance:
            self.issues.append(ValidationIssue(
                category="physics",
                severity=ValidationResult.FAIL,
                description=f"Gravity magnitude {gravity_magnitude:.3f} differs significantly from Earth's gravity (9.81 m/s²)",
                suggested_fix="Set gravity magnitude to approximately 9.81 m/s² for realistic simulation"
            ))
        
        time_step = physics_config.get('time_step', 0.001)
        if time_step > 0.01:  # Very large time step
            self.issues.append(ValidationIssue(
                category="physics",
                severity=ValidationResult.FAIL,
                description=f"Time step {time_step}s is too large for stable simulation",
                suggested_fix="Use a time step of 0.001s or smaller for stable physics simulation"
            ))
        elif time_step > 0.005:  # Large time step
            self.issues.append(ValidationIssue(
                category="physics",
                severity=ValidationResult.WARNING,
                description=f"Time step {time_step}s may cause stability issues",
                suggested_fix="Consider using a smaller time step (e.g., 0.001s) for better stability"
            ))
    
    def _validate_geometry(self, objects: List[Dict[str, Any]]):
        """Validate geometric properties of objects in the environment."""
        for i, obj in enumerate(objects):
            name = obj.get('name', f'object_{i}')
            
            # Check object size
            size = obj.get('size', [1.0, 1.0, 1.0])
            max_dimension = max(size)
            min_dimension = min(size)
            
            if max_dimension > self.config.maximum_object_size:
                self.issues.append(ValidationIssue(
                    category="geometry",
                    severity=ValidationResult.FAIL,
                    description=f"Object '{name}' has dimension {max_dimension}m which exceeds maximum of {self.config.maximum_object_size}m",
                    location=name,
                    suggested_fix=f"Reduce object size to be under {self.config.maximum_object_size}m in largest dimension"
                ))
            
            if min_dimension < self.config.minimum_object_size:
                self.issues.append(ValidationIssue(
                    category="geometry",
                    severity=ValidationResult.WARNING,
                    description=f"Object '{name}' has dimension {min_dimension}m which may be too small for reliable simulation",
                    location=name,
                    suggested_fix=f"Consider increasing object size to be at least {self.config.minimum_object_size}m in smallest dimension"
                ))
            
            # Check for non-physical shapes or properties
            mass = obj.get('mass', 1.0)
            if mass <= 0:
                self.issues.append(ValidationIssue(
                    category="geometry",
                    severity=ValidationResult.FAIL,
                    description=f"Object '{name}' has non-positive mass {mass}",
                    location=name,
                    suggested_fix="Set a positive mass value for the object"
                ))
    
    def _validate_lighting(self, lighting_config: Dict[str, Any]):
        """Validate lighting configuration."""
        lights = lighting_config.get('lights', [])
        
        for i, light in enumerate(lights):
            intensity = light.get('intensity', 100)
            name = light.get('name', f'light_{i}')
            
            if intensity < self.config.minimum_light_intensity:
                self.issues.append(ValidationIssue(
                    category="lighting",
                    severity=ValidationResult.WARNING,
                    description=f"Light '{name}' intensity {intensity} is too low for good vision in simulation",
                    location=name,
                    suggested_fix=f"Increase light intensity to at least {self.config.minimum_light_intensity}"
                ))
            elif intensity > self.config.maximum_light_intensity:
                self.issues.append(ValidationIssue(
                    category="lighting",
                    severity=ValidationResult.WARNING,
                    description=f"Light '{name}' intensity {intensity} may cause saturation in vision sensors",
                    location=name,
                    suggested_fix=f"Reduce light intensity to under {self.config.maximum_light_intensity} to avoid saturation"
                ))
    
    def _validate_sensors(self, robot_config: Dict[str, Any]):
        """Validate sensor configuration of the robot."""
        sensors = robot_config.get('sensors', [])
        
        for i, sensor in enumerate(sensors):
            sensor_type = sensor.get('type', 'unknown')
            name = sensor.get('name', f'sensor_{i}')
            
            if sensor_type == 'camera':
                fov = sensor.get('fov', 60)
                if fov < 30:
                    self.issues.append(ValidationIssue(
                        category="sensors",
                        severity=ValidationResult.WARNING,
                        description=f"Camera '{name}' has narrow field of view ({fov}°), may miss important information",
                        location=name,
                        suggested_fix="Consider increasing FOV to 60°-90° for better environmental awareness"
                    ))
                elif fov > 120:
                    self.issues.append(ValidationIssue(
                        category="sensors",
                        severity=ValidationResult.WARNING,
                        description=f"Camera '{name}' has wide field of view ({fov}°), may introduce distortion",
                        location=name,
                        suggested_fix="Consider decreasing FOV to under 120° to reduce distortion"
                    ))
            
            elif sensor_type == 'lidar':
                resolution = sensor.get('resolution', 1.0)
                if resolution > 5.0:  # Low resolution
                    self.issues.append(ValidationIssue(
                        category="sensors",
                        severity=ValidationResult.WARNING,
                        description=f"Lidar '{name}' has low angular resolution ({resolution}°), may miss obstacles",
                        location=name,
                        suggested_fix="Increase angular resolution to under 1.0° for better obstacle detection"
                    ))
    
    def _validate_sim_to_real_params(self, env_description: Dict[str, Any]):
        """Validate parameters that affect sim-to-real transfer."""
        # Check if the environment provides the necessary features for sim-to-real transfer
        features = env_description.get('features', [])
        
        # Check for domain randomization features
        if not any('domain_randomization' in feature.lower() for feature in features):
            self.issues.append(ValidationIssue(
                category="sim_to_real",
                severity=ValidationResult.WARNING,
                description="Environment lacks domain randomization features for robust sim-to-real transfer",
                suggested_fix="Consider adding domain randomization capabilities for lighting, textures, and physics parameters"
            ))
    
    def check_environment_suitability(
        self, 
        env_description: Dict[str, Any], 
        task_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if an environment is suitable for specific tasks.
        
        Args:
            env_description: Description of the environment
            task_requirements: Requirements for the task to be performed
            
        Returns:
            Dictionary with suitability assessment
        """
        result = {
            'suitable': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check if environment meets navigation requirements
        if task_requirements.get('requires_navigation', False):
            obstacles = env_description.get('obstacles', [])
            if not obstacles:
                result['recommendations'].append(
                    "Consider adding obstacles for realistic navigation training"
                )
        
        # Check if environment meets manipulation requirements
        if task_requirements.get('requires_manipulation', False):
            objects = env_description.get('objects', [])
            small_objects = [obj for obj in objects if obj.get('size', [1,1,1])[0] < 0.2]
            if not small_objects:
                result['issues'].append(
                    "Environment lacks small objects suitable for manipulation tasks"
                )
                result['suitable'] = False
        
        # Check if environment meets perception requirements
        if task_requirements.get('requires_perception', False):
            lighting = env_description.get('lighting', {})
            lights = lighting.get('lights', [])
            if not lights:
                result['issues'].append(
                    "Environment has no lighting, unsuitable for vision-based perception"
                )
                result['suitable'] = False
        
        return result
    
    def generate_validation_report(
        self, 
        env_description: Dict[str, Any], 
        robot_config: Dict[str, Any] = None
    ) -> str:
        """
        Generate a detailed validation report for an environment.
        
        Args:
            env_description: Description of the environment
            robot_config: Optional robot configuration
            
        Returns:
            Formatted validation report as a string
        """
        overall_result, issues = self.validate_environment(env_description, robot_config)
        
        report = ["Environment Validation Report", "="*25, ""]
        report.append(f"Overall Result: {overall_result.value}")
        report.append("")
        
        # Group issues by category
        issues_by_category = {}
        for issue in issues:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
        
        for category, category_issues in issues_by_category.items():
            report.append(f"Category: {category}")
            report.append("-" * (len(category) + 10))
            
            for issue in category_issues:
                report.append(f"  {issue.severity.value}: {issue.description}")
                if issue.location:
                    report.append(f"    Location: {issue.location}")
                if issue.suggested_fix:
                    report.append(f"    Suggestion: {issue.suggested_fix}")
            report.append("")
        
        return "\n".join(report)


class PhysicsAccuracyValidator:
    """Validator to check physics accuracy between simulation and reality."""
    
    def __init__(self, tolerance: float = 0.05):  # 5% tolerance
        self.tolerance = tolerance
    
    def validate_physics_model(
        self, 
        sim_data: Dict[str, List[float]], 
        real_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Validate physics model by comparing simulation and real data.
        
        Args:
            sim_data: Simulation data (e.g., {'position': [x1, x2, ...], 'velocity': [v1, v2, ...]})
            real_data: Real-world data for comparison
            
        Returns:
            Dictionary with error metrics
        """
        errors = {}
        
        for key in sim_data.keys():
            if key in real_data:
                sim_values = np.array(sim_data[key])
                real_values = np.array(real_data[key])
                
                # Ensure arrays are the same length
                min_len = min(len(sim_values), len(real_values))
                sim_values = sim_values[:min_len]
                real_values = real_values[:min_len]
                
                # Calculate mean absolute error
                mae = np.mean(np.abs(sim_values - real_values))
                
                # Calculate root mean square error
                rmse = np.sqrt(np.mean((sim_values - real_values)**2))
                
                # Calculate mean absolute percentage error
                mape = np.mean(np.abs((sim_values - real_values) / (np.abs(real_values) + 1e-8))) * 100
                
                errors[key] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'within_tolerance': mape <= self.tolerance * 100
                }
        
        return errors
    
    def validate_trajectory_accuracy(
        self, 
        sim_trajectory: List[Tuple[float, float, float]], 
        real_trajectory: List[Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """
        Validate trajectory accuracy between simulation and real movement.
        
        Args:
            sim_trajectory: List of (x, y, theta) positions from simulation
            real_trajectory: List of (x, y, theta) positions from real robot
            
        Returns:
            Dictionary with trajectory accuracy metrics
        """
        if len(sim_trajectory) != len(real_trajectory):
            raise ValueError("Trajectory lengths must match for comparison")
        
        sim_array = np.array(sim_trajectory)
        real_array = np.array(real_trajectory)
        
        # Calculate position errors
        position_errors = np.linalg.norm(sim_array[:, :2] - real_array[:, :2], axis=1)
        avg_position_error = np.mean(position_errors)
        max_position_error = np.max(position_errors)
        
        # Calculate orientation errors
        orientation_errors = np.abs(sim_array[:, 2] - real_array[:, 2])
        avg_orientation_error = np.mean(orientation_errors)
        max_orientation_error = np.max(orientation_errors)
        
        return {
            'avg_position_error': avg_position_error,
            'max_position_error': max_position_error,
            'avg_orientation_error': avg_orientation_error,
            'max_orientation_error': max_orientation_error,
            'within_tolerance': avg_position_error <= self.tolerance
        }


def example_usage():
    """Example of how to use the validation tools."""
    # Create a validator instance
    validator = SimulationEnvironmentValidator()
    
    # Example environment description
    env_desc = {
        'physics': {
            'gravity': [-9.81, 0, 0],
            'time_step': 0.001
        },
        'objects': [
            {'name': 'table', 'size': [1.0, 0.8, 0.8], 'mass': 10.0},
            {'name': 'box', 'size': [0.2, 0.2, 0.2], 'mass': 1.0}
        ],
        'lighting': {
            'lights': [
                {'name': 'main_light', 'intensity': 500}
            ]
        },
        'features': ['domain_randomization']
    }
    
    # Example robot configuration
    robot_config = {
        'sensors': [
            {'type': 'camera', 'name': 'rgb_camera', 'fov': 60},
            {'type': 'lidar', 'name': 'lidar', 'resolution': 1.0}
        ]
    }
    
    # Validate the environment
    result, issues = validator.validate_environment(env_desc, robot_config)
    
    # Print the report
    print("Validation Result:", result.value)
    print("\nDetailed Report:")
    print(validator.generate_validation_report(env_desc, robot_config))
    
    # Check environment suitability for specific tasks
    task_reqs = {
        'requires_navigation': True,
        'requires_manipulation': True,
        'requires_perception': True
    }
    
    suitability = validator.check_environment_suitability(env_desc, task_reqs)
    print("Environment Suitability:", json.dumps(suitability, indent=2))


if __name__ == "__main__":
    example_usage()