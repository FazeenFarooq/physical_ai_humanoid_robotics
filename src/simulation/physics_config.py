"""
Physics Parameter Configuration System for the Physical AI & Humanoid Robotics Course.

This module provides a configuration system for physics parameters across different 
simulation environments (Gazebo, Isaac Sim, Unity) and for real robot deployment.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class PhysicsParameter:
    """Represents a single physics parameter with metadata."""
    name: str
    value: Union[float, int, bool, str]
    description: str
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    unit: str = ""


@dataclass
class SimulationConfig:
    """Configuration for simulation physics parameters."""
    name: str
    gravity: float = -9.81  # m/s^2
    time_step: float = 0.001  # seconds
    solver_iterations: int = 50
    solver_velocity_iterations: int = 50
    contact_surface_layer: float = 0.001
    contact_max_correcting_vel: float = 100.0
    friction_pyramid_sides: int = 4
    collision_margin: float = 0.001
    sleep_threshold: float = 0.005
    contact_damping: float = 0.01
    contact_stiffness: float = 100000.0


@dataclass
class MaterialProperties:
    """Material properties for physics simulation."""
    name: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.2  # bounciness
    density: float = 1000.0  # kg/m^3
    youngs_modulus: float = 2e9  # Pa (for soft bodies)
    damping_coefficient: float = 0.1


@dataclass
class RobotPhysicsConfig:
    """Physics configuration for robot components."""
    robot_name: str
    base_mass: float = 10.0
    link_masses: Dict[str, float] = None
    joint_damping: Dict[str, float] = None
    joint_friction: Dict[str, float] = None
    max_joint_velocity: float = 10.0
    max_joint_effort: float = 100.0
    contact_parameters: Dict[str, float] = None
    
    def __post_init__(self):
        if self.link_masses is None:
            self.link_masses = {}
        if self.joint_damping is None:
            self.joint_damping = {}
        if self.joint_friction is None:
            self.joint_friction = {}
        if self.contact_parameters is None:
            self.contact_parameters = {}


class PhysicsConfigurationManager:
    """Manages physics configurations for different environments and scenarios."""
    
    def __init__(self):
        self.simulation_configs: Dict[str, SimulationConfig] = {}
        self.material_properties: Dict[str, MaterialProperties] = {}
        self.robot_configs: Dict[str, RobotPhysicsConfig] = {}
        self.environment_configs: Dict[str, Dict] = {}
    
    def add_simulation_config(self, config: SimulationConfig):
        """Add or update a simulation configuration."""
        self.simulation_configs[config.name] = config
    
    def add_material_properties(self, material: MaterialProperties):
        """Add or update material properties."""
        self.material_properties[material.name] = material
    
    def add_robot_config(self, config: RobotPhysicsConfig):
        """Add or update robot physics configuration."""
        self.robot_configs[config.robot_name] = config
    
    def get_simulation_config(self, name: str) -> Optional[SimulationConfig]:
        """Retrieve a simulation configuration by name."""
        return self.simulation_configs.get(name)
    
    def get_material_properties(self, name: str) -> Optional[MaterialProperties]:
        """Retrieve material properties by name."""
        return self.material_properties.get(name)
    
    def get_robot_config(self, name: str) -> Optional[RobotPhysicsConfig]:
        """Retrieve robot physics configuration by name."""
        return self.robot_configs.get(name)
    
    def save_config(self, config_name: str, filepath: str):
        """Save a configuration to a file."""
        config_data = {}
        
        if config_name in self.simulation_configs:
            config_data = {
                'type': 'simulation',
                'data': asdict(self.simulation_configs[config_name])
            }
        elif config_name in self.material_properties:
            config_data = {
                'type': 'material',
                'data': asdict(self.material_properties[config_name])
            }
        elif config_name in self.robot_configs:
            config_data = {
                'type': 'robot',
                'data': asdict(self.robot_configs[config_name])
            }
        else:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load a configuration from a file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        config_type = config_data.get('type')
        data = config_data.get('data', {})
        
        if config_type == 'simulation':
            config = SimulationConfig(**data)
            self.simulation_configs[config.name] = config
        elif config_type == 'material':
            config = MaterialProperties(**data)
            self.material_properties[config.name] = config
        elif config_type == 'robot':
            config = RobotPhysicsConfig(**data)
            self.robot_configs[config.robot_name] = config
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")
        
        return config
    
    def export_for_gazebo(self, config_name: str, output_path: str):
        """Export physics configuration in Gazebo-compatible format."""
        # Create the world file content
        if config_name in self.simulation_configs:
            sim_config = self.simulation_configs[config_name]
            
            gazebo_config = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="{config_name}">
    <physics type="ode">
      <gravity>{sim_config.gravity} 0 {sim_config.gravity}</gravity>
      <max_step_size>{sim_config.time_step}</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>{1.0/sim_config.time_step}</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>{sim_config.solver_iterations}</iters>
          <sor>{1.3}</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>{sim_config.contact_max_correcting_vel}</contact_max_correcting_vel>
          <contact_surface_layer>{sim_config.contact_surface_layer}</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
"""
            
            with open(output_path, 'w') as f:
                f.write(gazebo_config)
    
    def export_for_unity(self, config_name: str, output_path: str):
        """Export physics configuration in Unity-compatible format."""
        # Create a JSON configuration for Unity
        if config_name in self.simulation_configs:
            sim_config = self.simulation_configs[config_name]
            
            unity_config = {
                "gravity": {
                    "x": 0,
                    "y": sim_config.gravity,
                    "z": 0
                },
                "timeStep": sim_config.time_step,
                "solverIterations": sim_config.solver_iterations,
                "solverVelocityIterations": sim_config.solver_velocity_iterations
            }
            
            with open(output_path, 'w') as f:
                json.dump(unity_config, f, indent=2)
    
    def validate_config(self, config_name: str) -> List[str]:
        """Validate a configuration and return a list of issues."""
        issues = []
        
        if config_name in self.simulation_configs:
            config = self.simulation_configs[config_name]
            if config.time_step <= 0:
                issues.append("time_step must be positive")
            if config.solver_iterations <= 0:
                issues.append("solver_iterations must be positive")
            if config.gravity > 0:
                issues.append("gravity should typically be negative in simulation")
        elif config_name in self.material_properties:
            config = self.material_properties[config_name]
            if config.static_friction < 0:
                issues.append("static_friction should be non-negative")
            if config.restitution < 0 or config.restitution > 1:
                issues.append("restitution should be between 0 and 1")
        elif config_name in self.robot_configs:
            config = self.robot_configs[config_name]
            for joint, damping in config.joint_damping.items():
                if damping < 0:
                    issues.append(f"joint damping for {joint} should be non-negative")
        
        return issues
    
    def get_parameter_range(self, parameter_name: str) -> Dict[str, Union[float, int]]:
        """Get acceptable range for a physics parameter."""
        ranges = {
            "gravity": {"min": -20, "max": 0, "default": -9.81},
            "time_step": {"min": 0.0001, "max": 0.1, "default": 0.001},
            "solver_iterations": {"min": 1, "max": 500, "default": 50},
            "static_friction": {"min": 0, "max": 10, "default": 0.5},
            "restitution": {"min": 0, "max": 1, "default": 0.2}
        }
        
        return ranges.get(parameter_name, {"min": float('-inf'), "max": float('inf'), "default": 0})


def create_default_configs() -> PhysicsConfigurationManager:
    """Create a configuration manager with default physics configurations."""
    manager = PhysicsConfigurationManager()
    
    # Add default simulation configurations
    default_sim = SimulationConfig(
        name="default_simulation",
        gravity=-9.81,
        time_step=0.001,
        solver_iterations=50,
        solver_velocity_iterations=50,
        contact_surface_layer=0.001,
        contact_max_correcting_vel=100.0
    )
    manager.add_simulation_config(default_sim)
    
    # Add high-fidelity simulation config
    high_fidelity_sim = SimulationConfig(
        name="high_fidelity_simulation",
        gravity=-9.81,
        time_step=0.0005,
        solver_iterations=100,
        solver_velocity_iterations=100,
        contact_surface_layer=0.0005,
        contact_max_correcting_vel=100.0
    )
    manager.add_simulation_config(high_fidelity_sim)
    
    # Add default material properties
    default_material = MaterialProperties(
        name="default_material",
        static_friction=0.5,
        dynamic_friction=0.4,
        restitution=0.2,
        density=1000.0
    )
    manager.add_material_properties(default_material)
    
    # Add rubber material
    rubber = MaterialProperties(
        name="rubber",
        static_friction=0.9,
        dynamic_friction=0.8,
        restitution=0.7,
        density=1100.0
    )
    manager.add_material_properties(rubber)
    
    # Add metal material
    metal = MaterialProperties(
        name="metal",
        static_friction=0.7,
        dynamic_friction=0.6,
        restitution=0.1,
        density=7800.0
    )
    manager.add_material_properties(metal)
    
    # Add default robot config
    default_robot = RobotPhysicsConfig(
        robot_name="basic_humanoid",
        base_mass=10.0,
        link_masses={
            "torso": 5.0,
            "head": 1.0,
            "left_upper_arm": 0.8,
            "left_lower_arm": 0.6,
            "right_upper_arm": 0.8,
            "right_lower_arm": 0.6,
            "left_thigh": 1.5,
            "left_shin": 1.0,
            "right_thigh": 1.5,
            "right_shin": 1.0
        },
        joint_damping={
            "neck_joint": 0.5,
            "left_shoulder_joint": 1.0,
            "left_elbow_joint": 0.8,
            "right_shoulder_joint": 1.0,
            "right_elbow_joint": 0.8,
            "left_hip_joint": 1.5,
            "left_knee_joint": 1.2,
            "right_hip_joint": 1.5,
            "right_knee_joint": 1.2
        },
        max_joint_velocity=5.0,
        max_joint_effort=50.0
    )
    manager.add_robot_config(default_robot)
    
    return manager


if __name__ == "__main__":
    # Example usage
    config_manager = create_default_configs()
    
    # Validate a configuration
    issues = config_manager.validate_config("default_simulation")
    if issues:
        print(f"Validation issues found: {issues}")
    else:
        print("Configuration is valid!")
    
    # Save a configuration
    config_manager.save_config("default_simulation", "default_sim_config.json")
    print("Configuration saved to default_sim_config.json")
    
    # Export for Gazebo
    config_manager.export_for_gazebo("default_simulation", "gazebo_config.world")
    print("Gazebo configuration exported to gazebo_config.world")
    
    # Export for Unity
    config_manager.export_for_unity("default_simulation", "unity_config.json")
    print("Unity configuration exported to unity_config.json")
    
    # Print parameter ranges
    print("\nPhysics Parameter Ranges:")
    print(config_manager.get_parameter_range("gravity"))
    print(config_manager.get_parameter_range("time_step"))